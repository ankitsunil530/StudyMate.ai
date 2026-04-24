import os
import requests
import certifi
import uuid
import tempfile
import json
import re
from datetime import datetime, timezone
from datetime import timedelta
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from auth.routes import auth_bp, init_auth_routes
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import google.generativeai as genai
from pdf_pipeline.parser import PDFParser
from flask_bcrypt import Bcrypt

from flask_jwt_extended import JWTManager
from bson.errors import InvalidId
from rag.embedding import get_embedding
from rag.vector_store import VectorStore
from flask_jwt_extended import (
    JWTManager,
    get_jwt_identity,
    jwt_required,
    verify_jwt_in_request,
)
# --------------------------------------------------
# Load ENV
# --------------------------------------------------
load_dotenv()
app = Flask(__name__)

# CORS FIX: Proper configuration for credentials
CORS(
    app,
    origins=["*"],  # ya specific vercel URL baad me
    supports_credentials=True,
)

# JWT Config
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-prod")
app.config["JWT_HEADER_NAME"] = "Authorization"
app.config["JWT_HEADER_TYPE"] = "Bearer"
# Longer dev-friendly sessions; adjust for prod as needed.
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --------------------------------------------------
# MongoDB Atlas
# --------------------------------------------------
client = MongoClient(
    os.getenv("MONGO_URI"),
    tls=True,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    retryWrites=True
)
db = client["study"]

# Initialize and register auth routes (pass bcrypt & jwt)
init_auth_routes(db, bcrypt, jwt)
app.register_blueprint(auth_bp, url_prefix='/api/auth')

# Handle preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return "", 204

# --------------------------------------------------
# PDF Parser
# --------------------------------------------------
pdf_parser = PDFParser()

# --------------------------------------------------
# Cloudinary Config
# --------------------------------------------------
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# --------------------------------------------------
# Gemini Config
# --------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _strip_markdown_code_fences(text):
    if not text:
        return ""
    text = text.strip()
    # Remove opening fence like ``` or ```json
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _extract_first_json_block(text):
    """
    Best-effort extraction for model outputs that include extra text around JSON.
    Tries to isolate the first {...} or [...] block.
    """
    if not text:
        return ""
    text = text.strip()
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return text

    if obj_start == -1:
        start = arr_start
        end = text.rfind("]")
    elif arr_start == -1:
        start = obj_start
        end = text.rfind("}")
    else:
        start = min(obj_start, arr_start)
        end = text.rfind("}" if obj_start < arr_start else "]")

    if end == -1 or end <= start:
        return text[start:]
    return text[start : end + 1].strip()

def _utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _get_optional_user_id():
    try:
        verify_jwt_in_request(optional=True)
        return get_jwt_identity()
    except Exception:
        return None

def _serialize_conversation(doc):
    if not doc:
        return None
    return {
        "id": str(doc.get("_id")),
        "userId": doc.get("userId"),
        "pdfId": str(doc.get("pdfId")) if doc.get("pdfId") else None,
        "pdfFileName": doc.get("pdfFileName", ""),
        "title": doc.get("title", ""),
        "createdAt": doc.get("createdAt"),
        "updatedAt": doc.get("updatedAt"),
        "lastPageNo": doc.get("lastPageNo"),
        "messages": doc.get("messages", []),
    }

# --------------------------------------------------
# Prompt Builder
# --------------------------------------------------
def build_student_prompt(page_text, language):
    return f"""
TASK:
- Summarize and explain the content of this page for a student.
- The explanation should be very simple and easy to understand.
- Use real-life examples wherever necessary.
- Explain equations or diagrams step by step if present.
LANGUAGE RULE:
- hinglish → mix Hindi + English (casual)
- hindi → pure Hindi
- english → simple English
Requested Language: {language}
FORMAT:
- Use Markdown formatting with proper headings (## for main topics, ### for subtopics)
- Use **bold** for important terms
- Use bullet points (-) and numbered lists where appropriate
- Use proper structure with topics and subtopics
- Make it visually organized like ChatGPT responses
CONTENT:
{page_text}
"""

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def home():
    return jsonify({"message": "StudyMate.ai Backend is running! 🚀"}), 200

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "File missing"}), 400
        file = request.files["file"]
        upload_result = cloudinary.uploader.upload(
            file,
            resource_type="raw",
            folder="RagBot_PDFs"
        )
        user_id = _get_optional_user_id()
        pdf_data = {
            "fileName": file.filename,
            "pdfUrl": upload_result["url"],
            "ownerUserId": user_id,
            "createdAt": _utc_iso(),
            "pages": [],
            "chatHistory": [],
            # Student utilities (new docs will have these; old docs remain compatible)
            "revisionPacks": [],
            "doubtNotes": []
        }
        result = db.pdfs.insert_one(pdf_data)
        return jsonify({
            "message": "PDF Uploaded Successfully 🔥",
            "pdf_id": str(result.inserted_id)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/parse-page", methods=["POST"])
def parse_page():
    try:
        data = request.json
        pdf_id = data.get("pdf_id")
        page_no = int(data.get("page_no"))
        language = data.get("language", "english")
        
        # Validate page number
        if page_no <= 0:
            return jsonify({"error": "Invalid page number"}), 400
        
        # Check PDF
        try:
            pdf_id_obj = ObjectId(pdf_id)
        except InvalidId:
            return jsonify({"error": "Invalid PDF ID"}), 400

        pdf_entry = db.pdfs.find_one({"_id": pdf_id_obj})
        if not pdf_entry:
            return jsonify({"error": "PDF not found"}), 404
        # Already parsed?
        for page in pdf_entry.get("pages", []):
            if page["pageNumber"] == page_no:
                return jsonify({
                    "status": "already_parsed",
                    "pageNumber": page_no,
                    "text": page["text"],
                    "explanation": page["explanation"]
                }), 200
        # Download PDF with unique temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
        try:
            r = requests.get(pdf_entry["pdfUrl"], timeout=15)
            with open(temp_path, "wb") as f:
                f.write(r.content)

            result = pdf_parser.process_single_page(temp_path, page_no)

            # error handle
            if "error" in result:
                return jsonify({"error": result["error"]}), 400

            page_text = result["full_text"]
            chunks = result["chunks"]

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        # Gemini Prompt
        prompt = build_student_prompt(page_text, language)
        # Gemini Call
        response = model.generate_content(prompt)
        explanation = response.text or "Unable to generate explanation."
        # Save to DB
        db.pdfs.update_one(
            {"_id": ObjectId(pdf_id)},
            {
                "$push": {
                    "pages": {
                        "pageNumber": page_no,
                        "text": page_text,
                        "explanation": explanation
                    }
                }
            }
        )
        from rag.embedding import get_embedding

        chunk_data = []

        for chunk in chunks:
            try:
                emb = get_embedding(chunk)
                chunk_data.append({
                        "text": chunk,
                        "embedding": emb,
                        "page": page_no
                })
            except Exception as e:
                print("Embedding error:", e)

        # save chunks in DB
        if chunk_data:
            db.pdfs.update_one(
                {"_id": ObjectId(pdf_id)},
                {
                    "$push": {
                        "chunks": {"$each": chunk_data}
                    }
                }
            )
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            "status": "newly_parsed",
            "pageNumber": page_no,
            "text": page_text,
            "explanation": explanation
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- DOUBT CHAT ----------------
def format_recent_history(history, limit=5):
    recent = history[-(limit * 2):]
    text = ""
    for msg in recent:
        role = "STUDENT" if msg["role"] == "user" else "AI_TEACHER"
        text += f"\n<{role}>{msg['parts'][0]['text']}</{role}>\n"
    return text

def build_doubt_prompt(page_text, page_no, history_text, user_query):
    return f"""
<PAGE_CONTEXT>
Page {page_no}
{page_text}
</PAGE_CONTEXT>
<PREVIOUS_CONVERSATION>
{history_text}
</PREVIOUS_CONVERSATION>
<CURRENT_DOUBT>
{user_query}
</CURRENT_DOUBT>
Answer clearly like a teacher.
"""

@app.route("/pdf/<pdf_id>", methods=["GET"])
def get_pdf_info(pdf_id):
    try:
        try:
            pdf_id_obj = ObjectId(pdf_id)
        except InvalidId:
            return jsonify({"error": "Invalid PDF ID"}), 400

        pdf_entry = db.pdfs.find_one({"_id": pdf_id_obj})
        if not pdf_entry:
            return jsonify({"error": "PDF not found"}), 404
        
        # Get total pages by downloading and checking PDF
        try:
            r = requests.get(pdf_entry["pdfUrl"], timeout=10)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
            with open(temp_path, "wb") as f:
                f.write(r.content)
            doc = fitz.open(temp_path)
            total_pages = len(doc)
            doc.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Error getting total pages: {e}")
            total_pages = 1  # Default fallback
        
        return jsonify({
            "pdf_id": pdf_id,
            "fileName": pdf_entry.get("fileName", "Unknown"),
            "totalPages": total_pages,
            "parsedPages": len(pdf_entry.get("pages", [])),
            "pdfUrl": pdf_entry.get("pdfUrl", "")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pdf/<pdf_id>/page/<int:page_no>/image", methods=["GET"])
def get_pdf_page_image(pdf_id, page_no):
    try:
        try:
           pdf_id_obj = ObjectId(pdf_id)
        except InvalidId:
           return jsonify({"error": "Invalid PDF ID"}), 400

        pdf_entry = db.pdfs.find_one({"_id": pdf_id_obj})
        if not pdf_entry:
            return jsonify({"error": "PDF not found"}), 404
        
        # Download PDF
        r = requests.get(pdf_entry["pdfUrl"], timeout=10)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
        with open(temp_path, "wb") as f:
            f.write(r.content)
        
        # Open PDF and get page
        doc = fitz.open(temp_path)
        if page_no < 1 or page_no > len(doc):
            doc.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"error": "Invalid page number"}), 400
        
        page = doc[page_no - 1]  # 0-indexed
        
        # Render page to image (scale factor 2 for better quality)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to base64
        img_data = pix.tobytes("png")
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        img_url = f"data:image/png;base64,{img_base64}"
        
        doc.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({"image": img_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask-doubt", methods=["POST"])
def ask_doubt():
    try:
        data = request.json
        pdf_id = data.get("pdf_id")
        page_no = int(data.get("page_no"))
        query = data.get("query")
        language = data.get("language", "english")
        # Fetch PDF
        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
        if not pdf:
            return jsonify({"error": "PDF not found"}), 404
        # Fetch page text
        chunks = pdf.get("chunks", [])

        if not chunks:
           return jsonify({"error": "No chunks found. Please parse pages first."}), 400

        texts = [c["text"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]

        store = VectorStore()
        store.build_index(embeddings, texts)

        query_embedding = get_embedding(query)

        relevant_chunks = store.search(query_embedding, k=5)

        context = "\n\n".join(relevant_chunks)
        
        # Build last 5 chat turns
        history = pdf.get("chatHistory", [])
        recent_history = history[-10:]
        history_text = ""
        for msg in recent_history:
            role = "STUDENT" if msg["role"] == "user" else "AI_TEACHER"
            history_text += f"\n<{role}>\n{msg['parts'][0]['text']}\n</{role}>\n"
        # Build prompt with context and language
        prompt = f"""
LANGUAGE: {language} (hinglish = Hindi+English mix, hindi = pure Hindi, english = English)

You are a helpful teacher.

Use ONLY the context below.
If answer not found, say "Not found in document".

CONTEXT:
{context}

PREVIOUS CONVERSATION:
{history_text}

QUESTION:
{query}

Answer clearly like a teacher.
Use Markdown formatting:
- Use ## for main topics
- Use **bold** for important terms
- Use bullet points
"""
        # Call Gemini API
        response = model.generate_content(prompt)
        answer = response.text or "Unable to generate answer. Please try again."
        # Save Q&A to DB
        new_entries = [
            {"role": "user", "parts": [{"text": query}]},
            {"role": "model", "parts": [{"text": answer}]}
        ]
        db.pdfs.update_one(
            {"_id": ObjectId(pdf_id)},
            {"$push": {"chatHistory": {"$each": new_entries}}}
        )
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print("ask_doubt error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.json
        pdf_id = data.get("pdf_id")
        page_numbers = data.get("page_numbers", [])  # List of page numbers
        language = data.get("language", "english")
        num_questions = data.get("num_questions", 5)
        
        if not page_numbers:
            return jsonify({"error": "Please select at least one page"}), 400
        
        # Fetch PDF
        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
        if not pdf:
            return jsonify({"error": "PDF not found"}), 404
        
        # Collect text from selected pages
        selected_pages_text = ""
        for page_no in page_numbers:
            page = next((p for p in pdf.get("pages", []) if p["pageNumber"] == page_no), None)
            if page:
                selected_pages_text += f"\n\n--- PAGE {page_no} ---\n{page['text']}\n"
        
        if not selected_pages_text:
            return jsonify({"error": "Selected pages not parsed yet"}), 400
        
        # Build quiz generation prompt
        prompt = f"""
LANGUAGE: {language} (hinglish = Hindi+English mix, hindi = pure Hindi, english = English)

TASK: Generate {num_questions} multiple choice questions (MCQs) based on the following content from pages {', '.join(map(str, page_numbers))}.

CONTENT:
{selected_pages_text}

REQUIREMENTS:
- Generate exactly {num_questions} questions
- Each question should have 4 options (A, B, C, D)
- Mark the correct answer clearly
- Questions should test understanding, not just memorization
- Make questions clear and unambiguous
- Use the requested language ({language})

FORMAT (JSON):
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A",
      "explanation": "Brief explanation of why this is correct"
    }}
  ]
}}

Return ONLY valid JSON, no other text.
"""
        
        # Call Gemini API
        response = model.generate_content(prompt)
        quiz_text = response.text or "{}"
        
        # Try to parse JSON (Gemini might wrap it in markdown)
        quiz_text = _strip_markdown_code_fences(quiz_text)
        quiz_text = _extract_first_json_block(quiz_text)
        
        try:
            quiz_data = json.loads(quiz_text)
        except:
            # Fallback: return raw text if JSON parsing fails
            quiz_data = {"error": "Failed to parse quiz", "raw_response": quiz_text}
        
        return jsonify({"quiz": quiz_data}), 200
    except Exception as e:
        print("generate_quiz error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- REVISION PACK ----------------
@app.route("/generate-revision-pack", methods=["POST"])
def generate_revision_pack():
    """
    Generates a compact revision pack from selected pages and stores it on the PDF:
    - notes (Markdown)
    - key terms
    - flashcards (active recall)
    - exam-style questions
    """
    try:
        data = request.json or {}
        pdf_id = data.get("pdf_id")
        page_numbers = data.get("page_numbers", [])
        language = data.get("language", "english")
        title = (data.get("title") or "").strip()

        if not pdf_id:
            return jsonify({"error": "pdf_id is required"}), 400
        if not page_numbers:
            return jsonify({"error": "Please select at least one page"}), 400

        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
        if not pdf:
            return jsonify({"error": "PDF not found"}), 404

        # Collect text from selected pages (must be parsed already)
        selected_pages_text = ""
        for page_no in page_numbers:
            page = next((p for p in pdf.get("pages", []) if p["pageNumber"] == page_no), None)
            if page and page.get("text"):
                selected_pages_text += f"\n\n--- PAGE {page_no} ---\n{page['text']}\n"

        if not selected_pages_text:
            return jsonify({"error": "Selected pages not parsed yet"}), 400

        pack_id = uuid.uuid4().hex
        created_at = _utc_iso()
        effective_title = title or f"Revision Pack (Pages {', '.join(map(str, page_numbers))})"

        prompt = f"""
LANGUAGE: {language} (hinglish = Hindi+English mix, hindi = pure Hindi, english = English)

TASK: Create a compact "Revision Pack" for a student from the content below.
Make it highly exam-oriented and easy to revise quickly.

CONTENT:
{selected_pages_text}

REQUIREMENTS:
- Keep it short, but not vague (prioritize what is most likely asked in exams)
- Use simple language in the requested language ({language})
- Prefer active recall: many short Q/A flashcards
- Avoid hallucinations: only use info that appears in CONTENT

FORMAT (JSON):
{{
  "title": "Short title for this revision pack",
  "notes_markdown": "Markdown notes with headings and bullets",
  "key_terms": [
    {{ "term": "Term", "meaning": "Meaning" }}
  ],
  "flashcards": [
    {{ "front": "Question", "back": "Answer" }}
  ],
  "exam_questions": [
    {{ "question": "Exam style question", "answer_outline": "Bullet outline answer" }}
  ],
  "common_mistakes": [
    "Common mistake 1"
  ]
}}

Return ONLY valid JSON, no other text.
"""

        response = model.generate_content(prompt)

        pack_text = _strip_markdown_code_fences(response.text or "{}")
        pack_text = _extract_first_json_block(pack_text)
        try:
            pack_data = json.loads(pack_text)
        except Exception:
            pack_data = {"error": "Failed to parse revision pack", "raw_response": pack_text}

        saved_pack = {
            "packId": pack_id,
            "createdAt": created_at,
            "language": language,
            "pageNumbers": page_numbers,
            "title": pack_data.get("title", effective_title) if isinstance(pack_data, dict) else effective_title,
            "pack": pack_data,
        }

        db.pdfs.update_one(
            {"_id": ObjectId(pdf_id)},
            {"$push": {"revisionPacks": saved_pack}}
        )

        return jsonify({"revision_pack": saved_pack}), 200
    except Exception as e:
        print("generate_revision_pack error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/pdf/<pdf_id>/revision-packs", methods=["GET"])
def list_revision_packs(pdf_id):
    try:
        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
        if not pdf:
            return jsonify({"error": "PDF not found"}), 404
        packs = list(reversed(pdf.get("revisionPacks", [])))
        return jsonify({"revision_packs": packs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# Conversations (Saved chats)
# --------------------------------------------------
@app.route("/api/conversations", methods=["GET"])
@jwt_required()
def list_conversations():
    user_id = get_jwt_identity()
    try:
        docs = list(
            db.conversations.find({"userId": user_id}).sort("updatedAt", -1).limit(200)
        )
        return jsonify({"conversations": [_serialize_conversation(d) for d in docs]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations", methods=["POST"])
@jwt_required()
def create_conversation():
    user_id = get_jwt_identity()
    data = request.json or {}
    pdf_id = data.get("pdf_id")
    page_no = data.get("page_no")
    title = (data.get("title") or "").strip()

    if not pdf_id:
        return jsonify({"error": "pdf_id is required"}), 400

    try:
        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
    except Exception:
        return jsonify({"error": "Invalid pdf_id"}), 400

    if not pdf:
        return jsonify({"error": "PDF not found"}), 404

    owner = pdf.get("ownerUserId")
    if owner and owner != user_id:
        return jsonify({"error": "Not allowed"}), 403

    # Migration-friendly: claim older PDFs without owner
    if not owner:
        db.pdfs.update_one({"_id": pdf["_id"]}, {"$set": {"ownerUserId": user_id}})

    now = _utc_iso()
    effective_title = title or f"{pdf.get('fileName', 'PDF')}"
    if page_no is not None:
        try:
            effective_title = f"{effective_title} • Page {int(page_no)}"
        except Exception:
            pass

    conversation = {
        "userId": user_id,
        "pdfId": pdf["_id"],
        "pdfFileName": pdf.get("fileName", ""),
        "title": effective_title,
        "createdAt": now,
        "updatedAt": now,
        "lastPageNo": int(page_no) if str(page_no).isdigit() else None,
        "messages": [],
    }

    try:
        result = db.conversations.insert_one(conversation)
        conversation["_id"] = result.inserted_id
        return jsonify({"conversation": _serialize_conversation(conversation)}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>", methods=["GET"])
@jwt_required()
def get_conversation(conversation_id):
    user_id = get_jwt_identity()
    try:
        doc = db.conversations.find_one(
            {"_id": ObjectId(conversation_id), "userId": user_id}
        )
    except Exception:
        return jsonify({"error": "Invalid conversation id"}), 400

    if not doc:
        return jsonify({"error": "Conversation not found"}), 404

    return jsonify({"conversation": _serialize_conversation(doc)}), 200

@app.route("/api/conversations/<conversation_id>/messages", methods=["POST"])
@jwt_required()
def add_conversation_message(conversation_id):
    user_id = get_jwt_identity()
    data = request.json or {}
    query = (data.get("query") or "").strip()
    language = data.get("language", "english")

    try:
        page_no = int(data.get("page_no"))
    except Exception:
        page_no = None

    if not query:
        return jsonify({"error": "query is required"}), 400
    if not page_no or page_no <= 0:
        return jsonify({"error": "page_no is required"}), 400

    try:
        conv = db.conversations.find_one(
            {"_id": ObjectId(conversation_id), "userId": user_id}
        )
    except Exception:
        return jsonify({"error": "Invalid conversation id"}), 400

    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    pdf = db.pdfs.find_one({"_id": conv.get("pdfId")})
    if not pdf:
        return jsonify({"error": "PDF not found"}), 404

    owner = pdf.get("ownerUserId")
    if owner and owner != user_id:
        return jsonify({"error": "Not allowed"}), 403

    chunks = pdf.get("chunks", [])

    if not chunks:
       return jsonify({"error": "No chunks found. Please parse pages first."}), 400

    texts = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]

    store = VectorStore()
    store.build_index(embeddings, texts)

    query_embedding = get_embedding(query)

    relevant_chunks = store.search(query_embedding, k=5)

    context = "\n\n".join(relevant_chunks)
    if not page:
        return jsonify({"error": "Page not parsed yet"}), 400

    recent = (conv.get("messages") or [])[-10:]
    history_text = ""
    for msg in recent:
        role = "STUDENT" if msg.get("role") == "user" else "AI_TEACHER"
        history_text += f"\n<{role}>\n{msg.get('text', '')}\n</{role}>\n"

    prompt = f"""
LANGUAGE: {language}

You are a helpful teacher.

STRICT RULE:
- Answer ONLY from the given context
- If answer not found → say "Not found in document"
- Do NOT use outside knowledge

CONTEXT:
{context}

QUESTION:
{query}

Give clear, structured answer.
"""

    response = model.generate_content(prompt)
    answer = response.text or "Unable to generate answer. Please try again."

    now = _utc_iso()
    new_entries = [
        {"role": "user", "text": query, "pageNo": page_no, "createdAt": now},
        {"role": "model", "text": answer, "pageNo": page_no, "createdAt": now},
    ]

    try:
        db.conversations.update_one(
            {"_id": conv["_id"], "userId": user_id},
            {
                "$push": {"messages": {"$each": new_entries}},
                "$set": {"updatedAt": now, "lastPageNo": page_no},
            },
        )
    except Exception as e:
        return jsonify({"error": f"Failed to save message: {e}"}), 500

    return jsonify({"answer": answer}), 200

@app.route("/api/me", methods=["GET"])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    return jsonify({"userId": user_id}), 200

# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running on port {port}")
    app.run(host="0.0.0.0", port=port)

