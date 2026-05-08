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
from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
    verify_jwt_in_request,
)
# --------------------------------------------------
# Load ENV
# --------------------------------------------------
load_dotenv()
app = Flask(__name__)

def _get_cors_origins():
    origins = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]
    configured = os.getenv("CORS_ORIGINS") or os.getenv("FRONTEND_URL") or ""
    origins.extend(
        origin.strip().rstrip("/")
        for origin in configured.split(",")
        if origin.strip()
    )
    return sorted(set(origins))

# CORS FIX: Proper configuration for credentials.
# On Hugging Face, set CORS_ORIGINS to your frontend URL(s), comma-separated.
CORS(
    app,
    origins=_get_cors_origins(),
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    max_age=3600
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
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("MONGO_URI environment variable is required")

# MongoDB Atlas
# --------------------------------------------------
client = MongoClient(
    mongo_uri,
    tls=True,
    tlsCAFile=certifi.where(),
    connect=False,
    connectTimeoutMS=10000,
    serverSelectionTimeoutMS=10000,
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
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=gemini_api_key)
GEMINI_MODEL = "gemini-2.5-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

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

def _generate_gemini_text(prompt):
    response = gemini_model.generate_content(prompt)
    return getattr(response, "text", "") or ""

def _generate_gemini_json(prompt, fallback=None):
    fallback = fallback if fallback is not None else {}
    try:
        text = _strip_markdown_code_fences(_generate_gemini_text(prompt) or "{}")
        text = _extract_first_json_block(text)
        return json.loads(text)
    except Exception as e:
        print("Gemini JSON parse error:", e)
        return fallback

def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def _safe_percent(numerator, denominator):
    if not denominator:
        return 0
    return round((numerator / denominator) * 100)

def _page_lookup(pdf):
    return {p.get("pageNumber"): p for p in pdf.get("pages", [])}

def _extract_concepts_from_text(text, page_no, limit=8):
    words = re.findall(r"\b[A-Z][A-Za-z0-9+\-/]{3,}(?:\s+[A-Z][A-Za-z0-9+\-/]{2,}){0,3}\b", text or "")
    seen = set()
    concepts = []
    for word in words:
        key = word.strip().lower()
        if key in seen or len(key) < 4:
            continue
        seen.add(key)
        concepts.append({
            "name": word.strip()[:80],
            "page": page_no,
            "importance": "medium",
            "summary": "Detected from the page text.",
        })
        if len(concepts) >= limit:
            break
    return concepts

def _generate_page_learning_metadata(page_text, page_no, language):
    fallback = {
        "concepts": _extract_concepts_from_text(page_text, page_no),
        "learning_objectives": [],
        "exam_focus": [],
    }
    prompt = f"""
LANGUAGE: {language}
Analyze this PDF page for an adaptive learning system.

PAGE {page_no} TEXT:
{page_text[:9000]}

Return ONLY valid JSON:
{{
  "concepts": [
    {{
      "name": "Concept name",
      "summary": "One line meaning",
      "importance": "high|medium|low",
      "page": {page_no}
    }}
  ],
  "learning_objectives": ["What a student should learn"],
  "exam_focus": ["Likely exam angle"]
}}
"""
    data = _generate_gemini_json(prompt, fallback)
    concepts = data.get("concepts") if isinstance(data, dict) else []
    if not isinstance(concepts, list) or not concepts:
        data["concepts"] = fallback["concepts"]
    for concept in data.get("concepts", []):
        concept["page"] = concept.get("page") or page_no
    return data

def _build_concept_map(pdf):
    concepts_by_name = {}
    for page in pdf.get("pages", []):
        for concept in page.get("learningMetadata", {}).get("concepts", []):
            name = (concept.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            existing = concepts_by_name.setdefault(key, {
                "id": key.replace(" ", "-")[:80],
                "name": name,
                "pages": [],
                "importance": concept.get("importance", "medium"),
                "summary": concept.get("summary", ""),
            })
            if page.get("pageNumber") not in existing["pages"]:
                existing["pages"].append(page.get("pageNumber"))

    nodes = list(concepts_by_name.values())[:30]
    links = []
    page_groups = {}
    for node in nodes:
        for page_no in node.get("pages", []):
            page_groups.setdefault(page_no, []).append(node["id"])
    for ids in page_groups.values():
        for idx in range(len(ids) - 1):
            links.append({"source": ids[idx], "target": ids[idx + 1], "relation": "same page"})
    return {"nodes": nodes, "links": links[:40]}

def _classify_doubt(query):
    q = (query or "").lower()
    if any(word in q for word in ["formula", "equation", "solve", "numerical"]):
        return "formula"
    if any(word in q for word in ["example", "real life", "use case"]):
        return "example"
    if any(word in q for word in ["definition", "meaning", "what is"]):
        return "definition"
    if any(word in q for word in ["exam", "important", "question"]):
        return "exam-oriented"
    return "conceptual"

def _build_grounded_answer(page_text, page_no, history_text, query, language):
    fallback = {
        "answer": "Unable to generate answer. Please try again.",
        "doubt_type": _classify_doubt(query),
        "confidence": "medium" if page_text else "low",
        "source_pages": [page_no],
        "source_evidence": "",
        "suggested_next_steps": [],
    }
    prompt = f"""
LANGUAGE: {language} (hinglish = Hindi+English mix, hindi = pure Hindi, english = English)
You are a document-grounded AI tutor. Use ONLY PAGE_CONTEXT and PREVIOUS_CONVERSATION.
If the answer is not clearly present, say that clearly and keep confidence low.

<PAGE_CONTEXT>
Page {page_no}
{page_text[:9000]}
</PAGE_CONTEXT>
<PREVIOUS_CONVERSATION>
{history_text}
</PREVIOUS_CONVERSATION>
<CURRENT_DOUBT>
{query}
</CURRENT_DOUBT>

Return ONLY valid JSON:
{{
  "answer": "Markdown answer in requested language",
  "doubt_type": "conceptual|definition|formula|example|exam-oriented|other",
  "confidence": "high|medium|low",
  "source_pages": [{page_no}],
  "source_evidence": "short phrase or sentence from the page that supports the answer",
  "suggested_next_steps": ["one concrete revision suggestion"]
}}
"""
    data = _generate_gemini_json(prompt, fallback)
    if not isinstance(data, dict):
        return fallback
    data["answer"] = data.get("answer") or fallback["answer"]
    data["doubt_type"] = data.get("doubt_type") or fallback["doubt_type"]
    data["confidence"] = data.get("confidence") or fallback["confidence"]
    data["source_pages"] = data.get("source_pages") or [page_no]
    data["suggested_next_steps"] = data.get("suggested_next_steps") or []
    return data

def _serialize_attempt(doc):
    return {
        "id": str(doc.get("_id")),
        "userId": doc.get("userId"),
        "pdfId": str(doc.get("pdfId")) if doc.get("pdfId") else None,
        "pdfFileName": doc.get("pdfFileName", ""),
        "score": doc.get("score", 0),
        "total": doc.get("total", 0),
        "accuracy": _safe_percent(doc.get("score", 0), doc.get("total", 0)),
        "createdAt": doc.get("createdAt"),
        "topics": doc.get("topics", []),
        "weakTopics": doc.get("weakTopics", []),
    }

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

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "File missing"}), 400
        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF allowed"}), 400
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
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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
                    "explanation": page["explanation"],
                    "learningMetadata": page.get("learningMetadata", {})
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

            page_text = result.get("full_text", "")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        # Gemini Prompt
        prompt = build_student_prompt(page_text, language)
        # Gemini Call
        explanation = _generate_gemini_text(prompt) or "Unable to generate explanation."
        learning_metadata = _generate_page_learning_metadata(page_text, page_no, language)
        # Save to DB
        db.pdfs.update_one(
            {"_id": ObjectId(pdf_id)},
            {
                "$push": {
                    "pages": {
                        "pageNumber": page_no,
                        "text": page_text,
                        "explanation": explanation,
                        "learningMetadata": learning_metadata
                    }
                }
            }
        )
        return jsonify({
            "status": "newly_parsed",
            "pageNumber": page_no,
            "text": page_text,
            "explanation": explanation,
            "learningMetadata": learning_metadata
        }), 200
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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
            "pdfUrl": pdf_entry.get("pdfUrl", ""),
            "conceptMap": _build_concept_map(pdf_entry)
        }), 200
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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
        page = next((p for p in pdf.get("pages", []) if p["pageNumber"] == page_no), None)
        if not page:
            return jsonify({"error": "Page not parsed yet"}), 400
        # Build last 5 chat turns
        history = pdf.get("chatHistory", [])
        recent_history = history[-10:]
        history_text = ""
        for msg in recent_history:
            role = "STUDENT" if msg["role"] == "user" else "AI_TEACHER"
            history_text += f"\n<{role}>\n{msg['parts'][0]['text']}\n</{role}>\n"
        grounded = _build_grounded_answer(page.get("text", ""), page_no, history_text, query, language)
        answer = grounded["answer"]
        # Save Q&A to DB
        now = _utc_iso()
        new_entries = [
            {"role": "user", "parts": [{"text": query}]},
            {"role": "model", "parts": [{"text": answer}], "grounding": grounded}
        ]
        db.pdfs.update_one(
            {"_id": ObjectId(pdf_id)},
            {
                "$push": {
                    "chatHistory": {"$each": new_entries},
                    "doubtNotes": {
                        "query": query,
                        "pageNo": page_no,
                        "type": grounded.get("doubt_type"),
                        "confidence": grounded.get("confidence"),
                        "createdAt": now,
                    },
                }
            }
        )
        return jsonify({
            "answer": answer,
            "grounding": grounded,
            "doubt_type": grounded.get("doubt_type"),
            "confidence": grounded.get("confidence"),
            "source_pages": grounded.get("source_pages", [page_no]),
        }), 200
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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
- Include a topic and difficulty for mastery tracking

FORMAT (JSON):
{{
  "questions": [
    {{
      "question": "Question text here?",
      "topic": "Main concept tested",
      "difficulty": "easy|medium|hard",
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
        quiz_text = _generate_gemini_text(prompt) or "{}"
        
        # Try to parse JSON (Gemini might wrap it in markdown)
        quiz_text = _strip_markdown_code_fences(quiz_text)
        quiz_text = _extract_first_json_block(quiz_text)
        
        try:
            quiz_data = json.loads(quiz_text)
        except:
            # Fallback: return raw text if JSON parsing fails
            quiz_data = {"error": "Failed to parse quiz", "raw_response": quiz_text}
        if isinstance(quiz_data, dict):
            for question in quiz_data.get("questions", []) or []:
                question.setdefault("topic", "General")
                question.setdefault("difficulty", "medium")
        
        return jsonify({"quiz": quiz_data}), 200
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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

        pack_text = _strip_markdown_code_fences(_generate_gemini_text(prompt) or "{}")
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
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/pdf/<pdf_id>/revision-packs", methods=["GET"])
def list_revision_packs(pdf_id):
    try:
        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
        if not pdf:
            return jsonify({"error": "PDF not found"}), 404
        packs = list(reversed(pdf.get("revisionPacks", [])))
        return jsonify({"revision_packs": packs}), 200
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/pdf/<pdf_id>/concept-map", methods=["GET"])
def get_concept_map(pdf_id):
    try:
        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
        if not pdf:
            return jsonify({"error": "PDF not found"}), 404
        return jsonify({"conceptMap": _build_concept_map(pdf)}), 200
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/quiz-attempts", methods=["POST"])
@jwt_required()
def save_quiz_attempt():
    user_id = get_jwt_identity()
    data = request.json or {}
    pdf_id = data.get("pdf_id")
    questions = data.get("questions") or []
    answers = data.get("answers") or {}

    if not pdf_id or not questions:
        return jsonify({"error": "pdf_id and questions are required"}), 400

    try:
        pdf = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
    except Exception:
        return jsonify({"error": "Invalid pdf_id"}), 400
    if not pdf:
        return jsonify({"error": "PDF not found"}), 404

    results = []
    score = 0
    topic_stats = {}
    for idx, question in enumerate(questions):
        key = str(idx)
        selected = answers.get(key) or answers.get(idx)
        correct = question.get("correct_answer")
        is_correct = bool(selected and correct and selected == correct)
        if is_correct:
            score += 1
        topic = question.get("topic") or "General"
        stat = topic_stats.setdefault(topic, {"topic": topic, "correct": 0, "total": 0})
        stat["total"] += 1
        if is_correct:
            stat["correct"] += 1
        results.append({
            "question": question.get("question"),
            "topic": topic,
            "difficulty": question.get("difficulty", "medium"),
            "selected": selected,
            "correctAnswer": correct,
            "isCorrect": is_correct,
        })

    topics = []
    weak_topics = []
    for stat in topic_stats.values():
        stat["accuracy"] = _safe_percent(stat["correct"], stat["total"])
        topics.append(stat)
        if stat["accuracy"] < 70:
            weak_topics.append(stat["topic"])

    attempt = {
        "userId": user_id,
        "pdfId": pdf["_id"],
        "pdfFileName": pdf.get("fileName", ""),
        "score": score,
        "total": len(questions),
        "topics": topics,
        "weakTopics": weak_topics,
        "results": results,
        "createdAt": _utc_iso(),
    }
    result = db.quizAttempts.insert_one(attempt)
    attempt["_id"] = result.inserted_id

    return jsonify({
        "attempt": _serialize_attempt(attempt),
        "results": results,
        "recommendations": [
            f"Revise {topic} and ask one doubt before retrying."
            for topic in weak_topics[:3]
        ],
    }), 201

@app.route("/api/learning-dashboard", methods=["GET"])
@jwt_required()
def learning_dashboard():
    user_id = get_jwt_identity()
    try:
        conversations = list(db.conversations.find({"userId": user_id}).limit(500))
        pdfs = list(db.pdfs.find({"ownerUserId": user_id}).limit(200))
        attempts = list(db.quizAttempts.find({"userId": user_id}).sort("createdAt", -1).limit(200))

        topic_rollup = {}
        for attempt in attempts:
            for topic in attempt.get("topics", []):
                name = topic.get("topic") or "General"
                stat = topic_rollup.setdefault(name, {"topic": name, "correct": 0, "total": 0})
                stat["correct"] += _safe_int(topic.get("correct"))
                stat["total"] += _safe_int(topic.get("total"))

        topics = []
        for stat in topic_rollup.values():
            stat["accuracy"] = _safe_percent(stat["correct"], stat["total"])
            topics.append(stat)
        topics.sort(key=lambda item: item["accuracy"])

        parsed_pages = sum(len(pdf.get("pages", [])) for pdf in pdfs)
        doubt_count = sum(len(pdf.get("doubtNotes", [])) for pdf in pdfs)
        quiz_score = sum(a.get("score", 0) for a in attempts)
        quiz_total = sum(a.get("total", 0) for a in attempts)
        weak_topics = [t for t in topics if t["total"] and t["accuracy"] < 70][:6]
        strong_topics = [t for t in reversed(topics) if t["total"] and t["accuracy"] >= 80][:6]

        return jsonify({
            "summary": {
                "pdfsStudied": len(pdfs),
                "parsedPages": parsed_pages,
                "savedChats": len(conversations),
                "doubtsAsked": doubt_count,
                "quizAttempts": len(attempts),
                "quizAccuracy": _safe_percent(quiz_score, quiz_total),
            },
            "weakTopics": weak_topics,
            "strongTopics": strong_topics,
            "recentAttempts": [_serialize_attempt(a) for a in attempts[:8]],
            "recommendations": [
                f"Revise {topic['topic']} with a revision pack and retry a short quiz."
                for topic in weak_topics[:3]
            ] or ["Upload a PDF, explain 2 pages, and attempt one quiz to start mastery tracking."],
        }), 200
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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
        print("ERROR:", e)
        return jsonify({"error": "Internal server error"}), 500

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

    page = next((p for p in pdf.get("pages", []) if p["pageNumber"] == page_no), None)
    if not page:
        return jsonify({"error": "Page not parsed yet"}), 400

    recent = (conv.get("messages") or [])[-10:]
    history_text = ""
    for msg in recent:
        role = "STUDENT" if msg.get("role") == "user" else "AI_TEACHER"
        history_text += f"\n<{role}>\n{msg.get('text', '')}\n</{role}>\n"

    grounded = _build_grounded_answer(page.get("text", ""), page_no, history_text, query, language)
    answer = grounded["answer"]

    now = _utc_iso()
    new_entries = [
        {"role": "user", "text": query, "pageNo": page_no, "createdAt": now},
        {
            "role": "model",
            "text": answer,
            "pageNo": page_no,
            "createdAt": now,
            "grounding": grounded,
        },
    ]

    try:
        db.conversations.update_one(
            {"_id": conv["_id"], "userId": user_id},
            {
                "$push": {"messages": {"$each": new_entries}},
                "$set": {"updatedAt": now, "lastPageNo": page_no},
            },
        )
        db.pdfs.update_one(
            {"_id": pdf["_id"]},
            {
                "$push": {
                    "doubtNotes": {
                        "query": query,
                        "pageNo": page_no,
                        "type": grounded.get("doubt_type"),
                        "confidence": grounded.get("confidence"),
                        "createdAt": now,
                    }
                }
            },
        )
    except Exception as e:
        return jsonify({"error": f"Failed to save message: {e}"}), 500

    return jsonify({
        "answer": answer,
        "grounding": grounded,
        "doubt_type": grounded.get("doubt_type"),
        "confidence": grounded.get("confidence"),
        "source_pages": grounded.get("source_pages", [page_no]),
    }), 200

@app.route("/api/me", methods=["GET"])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    return jsonify({"userId": user_id}), 200

# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Running on port {port}")
    app.run(host="0.0.0.0", port=port)

