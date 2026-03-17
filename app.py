import os
import requests
import certifi
import uuid
import tempfile
import json
import re
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
from google import genai
from pdf_pipeline.parser import PDFParser
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

# --------------------------------------------------
# Load ENV
# --------------------------------------------------
load_dotenv()
app = Flask(__name__)

# CORS FIX: Proper configuration for credentials
CORS(
    app,
    origins=["http://localhost:5173", "http://localhost:3000"],
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    max_age=3600
)

# JWT Config
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-prod")
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
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"

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
        pdf_data = {
            "fileName": file.filename,
            "pdfUrl": upload_result["url"],
            "pages": [],
            "chatHistory": []
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
        pdf_entry = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
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
        r = requests.get(pdf_entry["pdfUrl"])
        with open(temp_path, "wb") as f:
            f.write(r.content)
        # Parse Page
        page_text = pdf_parser.process_single_page(temp_path, page_no)
        # Gemini Prompt
        prompt = build_student_prompt(page_text, language)
        # Gemini Call
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
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
        pdf_entry = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
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
        pdf_entry = db.pdfs.find_one({"_id": ObjectId(pdf_id)})
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
        # Build prompt with context and language
        prompt = f"""
LANGUAGE: {language} (hinglish = Hindi+English mix, hindi = pure Hindi, english = English)
<PAGE_CONTEXT>
Page {page_no}
{page['text']}
</PAGE_CONTEXT>
<PREVIOUS_CONVERSATION>
{history_text}
</PREVIOUS_CONVERSATION>
<CURRENT_DOUBT>
{query}
</CURRENT_DOUBT>
Answer clearly like a teacher in the requested language. Use Markdown formatting:
- Use ## for main topics, ### for subtopics
- Use **bold** for important terms
- Use bullet points (-) and numbered lists
- Make it well-structured and organized
"""
        # Call Gemini API
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
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
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        quiz_text = response.text or "{}"
        
        # Try to parse JSON (Gemini might wrap it in markdown)
        # Remove markdown code blocks if present
        quiz_text = re.sub(r'```json\n?', '', quiz_text)
        quiz_text = re.sub(r'```\n?', '', quiz_text)
        quiz_text = quiz_text.strip()
        
        try:
            quiz_data = json.loads(quiz_text)
        except:
            # Fallback: return raw text if JSON parsing fails
            quiz_data = {"error": "Failed to parse quiz", "raw_response": quiz_text}
        
        return jsonify({"quiz": quiz_data}), 200
    except Exception as e:
        print("generate_quiz error:", e)
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running on port {port}")
    app.run(host="0.0.0.0", port=port)