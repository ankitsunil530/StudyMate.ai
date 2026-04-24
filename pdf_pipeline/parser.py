import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
# import torch

IMAGE_DIR = "/tmp/images" if os.environ.get("RENDER") else "images"

# Optional imports
try:
    import pytesseract
except:
    pytesseract = None

try:
    import cv2
except:
    cv2 = None

from transformers import BlipProcessor, BlipForConditionalGeneration

# ======================================================
# 🔹 Device Setup
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE = "/tmp" if os.environ.get("RENDER") else None

# ======================================================
# 🔹 Load BLIP Model (Global)
# ======================================================
print("🔄 Loading BLIP model...")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir=MODEL_CACHE
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir=MODEL_CACHE
)
model.to(DEVICE)
model.eval()
print("✅ BLIP loaded on", DEVICE)

# ======================================================
# 🔹 Tesseract Setup (Windows)
# ======================================================
if pytesseract and os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class PDFParser:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ======================================================
    # 🔹 Chunking Function (RAG)
    # ======================================================
    def chunk_text(self, text):
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)

        return chunks

    # ======================================================
    # 🔹 Process Single Page (UPDATED)
    # ======================================================
    def process_single_page(self, pdf_path, page_no):
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return {"error": f"Error opening PDF: {str(e)}"}

        if page_no > len(doc) or page_no < 1:
            return {"error": "Invalid Page Number"}

        page = doc[page_no - 1]
        os.makedirs(IMAGE_DIR, exist_ok=True)

        # Extract content
        raw_text, page_image_data = self.extract_page_content(page, page_no, doc)

        # Process images
        image_results = self.process_images_for_page(page_image_data)

        # Replace placeholders
        final_text = raw_text
        for key, result in image_results.items():
            final_text = final_text.replace(f"[IMAGE_{key}]", result)

        final_text = final_text.strip()

        # 🔥 RAG READY: create chunks
        chunks = self.chunk_text(final_text)

        return {
            "full_text": final_text,
            "chunks": chunks
        }

    # ======================================================
    # 🔹 Extract Page Content
    # ======================================================
    def extract_page_content(self, page, page_no, doc):
        blocks = page.get_text("dict")["blocks"]
        page_text = ""
        img_counter = 0
        page_image_data = {}

        for block in blocks:
            # TEXT BLOCK
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        page_text += span.get("text", "") + " "

            # IMAGE BLOCK
            elif block.get("type") == 1:
                img_counter += 1
                img_name = f"page{page_no}_img{img_counter}"

                image_bytes = None
                ext = "png"

                # Try xref extraction
                if "xref" in block:
                    try:
                        base_image = doc.extract_image(block["xref"])
                        image_bytes = base_image["image"]
                        ext = base_image["ext"]
                    except:
                        pass

                # Fallback
                if image_bytes is None and "image" in block:
                    image_bytes = block["image"]
                    ext = block.get("ext", "png")

                # Save image
                if image_bytes:
                    img_path = f"{IMAGE_DIR}/{img_name}.{ext}"
                    try:
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)

                        page_text += f" [IMAGE_{img_name}] "
                        page_image_data[img_name] = img_path

                    except Exception as e:
                        print(f"Error saving image: {e}")

        return page_text, page_image_data

    # ======================================================
    # 🔹 Process Images (OCR + BLIP)
    # ======================================================
    def process_images_for_page(self, page_image_data):
        results = {}

        for key, img_path in page_image_data.items():
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                results[key] = f"(Image load error: {str(e)})"
                continue

            # OCR
            ocr_text = ""
            if pytesseract and cv2:
                try:
                    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                    ocr_text = pytesseract.image_to_string(gray, config="--psm 6").strip()
                except Exception as e:
                    print(f"OCR Error: {e}")

            # Decision
            if ocr_text and len(ocr_text.strip()) > 15:
                results[key] = f"(Text in Image: {ocr_text})"
            else:
                try:
                    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=30)

                    caption = processor.decode(output[0], skip_special_tokens=True)
                    results[key] = f"(Image Description: {caption})"

                except Exception as e:
                    results[key] = f"(Caption error: {str(e)})"

            # Cleanup
            try:
                os.remove(img_path)
            except:
                pass

        return results