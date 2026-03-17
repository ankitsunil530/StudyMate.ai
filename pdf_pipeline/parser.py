import os
import fitz  # PyMuPDF
from PIL import Image

try:
    import pytesseract
except:
    pytesseract = None
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# 🔹 Set Tesseract path
if pytesseract:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class PDFParser:
    def __init__(self, max_chunk_chars=1000):
        self.max_chunk_chars = max_chunk_chars
        # BLIP (load once)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.eval()

    # ======================================================
    # NEW FUNCTION: Sirf ek page parse karne ke liye
    # ======================================================
    def process_single_page(self, pdf_path, page_no):
        doc = fitz.open(pdf_path)
        # Check ki page number valid hai ya nahi
        if page_no > len(doc) or page_no < 1:
            return "Invalid Page Number"

        page = doc[page_no - 1] # Index 0 se shuru hota hai
        os.makedirs("images", exist_ok=True)

        # 1. Page se text aur image placeholders nikalo
        raw_text, page_image_data = self.extract_page_content(page, page_no, doc)

        # 2. Images ka OCR aur BLIP description nikalo
        image_results = self.process_images_for_page(page_image_data)

        # 3. Text mein [IMAGE_X] ki jagah description daalo
        final_text = raw_text
        for key, result in image_results.items():
            final_text = final_text.replace(f"[IMAGE_{key}]", result)

        return final_text.strip()

    # ======================================================
    # HELPER 1: Ek single page ka raw content nikalna
    # ======================================================
    def extract_page_content(self, page, page_no, doc):
        blocks = page.get_text("dict")["blocks"]
        page_text = ""
        img_counter = 0
        page_image_data = {}

        for block in blocks:
            if block.get("type") == 0: # Text
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        page_text += span.get("text", "") + " "
            
            elif block.get("type") == 1: # Image
                img_counter += 1
                img_name = f"page{page_no}_img{img_counter}"
                
                # Image extraction logic (Tera purana wala hi hai)
                image_bytes = None
                if "xref" in block:
                    try:
                        base_image = doc.extract_image(block["xref"])
                        image_bytes = base_image["image"]
                        ext = base_image["ext"]
                    except: pass
                
                if image_bytes is None and "image" in block:
                    image_bytes = block["image"]
                    ext = block.get("ext", "png")

                if image_bytes:
                    img_path = f"images/{img_name}.{ext}"
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    page_text += f" [IMAGE_{img_name.upper()}] "
                    page_image_data[img_name.upper()] = img_path

        return page_text, page_image_data

    # ======================================================
    # HELPER 2: Sirf us page ki images ko AI se process karna
    # ======================================================
    def process_images_for_page(self, page_image_data):
        results = {}
        for key, img_path in page_image_data.items():
            image = Image.open(img_path).convert("RGB")
            
            # OCR
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            ocr_text = pytesseract.image_to_string(gray, config="--psm 6").strip()

            if len(ocr_text) > 8:
                results[key] = f"(Equation/Text in Image: {ocr_text})"
            else:
                # BLIP Fallback
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    output = self.model.generate(**inputs)
                caption = self.processor.decode(output[0], skip_special_tokens=True)
                results[key] = f"(Image Description: {caption})"
            
            # Cleanup: Kaam hone ke baad image delete kar sakte ho storage bachane ke liye
            # os.remove(img_path) 
            
        return results
