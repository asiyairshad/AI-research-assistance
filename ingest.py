from model import DocumentChunk
import fitz
from PIL import Image
import io

def parse_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []

    for i in range(len(doc)):
        text = doc[i].get_text()
        if text.strip():
            chunks.append(
                DocumentChunk(text, "text", i+1)
            )
    return chunks


def extract_images_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            chunks.append(
                DocumentChunk(image, "image", page_num+1)
            )
    return chunks