import PyPDF2
from PIL import Image
import io
import base64


def process_pdf(pdf_file):
    """Process PDF file and return extracted text"""
    import PyPDF2

    # If pdf_file is a path string, open it. If it's already a file object, use it directly
    if isinstance(pdf_file, str):
        pdf = PyPDF2.PdfReader(open(pdf_file, 'rb'))
    else:
        pdf = PyPDF2.PdfReader(pdf_file)

    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

    return {"context": text}


def process_image(image_file):
    """Process image file and return HTML img tag with base64 encoded image"""
    try:
        # Handle both file object and file path
        if isinstance(image_file, str):
            image = Image.open(open(image_file, 'rb'))
        else:
            image = Image.open(image_file)

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        img_html = f"<img src='data:image/png;base64,{img_str}'>"

        return {
            "display_message": f"🖼️ Image loaded successfully!",
            "context": img_html
        }
    except Exception as e:
        return {
            "display_message": f"Error processing image: {str(e)}",
            "context": ""
        }
