import easyocr
import pytesseract
import cv2
from PIL import Image
import numpy as np

# Initialize EasyOCR (works well for handwritten and printed text)
reader = easyocr.Reader(['en'])

# Function to preprocess the image
def preprocess_image(image_path):
    # Load image with OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image

# Function to extract text using EasyOCR
def perform_easyocr(image_path):
    image = preprocess_image(image_path)
    result = reader.readtext(image)

    extracted_text = " ".join([entry[1] for entry in result])  # Extract text
    return extracted_text.strip()

# Function to extract text using Tesseract OCR
def perform_tesseract_ocr(image_path):
    image = preprocess_image(image_path)
    pil_image = Image.fromarray(image)  # Convert OpenCV image to PIL format
    extracted_text = pytesseract.image_to_string(pil_image)  # Perform OCR
    return extracted_text.strip()

# Function to classify between handwritten and printed text
def classify_text_type(image_path):
    text_handwritten = perform_easyocr(image_path)
    text_printed = perform_tesseract_ocr(image_path)

    len_handwritten = len(text_handwritten)
    len_printed = len(text_printed)

    print(f"Extracted (Handwritten): {text_handwritten}")
    print(f"Extracted (Printed): {text_printed}")

    # Compare text length for classification
    if  len_printed > len_handwritten:
        return "Handwritten", text_handwritten
    else:
        return "Printed", text_printed

# Run the OCR classification
if __name__ == "__main__":
    image_path = "print3.png"  # Change to your image path
    text_type, recognized_text = classify_text_type(image_path)

    print(f"Detected text type: {text_type}")
    print(f"Recognized text: {recognized_text}")
