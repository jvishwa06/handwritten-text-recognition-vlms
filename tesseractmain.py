import cv2
import pytesseract
import numpy as np
import os

# Set the path to Tesseract-OCR (update this if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows users

def preprocess_image(image_path):
    """Preprocess the image for better OCR results."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply binarization
    return binary

def extract_text(image_path):
    """Extract text using Tesseract OCR."""
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6')  # OCR extraction
    return text.strip()

def extract_image_features(image_path):
    """Extract image-based features to classify handwritten vs digital text."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Edge detection using Canny
    edges = cv2.Canny(image, 50, 150)  # Detect edges
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])  # Calculate edge density

    # HOG feature extraction
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    resized_image = cv2.resize(image, (64, 128))  # Resize for HOG
    hog_features = hog.compute(resized_image)
    hog_variance = np.var(hog_features)  # Variance of HOG features

    return edge_density, hog_variance

def classify_text(text, edge_density, hog_variance):
    """Classify as handwritten or digital based on text and image features."""
    digit_confidence = sum(char.isdigit() for char in text) / len(text) if text else 0
    non_alpha_count = sum(not char.isalnum() and char != ' ' for char in text) / len(text) if text else 0

    # Edge density higher than a threshold means more stroke variations (likely handwritten)
    # HOG variance lower means uniform text (likely printed)
    if edge_density > 0.02 and hog_variance > 0.01 and digit_confidence < 0.3 and non_alpha_count > 0.15:
        return "Handwritten Text"
    return "Digital/Printed Text"

def main():
    images = ["hand1.jpg", "digital.jpg"]  # Replace with actual file paths
    for image_path in images:
        if os.path.exists(image_path):
            print(f"\nProcessing: {image_path}")
            extracted_text = extract_text(image_path)
            edge_density, hog_variance = extract_image_features(image_path)
            classification = classify_text(extracted_text, edge_density, hog_variance)
            
            print(f"Extracted Text:\n{extracted_text}\n")
            print(f"Edge Density: {edge_density:.4f}, HOG Variance: {hog_variance:.4f}")
            print(f"Classification: {classification}")
        else:
            print(f"Error: File {image_path} not found!")

if __name__ == "__main__":
    main()
