from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor_handwritten = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model_handwritten = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

processor_printed = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model_printed = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

def perform_ocr_with_confidence(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    output = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

    generated_text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]

    log_probs = torch.mean(torch.cat(output.scores), dim=0).mean().item()

    return generated_text, log_probs

def classify_text_type(image_path):
    text_handwritten, confidence_handwritten = perform_ocr_with_confidence(image_path, processor_handwritten, model_handwritten)
    text_printed, confidence_printed = perform_ocr_with_confidence(image_path, processor_printed, model_printed)

    print(f"Confidence (Handwritten): {confidence_handwritten:.4f}")
    print(f"Confidence (Printed): {confidence_printed:.4f}")

    if confidence_handwritten > confidence_printed:
        return "Handwritten", text_handwritten
    else:
        return "Printed", text_printed

image_path = 'hand1.jpg' 
text_type, recognized_text = classify_text_type(image_path)

print(f"Detected text type: {text_type}")
print(f"Recognized text: {recognized_text}")
