# Handwritten vs Printed Text Recognition

Experiments in reading text from images and deciding whether it is handwritten or printed. Each script tries a different approach, from classic OCR to a vision-language model.

## Approaches

| Script | Approach |
|---|---|
| `qwenvl-image.py` | Prompts the vision-language model `Qwen/Qwen2-VL-7B-Instruct` (Hugging Face Transformers) to say whether the text in an image is handwritten or printed, and which part is which when both appear. |
| `transformersmain.py` | Runs Microsoft's `trocr-base-handwritten` and `trocr-base-printed` models on the image and compares their outputs and confidence scores to classify the text type. |
| `tesseractmain.py` | OCR with Tesseract (`--psm 6`), then classifies from image features: edge density and HOG variance. |
| `paddleocrmain.py` | Despite its name, it imports EasyOCR and pytesseract. It runs both on the image and classifies the text type from the results. |
| `tensorflowmain.py` | A Keras binary classifier (`PrintedWrittenClassifier`) that loads `model.h5` and returns 0 for printed or 1 for handwritten. |
| `youonlylookonce.py` | Runs a YOLO model (`best.pt`) on an image with Ultralytics and saves the annotated result to `detected/`. |
| `qwenvl-pdf.py` | Empty file. |

## Setup

Install the packages for the script you want to run. They use:

- `transformers`, `qwen_vl_utils`, `torch` (Qwen2-VL and TrOCR)
- `pytesseract` plus the Tesseract program, `easyocr`, `opencv-python`, `Pillow`, `numpy`
- `tensorflow` / `keras`
- `ultralytics`

`tesseractmain.py` sets the Windows install path of Tesseract (`C:\Program Files\Tesseract-OCR\tesseract.exe`), so change it on other systems. The model files `model.h5` and `best.pt` and the sample image are not downloaded by these scripts, so add your own.

## Run

```bash
python qwenvl-image.py
python transformersmain.py
```
