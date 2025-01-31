from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt

# Load the weights from our repository
model = YOLO('best.pt')

path = 'hand1.jpg'
# Do the predictions
res = model.predict(source=path, project='.',name='detected', exist_ok=True, save=True, show=False, show_labels=False, show_conf=False, conf=0.5, )
print(res)