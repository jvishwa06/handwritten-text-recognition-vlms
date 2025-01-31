import cv2
import numpy as np
import tensorflow as tf
import keras

class PrintedWrittenClassifier():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def preprocess(self, image):
        IMG_SIZE = (256, 256)
        im = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
        im = im / 255.0 
        im = np.expand_dims(im, axis=0)
        return im

    def predict(self, img):
        return self.model.predict(img)[0, 0]

    def postprocess(self, y_pred):
        print(f"Raw Model Output: {y_pred}")  
        if y_pred < 0.5:  
            return 0  # Printed Text
        return 1  # Handwritten Text


if __name__ == "__main__":
    model_path = "model.h5"
    classifier = PrintedWrittenClassifier(model_path)

    image_path = "hand3.png"
    image = cv2.imread(image_path)
    image = classifier.preprocess(image)

    y_pred = classifier.predict(image)
    output = classifier.postprocess(y_pred)

    print(f"Predicted Value: {y_pred}, Output Label: {output}")