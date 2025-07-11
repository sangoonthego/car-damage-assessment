import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from PIL import Image

from app.model_loader import ModelLoader
from app.predict_image import ImagePredictor
from scripts.utils import model_path, device, transform
from scripts.utils import test_dir

class CarDamagePredictor:
    def __init__(self, model_path, test_dir):
        self.class_names = sorted(os.listdir(test_dir))
        model_loader = ModelLoader(model_path, len(self.class_names))
        self.model = model_loader.load()

    def predict_image_bytes(self, image_bytes):
        predict_loader = ImagePredictor(self.model, self.class_names)
        pred_class, confidence = predict_loader.predict(image_bytes)
        print(f"{pred_class}: {confidence}")
        return pred_class, confidence

    def predict_image_file(self, image_path):
        if not os.path.exists(image_path):
            print("File not Found!!!")
            return None, None
        with open(image_path, "rb") as file:
            image_bytes = file.read()
        return self.predict_image_bytes(image_bytes)

    # def predict_directory(self, image_bytes, predict_dir):
    #     for img_name in os.listdir(predict_dir):
    #         if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
    #             path = os.path.join(predict_dir, img_name)
    #             predict_loader = ImagePredictor(image_bytes, self.model, self.class_names)
    #             label, confidence = predict_loader.predict()
    #             print(f"{label}: {confidence}")

    #             for i, name in enumerate(self.class_names):
    #                 print(f"{name}: {confidence[i]:.2f}")

    #             image = Image.open(path).convert("RGB")
    #             plt.imshow(image)
    #             plt.title(f"{img_name}: {label}")
    #             plt.axis("off")
    #             plt.show()

if __name__ == "__main__":
    predictor = CarDamagePredictor(model_path, test_dir)
    predictor.predict_image_file("image/10.jpeg")
    # predictor.predict_directory("predict_image")