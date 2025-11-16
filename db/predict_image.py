import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image
from scripts.utils import device, transform  

class ImagePredictor:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.transform = transform

    def predict(self, image_bytes):
        image = Image.open(image_bytes).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = F.softmax(output, dim=1)[0]
            pred_index = torch.argmax(prob).item()
            pred_class = self.class_names[pred_index]
            confidence = prob[pred_index].item()

        return pred_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]

    # Load model
    model_path = "runs/classify/train2/weights/best.pt"
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Class names 
    class_names = ["minor", "moderate", "severe"]  

    predictor = ImagePredictor(model, class_names)
    with open(image_path, "rb") as f:
        pred_class, confidence = predictor.predict(f)

    print(json.dumps({"class": pred_class, "confidence": confidence}))

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import torch
# import torch.nn.functional as F
# from PIL import Image
# from io import BytesIO
# from scripts.utils import device, transform

# class ImagePredictor:
#     def __init__(self, model, class_names):
#         self.model = model
#         self.class_names = class_names
#         self.device = device
#         self.transform = transform

#     def predict(self, image_bytes):
#         image = Image.open(BytesIO(image_bytes)).convert("RGB")
#         input_tensor = self.transform(image).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             output = self.model(input_tensor)
#             prob = F.softmax(output, dim=1)[0]
#             pred_index = torch.argmax(prob).item()
#             pred_class = self.class_names[pred_index]
#             confidence = prob[pred_index].item()

#         return pred_class, confidence