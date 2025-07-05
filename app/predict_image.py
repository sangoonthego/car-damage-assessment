import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from scripts.utils import device, transform

def predict_utils(image_bytes, model, class_names):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)[0]
        pred_index = torch.argmax(prob).item()
        pred_class = class_names[pred_index]
        confidence = prob[pred_index].item()

    return pred_class, confidence