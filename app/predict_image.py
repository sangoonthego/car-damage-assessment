import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_size = 224

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

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