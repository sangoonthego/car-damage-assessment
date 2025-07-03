import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model_loader import load_model
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from resnet_model import get_resnet18

img_size = 224
model_path = "models/car_resnet18_model_best.pth"
test_dir = "data_split/test"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class_names = sorted(os.listdir(test_dir))

model = load_model(model_path, len(class_names))

def predict_image(image_path, model, class_names):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)[0]
        pred_index = torch.argmax(prob).item()
        pred_class = class_names[pred_index]
        confidence = prob[pred_index].item()

        return pred_class, confidence
    
image_path = "image/10.jpeg"
pred_class, confidence = predict_image(image_path, model, class_names)
print(f"{pred_class}: {confidence}")

# test new downloaded image
# predict_dir = "predict_image"
# for img_name in os.listdir(predict_dir):
#     if img_name.lower().endswith(".jpg", ".jpeg", ".png"):
#         path = os.path.join(img_name, predict_dir)
#         label, confidence = predict_image(path, model, class_names)
#         print(f"{label}: {confidence}")

#         for i, name in enumerate(class_names):
#             print(f"{name}: {confidence[i]:.2f}")

#         image = Image.open(path).convert("RGB")
#         plt.imshow(image)
#         plt.title(f"{img_name}: {label}")
#         plt.axis("off")
#         plt.show()
