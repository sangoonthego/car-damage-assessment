import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_loader import load_model
from app.predict_image import predict_utils
from scripts.utils import model_path
from scripts.utils import test_dir

class_names = sorted(os.listdir(test_dir))

model = load_model(model_path, len(class_names))

image_path = "image/10.jpeg"

if not os.path.exists(image_path):
    print("FIle not Found!!!")

with open(image_path, "rb") as file:
    image_bytes = file.read()

pred_class, confidence = predict_utils(image_bytes, model, class_names)
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
