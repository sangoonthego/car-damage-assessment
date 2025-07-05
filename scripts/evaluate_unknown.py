import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_loader import load_model
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from scripts.utils import batch_size, threshold, device, model_path, transform

log_path = "unknown_evaluate.csv"
unknown_dir = "data_split/test_unknown/unknown"

class UnknownDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.image_paths = [
            os.path.join(folder, fname) for fname in os.listdir(folder)
            if fname.lower().endswith((".jpeg", ".png", ".jpg"))
        ]
        self.transform = transform
        self.loader = default_loader

    # DataLoader knows how many imgs to create batch 
    def __len__(self):
        return len(self.image_paths)
    
    # load imgs at idx pos
    def __getitem__(self, idx):
        image = self.loader(self.image_paths[idx])
        if self.transform:
            image = transform(image)

        return image, os.path.basename(self.image_paths[idx])
    
unknown_dataset = UnknownDataset(unknown_dir, transform)
unknown_loader = DataLoader(unknown_dataset, batch_size, shuffle=False)

class_names = sorted(os.listdir("data_split/train"))

model = load_model(model_path, len(class_names))

print(f"Sum of Unknown Images: {len(unknown_dataset)}")

results = []

with torch.no_grad():
    for images, img_names in unknown_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        for i in range(images.size(0)):
            pred_class = class_names[preds[i].item()]
            confidence = probs[i][preds[i].item()]
            img_name = img_names[i]

            results.append({
                "img_name": img_name, 
                "predicted_label": pred_class, 
                "confidence": confidence
            })

            if confidence < threshold:
                print(f"{img_name}: Predicted: {pred_class}, Confidence: {confidence:.2f} -> Low Confidence")
            else:
                print(f"{img_name}: Predicted: {pred_class}, Confidence: {confidence:.2f}")

df = pd.DataFrame(results)
df = df.sort_values(by="confidence", ascending=False)
df.to_csv(log_path, index=False)
            