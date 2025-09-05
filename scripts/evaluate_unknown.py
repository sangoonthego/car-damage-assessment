import os
import pandas as pd
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.model_loader import ModelLoader
from scripts.utils import batch_size, threshold, device, model_path, transform

class UnknownDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.image_paths = [
            os.path.join(folder, fname) for fname in os.listdir(folder)
            if fname.lower().endswith((".jpeg", ".png", ".jpg"))
        ]
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = self.loader(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(self.image_paths[idx])

class UnknownEvaluator:
    def __init__(self, unknown_dir, train_dir, model_path, log_path, batch_size, threshold, device, transform):
        self.unknown_dir = unknown_dir
        self.train_dir = train_dir
        self.model_path = model_path
        self.log_path = log_path
        self.batch_size = batch_size
        self.threshold = threshold
        self.device = device
        self.transform = transform

        self.class_names = sorted(os.listdir(self.train_dir))
        self.dataset = UnknownDataset(self.unknown_dir, self.transform)
        self.loader = DataLoader(self.dataset, self.batch_size, shuffle=False)

        model_loader = ModelLoader(self.model_path, len(self.class_names))
        self.model = model_loader.load()

    def evaluate(self):
        print(f"Sum of Unknown Images: {len(self.dataset)}")
        results = []
        with torch.no_grad():
            for images, img_names in self.loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                for i in range(images.size(0)):
                    pred_class = self.class_names[preds[i].item()]
                    confidence = probs[i][preds[i].item()]
                    img_name = img_names[i]

                    results.append({
                        "img_name": img_name, 
                        "predicted_label": pred_class, 
                        "confidence": confidence.item()
                    })

                    if confidence < self.threshold:
                        print(f"{img_name}: Predicted: {pred_class}, Confidence: {confidence:.2f} -> Low Confidence")
                    else:
                        print(f"{img_name}: Predicted: {pred_class}, Confidence: {confidence:.2f}")

        df = pd.DataFrame(results)
        df = df.sort_values(by="confidence", ascending=False)
        df.to_csv(self.log_path, index=False)

if __name__ == "__main__":
    unknown_dir = "data_split/test_unknown/unknown"
    train_dir = "data_split/train"
    log_path = "unknown_evaluate.csv"
    evaluator = UnknownEvaluator(
        unknown_dir=unknown_dir,
        train_dir=train_dir,
        model_path=model_path,
        log_path=log_path,
        batch_size=batch_size,
        threshold=threshold,
        device=device,
        transform=transform
    )
    evaluator.evaluate()