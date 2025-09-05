import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from scripts.model_loader import ModelLoader

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scripts.utils import batch_size, model_path, device, test_transform
from scripts.utils import test_dir

class Evaluator:
    def __init__(self, model_path, test_dir, batch_size, device, test_transform):
        self.test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.class_names = self.test_dataset.classes

        model_loader = ModelLoader(model_path, len(self.class_names))
        self.model = model_loader.load()
        self.device = device

    def evaluate(self):
        all_preds = []
        all_actuals = []
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_actuals.extend(labels.cpu().numpy())
        return all_actuals, all_preds

    def save_classification_report(self, all_actuals, all_preds, save_path="models/classification_report.csv"):
        report = classification_report(all_actuals, all_preds, target_names=self.class_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(save_path)
        print("Classification Report:\n", df_report)

    def save_confusion_matrix(self, all_actuals, all_preds, save_path="models/confusion_matrix.png"):
        cm = confusion_matrix(all_actuals, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    evaluator = Evaluator(model_path, test_dir, batch_size, device, test_transform)
    actuals, preds = evaluator.evaluate()
    evaluator.save_classification_report(actuals, preds)
    evaluator.save_confusion_matrix(actuals, preds)

