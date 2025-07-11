import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.resnet_model import ResNet18Classifier
from scripts.utils import device

class ModelLoader:
    def __init__(self, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = None

    def load(self):
        model_loader = ResNet18Classifier(num_classes=self.num_classes)
        self.model = model_loader.model
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        return self.model

    def get_model(self):
        if self.model is None:
            return self.load()
        return self.model