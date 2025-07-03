import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.resnet_model import get_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes):
    model = get_resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model