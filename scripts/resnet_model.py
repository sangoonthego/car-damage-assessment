import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad=False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model