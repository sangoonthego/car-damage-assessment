import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from scripts.utils import batch_size, train_transform, test_transform

def get_dataloader(train_dir, val_dir, test_dir, batch_size=batch_size):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

# train_loader, val_loader, test_loader, class_names = get_dataloader()
# print("Class:", class_names)
# for images, labels in train_loader:
#     print("Batch shape:", images.shape)
#     print("Labels:", labels[:8])
#     break