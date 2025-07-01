import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

img_size = 224
batch_size = 32
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def get_dataloader(train_transform=train_transform, test_transform=test_transform,batch_size=batch_size):
    train_dataset = datasets.ImageFolder("data_split/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("data_split/val", transform=test_transform)
    test_dataset = datasets.ImageFolder("data_split/test", transform=test_transform)

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