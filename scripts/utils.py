import torch
from torchvision import transforms

epochs = 30
img_size = 224
batch_size = 32
patience = 5
threshold = 0.5
lr = 0.001
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_dir = "data_split/train"
val_dir = "data_split/val"
test_dir = "data_split/test"
model_path = "models/car_resnet18_model_best.pth"
class_names = ["bumper_dent", "bumper_scratch", "door_dent", "door_scratch", "glass_shatter", "head_lamp", "tail_lamp"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])