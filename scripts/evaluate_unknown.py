import os
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image
from resnet_model import get_resnet18

img_size = 224
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/car_resnet18_model_best.pth"
unknown_dir = "data_split/test_unknown/unknown"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

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

def load_model(model_path, num_classes):
    model = get_resnet18(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(model_path, len(class_names))

print(f"Sum of Unknown Images: {len(unknown_dataset)}")

results = []

with torch.no_grad():
    for images, filenames in unknown_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        for i in range(images.size(0)):
            pred_class = class_names[preds[i].item()]
            confidence = probs[i][preds[i].item()]
            img_name = filenames[i]
            results.append((img_name, pred_class, confidence))

            if confidence < 0.5:
                print(f"{img_name}: Predicted: {pred_class}, Confidence: {confidence:.2f} -> Low Confidence")
            else:
                print(f"{img_name}: Predicted: {pred_class}, COnfidence: {confidence:.2f}")