import os 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from dataset_loader import get_dataloader
from resnet_model import get_resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau

epochs = 30
img_size = 224
batch_size = 32
patience = 5
lr = 0.001
model_path = "models/car_resnet18_model_best.pth"
log_path = "models/train_log.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_dir = "data_spilt/train"
val_dir = "data_split/val"
test_dir = "data_split/test"

train_loader, val_loader, _, class_names = get_dataloader(train_dir, val_dir, test_dir, batch_size)

model = get_resnet18(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

best_val_loss = float("inf")
early_stop_counter = 0
log = []

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0 ,0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_preds = outputs.argmax(dim=1)
        train_correct += (train_preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_preds = outputs.argmax(dim=1)
            val_correct += (val_preds == labels).sum().item()
            val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        pre_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        sub_lr = optimizer.param_groups[0]["lr"]
        
        if sub_lr < pre_lr:
            print(f"LR reduced from {pre_lr:.6f} to {sub_lr:.6f}")

        print(f"Train Loss: {train_loss:.2f} - Train Acc: {train_acc:.2f}")
        print(f"Val Loss: {val_loss:.2f} - Val Acc: {val_acc:.2f}")

        log.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

df_log = pd.DataFrame(log)
df_log.to_csv(log_path, index=False)

plt.plot(df_log["train_loss"], label="Train Loss")
plt.plot(df_log["val_loss"], label="Val Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("models/loss_plot.png")
plt.show()