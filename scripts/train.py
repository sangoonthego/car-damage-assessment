import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
from tqdm import tqdm

from dataset_loader import DatasetLoader
from resnet_model import ResNet18Classifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scripts.utils import epochs, batch_size, patience, lr, model_path, device
from scripts.utils import train_dir, val_dir, test_dir

class Trainer:
    def __init__(self, train_dir, val_dir, test_dir, batch_size, lr, model_path, device, epochs, patience, log_path):
        dataset = DatasetLoader(train_dir, val_dir, test_dir, batch_size)
        self.train_loader = dataset.train_loader
        self.val_loader = dataset.val_loader
        self.class_names = dataset.class_names

        data_loader = ResNet18Classifier(num_classes=len(self.class_names)).to(device)
        self.model = data_loader.model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.model_path = model_path
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.log_path = log_path
        self.log = []

    def train(self):
        best_val_loss = float("inf")
        early_stop_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_preds = outputs.argmax(dim=1)
                train_correct += (train_preds == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            val_loss, val_acc = self.validate()

            pre_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            sub_lr = self.optimizer.param_groups[0]["lr"]

            if sub_lr < pre_lr:
                print(f"LR reduced from {pre_lr:.6f} to {sub_lr:.6f}")

            print(f"Train Loss: {train_loss:.2f} - Train Acc: {train_acc:.2f}")
            print(f"Val Loss: {val_loss:.2f} - Val Acc: {val_acc:.2f}")

            self.log.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    break

        self.save_log()
        self.plot_loss()

    def validate(self):
        self.model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_preds = outputs.argmax(dim=1)
                val_correct += (val_preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        return val_loss, val_acc

    def save_log(self):
        df_log = pd.DataFrame(self.log)
        df_log.to_csv(self.log_path, index=False)

    def plot_loss(self):
        df_log = pd.DataFrame(self.log)
        plt.plot(df_log["train_loss"], label="Train Loss")
        plt.plot(df_log["val_loss"], label="Val Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("models/loss_plot.png")
        plt.show()

if __name__ == "__main__":
    log_path = "models/train_log.csv"
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    trainer = Trainer(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        lr=lr,
        model_path=model_path,
        device=device,
        epochs=epochs,
        patience=patience,
        log_path=log_path
    )
    trainer.train()