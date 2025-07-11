import os
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataSplitter:
    def __init__(self, csv_path, image_dir, output_dir, split_ratios=(0.7, 0.15, 0.15), seed=42, log_path="models/split_summary.csv"):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.split_ratios = split_ratios
        self.seed = seed
        self.log_path = log_path
        random.seed(self.seed)
        self.df = None
        self.log_data = []

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def is_valid_image(self, file_path):
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df["image"] = self.df["image"].apply(lambda x: os.path.basename(x))
        self.df = self.df[self.df["classes"] != "unknown"]
        self.df = self.df.rename(columns={"classes": "label"})
        print(f"Sum of Image (except Unknown): {len(self.df)}")
        print(f"Classes: {sorted(self.df['label'].unique())}")

    def plot_class_distribution(self):
        plt.figure(figsize=(8, 6))
        self.df["label"].value_counts().plot(kind="bar", color="skyblue")
        plt.title("Nums of Image per Class (before splitting)")
        plt.xlabel("Class")
        plt.ylabel("Nums of Images")
        plt.tight_layout()
        plt.savefig("class_distribution.png")

    def split_and_copy(self):
        for label in sorted(self.df["label"].unique()):
            df_label = self.df[self.df["label"] == label]
            print(f"Class: {label} | Sum of Image: {len(df_label)}")

            train_df, temp_df = train_test_split(
                df_label, test_size=1 - self.split_ratios[0], random_state=self.seed)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2]),
                random_state=self.seed)

            stats = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
            print(f"train: {stats['train']} | val: {stats['val']} | test: {stats['test']}")

            for subset, subset_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
                for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f" [{subset}/{label}]", leave=False):
                    src_path = os.path.join(self.image_dir, row["image"])
                    dst_dir = os.path.join(self.output_dir, subset, row["label"])
                    dst_path = os.path.join(dst_dir, row["image"])

                    self.create_dir(dst_dir)

                    if self.is_valid_image(src_path):
                        shutil.copy(src_path, dst_path)
                        self.log_data.append({"image": row["image"], "label": row["label"], "subset": subset})
                    else:
                        print(f"File not found: {src_path}")

        log_df = pd.DataFrame(self.log_data)
        log_df.to_csv(self.log_path, index=False)

    def print_summary(self):
        for subset in ["train", "val", "test"]:
            print(f"{subset.upper()}:")
            subset_path = os.path.join(self.output_dir, subset)
            if not os.path.exists(subset_path):
                continue
            for label in sorted(os.listdir(subset_path)):
                class_dir = os.path.join(subset_path, label)
                n = len(os.listdir(class_dir))
                print(f"{label}: {n} images")

    def run(self):
        self.load_and_prepare_data()
        self.plot_class_distribution()
        self.split_and_copy()
        self.print_summary()

if __name__ == "__main__":
    splitter = DataSplitter(
        csv_path="data.csv",
        image_dir="image/",
        output_dir="data_split/",
        split_ratios=(0.7, 0.15, 0.15),
        seed=42,
        log_path="models/split_summary.csv"
    )
    splitter.run()