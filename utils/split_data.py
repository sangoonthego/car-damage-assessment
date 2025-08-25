import os
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_valid_image(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    df = df[df["classes"] != "unknown"]
    df = df.rename(columns={"classes": "label"})
    print(f"Sum of Image (except Unknown): {len(df)}")
    print(f"Classes: {sorted(df['label'].unique())}")
    return df

def plot_class_distribution(df):
    plt.figure(figsize=(8, 6))
    df["label"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Nums of Image per Class (before splitting)")
    plt.xlabel("Class")
    plt.ylabel("Nums of Images")
    plt.tight_layout()
    plt.savefig("models/class_distribution.png")

def split_and_copy_data(df, image_dir, output_dir, split_ratios=(0.7, 0.15, 0.15), seed=42):
    log_data = []
    
    for label in sorted(df["label"].unique()):
        df_label = df[df["label"] == label]
        print(f"Class: {label} | Sum of Image: {len(df_label)}")

        train_df, temp_df = train_test_split(
            df_label, test_size=1 - split_ratios[0], random_state=seed)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]),
            random_state=seed)

        stats = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
        print(f"train: {stats['train']} | val: {stats['val']} | test: {stats['test']}")

        for subset, subset_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
            for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f" [{subset}/{label}]", leave=False):
                src_path = os.path.join(image_dir, row["image"])
                dst_dir = os.path.join(output_dir, subset, row["label"])
                dst_path = os.path.join(dst_dir, row["image"])

                create_dir(dst_dir)

                if is_valid_image(src_path):
                    shutil.copy(src_path, dst_path)
                    log_data.append({"image": row["image"], "label": row["label"], "subset": subset})
                else:
                    print(f"File not found: {src_path}")
    
    return log_data

def save_split_log(log_data, log_path):
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(log_path, index=False)

def print_summary(output_dir):
    for subset in ["train", "val", "test"]:
        print(f"{subset.upper()}:")
        subset_path = os.path.join(output_dir, subset)
        if not os.path.exists(subset_path):
            continue
        for label in sorted(os.listdir(subset_path)):
            class_dir = os.path.join(subset_path, label)
            n = len(os.listdir(class_dir))
            print(f"{label}: {n} images")

def split_data(csv_path, image_dir, output_dir, split_ratios=(0.7, 0.15, 0.15), random_state=42, log_path="models/split_summary.csv"):
    random.seed(random_state)
    df = load_and_prepare_data(csv_path)
    plot_class_distribution(df)
    
    log_data = split_and_copy_data(df, image_dir, output_dir, split_ratios, random_state)
    
    save_split_log(log_data, log_path)
    
    print_summary(output_dir)

if __name__ == "__main__":
    split_data(
        csv_path="data.csv",
        image_dir="image/",
        output_dir="data_split/",
        split_ratios=(0.7, 0.15, 0.15),
        seed=42,
        log_path="models/split_summary.csv"
    )