import os
import pandas as pd
import shutil
from tqdm import tqdm

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_unknown_data(csv_path):
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].apply(lambda x: os.path.basename(x))
    df_unknown = df[df["classes"] == "unknown"]
    return df_unknown

def copy_unknown_images(df_unknown, image_dir, output_dir):
    print(f"Sum of Unknown Images: {len(df_unknown)}")
    create_dir(output_dir)
    
    for _, row in tqdm(df_unknown.iterrows(), total=len(df_unknown), desc="Copying unknown images"):
        src_path = os.path.join(image_dir, row["image"])
        dst_path = os.path.join(output_dir, row["image"])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Missing file: {src_path}")

def split_unknown_data(csv_path, image_dir, output_dir):
    df_unknown = load_unknown_data(csv_path)
    copy_unknown_images(df_unknown, image_dir, output_dir)

if __name__ == "__main__":
    split_unknown_data(
        csv_path="data.csv",
        image_dir="image/",
        output_dir="data_split/test_unknown/unknown/"
    )