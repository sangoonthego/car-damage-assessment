import os
import pandas as pd
import shutil
from tqdm import tqdm

class UnknownImageSplitter:
    def __init__(self, csv_path, image_dir, output_dir):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.df = None
        self.df_unknown = None

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df["image"] = self.df["image"].apply(lambda x: os.path.basename(x))
        self.df_unknown = self.df[self.df["classes"] == "unknown"]

    def copy_unknown_images(self):
        print(f"Sum of Unknown Images: {len(self.df_unknown)}")
        self.create_dir(self.output_dir)
        for _, row in tqdm(self.df_unknown.iterrows(), total=len(self.df_unknown), desc="Copying unknown images"):
            src_path = os.path.join(self.image_dir, row["image"])
            dst_path = os.path.join(self.output_dir, row["image"])
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Missing file: {src_path}")

    def run(self):
        self.load_data()
        self.copy_unknown_images()

if __name__ == "__main__":
    splitter = UnknownImageSplitter(
        csv_path="data.csv",
        image_dir="image/",
        output_dir="data_split/test_unknown/unknown/"
    )
    splitter.run()