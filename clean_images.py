import os 
import shutil

class ImageCleaner:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.output_dir = os.path.join(root_dir, "clean_images")
        self.subfolders = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_images(self):
        for split in self.subfolders:
            split_path = os.path.join(self.root_dir, split)
            if not os.path.isdir(split_path):
                continue

            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                for filename in os.listdir(class_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        src_path = os.path.join(class_path, filename)
                        dest_path = os.path.join(self.output_dir, f"{split}_{class_name}_{filename}")
                        try:
                            shutil.copy2(src_path, dest_path)
                        except Exception as e:
                            print(f"Error Copy {src_path}: {e}")

if __name__ == "__main__":
    root_dir = r"D:\saves\AI Research\Project\car_damage_assessment\data_split"
    cleaner = ImageCleaner(root_dir)
    cleaner.clean_images()
