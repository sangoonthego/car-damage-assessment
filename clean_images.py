import os 
import shutil

root_dir = r"D:\saves\AI Research\Project\car_damage_assessment\data_split"
output_dir = os.path.join(root_dir, "clean_images")

os.makedirs(output_dir, exist_ok=True)

subfolders = ["train", "val", "test"]

for split in subfolders:
    split_path = os.path.join(root_dir, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                src_dir = os.path.join(class_path, filename)
                dest_dir = os.path.join(output_dir, f"{split}_{class_name}_{filename}")
                shutil.copy2(src_dir, dest_dir)