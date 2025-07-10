import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
import uuid
from PIL import Image
from ultralytics import YOLO

segment_model_path = "runs/segment/train2/weights/best.pt"
segment_model = YOLO(segment_model_path)

seg_output_dir = "static/segment_masks"
os.makedirs(seg_output_dir, exist_ok=True)

def segment_objects(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    results = segment_model.predict(source=image_np, save=False, conf=0.25, verbose=False)
    
    segmentations = []
    for result in results:
        boxes = result.boxes
        masks = result.masks
        names = result.names

        if masks is None:
            continue

        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = names[class_id]

            mask = masks.data[i].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            mask_filename = f"{uuid.uuid4().hex}_{class_name}.png"
            mask_path = os.path.join(seg_output_dir, mask_filename)

            cv2.imwrite(mask_path, mask)

            segmentations.append({
                "class": class_name,
                "confidence": confidence,
                "mask_path": mask_path.replace("\\", "/")
            })

    return segmentations


    