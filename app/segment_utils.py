import os
import sys
import uuid
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from app.severity_level import SeverityLevel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class ObjectSegmenter:
    def __init__(self):
        pass
    
    def __init__(self, model_path="runs/segment/train2/weights/best.pt", output_dir="static/segment_masks"):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def segment_objects(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        results = self.model.predict(source=image_np, save=False, conf=0.25, verbose=False)
        
        segmentations = []
        for result in results:
            boxes = result.boxes
            masks = result.masks

            if masks is None:
                continue

            for i, box in enumerate(boxes):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                segment_class = self.model.names[class_id]
                level = SeverityLevel(segment_class)
                severity = level.severity

                mask = masks.data[i].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)

                mask_filename = f"{uuid.uuid4().hex}_{segment_class}.png"
                mask_path = os.path.join(self.output_dir, mask_filename)

                cv2.imwrite(mask_path, mask)

                segmentations.append({
                    "class": segment_class,
                    "confidence": confidence,
                    "severity": severity,
                    "mask_path": mask_path.replace("\\", "/"),
                })

        return segmentations
