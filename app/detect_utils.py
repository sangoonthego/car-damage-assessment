import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from ultralytics import YOLO

detect_model_path = "runs/detect/train2/weights/best.pt"
detect_model = YOLO(detect_model_path)

def detect_objects(image_path, save_annotated=True):
    detect_model.conf = 0.25
    results = detect_model(image_path)

    detections = []
    result = results[0]
    
    boxes = result.boxes
    for box in boxes:
        detect_class = int(box.cls.item())
        confidence = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "class": detect_model.names[detect_class],
            "confidence": confidence,
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3]
        })


    if save_annotated:
        os.makedirs("static/output", exist_ok=True)
        annotated_img = result.plot()
        output_path = f"static/output/{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated_img)

    return detections