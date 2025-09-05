import os
import sys

import cv2
from ultralytics import YOLO
from yolov8.car_severity.severity_level import SeverityLevel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class ObjectDetector:
    def __init__(self, model_path="runs/detect/train2/weights/best.pt", conf=0.25):
        self.model = YOLO(model_path)
        self.model.conf = conf

    def detect_objects(self, image_path, save_annotated=True):
        results = self.model(image_path)
        detections = []
        result = results[0]

        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            detect_class = self.model.names[class_id]
            level = SeverityLevel(detect_class)
            severity = level.severity
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class": detect_class,
                "confidence": confidence,
                "severity": severity,
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
