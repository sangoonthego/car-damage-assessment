import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.predict_query import PredictionLogManager, UpdateLog
from app.detect_query import DetectionLogManager
from app.segment_query import SegmentationLogManager

from app.model_loader import ModelLoader

from app.predict_image import ImagePredictor
from app.detect_utils import ObjectDetector
from app.segment_utils import ObjectSegmenter

from scripts.utils import model_path, class_names

class CarDamageAssessmentAPI:
    def __init__(self):
        self.app = FastAPI()
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        model_loader = ModelLoader(model_path, num_classes=len(class_names))
        self.model = model_loader.load()
        self.class_names = class_names
        self.register_routes()

    async def save_temp_file(self, upload_file: UploadFile, temp_dir: str) -> str:
        os.makedirs(temp_dir, exist_ok=True)
        img_path = os.path.join(temp_dir, upload_file.filename)
        with open(img_path, "wb") as file:
            file.write(await upload_file.read())
        return img_path

    def register_routes(self):
        app = self.app

        # Segmentations
        @app.post("/segment")
        async def segment_api(img_file: UploadFile = File(...)):
            temp_dir = "temp_seg_images"
            img_path = await self.save_temp_file(img_file, temp_dir)

            segment = ObjectSegmenter()
            segmentations = segment.segment_objects(img_path)

            segment_log = SegmentationLogManager()
            segment_log.save_segmentation_log(img_file.filename, segmentations)

            os.remove(img_path)
            return {
                "filename": img_file.filename, 
                "segmentations": segmentations
            }

        @app.get("/get/segmentations")
        def get_all_segment():
            segment_log = SegmentationLogManager()
            get_segment = segment_log.get_segmentation()
            if not get_segment:
                raise HTTPException(status_code=404, detail="No segment image Exist")
            return get_segment

        @app.delete("/delete/segmentations/{id}")
        def delete_segment(id: int):
            segment_log = SegmentationLogManager()
            delete_log = segment_log.delete_segmentation(id)
            if not delete_log:
                raise HTTPException(status_code=404, detail="ID {id} not Exist")
            return delete_log

        # Detections
        @app.post("/detect")
        async def detect_api(img_file: UploadFile = File(...)):
            temp_dir = "temp"
            img_path = await self.save_temp_file(img_file, temp_dir)

            detect = ObjectDetector()
            detections = detect.detect_objects(img_path)

            detect_log = DetectionLogManager()
            detect_log.save_detection_log(img_file.filename, detections)

            output_url = f"static/output/{img_file.filename}"
            os.remove(img_path)
            return {
                "filename": img_file.filename,
                "detections": detections,
                "output_image": output_url
            }

        @app.get("/get/detections")
        def get_all_detect():
            detect_log = DetectionLogManager() 
            get_detect = detect_log.get_detection()
            if not get_detect:
                raise HTTPException(status_code=404, detail="Not detected image Exist")
            return get_detect

        @app.get("/get/detections/label/{detected_class}")
        def get_detect_by_label(detected_class: str):
            detect_log = DetectionLogManager()
            get_detect_label = detect_log.get_detection_label(detected_class)
            if not get_detect_label:
                raise HTTPException(status_code=404, detail=f"{detected_class} not Exist")
            return get_detect_label

        @app.get("/get/detections/image/{img_name}")
        def get_detect_by_name(img_name: str):
            detect_log = DetectionLogManager()
            get_detect_name = detect_log.get_detection_image(img_name)
            if not get_detect_name:
                raise HTTPException(status_code=404, detail=f"{img_name} not Exist")
            return get_detect_name

        @app.delete("/delete/detections/{id}")
        def delete_detect(id: int):
            detect_log = DetectionLogManager()
            delete_log = detect_log.delete_detection(id)
            if not delete_log:
                raise HTTPException(status_code=404, detail=f"ID {id} not Exist")
            return delete_log

        # Predictions
        @app.post("/predict")
        async def predict_api(img_file: UploadFile = File(...)):
            image_bytes = await img_file.read()

            predict = ImagePredictor(self.model, self.class_names)
            pred_class, confidence = predict.predict(image_bytes)

            predict_log = PredictionLogManager()
            predict_log.save_prediction_log(img_file.filename, pred_class, confidence)
            return {
                "filename": img_file.filename,
                "predicted": pred_class,
                "confidence": confidence
            }

        @app.get("/get/predictions")
        def get_all_pred():
            predict_log = PredictionLogManager()
            get_log = predict_log.get_prediction()
            if not get_log:
                raise HTTPException(status_code=404, detail="Not Image Exist")
            return get_log

        @app.get("/get/predictions/confidence/{confidence}")
        def get_best_accuracy_pred(confidence: float):
            predict_log = PredictionLogManager()
            get_best_conf = predict_log.get_best_confidence(confidence)
            if not get_best_conf:
                raise HTTPException(status_code=404, detail=f"Not Prediction > {confidence}")
            return get_best_conf

        @app.get("/get/predictions/label/{pred_class}")
        def get_pred_by_label(pred_class: str):
            predict_log = PredictionLogManager()
            get_pred_class = predict_log.get_prediction_label(pred_class)
            if not get_pred_class:
                raise HTTPException(status_code=404, detail=f"Label Named {pred_class} not Exist")
            return get_pred_class

        @app.get("/get/predictions/label/{pred_class}/confidence/{confidence}")
        def get_best_pred(pred_class: str, confidence: float):
            predict_log = PredictionLogManager()
            get_best_pred = predict_log.get_best_prediction_label(pred_class, confidence)
            if not get_best_pred:
                raise HTTPException(status_code=404, detail=f"Not Exist Prediction > {confidence}")
            return get_best_pred

        @app.get("/get/predictions/image/{img_name}")
        def get_pred_by_name(img_name: str):
            predict_log = PredictionLogManager()
            get_img_name = predict_log.get_prediction_image(img_name)
            if not get_img_name:
                raise HTTPException(status_code=404, detail=f"Image Named {img_name} not Exist")
            return get_img_name

        @app.put("/update/predictions/{id}")
        def update_pred(id: int, update: UpdateLog):
            predict_log = PredictionLogManager()
            update_log = predict_log.update_prediction(id, update.corrected_label)
            if not update_log:
                raise HTTPException(status_code=404, detail=f"{id} Not Exist")
            return {"message": "Updated"}

        @app.delete("/delete/predictions/{id}")
        def delete_pred(id: int):
            predict_log = PredictionLogManager()
            delete_log = predict_log.delete_uncorrect_prediction(id)
            if not delete_log:
                raise HTTPException(status_code=404, detail=f"{id} Not Exist")
            return {"message": "Deleted"}

api = CarDamageAssessmentAPI()
app = api.app
