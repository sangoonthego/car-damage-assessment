import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles

from app.predict_query import save_prediction_log, get_prediction, get_best_confidence, get_prediction_label, get_best_prediction_label, get_prediction_image, update_prediction, delete_uncorrect_prediction
from app.detect_query import save_detection_log, get_detection, get_detection_image, get_detection_label, delete_detection
from app.segment_query import save_segmentation_log
from app.predict_query import UpdateLog
from app.model_loader import load_model

from app.predict_image import predict_utils
from app.detect_utils import detect_objects
from app.segment_utils import segment_objects

from scripts.utils import model_path, class_names

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
    
model = load_model(model_path, num_classes=len(class_names))

# segmentations
@app.post("/segment")
async def segment_api(img_file: UploadFile = File(...)):
    temp_dir = "temp_seg_images"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, img_file.filename)

    with open(img_path, "wb") as file:
        file.write(await img_file.read())

    segmentations = segment_objects(img_path)

    save_segmentation_log(img_file.filename, segmentations)

    os.remove(img_path)

    return {
        "filename": img_file.filename,
        "segmentations": segmentations
    }

# detections
@app.post("/detect")
async def detect_api(img_file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, img_file.filename)

    with open(img_path, "wb") as file:
        file.write(await img_file.read())

    detections = detect_objects(img_path)

    save_detection_log(img_file.filename, detections)

    output_url = f"static/output/{img_file.filename}"
    os.remove(img_path)

    return {
        "filename": img_file.filename,
        "detections": detections,
        "output_image": output_url
    }

@app.get("/get/detections")
def get_all_detect():
    get_detect = get_detection()

    if not get_detect:
        raise HTTPException(status_code=404, detail="Not detected image Exist")
    
    return get_detect

@app.get("/get/detections/label/{detected_class}")
def get_detect_by_label(detected_class: str):
    get_detect_label = get_detection_label(detected_class)

    if not get_detect_label:
        raise HTTPException(status_code=404, detail=f"{detected_class} not Exist")
    
    return get_detect_label

@app.get("/get/detections/image/{img_name}")
def get_detect_by_name(img_name: str):
    get_detect_name = get_detection_image(img_name)

    if not get_detect_name:
        raise HTTPException(status_code=404, detail=f"{img_name} not Exist")
    
    return get_detect_name

@app.delete("/delete/detections/{id}")
def delete_detect(id: int):
    delete_log = delete_detection(id)

    if not delete_log:
        raise HTTPException(status_code=404, detail=f"ID {id} not Exist")
    
    return delete_log

# predictions
@app.post("/predict")
async def predict_api(img_file: UploadFile = File(...)):
    image_bytes = await img_file.read()
    pred_class, confidence = predict_utils(image_bytes, model, class_names)
    
    save_prediction_log(img_file.filename, pred_class, confidence)

    return {
        "filename": img_file.filename,
        "predicted": pred_class,
        "confidence": confidence
    }

@app.get("/get/predictions")
def get_all_pred():
    get_log = get_prediction()

    if not get_log:
        raise HTTPException(status_code=404, detail="Not Image Exist")
    
    return get_log

@app.get("/get/predictions/confidence/{confidence}")
def get_best_accuracy_pred():
    confidence = 0.9
    get_best_conf = get_best_confidence(confidence)

    if not get_best_conf:
        raise HTTPException(status_code=404, detail=f"Not Prediction > 0.9")

    return get_best_conf 

@app.get("/get/predictions/label/{pred_class}")
def get_pred_by_label(pred_class: str):
    get_pred_class = get_prediction_label(pred_class)

    if not get_pred_class:
        raise HTTPException(status_code=404, detail=f"Label Named {pred_class} not Exist")
    
    return get_pred_class

@app.get("/get/predictions/label/{pred_class}/confidence/{confidence}")
def get_best_pred(pred_class: str, confidence: float):
    get_best_pred = get_best_prediction_label(pred_class, confidence)

    if not get_best_pred:
        raise HTTPException(status_code=404, detail=f"Not Exist Prediction > 0.9")
    
    return get_best_pred

@app.get("/get/predictions/image/{img_name}")
def get_pred_by_name(img_name: str):
    get_img_name = get_prediction_image(img_name)

    if not get_img_name:
        raise HTTPException(status_code=404, detail=f"Image Named {img_name} not Exist")
    
    return get_img_name

@app.put("/update/predictions/{id}")
def update_pred(id: int, update: UpdateLog):
    update_log = update_prediction(id, update.corrected_label)

    if not update_log:
        raise HTTPException(status_code=404, detail=f"{id} Not Exist")
    
    return {
        "message": "Updated"
    }

@app.delete("/delete/predictions/{id}")
def delete_pred(id: int):
    delete_log = delete_uncorrect_prediction(id)
    
    if not delete_log:
        raise HTTPException(status_code=404, detail=f"{id} Not Exist")
    
    return {
        "message": "Deleted"
    }

