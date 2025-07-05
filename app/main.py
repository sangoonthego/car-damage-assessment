import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, UploadFile, File
from app.database import save_prediction_log, get_prediction_log, get_prediction_log_by_image, update_prediction_log, delete_uncorrect_prediction_log
from app.database import UpdateLog
from app.model_loader import load_model
from app.predict_image import predict_utils
import io
import json

app = FastAPI()

model_path = "models/car_resnet18_model_best.pth"
class_names = ["bumper_dent", "bumper_scratch", "door_dent", "door_scratch", "glass_shatter", "head_lamp", "tail_lamp"]
with open("class_names.json", "w") as file:
    json.dump(class_names, file)
    
model = load_model(model_path, num_classes=len(class_names))

with open("class_names.json", "r") as file:
    class_names = json.load(file)

@app.post("/predict_logs")
async def predict_log(img_file: UploadFile = File(...)):
    image_bytes = await img_file.read()
    pred_class, confidence = predict_utils(image_bytes, model, class_names)
    
    save_prediction_log(img_file.filename, pred_class, confidence)

    return {
        "filename": img_file.filename,
        "predicted": pred_class,
        "confidence": confidence
    }

@app.get("/get_logs")
def get_all_logs():
    get_log = get_prediction_log()

    if not get_log:
        raise HTTPException(status_code=404, detail="Not Image Exist")
    
    return get_log

@app.get("/get_logs/{img_name}")
def get_logs_by_name(img_name: str):
    get_img_name = get_prediction_log_by_image(img_name)

    if not get_img_name:
        raise HTTPException(status_code=404, detail=f"Image Named {img_name} not Exist")
    
    return get_img_name

@app.put("/update_logs/{id}")
def update_logs(id: int, update: UpdateLog):
    update_log = update_prediction_log(id, update.corrected_label)

    if not update_log:
        raise HTTPException(status_code=404, detail=f"{id} Not Exist")
    
    return {
        "message": "Updated"
    }

@app.delete("/delete_logs/{id}")
def delete_logs(id: int):
    delete_log = delete_uncorrect_prediction_log(id)
    
    if not delete_log:
        raise HTTPException(status_code=404, detail=f"{id} Not Exist")
    
    return {
        "message": "Deleted"
    }

