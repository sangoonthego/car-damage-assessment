import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, UploadFile, File
from app.query import save_prediction_log, get_prediction, get_prediction_label, get_prediction_image, update_prediction, delete_uncorrect_prediction
from app.query import UpdateLog
from app.model_loader import load_model
from app.predict_image import predict_utils
import json
from scripts.utils import model_path, class_names

app = FastAPI()

with open("class_names.json", "w") as file:
    json.dump(class_names, file)
    
model = load_model(model_path, num_classes=len(class_names))

with open("class_names.json", "r") as file:
    class_names = json.load(file)

@app.post("/predict")
async def predict(img_file: UploadFile = File(...)):
    image_bytes = await img_file.read()
    pred_class, confidence = predict_utils(image_bytes, model, class_names)
    
    save_prediction_log(img_file.filename, pred_class, confidence)

    return {
        "filename": img_file.filename,
        "predicted": pred_class,
        "confidence": confidence
    }

@app.get("/get")
def get_all():
    get_log = get_prediction()

    if not get_log:
        raise HTTPException(status_code=404, detail="Not Image Exist")
    
    return get_log

@app.get("/get/{pred_class}")
def get_by_label(pred_class: str):
    get_pred_class = get_prediction_label(pred_class)

    if not get_pred_class:
        raise HTTPException(status_code=404, detail=f"Label Named {pred_class} not Exist")
    
    return get_pred_class

@app.get("/get/{img_name}")
def get_by_name(img_name: str):
    get_img_name = get_prediction_image(img_name)

    if not get_img_name:
        raise HTTPException(status_code=404, detail=f"Image Named {img_name} not Exist")
    
    return get_img_name

@app.put("/update/{id}")
def update(id: int, update: UpdateLog):
    update_log = update_prediction(id, update.corrected_label)

    if not update_log:
        raise HTTPException(status_code=404, detail=f"{id} Not Exist")
    
    return {
        "message": "Updated"
    }

@app.delete("/delete/{id}")
def delete(id: int):
    delete_log = delete_uncorrect_prediction(id)
    
    if not delete_log:
        raise HTTPException(status_code=404, detail=f"{id} Not Exist")
    
    return {
        "message": "Deleted"
    }

