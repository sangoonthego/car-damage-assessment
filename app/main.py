from fastapi import FastAPI, UploadFile, File
from app.database import save_prediction_log
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
