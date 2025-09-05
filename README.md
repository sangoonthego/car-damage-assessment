# Car Damage Assessment

## Author

- Name: Nguyen Tuan Ngoc
- University: Danang University of Science and Technology (DUT)
- Student ID: 102230087

End-to-end car damage assessment using deep learning:
- Damage classification (ResNet18)
- Object detection and segmentation (YOLOv8)
- Severity estimation
- Web demo (Streamlit) and REST API (FastAPI)

## Tech Stack

- Python 3.9+
- PyTorch, TorchVision
- Ultralytics YOLOv8
- OpenCV, Pillow
- FastAPI, Uvicorn
- Streamlit
- Pandas, scikit-learn, Seaborn, Matplotlib

## Project Structure
```
car_damage_assessment/
├── app/
│   ├── __init__.py
│   └── main.py                    # FastAPI app: classification, detection, segmentation APIs
├── db/
│   ├── __init__.py
│   ├── database.py                # SQLite/MySQL helpers for logs
│   ├── predict_image.py           # Classification inference helper
│   └── predict_query.py           # Pydantic models + log manager
├── scripts/
│   ├── dataset_loader.py          # ImageFolder loaders
│   ├── evaluate.py                # Evaluate classifier, save report + CM
│   ├── evaluate_unknown.py        # Evaluate on unknown split
│   ├── model_loader.py            # Load ResNet18 weights
│   ├── predict.py                 # Single-image prediction helper
│   ├── resnet_model.py            # ResNet18 wrapper
│   ├── train.py                   # Train classifier
│   └── utils.py                   # Hyperparams, transforms, paths
├── yolov8/
│   ├── car_detection/
│   │   └── data.yaml              # YOLO detection dataset config
│   ├── car_segmentation/
│   │   └── data.yaml              # YOLO segmentation dataset config
│   ├── car_severity/
│   │   └── severity_level.py      # Severity mapping utilities
│   └── utils/
│       ├── detect_query.py        # Detection log manager
│       ├── detect_utils.py        # Detection entrypoints
│       ├── segment_query.py       # Segmentation log manager
│       └── segment_utils.py       # Segmentation entrypoints
├── data_split/                    # Dataset splits for classifier
│   ├── clean_images/
│   ├── train/                     # bumper_dent, bumper_scratch, ...
│   ├── val/
│   ├── test/
│   └── test_unknown/
├── models/                        # Place downloaded weights here
│   ├── car_resnet18_model_best.pth
│   ├── yolov8s.pt
│   └── yolov8n-seg.pt
├── static/
│   ├── output/                    # Detection output images
│   └── segment_masks/             # Segmentation mask images
├── screenshots/                   # Demo screenshots
├── notebooks/
├── runs/
├── streamlit_app.py               # Streamlit UI
├── requirements.txt
├── Dockerfile
├── TRAINING.md
├── PIPELINE_ANALYSIS.md
└── LICENSE
```

## Installation

1) Create environment and install dependencies
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2) Download model weights and place in `models/`
- `car_resnet18_model_best.pth`
- `yolov8s.pt`
- `yolov8n-seg.pt`

## Running the Apps

### Streamlit UI
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`.

### FastAPI server
```bash
uvicorn app.main:app --reload --port 8000
```
Open docs at `http://localhost:8000/docs`.

## REST API Endpoints (FastAPI)

- POST `/predict` (form-data file: `img_file`)
  - Response: `{ filename, predicted, confidence }`

- POST `/detect` (form-data file: `img_file`)
  - Response: `{ filename, detections, output_image }`

- POST `/segment` (form-data file: `img_file`)
  - Response: `{ filename, segmentations }`

- GET `/get/predictions` — all classification logs
- GET `/get/predictions/label/{pred_class}`
- GET `/get/predictions/confidence/{confidence}`
- GET `/get/predictions/label/{pred_class}/confidence/{confidence}`
- GET `/get/predictions/image/{img_name}`
- PUT `/update/predictions/{id}`
- DELETE `/delete/predictions/{id}`

- GET `/get/detections` — all detection logs
- GET `/get/detections/label/{detected_class}`
- GET `/get/detections/image/{img_name}`
- GET `/get/detectiosn/label/{detected_class}/level/{severity}`  
  Note: path name contains `detectiosn` as implemented.
- DELETE `/delete/detections/{id}`

- GET `/get/segmentations` — all segmentation logs
- GET `/get/segmentations/class/{predicted_class}/level/{severity}`
- DELETE `/delete/segmentations/{id}`

Minimal cURL examples
```bash
curl -F "img_file=@image/6.jpeg" http://localhost:8000/predict
curl -F "img_file=@image/6.jpeg" http://localhost:8000/detect
curl -F "img_file=@image/6.jpeg" http://localhost:8000/segment
```

## Training, Evaluation, Inference (Classifier)

Paths and hyperparameters are defined in `scripts/utils.py`.

- Train
```bash
python scripts/train.py
```

- Evaluate (saves `models/classification_report.csv` and `models/confusion_matrix.png`)
```bash
python scripts/evaluate.py
```

- Predict a single image (prints label and confidence)
```bash
python scripts/predict.py
```

Dataset splits expected:
```
data_split/
  train/<class_name>/*.jpeg
  val/<class_name>/*.jpeg
  test/<class_name>/*.jpeg
```

## Docker

Build image
```bash
docker build -t car-damage-assessment .
```

Run Streamlit UI
```bash
docker run --rm -p 8501:8501 -v %cd%/models:/app/models car-damage-assessment   # Windows cmd
# or
docker run --rm -p 8501:8501 -v $(pwd)/models:/app/models car-damage-assessment  # bash/WSL
```

The container starts Streamlit by default at `http://localhost:8501`.

## Screenshots

![Streamlit Web UI](screenshots/streamlit_ui.png)
![Object Detection](screenshots/object_detect.png)
![Segmentation & Severity Estimation](screenshots/segment_estimat.png)

## License

This project is for educational and research purposes.
