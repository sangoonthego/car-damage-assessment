import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from yolov8.utils.detect_utils import ObjectDetector
from yolov8.utils.segment_utils import ObjectSegmenter
from yolov8.car_severity.severity_level import SeverityLevel

# Import classification predictor
from scripts.utils import model_path, test_dir
from scripts.predict import CarDamagePredictor

st.set_page_config(page_title="Car Damage Assessment", layout="wide")
st.title("Car Damage Detection, Segmentation and Classification")

object_detector = ObjectDetector()
object_segmentor = ObjectSegmenter()
classifier = CarDamagePredictor(model_path, test_dir)

uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Selected Image", use_container_width=True)
    temp_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as file:
        file.write(uploaded_file.read())

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    # Object Detection
    with col1:
        st.subheader("Object Detection")
        detections = object_detector.detect_objects(temp_path)

        if detections:
            for detection in detections:
                st.write(f"Class: {detection['class']}")
                st.write(f"Confidence: {detection['confidence']:.2f}")
        else:
            st.warning("No Object Detected")

        output_img_path = f"static/output/{uploaded_file.name}"
        if os.path.exists(output_img_path):
            st.image(output_img_path, caption="Detection Result", use_container_width=True)

    # Object Segmentation
    with col2:
        st.subheader("Object Segmentation")
        segmentations = object_segmentor.segment_objects(temp_path)
        
        if segmentations:
            for seg in segmentations:
                st.write(f"Class: {seg['class']}")
                st.write(f"Confidence: {seg['confidence']:.2f}")
                st.write(f"Severity: {seg['severity']}")
                st.image(seg["mask_path"], caption=seg["class"], use_container_width=True)
        else:
            st.warning("No Object Segmented")

    # Classification
    with col3:
        st.subheader("Damage Classification")
        pred_class, confidence = classifier.predict_image_file(temp_path)
        if pred_class:
            st.success(f"Prediction: {pred_class} ({confidence:.2f})")
        else:
            st.warning("Classification failed")

    os.remove(temp_path)
