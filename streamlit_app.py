import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.detect_utils import ObjectDetector
from app.segment_utils import ObjectSegmenter
from app.severity_level import SeverityLevel

st.set_page_config(page_title="Car Damage Assessment", layout="wide")
st.title("Car Damage Detection and Segmentation")

object_detector = ObjectDetector()
object_segmentor = ObjectSegmenter()

uploaded_file = st.file_uploader("Up an image", type=["jpeg", "jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Seleted Image", use_container_width=True)
    temp_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as file:
        file.write(uploaded_file.read())

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Object Detection")
        detections = object_detector.detect_objects(temp_path)

        if detections:
            for detection in detections:
                st.write(f"Class: {detection["class"]}")
                st.write(f"Confidence: {detection["confidence"]:.2f}")
                st.write(f"Severity: {detection["severity"]}")
        else:
            st.warning("No Object Detected")

        output_img_path = f"static/output/{uploaded_file.name}"
        if os.path.exists(output_img_path):
            st.image(output_img_path, caption=detection["class"], use_container_width=True)
    
    with col2:
        st.subheader("Object Segmentation")
        segmentations = object_segmentor.segment_objects(temp_path)
        
        if segmentations:
            for seg in segmentations:
                st.write(f"Class: {seg["class"]}")
                st.write(f"Confidence: {seg["confidence"]:.2f}")
                st.write(f"Severity: {seg["severity"]}")
                st.image(seg["mask_path"], caption=seg["class"], use_container_width=True)
        else:
            st.warning("No Object Segmented")

    os.remove(temp_path)
