# Pipeline, Model, and Result Analysis

## 1. Pipeline Overview

The following diagram illustrates the main processing pipeline of the Car Damage Assessment system:

```mermaid
flowchart TD
    A[Input Image] --> B[Preprocessing]
    B --> C[Classification (ResNet18)]
    B --> D[Detection/Segmentation (YOLOv8)]
    C --> E[Damage Type Prediction]
    D --> F[Segmented Damage Area]
    F --> G[Severity Estimation]
    E --> H[Result Display]
    G --> H[Result Display]
```

## 2. Model and Processing Steps

- **Input:** User uploads a car image via the web interface.
- **Preprocessing:** Image is resized, normalized, and optionally augmented.
- **Classification:**
    - Model: ResNet18 (fine-tuned on 7 car damage classes)
    - Output: Predicted damage type and confidence score
- **Detection/Segmentation:**
    - Model: YOLOv8 (for object detection and segmentation)
    - Output: Bounding boxes and segmentation masks for damaged areas
- **Severity Estimation:**
    - Rule-based or model-based estimation of damage severity (to be improved in future work)
- **Result Display:**
    - The system displays predicted class, confidence, segmented area, and severity on the web interface.

## 3. Result Analysis, Baseline Comparison, and Challenges

### Result Analysis
- The model achieves reasonable accuracy on the test set, with per-class precision, recall, and F1-score reported in `TRAINING.md`.
- However, the model's confidence is inconsistent: some images yield very high confidence, while others (even when correctly predicted) have low confidence scores.

### Baseline Comparison
- The current baseline is ResNet18 for classification and YOLOv8 for detection/segmentation.
- No simpler baseline (e.g., logistic regression, SVM) was used, as the problem is image-based and requires deep learning.

### Challenges and Solutions
- **Challenge:** Model confidence is not well-calibrated; some correct predictions have low confidence.
    - **Solution:** Further training, data augmentation, and possibly calibration techniques (e.g., temperature scaling) are needed.
- **Challenge:** Severity estimation is currently basic and needs improvement.
    - **Solution:** Plan to develop a more robust severity estimation model and train it more thoroughly in the future.
- **Challenge:** Data imbalance and image quality can affect model performance.
    - **Solution:** Data cleaning, augmentation, and possibly collecting more labeled data.
    
---
