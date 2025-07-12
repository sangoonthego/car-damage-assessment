# TRAINING.md

## Data Processing
- The dataset consists of car images labeled by damage type (e.g., bumper dent, glass shatter, etc.).
- Data is split into train, validation, and test sets using `split_data.py`.
- Images are resized to 224x224 and normalized with ImageNet mean/std.
- Data augmentation: random horizontal flip, rotation, color jitter (for training).

## Training Algorithm
- Model: ResNet18 (PyTorch, pretrained on ImageNet, last layer fine-tuned for 7 classes).
- Loss: CrossEntropyLoss
- Optimizer: Adam (learning rate 0.001)
- Scheduler: ReduceLROnPlateau (reduce LR on plateau of validation loss)
- Early stopping: patience of 5 epochs
- Training script: `scripts/train.py`

## Metrics
- Accuracy, Precision, Recall, F1-score (per class and overall)
- Confusion matrix
- Training/validation loss and accuracy curves

## Results
- Example (from evaluation):

| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| bumper_dent     | 0.78      | 0.70   | 0.74     | 20      |
| bumper_scratch  | 0.86      | 0.75   | 0.80     | 24      |
| door_dent       | 0.75      | 0.62   | 0.68     | 29      |
| door_scratch    | 0.69      | 0.83   | 0.75     | 24      |
| glass_shatter   | 0.89      | 0.76   | 0.82     | 21      |
| head_lamp       | 0.64      | 0.80   | 0.71     | 20      |
| tail_lamp       | 0.71      | 0.81   | 0.76     | 21      |
| **accuracy**    |           |        | **0.75** | 159     |

- Training/validation accuracy: up to ~83%/75%
- See `models/train_log.csv` and `models/loss_plot.png` for full logs and loss curves.

## How to Train
1. Prepare data in `data_split/` (see `split_data.py`).
2. Run:
   ```bash
   python scripts/train.py
   ```
3. Logs and best model will be saved in `models/`.
