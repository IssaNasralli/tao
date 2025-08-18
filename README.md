# TAO-Net: Trajectory, Action, and Orientation Network

This repository contains the official implementation of **TAO-Net**, a multimodal deep model for **pedestrian trajectory prediction** and **action (walking/standing) classification**.  
TAO-Net fuses **pose**, **appearance features**, and **orientation angles** within a context-aware LSTM decoder.

---

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.15
- MediaPipe
- pandas
- numpy
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

Create `requirements.txt` (if you need it):

```
tensorflow==2.15.0
mediapipe
pandas
numpy
matplotlib
```

---

## 📂 Repository Structure

```
TAO-Net/
│── README.md
│── requirements.txt
│── extract_features.py        # Iterate JAAD/PIE and export CSV features for training
│── cross_val.py               # Training (cross-validation)   ← rename to this if your file is `corss_val.py`
│── predict.py                 # Inference: trajectory + action classification
│── weight.h5                  # Best pretrained weights
│
├── JAAD/                      # Subset of JAAD (5 example videos + annotations)
│   ├── videos/                # e.g., *.mp4
│   └── annotations/           # e.g., *.json / *.csv
│
├── data45_jaad/               # Pre-extracted training features (sequence length = 45)
│   ├── pose.csv               # Pose keypoints
│   ├── features.csv           # EfficientNet-Lite0 feature vectors
│   ├── angle.csv              # Quaternion-based orientation angles
│   ├── x_traject.csv          # X centers of bboxes (obs+future)
│   └── y_traject.csv          # Y centers of bboxes (obs+future)
```

> ⚠️ The JAAD subset is provided for **demo/repro** only. For full datasets, obtain JAAD/PIE from their official sources and update paths accordingly.

---

## 🚀 Quick Start

### 1) Feature Extraction

Extract pose (MediaPipe), appearance (EfficientNet-Lite0 feature-vector), angles, and trajectories.  
Edit input/output paths inside `extract_features.py` if needed.

```bash
python extract_features.py
```

This produces CSVs under `data45_jaad/` (or your chosen output dir).

---

### 2) Train (Cross-Validation)

Run training and hyperparameter search (example):

```bash
python cross_val.py
```

- Default sequence length: `45` (observe 15, predict 45; evaluation can use 15/30/45 horizons).
- Typical strong hyperparameters (as in the paper):
  - LSTM units: `1024`
  - Batch size: `512`
  - Loss weight α (for action): `1000`

Training logs/results are written to:

```
training_history_<sequence_length>/training_results_<dataset>.txt
```

---

### 3) Inference (Prediction + Action)

Use pretrained weights (`weight.h5`) or your own checkpoint:

```bash
python predict.py
```

Outputs:
- Future bounding-box trajectories at 0.5s / 1.0s / 1.5s
- Action class (Walking / Standing)

---

## 🧰 Notes on Features

- **Appearance**: EfficientNet-Lite0 feature vector from TF-Hub (`efficientnet/lite0/feature-vector/2`) produces a 1280-d embedding per crop. We then apply **1D max-pooling (size=40)** to reduce temporal noise before feeding the action LSTM.
- **Pose**: MediaPipe (33 keypoints). We retain **12 locomotion-indicative joints** (elbows, wrists, hips/knees/ankles, heels, foot indices) after analyzing walking sequences.
- **Orientation**: 2D-pose → quaternion-based yaw extraction following the economical method of Radhakrishna *et al.* (2023); provides a robust heading signal even with monocular video.

---

## 📊 Pretrained Weights

A ready-to-use checkpoint is provided:

```
weight.h5
```

Load example:

```python
from tensorflow.keras.models import load_model
model = load_model("weight.h5", compile=False)
```

---

## 🔁 Reproducing Paper Settings

- **Optimizer**: Adam
- **Epochs**: up to 200 (with early stopping)
- **Early Stopping**: monitor `val_loss`, patience `20`, `restore_best_weights=True`
- **Metrics**: MSE (bbox), CMSE / CFMSE (centroid), and P/R/F1/Accuracy for action

---

## 🧪 Data Files (CSV format)

Each row corresponds to one sequence:
- `pose.csv`: pose keypoints (e.g., 33×2×15 observed + optionally future)
- `features.csv`: 1280-d vectors per observed frame (after pooling)
- `angle.csv`: per-frame orientation angles
- `x_traject.csv`, `y_traject.csv`: bbox center coordinates (observed+future)

Make sure all CSVs are **aligned** (same number/order of sequences).

---

## ❓ Troubleshooting

- **OOM during training**: reduce batch size (`512 → 256 → 128`) or LSTM units (`1024 → 512`).
- **Mismatched lengths**: verify the number of sequences/frames per CSV.
- **MediaPipe GPU issues**: switch to CPU or pin versions in `requirements.txt`.

---

## 🧾 Citation

If you use **TAO-Net**, please cite:

```bibtex
@article{taonet2025,
  title   = {TAO-Net: Multimodal Trajectory, Action, and Orientation Network for Pedestrian Forecasting},
  author  = {Your Name and Co-authors},
  journal = {Turkish Journal of Mathematics and Computer Science},
  year    = {2025}
}
```

---

## 📬 Contact

For questions or issues, please open a GitHub Issue or email: **aissanasralli@gmail.com**.

---

