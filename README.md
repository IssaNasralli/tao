# 🚶 TAO-Net: Pedestrian Trajectory & Action Prediction

**A lightweight repository for testing the pretrained TAO-Net model**, with ongoing extensions for preprocessing, model training, and ablation studies.

TAO-Net predicts future pedestrian bounding-box trajectories and classifies pedestrian actions using sample sequences from the JAAD dataset.

> 📦 **Pretrained weights are hosted externally** because of GitHub repository size restrictions.

## 🚀 Quick Start

### 1. 📂 Prepare the JAAD Dataset

Ensure that the `JAAD/` folder contains the demo subset, including `videos/` and `annotations/`.

For full evaluation, download the dataset from the [official JAAD website](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

### 2. ⬇️ Download the Pretrained Weights

Download [`weight.h5` from Google Drive](https://drive.google.com/file/d/1Wb8Jh46_M5lzV-a603ruT8XiiR9wCmVb/view?usp=sharing).

Place the downloaded file in the repository root, alongside `predict.py`.

```text
TAO-Net-Test/
├── predict.py
├── ato.py
└── weight.h5
```

### 3. ▶️ Run Prediction

Run the prediction script using the pretrained weights:

```bash
python predict.py --weights weight.h5
```

If the script defines a default weights path, you may also run:

```bash
python predict.py
```

**Expected outputs:**

- 📈 **Pedestrian trajectory forecasting:** predicted bounding-box trajectories at short-, mid-, and long-term horizons.
- 🚶 **Pedestrian action classification:** walking or standing.

---

## 🧭 Repository Guide

Explore the different components of the project:

| Component | Description | Documentation |
|---|---|---|
| 🧪 **Testing** | Run the pretrained model for trajectory forecasting and action classification. | [`predict.py`](predict.py) |
| ⚙️ **Preprocessing** | Extract appearance features, pose landmarks, orientation angles, and bounding-box trajectories. | [Preprocessing README](Preprocessing/README.md) |
| 🏋️ **Training** | Train the model using preprocessed multimodal data and evaluate trained checkpoints. | [Training folder](Training/) · [Training README](Training/README.md) |
| 🔬 **Ablation Studies** | Compare model variants and investigate the contribution of attention and different input modalities. | [Ablation folder](Ablation/) · [Ablation README](Ablation/README.md) |

> 💡 Start with **Quick Start** if you want to test the pretrained model. For dataset preparation, see [Preprocessing](Preprocessing/README.md). For model development, explore [Training](Training/) and [Ablation](Ablation/).

## 📋 Requirements

- 🐍 Python 3.8+
- 🧠 TensorFlow 2.15
- 🦾 MediaPipe
- 🔗 TensorFlow Hub
- 🎥 OpenCV
- 🗃️ pandas
- 🔢 NumPy
- 📊 Matplotlib

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Repository Structure

```text
TAO-Net-Test/
├── README.md
├── requirements.txt
├── predict.py
├── ato.py
├── weight.h5                 # Download separately
├── JAAD/
│   ├── videos/
│   └── annotations/
├── Preprocessing/
│   ├── README.md
│   ├── Gathering_Data.py
│   ├── data45_jaad/
│   │   ├── angle.csv
│   │   ├── features.csv
│   │   ├── pose.csv
│   │   ├── x_traject.csv
│   │   └── y_traject.csv
│   └── data45_jaad.zip
├── Training/
│   ├── README.md
│   ├── cross_val.py
│   ├── csv_features.py
│   └── ato.py
└── Ablation/
    ├── README.md
    ├── ato.py
    ├── ato_attention.py
    ├── ato_attention_angle.py
    ├── ato_attention_angle_feature.py
    └── ato_attention_angle_feature_pose.py
```

**Note:** `weight.h5` is hosted externally and must be downloaded separately. The `Training/` and `Ablation/` directories contain the scripts and documentation for model development and experiments.

---

## ⚙️ 1. Preprocessing

The preprocessing pipeline extracts multimodal pedestrian information from JAAD videos and their XML annotations.

### 📄 Generated Data

| File | Description |
|---|---|
| `angle.csv` | Pedestrian orientation angles |
| `features.csv` | Visual appearance features extracted using EfficientNet-Lite0 |
| `pose.csv` | Human pose landmarks estimated using MediaPipe Pose |
| `x_traject.csv` | Observed pedestrian bounding boxes |
| `y_traject.csv` | Future pedestrian bounding-box trajectories |

### 📦 Prepared Dataset

The prepared preprocessing data is provided in:

`Preprocessing/data45_jaad.zip`

Extract the archive into the `Preprocessing/` directory:

```bash
cd Preprocessing
unzip data45_jaad.zip
```

To generate or extend the dataset from raw videos, follow the [Preprocessing documentation](Preprocessing/README.md).

**Default configuration:**

- 👀 **Observation window:** 15 frames.
- 🔮 **Prediction horizon:** 45 future frames.

---

## 🏋️ 2. Training

The training pipeline supports training TAO-Net using the generated multimodal CSV files.

The workflow includes:

- 📥 Loading and validating preprocessed data.
- 🧩 Preparing model inputs and trajectory targets.
- 🔀 Splitting data into training, validation, and test sets.
- 🧠 Training the model and saving checkpoints.
- 📉 Recording training history and losses.
- 📊 Evaluating trained models.

For training scripts, configuration details, and execution instructions, see the **[Training folder](Training/)** and its [README](Training/README.md).

---

## 🔬 3. Ablation Studies

The ablation study investigates the contribution of different model components and input modalities.

The available model implementations include:

- `ato.py` — baseline model implementation.
- `ato_attention.py` — attention variant.
- `ato_attention_angle.py` — attention and angle variant.
- `ato_attention_angle_feature.py` — attention, angle, and feature variant.
- `ato_attention_angle_feature_pose.py` — attention, angle, feature, and pose variant.

These descriptions follow the filenames; consult the implementations to confirm the exact architectural differences.

The experiments aim to evaluate how different components affect:

- 📈 Trajectory forecasting performance.
- 🚶 Pedestrian action classification.
- ⚖️ The trade-off between predictive performance and model complexity.

For model variants and experiment instructions, visit the **[Ablation folder](Ablation/)** and its [README](Ablation/README.md).

---

## 📚 Dataset

The [Joint Attention in Autonomous Driving (JAAD) dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) provides pedestrian videos and annotations for behavior analysis in driving scenarios.

- 🎥 **Raw data:** JAAD videos and XML annotations.
- 🧮 **Preprocessed data:** `Preprocessing/data45_jaad.zip`.
- 🧠 **Pretrained weights:** [Download `weight.h5` from Google Drive](https://drive.google.com/file/d/1Wb8Jh46_M5lzV-a603ruT8XiiR9wCmVb/view?usp=sharing).

Please follow the dataset's terms of use when downloading and using JAAD.

---

## 📝 Citation

If you use TAO-Net in your research, please cite the associated publication once its bibliographic details are available.

```bibtex
@article{taonet2026,
  title   = {TAO-Net: Multimodal Trajectory, Action, and Orientation Network for Pedestrian Forecasting},
  author  = {},
  journal = {},
  year    = {}
}
```

Replace the placeholder fields with the verified publication details before submitting or publishing the repository.

---

## 🤝 Project Overview

TAO-Net brings together pedestrian trajectory forecasting and action classification through multimodal information. This repository provides a starting point for testing the pretrained model and exploring the preprocessing, training, and ablation workflows.

**Useful links:** [🚀 Quick Start](#-quick-start) · [⚙️ Preprocessing](Preprocessing/README.md) · [🏋️ Training](Training/README.md) · [🔬 Ablation Studies](Ablation/README.md)
