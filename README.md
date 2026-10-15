# TAO-Net: Pedestrian Trajectory & Action Prediction (Testing Repository)

This repository provides a **lightweight version** of TAO-Net, focused primarily on **testing the pretrained model**. You can run trajectory forecasting and action classification on JAAD sample sequences using the provided **pretrained weights**.

Due to GitHub repository size restrictions, the pretrained weights are hosted on Google Drive.

The repository is also being extended to include **preprocessing, training, and ablation studies**.

---

## Quick Start

### 1. Prepare JAAD Data

Ensure the `JAAD/` folder contains the demo subset (`videos/` and `annotations/`).

For full evaluation, download JAAD from its [official source](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

---

### 2. Download the Pretrained Weights

Download [`weight.h5`](https://drive.google.com/file/d/1Wb8Jh46_M5lzV-a603ruT8XiiR9wCmVb/view?usp=sharing) from Google Drive.

Place the downloaded file in the repository root, alongside `predict.py`.

```text
TAO-Net-Test/
├── predict.py
├── ato.py
└── weight.h5
```

---

### 3. Run Prediction

Use the pretrained weights to test trajectory forecasting and action classification:

```bash
python predict.py --weights weight.h5
```

If the script defines a default weights path, you may also run:

```bash
python predict.py
```

This will output:

- **Predicted bounding-box trajectories** (short/mid/long horizons).
- **Pedestrian action classification** (Walking / Standing).

---

## Requirements

- Python 3.8+
- TensorFlow 2.15
- MediaPipe
- TensorFlow Hub
- OpenCV
- pandas
- NumPy
- Matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
TAO-Net-Test/
├── README.md
├── requirements.txt
├── predict.py
├── ato.py
├── JAAD/
│   ├── videos/
│   └── annotations/
├── Preprocessing/
│   ├── README.md
│   ├── Gathering_Data.py
│   ├── data45_JAAD/
│   │   ├── angle.csv
│   │   ├── features.csv
│   │   ├── pose.csv
│   │   ├── x_traject.csv
│   │   └── y_traject.csv
│   └── data45_jaad.zip
├── Training/
└── Ablation/
```

**Note:** The pretrained `weight.h5` file is hosted externally and must be downloaded separately. The `Training/` and `Ablation/` directories are reserved for the planned extensions.

---

## 1. Preprocessing

The preprocessing pipeline extracts multimodal pedestrian information from videos and their XML annotations.

The generated data includes:

| File | Description |
|---|---|
| `angle.csv` | Pedestrian orientation angles |
| `features.csv` | Visual appearance features extracted using EfficientNet-Lite0 |
| `pose.csv` | Human pose landmark coordinates estimated using MediaPipe Pose |
| `x_traject.csv` | Observed pedestrian bounding boxes |
| `y_traject.csv` | Future pedestrian bounding-box trajectories |

The prepared JAAD preprocessing data is provided in:

`Preprocessing/data45_jaad.zip`

To use the prepared data, extract the archive into the `Preprocessing/` directory.

```bash
cd Preprocessing
unzip data45_jaad.zip
```

To generate or extend the dataset from raw videos, follow [`Preprocessing/README.md`](Preprocessing/README.md).

The default configuration collects 15 observed frames and uses a prediction horizon of 45 future frames.

---

## 2. Training

The training component will extend the repository to support training TAO-Net using the generated multimodal CSV files.

The planned workflow includes:

- Loading and validating preprocessed data.
- Preparing model inputs and trajectory targets.
- Configuring training and validation datasets.
- Training the model and saving checkpoints.
- Evaluating the trained model.

Training scripts and instructions will be documented in the `Training/` directory as they are implemented.

---

## 3. Ablation Studies

The ablation component will support experiments investigating the contribution of individual modalities and model components.

Planned experiments include:

- Removing visual appearance features.
- Removing human pose information.
- Removing orientation information.
- Comparing different combinations of input modalities.
- Evaluating the impact on trajectory forecasting and action classification.

The experimental configurations, metrics, and commands will be documented in the `Ablation/` directory as they become available.

---

## Dataset

JAAD is available from its [official website](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

- **Raw data:** JAAD videos and XML annotations.
- **Preprocessed data:** `Preprocessing/data45_jaad.zip`.
- **Pretrained weights:** [Download `weight.h5` from Google Drive](https://drive.google.com/file/d/1Wb8Jh46_M5lzV-a603ruT8XiiR9wCmVb/view?usp=sharing).

Please follow the dataset's terms of use when downloading and using JAAD.

---

## Citation

If you use TAO-Net, please cite the associated publication when its bibliographic details are available.

```bibtex
@article{taonet2026,
  title   = {TAO-Net: Multimodal Trajectory, Action, and Orientation Network for Pedestrian Forecasting},
  author  = {},
  journal = {},
  year    = {}
}
```
