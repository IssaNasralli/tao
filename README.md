# TAO-Net: Pedestrian Trajectory & Action Prediction (Testing Repository)

This repository provides a **lightweight version** of TAO-Net, focused only on **testing the pretrained model**.  
You can run trajectory forecasting and action classification on JAAD sample sequences using the provided **pretrained weights**.

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

Create `requirements.txt`:

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
TAO-Net-Test/
│── README.md
│── predict.py       # Run testing: trajectory prediction + action classification
│── ato.py           # Model definition
└── JAAD/            # Subset of JAAD dataset (10 demo videos + annotations)

```

---

## 🚀 Quick Start

### 1) Prepare JAAD Data
Ensure the **JAAD/** folder contains the demo subset (`videos/` and `annotations/`).  
For full evaluation, download JAAD from its [official source](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

---

### 2) Run Prediction
Use the pretrained weights [`weight.h5`](https://drive.google.com/file/d/1Wb8Jh46_M5lzV-a603ruT8XiiR9wCmVb/view?usp=sharing) to test trajectory + action classification:

```bash
python predict.py --weights weight.h5


```bash
python predict.py
```

This will output:
- **Predicted bounding-box trajectories** (short/mid/long horizons).
- **Pedestrian action classification** (Walking / Standing).

---

## 🧾 Citation

If you use **TAO-Net**, please cite:

```bibtex
@article{taonet2026,
  title   = {TAO-Net: Multimodal Trajectory, Action, and Orientation Network for Pedestrian Forecasting},
  author  = {},
  journal = {},
  year    = {}
}
```

---

