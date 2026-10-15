# TAO-Net: Preprocessing

This directory contains the preprocessing pipeline for TAO-Net. The pipeline extracts pedestrian appearance, pose, orientation, and trajectory information from video sequences and their corresponding annotations.

The prepared JAAD dataset is provided as `data45_jaad.zip`, containing the five CSV files required by the downstream pipeline.

---

## Repository Structure

```text
Preprocessing/
├── README.md
├── Gathering_Data.py
├── data45_JAAD/
│   ├── angle.csv
│   ├── features.csv
│   ├── pose.csv
│   ├── x_traject.csv
│   └── y_traject.csv
└── data45_jaad.zip
```

---

## Requirements

- Python 3.8+
- TensorFlow 2.15
- TensorFlow Hub
- MediaPipe
- OpenCV
- NumPy
- pandas
- Matplotlib, if required by the supporting utilities

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

---

## 1. Use the Prepared JAAD Data

The prepared CSV dataset is available in:

`Preprocessing/data45_jaad.zip`

To extract the data, run:

```bash
cd Preprocessing
unzip data45_jaad.zip
```

After extraction, ensure that the following files are present:

```text
data45_JAAD/
├── angle.csv
├── features.csv
├── pose.csv
├── x_traject.csv
└── y_traject.csv
```

If the archive already contains the `data45_JAAD/` directory, avoid creating a second nested directory.

**If you only need the prepared data, you do not need to rerun the preprocessing script.**

---

## 2. Prepare the Raw JAAD Dataset

To generate new data, download JAAD from the [official dataset website](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

The preprocessing pipeline requires:

- MP4 pedestrian video sequences.
- XML annotations containing pedestrian tracks.
- Frame-level bounding boxes.
- Action attributes for the annotated pedestrians.

The current implementation constructs the annotation path by replacing `.mp4` with `.xml`. Therefore, it expects the corresponding XML annotation to be in the same directory as the video.

Example:

```text
JAAD/
└── videos/
    ├── 00001.mp4
    ├── 00001.xml
    ├── 00002.mp4
    └── 00002.xml
```

This example illustrates the expected layout of the current implementation. If your downloaded dataset stores videos and annotations separately, update the annotation path in `Gathering_Data.py`.

---

## 3. Preprocessing Pipeline

The main function in `Gathering_Data.py` is:

```python
process_video_and_annotations(
    dataset,
    sequence_length,
    video_folder,
    checkpoint_file,
    checkpoint_track
)
```

The function processes videos and pedestrian tracks to generate multimodal training data.

### Processing steps

1. Load checkpoint files to identify previously processed videos and tracks.
2. Initialize the pretrained EfficientNet-Lite0 feature extractor using TensorFlow Hub.
3. Initialize MediaPipe Pose.
4. Read and sort MP4 video files.
5. Parse pedestrian tracks and bounding boxes from XML annotations.
6. Extract pedestrian crops, expanding each bounding box by approximately 20%.
7. Extract visual appearance features from each crop.
8. Estimate pose landmarks and pedestrian orientation.
9. Collect observed bounding boxes and future trajectory targets.
10. Save valid samples using the project's data-saving utilities.

### Configuration

The default configuration uses 15 observed frames and a prediction horizon of 45 future frames.

Example:

```python
dataset = "jaad"
dataset_folder = "JAAD/videos"
sequence_length = 45

process_video_and_annotations(
    dataset,
    sequence_length,
    dataset_folder,
    checkpoint_file=f"checkpoint_file_{sequence_length}.txt",
    checkpoint_track=f"checkpoint_track_{sequence_length}.txt"
)
```

Adjust the paths and dataset identifier to match your local environment and the conventions used by the supporting functions.

**Important:** The example assumes that XML annotations are stored alongside the videos. Update the script if your dataset uses separate directories.

---

## 4. Run Preprocessing

From the `Preprocessing/` directory, execute:

```bash
python Gathering_Data.py
```

Ensure that the dataset paths and sequence length are configured correctly before running the script.

Processing a full dataset may take considerable time because visual feature extraction and pose estimation are performed on pedestrian image crops.

---

## 5. Generated CSV Files

The preprocessing pipeline produces five CSV files.

| File | Description | Purpose |
|---|---|---|
| `angle.csv` | Pedestrian orientation angles | Orientation input |
| `features.csv` | Appearance feature vectors | Visual input |
| `pose.csv` | Selected pose landmark coordinates | Pose input |
| `x_traject.csv` | Observed bounding boxes | Historical observations |
| `y_traject.csv` | Future bounding boxes | Prediction targets |

### Appearance features

Visual features are extracted using the pretrained EfficientNet-Lite0 feature-vector model from TensorFlow Hub:

```python
model = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
)
```

The pedestrian image crop is expanded by approximately 20% around the annotated bounding box, subject to the image boundaries.

### Pose and orientation

MediaPipe Pose is used to estimate human landmarks. Selected landmark coordinates are saved as pose features.

If pose estimation fails, the current implementation assigns `-1` to the selected landmark coordinates and `-360` to the orientation value.

The downstream training pipeline should handle these sentinel values consistently.

### Trajectory information

- `x_traject.csv` contains the observed bounding boxes.
- `y_traject.csv` contains the future bounding-box trajectories.

The future trajectory horizon is controlled by `sequence_length`.

---

## 6. Checkpoints and Recovery

The script uses two checkpoint files:

```text
checkpoint_file_45.txt
checkpoint_track_45.txt
```

The video checkpoint records processed video filenames. The track checkpoint records processed pedestrian track identifiers.

These files help avoid repeating work when processing large datasets.

Before resuming an interrupted run, check that the checkpoint entries correspond to data successfully written to the CSV files.

**Implementation note:** The current script removes the track checkpoint inside the video-processing loop if it exists. This can interfere with resuming track-level processing. For reliable recovery, checkpoint cleanup should occur after successful completion, and tracks should only be marked as processed after their outputs have been saved successfully.

---

## 7. Troubleshooting

### XML annotations cannot be found

Verify that the XML files exist and that their paths match the annotation path construction in `Gathering_Data.py`.

### No valid pedestrian samples are generated

Check the pedestrian track labels, bounding boxes, frame indices, action attributes, and the number of available frames.

### CSV files contain inconsistent sample counts

Inspect the `save_data()` implementation and ensure that all modalities are saved together for each valid sample.

### Previously processed tracks are skipped

Inspect the checkpoint files before restarting. Remove checkpoint entries only when you have verified that the corresponding samples need to be regenerated.

### TensorFlow Hub cannot load the feature extractor

The EfficientNet-Lite0 feature extractor is loaded from TensorFlow Hub and may require internet access on the first run.

---

## 8. Next Steps

The generated CSV files are intended for use by the TAO-Net training and evaluation pipelines.

The planned next stages include:

- **Training:** Loading the CSV files and training the trajectory forecasting and action classification model.
- **Ablation studies:** Evaluating the contribution of appearance, pose, orientation, and other model components.
