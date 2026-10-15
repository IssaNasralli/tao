# TAO-Net: Ablation Study

## Overview

This folder contains the model implementations used to investigate the contribution of different components of TAO-Net. The ablation study compares model variants that progressively introduce attention mechanisms and input modalities, helping assess how each component contributes to pedestrian trajectory prediction and pedestrian action classification.

The implementations are:

- `ato.py`
- `ato_attention.py`
- `ato_attention_angle.py`
- `ato_attention_angle_feature.py`
- `ato_attention_angle_feature_pose.py`

The goal is to compare these variants under consistent experimental conditions and identify which components provide the greatest benefit.

## Repository Structure

```text
Ablation/
├── ato.py
├── ato_attention.py
├── ato_attention_angle.py
├── ato_attention_angle_feature.py
├── ato_attention_angle_feature_pose.py
└── README.md
```

## Model Variants

The filenames suggest a sequence of experiments that investigate attention and the inclusion of different input modalities. The precise differences should be confirmed by inspecting each model's inputs, layers, and output heads.

| File | Ablation variant | Purpose |
|---|---|---|
| `ato.py` | Baseline | Establishes the baseline model performance. |
| `ato_attention.py` | Attention | Investigates the effect of adding attention. |
| `ato_attention_angle.py` | Attention + angle | Investigates the contribution of angle information. |
| `ato_attention_angle_feature.py` | Attention + angle + features | Investigates the contribution of additional feature inputs. |
| `ato_attention_angle_feature_pose.py` | Attention + angle + features + pose | Investigates the contribution of pose information. |

**Important:** This table is a filename-based interpretation, not a verified architectural specification. Confirm which inputs and attention modules are actually used in each implementation before reporting the results.

## Ablation Objectives

The experiments are intended to answer the following questions:

1. Does attention improve prediction performance compared with the baseline?
2. Does adding angle information improve trajectory prediction or action classification?
3. Do additional feature inputs provide complementary information?
4. Does pose information improve pedestrian behavior recognition and future trajectory prediction?
5. Which model variant offers the best balance between predictive performance and model complexity?

## Tasks and Outputs

The model variants should be evaluated on the same prediction tasks wherever their architectures support them:

- **Pedestrian trajectory prediction:** Predict future pedestrian bounding boxes.
- **Pedestrian action classification:** Classify pedestrian behavior, such as walking or standing, if the variant includes a classification output.

Use the same evaluation metrics across comparable variants. For trajectory prediction, report the metrics used by the project, such as average displacement error (ADE) and final displacement error (FDE), if these are implemented in the evaluation pipeline. For classification, report appropriate metrics such as accuracy, precision, recall, and F1-score when available.

Do not compare metrics calculated over different samples, prediction horizons, or evaluation procedures as if they were directly equivalent.

## Experimental Setup

For a fair ablation study, keep the following conditions consistent across variants:

- **Dataset:** Use the same dataset and data preparation pipeline.
- **Data split:** Use identical training, validation, and test samples.
- **Input history:** Use the same observation window wherever applicable.
- **Prediction horizon:** Evaluate the same future time steps.
- **Training configuration:** Keep the optimizer, learning-rate policy, batch size, and maximum number of epochs consistent unless one is the explicit subject of the ablation.
- **Model selection:** Select checkpoints using validation performance.
- **Evaluation:** Evaluate each selected checkpoint on the same held-out test set.
- **Randomness:** Fix random seeds and, ideally, repeat experiments with multiple seeds.

If a variant requires an input that another variant does not use, keep the underlying dataset samples and target labels aligned. Document any differences that prevent a strictly controlled comparison.

## Running the Experiments

The files in this folder define model variants. The exact training command depends on how the training pipeline imports and instantiates each model.

Before running an experiment, check:

1. Whether the script defines a model-building function or also contains training code.
2. Which input tensors the model expects.
3. Which outputs and loss functions it uses.
4. Which training script imports the implementation.
5. Where checkpoints, logs, and evaluation results are saved.

If the training pipeline supports selecting the model implementation, run the same training and evaluation procedure for each variant, changing only the selected model.

For example, if the training script exposes a model selection option, the experiments could be organized as follows:

```text
Baseline:              ato.py
Attention:             ato_attention.py
Attention + angle:     ato_attention_angle.py
Attention + features:  ato_attention_angle_feature.py
Full variant:          ato_attention_angle_feature_pose.py
```

These labels are illustrative; use the verified architectural differences when naming experiments in your results.

## Results

Record the results for every variant in a single table. Replace the placeholder values with results from actual runs.

| Model variant | Trajectory metric 1 | Trajectory metric 2 | Classification metric |
|---|---:|---:|---:|
| `ato.py` | — | — | — |
| `ato_attention.py` | — | — | — |
| `ato_attention_angle.py` | — | — | — |
| `ato_attention_angle_feature.py` | — | — | — |
| `ato_attention_angle_feature_pose.py` | — | — | — |

Clearly state the metric definitions, units, prediction horizon, and whether higher or lower values are better. Include model parameter counts and inference time if computational efficiency is part of the comparison.

## Reproducibility and Reporting

For each experiment, save:

- The model variant and its configuration.
- The dataset split and random seed.
- Training and validation histories.
- The selected checkpoint.
- Final test metrics.
- Any relevant model parameter counts.

Use distinct checkpoint and log filenames for each variant to avoid accidentally overwriting results or loading weights from a different experiment.

Hyperparameters should not be selected based on repeated evaluation against the held-out test set. Use validation performance for model selection and reserve the test set for final comparison.

## Notes

- Verify the architectural differences directly in the Python implementations before interpreting the ablation results.
- Ensure all variants are trained and evaluated using the same data preprocessing conventions.
- Check that checkpoint filenames uniquely identify the model variant and experiment configuration.
- Document any differences in trainable parameters, input dimensions, loss weighting, or training procedure that could affect the comparison.

