# 3D MRI Brain Tumor Segmentation

A deep-learning project for **3D brain tumor segmentation from multi-modal MRI volumes**. The project uses a custom 3D Attention Residual U-Net with Dense Inception blocks and multi-scale input features to predict voxel-wise tumor labels.

> **Note:** This repository contains the model implementation, data generator, loss functions, metrics, and a Jupyter notebook. The MRI dataset is not included in the supplied project archive and must be obtained and prepared separately.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [MRI Modalities and Segmentation Classes](#mri-modalities-and-segmentation-classes)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Environment and Dependencies](#environment-and-dependencies)
- [Running the Project](#running-the-project)
- [Training Configuration](#training-configuration)
- [Loss Functions and Evaluation Metrics](#loss-functions-and-evaluation-metrics)
- [Prediction and Visualization](#prediction-and-visualization)
- [Checkpointing](#checkpointing)
- [Limitations and Notes](#limitations-and-notes)

## Overview

The goal is to segment brain tumor regions in volumetric MRI scans. Instead of predicting a label independently for each 2D slice, the network processes a 3D volume and produces a 3D segmentation map.

The supplied notebook is configured for input volumes of **128 × 128 × 128 voxels with 3 channels** and **4 output classes**. It includes data loading, model construction, training/resuming, metric evaluation, inference timing, and visual inspection of predictions.

## Model Architecture

The network is implemented in `model1.py` as `unet_3d()`.

Its main components are:

1. **3D convolutional encoder-decoder:** A contracting path extracts features at progressively lower spatial resolutions, and an expanding path reconstructs a full-resolution segmentation.
2. **Dense Inception Residual Blocks:** Each block applies parallel 1×1×1, 3×3×3, and 5×5×5 convolutions, concatenates their outputs with the input, projects the result, and adds a residual projection of the input.
3. **Multi-scale input:** Average-pooled versions of the original input are concatenated with encoder features at multiple resolutions.
4. **Attention gates:** Decoder features guide attention on corresponding encoder skip features before concatenation.
5. **Softmax output:** A final 1×1×1 3D convolution predicts class probabilities for every voxel.

The notebook constructs the network with:

- Input shape: `(128, 128, 128, 3)`
- Number of classes: `4`
- Base encoder filters: `16`, increasing through the network to a bottleneck with `256` filters
- Data format: `channels_last`

## MRI Modalities and Segmentation Classes

The visualization code uses the following channel mapping:

| Channel | MRI modality |
|---|---|
| 0 | FLAIR |
| 1 | T1CE |
| 2 | T2 |

The model predicts four classes:

| Label | Region |
|---:|---|
| 0 | Background |
| 1 | NCR / NET (as named in the notebook) |
| 2 | Edema (ED) |
| 3 | Enhancing Tumor (ET) |

The training notebook expects masks to be one-hot encoded with four channels. During visualization and some metric calculations, `argmax` is applied over the class-channel axis to obtain integer label maps.

## Project Structure

```text
MRI-Segmentation-Project-main/
├── Main.ipynb          # Data setup, training, evaluation and visualization workflow
├── model1.py           # 3D Attention Residual U-Net architecture
├── custom_datagen.py   # NumPy volume loader and batch generator
├── losses.py           # Segmentation loss functions and loss classes
└── metrics.py          # IoU, Dice, F-score, precision and recall metrics
```

## Dataset Preparation

The notebook expects the dataset to be arranged into training and validation image/mask directories. Its Google Drive configuration uses:

```text
input_data_128/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

The data generator loads `.npy` files. Image and mask filenames are collected separately, sorted, and paired by their order in the two lists. Therefore, ensure that each image corresponds to the mask at the same sorted position and that both directories contain matching numbers of files.

Expected data characteristics based on the notebook and generator:

- Each image is a NumPy volume with shape compatible with `(128, 128, 128, 3)`.
- Each mask is a one-hot encoded NumPy volume with shape compatible with `(128, 128, 128, 4)`.
- Image and mask arrays are converted to `float32` when loaded.
- The notebook uses a batch size of `2`.

The archive does not include the dataset or a preprocessing script. Confirm that your data has already been resized, normalized as required by your dataset workflow, and encoded in the expected format before training.

## Environment and Dependencies

The code uses Python with the following libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- tqdm

A GPU is recommended because 3D convolutional networks on 128³ volumes can require substantial memory and computation. The notebook contains Google Colab/Google Drive setup and enables GPU memory growth when a GPU is available.

Install the main packages in your Python environment:

```bash
pip install tensorflow keras numpy matplotlib tqdm
```

The project does not currently include a pinned `requirements.txt`; package versions may need adjustment to match your TensorFlow/Keras and CUDA environment.

## Running the Project

### Option 1: Google Colab

1. Upload or extract the repository into your Colab working environment.
2. Place the prepared `input_data_128` dataset in Google Drive using the directory structure shown above.
3. Open `Main.ipynb` in Google Colab.
4. Run the drive-mount and GPU-check cells.
5. Verify or update the dataset paths in the data-path configuration cell.
6. Run the import, data-generator, model-setup, and training cells in order.
7. Review the saved checkpoints and inspect the prediction visualizations.

### Option 2: Local Jupyter Environment

1. Extract the repository and install the dependencies.
2. Start Jupyter from the project directory:

   ```bash
   jupyter notebook
   ```

3. Open `Main.ipynb`.
4. Replace the Google Drive paths with paths to your local dataset.
5. Ensure `model1.py`, `custom_datagen.py`, `losses.py`, and `metrics.py` are importable from the notebook's working directory.
6. Run the notebook cells in sequence, adjusting the checkpoint path if resuming training.

### Inspect the model independently

You can print the model summary by running:

```bash
python model1.py
```

This invokes the example model configuration in `model1.py` and prints its input/output shapes and layer summary. It requires TensorFlow and Keras to be installed.

## Training Configuration

The notebook's model setup currently specifies:

| Parameter | Value in notebook |
|---|---:|
| Input height | 128 |
| Input width | 128 |
| Input depth | 128 |
| Input channels | 3 |
| Output classes | 4 |
| Batch size | 2 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Configured epochs | 155 |
| Initial epoch | 149 |

The notebook loads weights from `model_epoch_155.keras` before its training section and sets `initial_epoch=149`. This is a **resume-training configuration**, not a clean-start configuration. To train from scratch, review the weight-loading cell and the `initial_epoch` setting before running the notebook. Make sure the checkpoint exists at the configured path if you intend to resume.

The notebook also defines an `AdaptiveLoss` that combines Dice loss and categorical focal cross-entropy. Its weights change as the epoch variable is updated by a callback at the start of each epoch.

## Loss Functions and Evaluation Metrics

`losses.py` provides segmentation losses and helper functions, including Dice/Jaccard-style losses, binary and categorical cross-entropy, and focal-loss variants. The notebook's training configuration uses a combination of `DiceLoss` and categorical focal cross-entropy through `AdaptiveLoss`.

`metrics.py` implements metrics including:

- Intersection over Union (IoU)
- Dice score
- F-score
- Precision
- Recall
- Macro Dice over selected tumor classes

The notebook configures class-specific Dice metrics for labels 1, 2, and 3, as well as IoU-related metrics. It also calculates a Mean IoU value for a validation batch in a separate evaluation cell.

## Prediction and Visualization

The notebook includes cells to:

- Load a selected image and its ground-truth mask from `.npy` files.
- Run model inference using `model.predict()`.
- Convert predicted probabilities to class labels with `argmax`.
- Compare MRI slices, ground-truth labels, and predicted segmentation maps.
- Display class contours for NCR/NET, edema, and enhancing tumor.
- Visualize axial, coronal, and sagittal views.
- Save a multi-view contour figure as PNG and PDF.

The visualization cells use example image numbers and slice indices. Update these values to match the files available in your dataset.

## Checkpointing

The notebook creates local output directories:

```text
savepoints2/
├── model/
└── history/
```

A Keras `ModelCheckpoint` callback saves a full model after each epoch. A custom callback also writes training history to JSON files. Ensure the output directory is writable and has enough disk space before training.

## Limitations and Notes

- The dataset is external and is not included in this repository archive.
- Dataset preprocessing and normalization are not implemented as a complete pipeline in the supplied files; the notebook assumes prepared `.npy` volumes.
- Data pairing depends on separately sorted image and mask filename lists. Matching names explicitly would make this safer if filenames are not perfectly aligned.
- Several notebook paths are specific to Google Drive or a particular local machine and must be changed for another environment.
- The notebook contains a checkpoint-loading step and starts at epoch 149; adjust this for a fresh training run.
- No trained model weights, formal test-set report, or numerical performance results are included in the supplied archive. Do not interpret the example visualizations or configured metrics as a reported benchmark.
- This project is for research and educational use. It is not a clinically validated diagnostic tool and must not be used as a substitute for professional medical assessment.

## Acknowledgements

Built with TensorFlow/Keras, NumPy, and Matplotlib. The model and training workflow are implemented in the Python modules and notebook included in this repository.
