
# Low Frame Rate Vehicle Tracking: MBR_4B

## Introduction

The `MBR_4B` is designed to track vehicles in low-frame-rate videos, a challenging task due to significant object displacements and visibility changes between frames. This project leverages deep learning techniques to perform robust vehicle tracking, specifically tailored for the AI Cup dataset, with applications in surveillance, traffic monitoring, and vehicle re-identification (Re-ID). It employs a two-stage training process: initial training on the VeRi-776 dataset using ImageNet pretrained model weights, followed by fine-tuning on the AI Cup dataset. The `aicupTracking.py` script facilitates both single-camera and cross-camera tracking, enabling the identification of vehicles across multiple camera views.

## Purpose

The primary goal of this project is to develop an effective vehicle tracking system for low-frame-rate videos, which is critical for deploying tracking solutions on resource-constrained edge devices. Low-frame-rate scenarios, such as those at 1 frame per second (fps), pose unique challenges, including:
- Large displacements of vehicles between consecutive frames.
- Rapid changes in appearance and visibility.
- Difficulty in maintaining consistent object identities across cameras.

By training on the VeRi-776 dataset, a benchmark for vehicle Re-ID with over 49,000 images of 776 vehicles across 20 cameras, and fine-tuning on the AI Cup dataset, the project aims to provide a robust solution for these challenges. The approach supports applications requiring continuous vehicle tracking in real-world settings, such as urban traffic analysis and security surveillance.


## Usage


### Dataset Preparation

The project requires two datasets:
- **VeRi-776 Dataset**:
  - Download from the official source, such as [Papers With Code](https://paperswithcode.com/dataset/veri-776).
  - Place the dataset in `./datasets/veri776`.
  - Expected structure includes images and annotations for vehicle Re-ID.

- **AI Cup Dataset**:
  - Obtain from the AI Cup competition website (e.g., [TBrain Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/33)).
  - Place in `./datasets/aicup`.
  - Expected structure:
    ```
    datasets/aicup/
    ├── train/
    │   ├── images/<timestamp>/*.jpg
    │   ├── labels/<timestamp>/*.txt
    ├── valid/
    │   ├── images/<timestamp>/*.jpg
    │   ├── labels/<timestamp>/*.txt
    ```
    Label files contain bounding box annotations in the format: `class center_x center_y width height track_ID`.

### Training

The training process involves two stages:

1. **Train on VeRi-776**:
   - Use ImageNet pretrained weights to initialize the model.
   - Run the training script (adjust based on actual script name):
     ```bash
     python main.py --model_arch MBR_4B --config ./config/config_BoT_Veri776.yaml
     ```
   - This step trains the model for vehicle Re-ID, leveraging the VeRi-776 dataset’s diverse camera views and annotations.

2. **Fine-tune on AI Cup**:
   - Use the trained model weights from VeRi-776.
   - Fine-tune on the AI Cup dataset to adapt to its low-frame-rate characteristics:
     ```bash
     python main.py --model_arch MBR_4B --config ./config/config_BoT_AICUP.yaml
     ```

### Tracking

- **Run Tracking**:
  - Use the `aicupTracking.py` script for single-camera or cross-camera tracking:
    ```bash
    python aicupTracking.py -i test_image_root -l test_label_root -s save_path -p model_weight_path
    ```


