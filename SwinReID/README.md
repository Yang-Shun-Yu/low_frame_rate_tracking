# SwinReID Component in Low-Frame-Rate Tracking Project

## Introduction

The SwinReID component is a key part of the [low_frame_rate_tracking](https://github.com/Yang-Shun-Yu/low_frame_rate_tracking) project, designed to perform vehicle re-identification (ReID) in low-frame-rate videos. It leverages the Swin Transformer architecture, known for its ability to capture long-range dependencies in images, to extract robust features from vehicle images. This enables consistent identification of vehicles across different cameras and times, addressing challenges like large movements between frames and varying appearances. The model is initially pretrained on ImageNet, trained on the [VeRi-776 dataset](https://github.com/JDAI-CV/VeRidataset), and fine-tuned on the AI Cup dataset to adapt to specific tracking requirements.

## Purpose

SwinReID aims to enhance vehicle tracking in low-frame-rate scenarios, such as traffic surveillance at 1 frame per second, where traditional methods struggle due to significant vehicle displacements and appearance changes. By providing accurate re-identification, SwinReID supports both single-camera and cross-camera tracking, making it suitable for applications like urban traffic monitoring and security surveillance. Its integration into the tracking pipeline ensures robust performance in maintaining vehicle identities, crucial for real-world deployment on resource-constrained devices.

## Datasets

The SwinReID component is trained and fine-tuned on the following datasets:

| Dataset   | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| VeRi-776  | A large-scale vehicle ReID dataset with over 49,000 images of 776 vehicles across 20 cameras. Available at [VeRi-776](https://github.com/JDAI-CV/VeRidataset). |
| AI Cup    | A custom dataset from the AI Cup competition, containing vehicle images with annotations for tracking and ReID. |

## Training Process

The training process for SwinReID involves three stages:

1. **Initial Pretraining on ImageNet**:
   - The Swin Transformer model is pretrained on ImageNet to learn general image features, providing a strong foundation for subsequent training.
   - Pretrained weights are typically sourced from repositories like [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

2. **Training on VeRi-776**:
   - The pretrained model is trained on the VeRi-776 dataset to learn vehicle-specific features, leveraging its diverse camera views and annotations.
   - This step builds a robust feature extractor for vehicle ReID.

3. **Fine-Tuning on AI Cup Dataset**:
   - The model is fine-tuned on the AI Cup dataset to adapt to the competition’s specific data characteristics, such as low-frame-rate video challenges.
   - This ensures optimal performance for the target tracking task.

## Integration with Tracking

SwinReID is integrated into the tracking pipeline to:
- **Extract Features**: Generate feature embeddings from vehicle images for ReID.
- **Compute Similarity Scores**: Compare features to associate detections across frames and cameras.
- **Maintain Identities**: Ensure consistent vehicle identities in single-camera and cross-camera tracking scenarios.

The component is used within the `aicupTracking.py` script, which handles both single-camera and cross-camera tracking, making it versatile for various tracking applications.


4. **Prepare Datasets**:
   - Download the [VeRi-776 dataset](https://github.com/JDAI-CV/VeRidataset) and place it in `./data/veri776`.
   - Obtain the AI Cup dataset and organize it in `./data/aicup` with the structure:
     ```
     data/aicup/
     ├── train/
     │   ├── images/<timestamp>/*.jpg
     │   ├── labels/<timestamp>/*.txt
     ├── valid/
     │   ├── images/<timestamp>/*.jpg
     │   ├── labels/<timestamp>/*.txt
     ```

5. **Download Pretrained Models**:
   - Download Swin Transformer pretrained weights 
   - Place them in the appropriate directory as specified in the configuration files.

## Usage

### Training on VeRi-776
Train the SwinReID model on the VeRi-776 dataset:
```bash
python train.py --dataset veri776_datasets --save_dir save_path --backbone model_architecture --model_weights mdoel_weight_path
```

### Fine-Tuning on AI Cup
Fine-tune the model on the AI Cup dataset:
```bash
python train.py --dataset AICUP_datasets --save_dir save_path --backbone model_architecture --model_weights mdoel_weight_path
```

### Running Tracking
Use the trained SwinReID model in the tracking pipeline:
```bash
python main.py \
  --weights-path /home/eddy/.../swin_center_loss_best.pth \
  --image-root   /home/eddy/.../AICUP_datasets/test/images \
  --label-root   /home/eddy/.../AICUP_datasets/test/labels \
  --output-root  /home/eddy/.../reid_tracking \
  --batch-size   32 \
  --buffer-size  5 \
  --threshold    0.6
```


