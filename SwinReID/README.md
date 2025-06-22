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



## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Yang-Shun-Yu/low_frame_rate_tracking.git
   cd low_frame_rate_tracking/SwinReID
   ```

2. **Create a Conda Environment**:
   ```bash
   conda create --name swinreid_env python=3.8
   conda activate swinreid_env
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch==1.7.1 torchvision==0.8.2 timm==0.3.2
   ```

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
   - Download Swin Transformer pretrained weights (e.g., Swin-Base) from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).
   - Place them in the appropriate directory as specified in the configuration files.

## Usage

### Training on VeRi-776
Train the SwinReID model on the VeRi-776 dataset:
```bash
python train_swinreid.py --config_file configs/swinreid_veri776.yml
```

### Fine-Tuning on AI Cup
Fine-tune the model on the AI Cup dataset:
```bash
python fine_tune_swinreid.py --config_file configs/swinreid_aicup.yml
```

### Running Tracking
Use the trained SwinReID model in the tracking pipeline:
```bash
python aicupTracking.py --config_file configs/tracking_with_swinreid.yml
```

### Configuration Files
The project uses YAML configuration files in the `configs` directory to set parameters for training, fine-tuning, and tracking. Key files include:
- `swinreid_veri776.yml`: Settings for VeRi-776 training.
- `swinreid_aicup.yml`: Settings for AI Cup fine-tuning.
- `tracking_with_swinreid.yml`: Settings for tracking with SwinReID.

Modify these files to adjust parameters like learning rate, batch size, or model architecture.

## Example Workflow

1. **Setup Environment**:
   ```bash
   git clone https://github.com/Yang-Shun-Yu/low_frame_rate_tracking.git
   cd low_frame_rate_tracking/SwinReID
   conda create --name swinreid_env python=3.8
   conda activate swinreid_env
   pip install -r requirements.txt
   ```

2. **Prepare Datasets**:
   - Place VeRi-776 in `./data/veri776`.
   - Organize AI Cup dataset in `./data/aicup`.

3. **Train and Fine-Tune**:
   ```bash
   python train_swinreid.py --config_file configs/swinreid_veri776.yml
   python fine_tune_swinreid.py --config_file configs/swinreid_aicup.yml
   ```

4. **Run Tracking**:
   ```bash
   python aicupTracking.py --config_file configs/tracking_with_swinreid.yml
   ```

## Output

Running the tracking script with SwinReID generates:
- **Feature Embeddings**: Vehicle feature vectors for ReID.
- **Tracking Results**: Videos or sequences with bounding boxes and consistent vehicle IDs.
- **Log Files**: Performance metrics, such as detection confidence and association scores.

## Notes

- **Low-Frame-Rate Challenges**: SwinReID addresses issues like large vehicle displacements and appearance changes in low-frame-rate videos (e.g., 1 fps).
- **Swin Transformer**: The architecture’s patch-based processing and hierarchical design make it effective for ReID tasks, as noted in [TransReID](https://www.researchgate.net/publication/358999846_TransReID_Transformer-based_Object_Re-Identification).
- **Reproducibility**: Set a random seed (e.g., `random.seed(42)`) in scripts for consistent results.
- **GPU Requirements**: A GPU with at least 12GB memory is recommended for training on vehicle datasets.
- **Limitations**: Performance depends on dataset quality and configuration settings. Users should verify paths and weights.

## Troubleshooting

| Issue                              | Possible Solution                                                                 |
|------------------------------------|-----------------------------------------------------------------------------------|
| Missing dependencies               | Run `pip install -r requirements.txt` or install specific packages (e.g., `torch`). |
| Dataset not found                  | Verify dataset paths in `./data/veri776` and `./data/aicup`.                      |
| Tracking script fails              | Check `tracking_with_swinreid.yml` for correct paths and parameters.               |
| Poor ReID performance              | Ensure fine-tuning on AI Cup dataset; adjust hyperparameters in config files.      |

## Conclusion

SwinReID enhances the low-frame-rate tracking project by providing robust vehicle re-identification, crucial for maintaining consistent identities in challenging scenarios. Its integration with the tracking pipeline, supported by a three-stage training process, ensures high accuracy and reliability for both single-camera and cross-camera tracking tasks.
