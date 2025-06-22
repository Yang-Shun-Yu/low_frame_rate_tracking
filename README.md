# PreprocessDataset : AI Cup to VeRi-776 Dataset Conversion Script
## Introduction

This Python script converts the AI Cup dataset into the VeRi-776 dataset format, a widely adopted benchmark for vehicle re-identification (Re-ID) tasks. The [VeRi-776 dataset](https://paperswithcode.com/dataset/veri-776) contains over 49,000 images of 776 vehicles captured by 20 cameras, with annotations for bounding boxes, vehicle types, colors, and brands. By converting the AI Cup dataset to this format, researchers and developers can leverage existing VeRi-776-compatible tools and models for vehicle Re-ID studies, enabling seamless integration and comparison with established benchmarks.

## Purpose

The script aims to facilitate the use of the AI Cup dataset in vehicle re-identification research by transforming it into the VeRi-776 format. This conversion is valuable for:
- **Researchers**: Evaluate models on a dataset structured similarly to VeRi-776, enabling direct comparisons with state-of-the-art methods.
- **Developers**: Integrate the AI Cup dataset into pipelines or tools designed for VeRi-776, streamlining development workflows.
- **Benchmarking**: Support cross-dataset evaluations in vehicle Re-ID, enhancing the robustness of research findings.

## Dependencies

Install all dependencies using:
```bash
pip install -r requirements.txt
```


## Usage

Run the script from the command line, providing paths to the input dataset and output directory:

```bash
python convert_to_veri776.py --dataset_dir /path/to/ai_cup_dataset --output_dir /path/to/output_directory
```

### Command-Line Arguments
- **`--dataset_dir`**: Path to the AI Cup dataset directory, which must have the following structure:
  - `train/images/` and `train/labels/`: Contain timestamp folders (e.g., `0902_150000_151900`) with image files (e.g., `0_00001.jpg`) and corresponding label files (e.g., `0_00001.txt`).
  - `valid/images/` and `valid/labels/`: Similar structure for validation (test) data.
- **`--output_dir`**: Path where the converted dataset will be saved.

### Example
To convert a dataset located at `/home/user/datasets/ai_cup` and save the output to `/home/user/datasets/veri776`, execute:
```bash
python convert_to_veri776.py --dataset_dir /home/user/datasets/ai_cup --output_dir /home/user/datasets/veri776
```

### Input Dataset Structure
The AI Cup dataset must follow this structure:

| Directory Path                     | Content Description                                      |
|------------------------------------|----------------------------------------------------------|
| `train/images/<timestamp>/`        | JPEG images (e.g., `0_00001.jpg`)                        |
| `train/labels/<timestamp>/`        | Text files with bounding box annotations (e.g., `0_00001.txt`) |
| `valid/images/<timestamp>/`        | JPEG images for validation                               |
| `valid/labels/<timestamp>/`        | Text files with bounding box annotations for validation  |

Each label file contains lines in the format:
```
class center_x center_y width height track_ID
```
- `center_x`, `center_y`, `width`, `height`: Normalized coordinates (0 to 1) for the bounding box.
- `track_ID`: Used as the `vehicleID` to identify unique vehicles.

## Output

The script generates the following files and directories in the `output_dir`:

| File/Directory         | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `image_train/`         | Cropped training images in VeRi-776 naming style.                           |
| `image_test/`          | Cropped test images from the validation set.                                |
| `image_query/`         | Query images (approximately one-third per vehicle, randomly selected from test set). |
| `train_label.xml`      | XML annotations for training images.                                        |
| `test_label.xml`       | XML annotations for test images.                                            |
| `name_train.txt`       | Text file listing training image names (one per line).                      |
| `name_test.txt`        | Text file listing test image names (one per line).                          |
| `name_query.txt`       | Text file listing query image names (one per line).                         |
| `gt_index.txt`         | Ground truth indices for evaluation, listing test image indices per query.  |
| `jk_index.txt`         | Empty file (not used in this script, possibly for future extensions).       |





