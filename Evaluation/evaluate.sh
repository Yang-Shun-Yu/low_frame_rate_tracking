#!/bin/bash

# Define directory paths
TS_RESULTS_DIR='/home/eddy/Desktop/MasterThesis/mainProgram/ts_results'

GT_RESULT_DIR='/home/eddy/Desktop/MasterThesis/mainProgram/gt_results'

SOURCE_LABELS='/home/eddy/Desktop/MasterThesis/mainProgram/low_frame_rate_tracking/Result/merge_labels/'

TARGET_LABELS='/home/eddy/Desktop/MasterThesis/mainProgram/low_frame_rate_tracking/Result/labels/'

# Ensure TARGET_LABELS exists (create it if it doesn't)
if [ ! -d "$TARGET_LABELS" ]; then
    echo "Creating target labels directory: $TARGET_LABELS"
    mkdir -p "$TARGET_LABELS" || {
        echo "Failed to create directory: $TARGET_LABELS"
        exit 1
    }
fi
cp -r "${SOURCE_LABELS}"* "${TARGET_LABELS}"

# Clear the ts_results directory
if [ -d "$TS_RESULTS_DIR" ]; then
    rm -rf "$TS_RESULTS_DIR"/*
    echo "Cleared directory: $TS_RESULTS_DIR"
else
    echo "Directory does not exist: $TS_RESULTS_DIR"
    exit 1
fi

# Clear the gt_results directory
if [ -d "$GT_RESULT_DIR" ]; then
    rm -rf "$GT_RESULT_DIR"/*
    echo "Cleared directory: $GT_RESULT_DIR"
else
    echo "Directory does not exist: $GT_RESULT_DIR"
    exit 1
fi

# Run the first Python script
python3 /home/eddy/Desktop/MasterThesis/mainProgram/tools/datasets/AICUP_to_MOT15.py \
    --AICUP_dir '/home/eddy/Desktop/MasterThesis/mainProgram/low_frame_rate_tracking/Result/labels' \
    --MOT15_dir "$TS_RESULTS_DIR"

# Check if the previous script executed successfully
if [ $? -ne 0 ]; then
    echo "First script failed. Terminating subsequent operations."
    exit 1
fi

# Run the second Python script
python3 /home/eddy/Desktop/MasterThesis/mainProgram/tools/datasets/AICUP_to_MOT15.py \
    --AICUP_dir '/home/eddy/Desktop/train/test/labels' \
    --MOT15_dir "$GT_RESULT_DIR"

# Check if the previous script executed successfully
if [ $? -ne 0 ]; then
    echo "Second script failed. Terminating subsequent operations."
    exit 1
fi

# Run the third Python script
python3 /home/eddy/Desktop/MasterThesis/mainProgram/tools/evaluate.py \
    --gt_dir "$GT_RESULT_DIR/" \
    --ts_dir "$TS_RESULTS_DIR/"
    # --mode "single_cam"

# Check if the previous script executed successfully
if [ $? -ne 0 ]; then
    echo "Third script failed."
    exit 1
fi

echo "All scripts executed successfully."
