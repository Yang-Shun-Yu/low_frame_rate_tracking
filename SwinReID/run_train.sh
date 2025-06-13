#!/bin/bash

# Set dataset path and other common parameters

DATASET_PATH="/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets"
WORKERS=20
SMOOTHING=0.1
LR=1e-4
MODELWEIGHTPATH='/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_centerloss_best.pth'

# Define the list of backbones to iterate over
# BACKBONES=( "densenet" "seresnet_a")
BACKBONES=( "swin" )

# Declare an associative array to map backbones to their respective save directories
# declare -A SAVE_DIRS=(
#     ["resnet_a"]="resnet_a_smoothing_0.1"
#     ["resnet_a_center"]="resnet_a_center_lr_0.5_loss_3e-4_smoothing_0.1"
#     ["resnet_b"]="resnet_b_smoothing_0.1"
#     ["resnet_b_center"]="resnet_b_center_lr_0.5_loss_3e-4_smoothing_0.1"
# )
# declare -A SAVE_DIRS=(

#     # ["densenet_center"]="densenet_center_lr_0.5_loss_3e-4_smoothing_0.1"

#     ["swin_center"]="/home/eddy/Desktop/MasterThesis/mainProgram/revise_AICUP_train/model_weight/swin_center_lr_0.5_loss_3e-4_smoothing_0.1_aicup"
# )
SAVE_DIR="/home/eddy/Desktop/MasterThesis/mainProgram/revise_AICUP_train/model_weight/swin_center_lr_0.5_loss_3e-4_smoothing_0.1_aicup"


# Loop over each backbone
for BACKBONE in "${BACKBONES[@]}"; do
    # echo "Running training with backbone: $BACKBONE (without center loss)"
    
    # # Set the save directory for the current backbone without center loss
    # SAVE_DIR="${SAVE_DIRS[$BACKBONE]}"
    
    # Run the training without center loss
    # python train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --smoothing $SMOOTHING
    
    echo "Running training with backbone: $BACKBONE (with center loss)"
    
    # Set the save directory for the current backbone with center loss
    # SAVE_DIR="${SAVE_DIRS[${BACKBONE}_center]}"
    
    # Run the training with center loss
    # python /home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --center_loss --smoothing $SMOOTHING 

    python /home/eddy/Desktop/MasterThesis/mainProgram/revise_AICUP_train/train.py --dataset "$DATASET_PATH" --workers $WORKERS --save_dir "$SAVE_DIR" --backbone "$BACKBONE" --center_loss --smoothing $SMOOTHING --model_weights $MODELWEIGHTPATH --check_init
done
