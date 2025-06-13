import os
from PIL import Image
from tqdm import tqdm
import argparse
import torch
import yaml
from torchvision import transforms
import torch.nn.functional as F
from model import make_model
import Transforms
from typing import OrderedDict
import math


"""
Tracking pipeline for re-identification:
- Preprocess features from images/labels.
- Perform forward and reverse tracking using a buffering mechanism.
- Merge the tracking storages.
"""
import logging
import os
import math
import copy
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Custom modules
from aicupDatasets import create_label_feature_map



import sys


# Configure logging: you can adjust level to DEBUG for detailed logs
# Configure logging
# Configure logging
log_filename = 'tracking_pipeline.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Example usage
logger.info('Tracking pipeline started.')

# =============================================================================
# Helper Classes and Functions
# =============================================================================
class OrderedDefaultdict(OrderedDict):
    """
    An OrderedDict that creates default values for missing keys.
    """
    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("first argument must be callable or None")
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value


def cosine_similarity(tensor1, tensor2):
    """Wrapper around PyTorch cosine similarity."""
    return F.cosine_similarity(tensor1, tensor2)


def write_buffer_to_disk(buffer):
    """
    Write the oldest buffer entry to disk.
    (Currently commented out file writing details can be enabled as needed.)
    """
    buffer_path, buffer_feature_id_list = buffer.popitem(last=False)
    folder = os.path.basename(os.path.dirname(buffer_path))
    file_name = os.path.basename(buffer_path)
    folder_path = os.path.join('/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/labels', folder)
    file_path = os.path.join(folder_path, file_name)
    os.makedirs(folder_path, exist_ok=True)
    with open(file_path, 'w') as f:
        for _, obj_id, _, _, _, _ in buffer_feature_id_list:
            f.write(f"{obj_id}\n")


def save_buffer_to_storage(buffer, storage):
    """
    Flush the oldest buffer entry into the provided storage dictionary.
    """
    buffer_path, buffer_feature_id_list = buffer.popitem(last=False)
    folder = os.path.basename(os.path.dirname(buffer_path))
    file_name = os.path.basename(buffer_path)
    folder_path = os.path.join('/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/labels', folder)
    file_path = os.path.join(folder_path, file_name)
    for buffer_feature, buffer_id, center_x_ratio, center_y_ratio, _, _ in buffer_feature_id_list:
        storage[file_path].append((buffer_feature,buffer_id, center_x_ratio, center_y_ratio))


def write_storage(merge_storage, storage_forward, storage_reverse):
    """
    Write all storage dictionaries to disk.
    The folder name is changed based on label type.
    """

    for storage, label_folder in zip(
        [merge_storage, storage_forward, storage_reverse],
        ['merge_labels', 'forward_labels', 'reverse_labels']
    ):
        for file_path, entries in storage.items():

            parts = file_path.split(os.sep)
            parts[-3] = label_folder  # Change folder name to match label type
            folder_path = os.sep.join(parts[:-1])
            os.makedirs(folder_path, exist_ok=True)
            final_file_path = os.sep.join(parts)

            with open(final_file_path, 'w') as f:
                for _,obj_id, _, _ in entries:
                    f.write(f"{obj_id}\n")



def update_labels(target_labels_folder, source_labels_folder):
    # Iterate through each subfolder in target_labels
    for subfolder in os.listdir(target_labels_folder):
        target_subfolder_path = os.path.join(target_labels_folder, subfolder)
        source_subfolder_path = os.path.join(source_labels_folder, subfolder)

        # Check if corresponding subfolder exists in source_labels
        if not os.path.exists(source_subfolder_path):
            # print(f"Skipping {subfolder}, corresponding folder not found in source labels.")
            continue

        # Iterate through all text files in target_labels subfolder
        for txt_file in os.listdir(target_subfolder_path):
            target_file_path = os.path.join(target_subfolder_path, txt_file)
            source_file_path = os.path.join(source_subfolder_path, txt_file)

            # Check if the corresponding source labels file exists
            if not os.path.exists(source_file_path):
                # print(f"Skipping {txt_file} in {subfolder}, corresponding file not found in source labels.")
                continue
            
            # Read the contents of the source labels file
            with open(source_file_path, "r") as source_file:
                source_lines = source_file.readlines()

            # Read the contents of the target labels file
            with open(target_file_path, "r") as target_file:
                target_lines = target_file.readlines()

            # **Remove 'None\n' lines from source_lines**
            source_lines = [line for line in source_lines if line.strip() != "None"]

            # If source_lines is now empty, clear the target file and continue
            if not source_lines:
                # print(f"File {source_file_path} contains only 'None'. Clearing target file {target_file_path}.")
                with open(target_file_path, "w") as target_file:
                    target_file.write("")  # Empty the file
                continue

            # Ensure target_labels has the same number of lines as source_labels
            if len(source_lines) != len(target_lines):
                print(f"Warning: {txt_file} in {subfolder} has a mismatch in line count ({len(source_lines)} vs {len(target_lines)}). Adjusting to minimum available lines.")
                min_lines = min(len(source_lines), len(target_lines))
                source_lines = source_lines[:min_lines]
                target_lines = target_lines[:min_lines]

            # Process each line
            updated_lines = []
            for i, source_line in enumerate(source_lines):
                source_parts = source_line.strip().split()
                target_value = target_lines[i].strip()  # The value from target_labels

                # Replace the last value in source_parts with the value from target_labels
                source_parts[-1] = target_value

                # Reconstruct the full line correctly
                updated_lines.append(" ".join(source_parts))

            # Write the modified content back to target_labels file
            with open(target_file_path, "w") as target_file:
                target_file.write("\n".join(updated_lines) + "\n")

            # print(f"Updated: {target_file_path}")


def write_id_mapping_to_txt(id_mapping, output_file):
    """
    Write the id mapping dictionary to a text file.
    Format: (time, cam, reverse_buffer_id): forward_buffer_id
    """
    with open(output_file, "w") as f:
        for key, value in id_mapping.items():
            f.write(f"{key}: {value}\n")


def resolve_duplicates(assignments, buffer, threshold, time_range, cam_id, id_counter):
    """
    Resolve duplicate IDs in the assignments list.
    
    Each assignment is a tuple:
        (feature, assigned_id, center_x_ratio, center_y_ratio, disp_x, disp_y)
    
    Returns:
        Updated assignments and the current id_counter.
    """
    cont = 0
    while True:
        cont += 1
        assigned_id_dict = defaultdict(list)
        for idx, (_, assigned_id, center_x, center_y, _, _) in enumerate(assignments):
            assigned_id_dict[assigned_id].append(idx)

        # Find duplicate IDs
        duplicates = {key: idx_list for key, idx_list in assigned_id_dict.items() if len(idx_list) > 1}
        if not duplicates:
            break

        # For each duplicate, decide which assignment to keep
        for dup_id, indices in duplicates.items():
            # Get the latest buffer entry (assumed most relevant)
            _, buffer_feature_list = next(reversed(buffer.items()))
            keep_index = -1
            # Use either displacement or similarity to choose best candidate
            for buffer_feature, buffer_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buffer_feature_list:
                if buffer_id != dup_id:
                    continue
                if buf_disp_x is not None and buf_disp_y is not None:
                    min_distance = float('inf')
                    for idx in indices:
                        predict_x = buf_center_x + buf_disp_x
                        predict_y = buf_center_y + buf_disp_y
                        delta_x = assignments[idx][2] - predict_x
                        delta_y = assignments[idx][3] - predict_y
                        distance = math.sqrt(delta_x**2 + delta_y**2)
                        if distance < min_distance:
                            keep_index = idx
                            min_distance = distance
                else:
                    max_sim = -1
                    for idx in indices:
                        sim = cosine_similarity(assignments[idx][0], buffer_feature).squeeze().item()
                        if sim > max_sim:
                            keep_index = idx
                            max_sim = sim
            if keep_index in duplicates[dup_id]:
                duplicates[dup_id].remove(keep_index)

        # Reassign new IDs for duplicates (other than the kept index)
        for dup_id, indices in duplicates.items():
            _, buffer_feature_list = next(reversed(buffer.items()))
            for idx in indices:
                feature, _, center_x, center_y, disp_x, disp_y = assignments[idx]
                new_id = None
                sim_matrix = []
                for buf_feature, buf_id, buf_center_x, buf_center_y, _, _ in buffer_feature_list:
                    if buf_id == dup_id:
                        continue
                    if buf_feature is None:
                        continue  # Skip if buf_feature is None
                    sim = cosine_similarity(feature, buf_feature).squeeze().item()
                    sim_matrix.append((sim, buf_id, buf_center_x, buf_center_y))
                sim_matrix.sort(key=lambda x: x[0], reverse=True)
                if len(sim_matrix) >= cont:
                    candidate_sim, candidate_id, cand_center_x, cand_center_y = sim_matrix[cont - 1]
                    if candidate_sim > threshold:
                        new_id = candidate_id
                        disp_x = center_x - cand_center_x
                        disp_y = center_y - cand_center_y
                if new_id is None:
                    new_id = id_counter
                    id_counter += 1
                assignments[idx] = (feature, new_id, center_x, center_y, disp_x, disp_y)
    return assignments, id_counter

def merge_storages(storage_forward, storage_reverse):
    """
    Merge forward and reverse storages based on coordinate clusters.
    
    The function builds clusters from the forward and reverse storages and
    creates an ID mapping when at least two coordinates match.
    """
    merge_storage = copy.deepcopy(storage_forward)
    forward_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    reverse_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Build clusters for forward storage
    for file_path, entries in tqdm(storage_forward.items(), desc="Processing forward storage"):
        parts = file_path.split(os.sep)
        time_range = parts[-2]
        cam = parts[-1].split('_')[0]
        for _,buffer_id, center_x, center_y in entries:
            forward_cluster[time_range][cam][buffer_id].append((center_x, center_y))

    # Build clusters for reverse storage
    for file_path, entries in tqdm(storage_reverse.items(), desc="Processing reverse storage"):
        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue
        time_range = parts[-2]
        cam = parts[-1].split('_')[0]
        for _,buffer_id, center_x, center_y in entries:
            reverse_cluster[time_range][cam][buffer_id].append((center_x, center_y))

    id_mapping = {}
    # For each time and camera, match IDs between clusters
    for time_range in tqdm(forward_cluster, desc="Mapping IDs over times"):
        for cam in tqdm(forward_cluster[time_range], desc=f"Time {time_range} cameras", leave=False):
            for r_id, r_coords in reverse_cluster[time_range][cam].items():
                best_count = 0
                best_f_id = None
                for f_id, f_coords in forward_cluster[time_range][cam].items():
                    count = 0
                    for (rx, ry) in r_coords:
                        for (fx, fy) in f_coords:
                            if rx == fx and ry == fy:
                                count += 1
                    if count > best_count:
                        best_count = count
                        best_f_id = f_id
                if best_count >= 2:
                    id_mapping[(time_range, cam, r_id)] = best_f_id

    write_id_mapping_to_txt(id_mapping, "id_mapping.txt")

    # Update merge_storage using id_mapping
    for file_path, entries in tqdm(storage_reverse.items(), desc="Updating merge storage"):
        current_entries = merge_storage[file_path]
        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue
        time_range = parts[-2]
        cam = parts[-1].split('_')[0]
        for (_,buffer_id, center_x, center_y) in entries:
            key = (time_range, cam, buffer_id)
            if key in id_mapping:
                new_id = id_mapping[key]
                if any(f_id == new_id and (f_x != center_x or f_y != center_y) for _,f_id, f_x, f_y in current_entries):
                    id_mapping.pop(key, None)
                    continue
                updated_entries = []
                for feature,f_id, f_x, f_y in current_entries:
                    if center_x == f_x and center_y == f_y :
                        # Check if any key in id_mapping maps to f_id.
                        updated_entries.append((feature,new_id, f_x, f_y))
                        # if not any(mapped_id ==f_id for mapped_id in id_mapping.values()):
                        #     updated_entries.append((feature,new_id, f_x, f_y))
                        # else:
                        #     updated_entries.append((feature,f_id, f_x, f_y))
                    else:
                        updated_entries.append((feature,f_id, f_x, f_y))
                current_entries = updated_entries
        merge_storage[file_path] = current_entries
    return merge_storage


# def process_label_features(label_to_feature_map, single_thresholds, buffer_size=5, id_counter_start=1):
#     """
#     Process a mapping of labels to feature objects, assigning tracking IDs using a buffer.
    
#     Args:
#         label_to_feature_map (dict): Mapping of label_path to list of feature objects.
#         single_thresholds (dict): Dictionary of thresholds keyed by time_range and cam_id.
#         buffer_size (int): Maximum number of frames to keep in the buffer.
#         id_counter_start (int): Initial ID to assign.
        
#     Returns:
#         storage (defaultdict): Storage dictionary with assigned IDs.
#     """
#     storage = defaultdict(list)
#     buffer = OrderedDefaultdict(list)
#     id_counter = id_counter_start
#     old_cam = None
#     first_frame = True
#     old_time_range = None

#     for label_path, objects in label_to_feature_map.items():
#         # Extract time_range and camera id from label_path
#         time_range = "_".join(os.path.basename(os.path.dirname(label_path)).split('_')[1:])
#         # time_range = os.path.basename(os.path.dirname(label_path))

#         cam_id = os.path.basename(label_path).split('_')[0]

#         # Reset buffer if camera changes
#         if old_time_range is None:
#             old_time_range = time_range
#         elif old_time_range != time_range:
#             id_counter = 1
#             old_time_range = time_range

#         if old_cam is None:
#             old_cam = cam_id
#         elif old_cam != cam_id:
#             while buffer:
#                 save_buffer_to_storage(buffer, storage)
#             # id_counter = 1
#             old_cam = cam_id
#             first_frame = True

#         # Flush buffer if size exceeded
#         if len(buffer) > buffer_size:
#             save_buffer_to_storage(buffer, storage)

#         # If no objects found, add a dummy entry
#         if objects == [[]]:
#             buffer[label_path].append((None, None, None, None, None, None))
#             continue

#         temp_assignments = []
#         for obj in objects:
#             feature = obj["feature"]
#             center_x = obj["center_x_ratio"]
#             center_y = obj["center_y_ratio"]
#             assigned_id = None
#             disp_x = None
#             disp_y = None

#             if first_frame:
#                 assigned_id = id_counter
#                 id_counter += 1
#             else:
#                 candidates = []
#                 found = False
#                 # Iterate over buffered frames in reverse order
#                 for _, buf_feature_list in reversed(buffer.items()):
#                     for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
#                         if buf_feature is not None:
#                             sim = cosine_similarity(feature, buf_feature).squeeze()
#                             if sim > single_thresholds[time_range][cam_id]:
#                                 # Calculate angles (in radians) for the predicted displacement (buffer) and the actual displacement.
#                                 # if buf_disp_x is not None and buf_disp_y is not None:
                                
#                                 #     angle_pred = math.atan2(buf_disp_y, buf_disp_x)
#                                 #     angle_actual = math.atan2(center_y - buf_center_y, center_x - buf_center_x)
#                                 #     # Compute the absolute difference between the angles.
#                                 #     angle_diff = abs(angle_pred - angle_actual)
#                                 #     # Normalize the angle difference to be within [0, pi]
#                                 #     if angle_diff > math.pi:
#                                 #         angle_diff = 2 * math.pi - angle_diff

#                                 #     # Set an acceptable threshold (e.g., 60 degrees in radians).
#                                 #     angle_threshold = math.radians(90)
#                                 #     if angle_diff < angle_threshold:
#                                 #         # Skip this candidate if the angle difference is too large.
#                                 #         found = True
#                                 #         candidates.append((buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y))
#                                 # else:
#                                 found = True
#                                 candidates.append((buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y))

                                    
#                     # if previous frame match stop at this frame
#                     if found:
#                         break
                
                
#                 if candidates:
#                     # if candidates larger than 1 , we use the motion predition method to find out the matching id
#                     if len(candidates) > 1:
#                         min_distance = float("inf")
#                         for cand in candidates:
#                             cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y = cand
#                             if cand_disp_x is None or cand_disp_y is None:
#                                 continue
#                             # Check if the displacement direction is consistent
#                             temp_disp = torch.tensor([center_x - cand_center_x, center_y - cand_center_y], dtype=torch.float32)
#                             buf_disp = torch.tensor([cand_disp_x, cand_disp_y], dtype=torch.float32)
#                             if torch.dot(temp_disp, buf_disp) < 0:
#                                 continue
#                             # Calculate angles (in radians) for the predicted displacement (buffer) and the actual displacement.
#                             # angle_pred = math.atan2(cand_disp_y, cand_disp_x)
#                             # angle_actual = math.atan2(center_y - cand_center_y, center_x - cand_center_x)

#                             # # Compute the absolute difference between the angles.
#                             # angle_diff = abs(angle_pred - angle_actual)
#                             # # Normalize the angle difference to be within [0, pi]
#                             # if angle_diff > math.pi:
#                             #     angle_diff = 2 * math.pi - angle_diff

#                             # # Set an acceptable threshold (e.g., 60 degrees in radians).
#                             # angle_threshold = math.radians(60)

#                             # if angle_diff > angle_threshold:
#                             #     continue  # Skip this candidate if the angle difference is too large.
#                             predict_x = cand_center_x + cand_disp_x
#                             predict_y = cand_center_y + cand_disp_y
#                             delta_x = center_x - predict_x
#                             delta_y = center_y - predict_y
#                             distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
#                             if distance < min_distance:
#                                 assigned_id = cand_id
#                                 min_distance = distance
#                                 disp_x = center_x - cand_center_x
#                                 disp_y = center_y - cand_center_y
#                     # if we only have one candidate we just assign the matching id
#                     elif len(candidates) == 1:
#                         assigned_id = candidates[0][0]
#                         disp_x = center_x - candidates[0][1]
#                         disp_y = center_y - candidates[0][2]

#                 # If still not assigned, try to find the closest feature based on displacement , use the motion prediction and lower the 
#                 # threshold value to matching
#                 if assigned_id is None:
#                     _, buf_feature_list = next(reversed(buffer.items()))
#                     min_distance = float("inf")
#                     close_feature = None
#                     for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
#                         if buf_disp_x is not None and buf_disp_y is not None:
#                             temp_disp = torch.tensor([center_x - buf_center_x, center_y - buf_center_y], dtype=torch.float32)
#                             buf_disp = torch.tensor([buf_disp_x, buf_disp_y], dtype=torch.float32)
#                             # make sure the predivtion motion are align with the new car motion , example if we predit the car would move to 
#                             # right , however the new car appera at the left , this must not be the candidate 
#                             if torch.dot(temp_disp, buf_disp) < 0:
#                                 continue

#                             # angle_pred = math.atan2(buf_disp_y, buf_disp_x)
#                             # angle_actual = math.atan2(center_y - buf_center_y, center_x - buf_center_x)

#                             # # Compute the absolute difference between the angles.
#                             # angle_diff = abs(angle_pred - angle_actual)
#                             # # Normalize the angle difference to be within [0, pi]
#                             # if angle_diff > math.pi:
#                             #     angle_diff = 2 * math.pi - angle_diff

#                             # # Set an acceptable threshold (e.g., 60 degrees in radians).
#                             # angle_threshold = math.radians(90)

#                             # if angle_diff > angle_threshold:
#                             #     continue  # Skip this candidate if the angle difference is too large.

#                             predict_x = buf_center_x + buf_disp_x
#                             predict_y = buf_center_y + buf_disp_y
#                             delta_x = center_x - predict_x
#                             delta_y = center_y - predict_y
#                             distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
#                             # find the most cloest prediction car 
#                             if distance < min_distance:
#                                 assigned_id = buf_id
#                                 min_distance = distance
#                                 disp_x = center_x - buf_center_x
#                                 disp_y = center_y - buf_center_y
#                                 close_feature = buf_feature
#                     # Check similarity to decide if a new ID is needed
#                     if assigned_id is None:
#                         assigned_id = id_counter
#                         id_counter += 1
#                     else:
#                         sim = cosine_similarity(feature, close_feature).squeeze()
#                         # lower the threshold and compare again
#                         if sim < single_thresholds[time_range][cam_id]/3:
#                             assigned_id = id_counter
#                             id_counter += 1

#             temp_assignments.append((feature, assigned_id, center_x, center_y, disp_x, disp_y))

#         # Resolve duplicates within the same frame
#         temp_assignments, id_counter = resolve_duplicates(
#             temp_assignments,
#             buffer,
#             single_thresholds[time_range][cam_id],
#             time_range,
#             cam_id,
#             id_counter
#         )
#         first_frame = False
#         buffer[label_path].extend(temp_assignments)

#     # Flush remaining buffer entries to storage
#     while buffer:
#         save_buffer_to_storage(buffer, storage)
#     return storage
def compute_candidate_cost(center_x, center_y, candidate, lower_bound=0.0, upper_bound=1.0):
    """
    Compute a cost for a candidate based on its predicted position compared to the observed center (center_x, center_y).
    If the predicted position (candidate's center + displacement) falls outside the specified boundaries,
    the cost is set to float('inf').
    
    If displacement is missing, the candidate is treated as stationary (predicted position equals its center),
    with an added penalty to denote lower confidence.
    
    Args:
        center_x, center_y (float): Observed center coordinates.
        candidate (tuple): (cand_f, cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y)
        lower_bound (float): Lower limit of valid coordinate range.
        upper_bound (float): Upper limit of valid coordinate range.
        
    Returns:
        (cost, cand_f) (tuple): 
            cost (float): Lower cost indicates a better match; float('inf') if out of bounds.
            cand_f: The candidate's feature (passed through unchanged).
    """
    cand_f, cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y = candidate

    # Handle missing displacement by treating candidate as stationary.
    if cand_disp_x is None or cand_disp_y is None:
        predicted_x = cand_center_x
        predicted_y = cand_center_y
        distance = math.sqrt((center_x - predicted_x) ** 2 + (center_y - predicted_y) ** 2)
        penalty = 0.0  # Adjust penalty if needed.
        return distance + penalty, cand_f

    # When displacement is available, compute predicted position.
    predicted_x = cand_center_x + cand_disp_x
    predicted_y = cand_center_y + cand_disp_y

    # Check boundaries for the predicted position.
    if predicted_x < lower_bound or predicted_x > upper_bound or predicted_y < lower_bound or predicted_y > upper_bound:
        return float('inf'), cand_f

    # Euclidean distance between predicted and observed centers.
    distance = math.sqrt((center_x - predicted_x) ** 2 + (center_y - predicted_y) ** 2)

    # Create vectors for candidate's displacement and the expected displacement.
    vec_candidate = torch.tensor([cand_disp_x, cand_disp_y], dtype=torch.float32)
    vec_expected = torch.tensor([center_x - cand_center_x, center_y - cand_center_y], dtype=torch.float32)

    # Compute the angular difference only if the vectors are large enough.
    if vec_candidate.norm().item() < 1e-6 or vec_expected.norm().item() < 1e-6:
        angle_penalty = 0.0
    else:
        cos_sim = torch.dot(vec_candidate, vec_expected) / (vec_candidate.norm() * vec_expected.norm())
        # Clamp cosine to [-1, 1] to handle numerical issues.
        cos_sim = max(min(cos_sim.item(), 1.0), -1.0)
        angle_diff = math.degrees(math.acos(cos_sim))
        angle_penalty = angle_diff

    # Reject candidates if the angular error exceeds threshold.
    angle_threshold = 60  # in degrees
    if angle_penalty > angle_threshold and distance > 0.05:
        return float('inf'), cand_f

    alpha = 0.0  # Weight for angular penalty; adjust as needed.
    cost = distance + alpha * angle_penalty

    return cost, cand_f



def process_label_features(label_to_feature_map, thresholds, buffer_size=5, id_counter_start=1):
    """
    Process a mapping of labels to feature objects, assigning tracking IDs using a buffer.
    Logs predicted vehicle positions, true centers, displacement differences,
    Euclidean distance, and predicted position accuracy percentage (預測位置精確度 %).
    Also logs the overall average accuracy across all predictions.
    """
    storage = defaultdict(list)
    buffer = OrderedDefaultdict(list)
    id_counter = id_counter_start
    new_cam = None
    first_frame = True
    new_time_range = None

    # Set a maximum error threshold (this is an example value; adjust as needed)
    max_error = 0.2

    # ---- NEW: Keep track of accuracy sums and counts ----
    total_accuracy_sum = 0.0
    accuracy_count = 0
    cont = 0

    for label_path, objects in label_to_feature_map.items():

        # Extract time_range and camera id from label_path
        time_range = "_".join(os.path.basename(os.path.dirname(label_path)).split('_')[1:])
        cam_id = os.path.basename(label_path).split('_')[0]

        # Reset buffer if camera or time changes
        if new_time_range is None:
            new_time_range = time_range
        elif new_time_range != time_range:
            id_counter = 1
            new_time_range = time_range

        if new_cam is None:
            new_cam = cam_id
        elif new_cam != cam_id:
            while buffer:
                save_buffer_to_storage(buffer, storage)
            new_cam = cam_id
            first_frame = True

        # Flush buffer if size exceeded
        if len(buffer) > buffer_size:
            save_buffer_to_storage(buffer, storage)

        # If no objects found, add a dummy entry
        if objects == []:
            buffer[label_path].append((None, None, None, None, None, None))
            continue

        temp_assignments = []
        for obj in objects:
            feature = obj["feature"]
            center_x = obj["center_x_ratio"]
            center_y = obj["center_y_ratio"]
            assigned_id = None
            disp_x = None
            disp_y = None
            position_accuracy = None  # We'll calculate this if/when relevant
            best_candidate = None

            if first_frame:
                assigned_id = id_counter
                id_counter += 1
            else:
                candidates = []
                found = False
                # Iterate over buffered frames in reverse order
                for _, buf_feature_list in reversed(buffer.items()):
                    for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
                        if buf_feature is not None:
                            sim = cosine_similarity(feature, buf_feature).squeeze()
                            if sim > thresholds:
                                found = True
                                candidates.append((buf_feature,buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y))
                    if found:
                        break

  
                if candidates:
                    best_cost = float("inf")
                    tmp_candidate = None
                    for candidate in candidates:
                        cost,_ = compute_candidate_cost(center_x, center_y, candidate)
                        if cost < best_cost:
                            best_cost = cost
                            tmp_candidate = candidate

                    if tmp_candidate is not None and best_cost < float("inf"):
                        cand_f,cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y = tmp_candidate
                        assigned_id = cand_id
                        disp_x = center_x - cand_center_x
                        disp_y = center_y - cand_center_y
                        best_candidate = (cand_center_x, cand_disp_x, cand_center_y, cand_disp_y)
                    else:
                        # Fallback action if no candidate meets the criteria.
                        assigned_id = None  # Or create a new ID if appropriate.

                candidates = []
                # If still not assigned, try to find the closest feature using the buffered frame
                if assigned_id is None:
                    _, buf_feature_list = next(reversed(buffer.items()))
                    close_feature = None
                    for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
                        if buf_feature is None:
                            continue
                        candidates.append((buf_feature,buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y))
                    
                    best_cost = float("inf")
                    tmp_candidate = None
                    for candidate in candidates:
                        cost,f = compute_candidate_cost(center_x, center_y, candidate)
                        if cost < best_cost:
                            best_cost = cost
                            tmp_candidate = candidate
                            close_feature = f
                       
                    # else:
                    
                    if close_feature is not None:
                        sim = cosine_similarity(feature, close_feature).squeeze()
                        if sim > thresholds / 2:
                            cand_f,cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y = tmp_candidate
                            assigned_id = cand_id
                            disp_x = center_x - cand_center_x
                            disp_y = center_y - cand_center_y
                            best_candidate = (cand_center_x, cand_disp_x, cand_center_y, cand_disp_y)
                        else:
                            assigned_id = id_counter
                            id_counter += 1
                            best_candidate = None
                            disp_x = None
                            disp_y = None

                    else :
                        assigned_id = id_counter
                        id_counter += 1
                        best_candidate = None
                        disp_x = None
                        disp_y = None
          

            if best_candidate is not None and best_candidate[1] is not None:
                pred_x = best_candidate[0] + best_candidate[1]
                pred_y = best_candidate[2] + best_candidate[3]
                # Calculate predicted position accuracy percentage
                delta_x = center_x - pred_x
                delta_y = center_y - pred_y
                distance = math.sqrt(delta_x**2 + delta_y**2)

                position_accuracy = max(0, min(100, (1 - (distance / max_error)) * 100))
                logger.info(
                    "Label %s: Predicted vehicle position: (%.4f, %.4f), True center: (%.4f, %.4f), "
                    "Minimum disp_x: %.4f, Minimum disp_y: %.4f, Distance: %.4f, 預測位置精確度: %.2f%%", 
                    label_path, pred_x, pred_y, center_x, center_y, disp_x, disp_y, distance, position_accuracy
                )
            # ---- NEW: Accumulate total accuracy if we computed one ----
            if position_accuracy is not None:
                total_accuracy_sum += position_accuracy
                accuracy_count += 1

            temp_assignments.append((feature, assigned_id, center_x, center_y, disp_x, disp_y))

        # Resolve duplicates within the same frame and update the buffer
        temp_assignments, id_counter = resolve_duplicates(
            temp_assignments,
            buffer,
            thresholds,
            time_range,
            cam_id,
            id_counter
        )
        first_frame = False
        buffer[label_path].extend(temp_assignments)

    # Flush remaining buffer entries to storage
    while buffer:
        save_buffer_to_storage(buffer, storage)

    # ---- NEW: Log the average accuracy across all predicted frames ----
    if accuracy_count > 0:
        avg_accuracy = total_accuracy_sum / accuracy_count
        logger.info("Overall average position accuracy: %.2f%% (based on %d predictions)", avg_accuracy, accuracy_count)
    else:
        logger.info("No valid accuracy measurements were computed.")

    return storage

import numpy as np
# kalman filter method
# =============================================================================
# Kalman Filter Class for Motion Prediction
# =============================================================================
class KalmanFilter:
    """
    Simple Kalman filter for tracking vehicle positions.
    State vector: [x, y, vx, vy]
    """
    def __init__(self, dt=1, process_noise=1e-2, measurement_noise=1e-1):
        self.dt = dt
        self.x = np.zeros((4, 1))  # initial state
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4)
        self.Q = process_noise * np.eye(4)
        self.R = measurement_noise * np.eye(2)

    def initialize(self, pos, velocity):
        self.x[0:2] = np.array(pos).reshape((2,1))
        self.x[2:4] = np.array(velocity).reshape((2,1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:2].flatten()

    def update(self, measurement):
        z = np.array(measurement).reshape((2,1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# =============================================================================
# Revised process_label_features with Kalman Filter Integration
# =============================================================================
# def process_label_features(label_to_feature_map, single_thresholds, buffer_size=5, id_counter_start=1):
#     """
#     Process a mapping of labels to feature objects, assigning tracking IDs using a buffer.
#     Uses a Kalman filter for motion prediction when a vehicle exhibits displacement.
#     """
#     storage = defaultdict(list)
#     buffer = OrderedDefaultdict(list)
#     id_counter = id_counter_start
#     old_cam = None
#     first_frame = True
#     old_time_range = None

#     max_error = 0.2

#     # NEW: Dictionary to hold Kalman filter instances for each vehicle ID
#     kalman_filters = {}

#     total_accuracy_sum = 0.0
#     accuracy_count = 0

#     for label_path, objects in label_to_feature_map.items():
#         # Extract time_range and camera id from label_path
#         time_range = "_".join(os.path.basename(os.path.dirname(label_path)).split('_')[1:])
#         cam_id = os.path.basename(label_path).split('_')[0]

#         if old_time_range is None:
#             old_time_range = time_range
#         elif old_time_range != time_range:
#             id_counter = 1
#             old_time_range = time_range

#         if old_cam is None:
#             old_cam = cam_id
#         elif old_cam != cam_id:
#             while buffer:
#                 save_buffer_to_storage(buffer, storage)
#             old_cam = cam_id
#             first_frame = True

#         if len(buffer) > buffer_size:
#             save_buffer_to_storage(buffer, storage)

#         if objects == [[]]:
#             buffer[label_path].append((None, None, None, None, None, None))
#             continue

#         temp_assignments = []
#         for obj in objects:
#             feature = obj["feature"]
#             center_x = obj["center_x_ratio"]
#             center_y = obj["center_y_ratio"]
#             assigned_id = None
#             disp_x = None
#             disp_y = None
#             position_accuracy = None
#             true_kf = None
#             best_candidate = None

#             if first_frame:
#                 assigned_id = id_counter
#                 id_counter += 1
#             else:
#                 candidates = []
#                 found = False
#                 # Iterate over buffered frames in reverse order
#                 for _, buf_feature_list in reversed(buffer.items()):
#                     for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
#                         if buf_feature is not None:
#                             sim = cosine_similarity(feature, buf_feature).squeeze()
#                             if sim > single_thresholds[time_range][cam_id]:
#                                 found = True
#                                 candidates.append((buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y))
#                     if found:
#                         break

#                 # Use Kalman filter prediction if multiple candidates are found
                
#                 if candidates:
#                     if len(candidates) > 1:
#                         min_distance = float("inf")
#                         for cand in candidates:
#                             cand_id, cand_center_x, cand_center_y,b_y is None:
#                                 continue
#                             # Check if the displacement direction is consistent
#                             temp_disp = torch.tensor([center_x - cand_center_x, center_y - cand_center_y], dtype=torch.float32)
#                             buf_disp = torch.tensor([cand_disp_x, cand_disp_y], dtype=torch.float32)
#                             if torch.dot(temp_disp, buf_disp) < 0:
#                                 continue
#                             if cand_id not in kalman_filters:
#                                 kf = KalmanFilter(dt=1)
#                                 initial_velocity = (cand_disp_x, cand_disp_y)
#                                 kf.initialize((cand_center_x, cand_center_y), initial_velocity)
#                                 kalman_filters[cand_id] = kf
#                             else:
#                                 kf = kalman_filters[cand_id]
#                             pred = kf.predict()
#                             predict_x, predict_y = pred[0], pred[1]
        
#                             delta_x = center_x - predict_x
#                             delta_y = center_y - predict_y
#                             distance = math.sqrt(delta_x**2 + delta_y**2)
#                             if distance < min_distance:
#                                 true_kf = kf
#                                 min_distance = distance
#                                 assigned_id = cand_id
#                                 disp_x = center_x - cand_center_x
#                                 disp_y = center_y - cand_center_y
#                                 best_candidate = (cand_centelabel_pathr_x, cand_center_y), initial_velocity)
#                                 kalman_filters[cand_id] = kf
#                             else:
#                                 kf = kalman_filters[cand_id]
#                             pred = kf.predict()
#                             pred_x, pred_y = pred[0], pred[1]
#                             true_kf = kf
#                             best_candidate = (cand_center_x, pred_x, cand_center_y, pred_y)



#                 # If no candidate was matched, fall back to a simple buffer search
#                 if assigned_id is None:
#                     _, buf_feature_list = next(reversed(buffer.items()))
#                     min_distance = float("inf")
#                     close_feature = None

#                     for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
#                         if buf_disp_x is not None and buf_disp_y is not None:

#                             temp_disp = torch.tensor([center_x - buf_center_x, center_y - buf_center_y], dtype=torch.float32)
#                             buf_disp = torch.tensor([buf_disp_x, buf_disp_y], dtype=torch.float32)
#                             # make sure the predivtion motion are align with the new car motion , example if we predit the car would move to 
#                             # right , however the new car appera at the left , this must not be the candidate 
#                             if torch.dot(temp_disp, buf_disp) < 0:
#                                 continue
#                             if buf_id not in kalman_filters:
#                                 kf = KalmanFilter(dt=1)
#                                 initial_velocity = (buf_disp_x, buf_disp_y)
#                                 kf.initialize((buf_center_x, buf_center_y), initial_velocity)
#                                 kalman_filters[buf_id] = kf
#                             else:
#                                 kf = kalman_filters[buf_id]
#                             pred = kf.predict()
#                             predict_x, predict_y = pred[0], pred[1]

#                             delta_x = center_x - predict_x
#                             delta_y = center_y - predict_y
#                             distance = math.sqrt(delta_x**2 + delta_y**2)
#                             if distance < min_distance:
#                                 true_kf = kf
#                                 min_distance = distance
#                                 assigned_id = buf_id
#                                 disp_x = center_x - buf_center_x
#                                 disp_y = center_y - buf_center_y
#                                 close_feature = buf_feature
#                                 best_candidate = (cand_center_x, predict_x, cand_center_y, predict_y)

#                     if assigned_id is None:
#                         _, buf_feature_list = next(reversed(buffer.items()))
#                         min_distance = float("inf")
#                         for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
#                             if buf_center_x is not None and buf_center_y is not None:
#                                 delta_x = center_x - buf_center_x
#                                 delta_y = center_y - buf_center_y
#                                 distance = math.sqrt(delta_x**2 + delta_y**2)
#                                 if distance < min_distance:
#                                     min_distance = distance
#                                     assigned_id = buf_id
#                                     disp_x = center_x - buf_center_x
#                                     disp_y = center_y - buf_center_y
#                                     close_feature = buf_feature
#                         if close_feature is not None:
#                             sim = cosine_similarity(feature, close_feature).squeeze()
                        
#                         if close_feature is None or sim < single_thresholds[time_range][cam_id] / 2:
#                             assigned_id = id_counter
#                             id_counter += 1
#                             disp_x = None
#                             disp_y = None
#                         # assigned_id = id_counter
#                         # id_counter += 1
#                         # disp_x = None 
#                         # disp_y = None

#                         # assigned_id = id_counter
#                         # id_counter += 1
#                     else:
#                         sim = cosine_similarity(feature, close_feature).squeeze()
#                         if sim < single_thresholds[time_range][cam_id] / 3:
#                             assigned_id = id_counter
#                             id_counter += 1
#                             true_kf = None
#                             best_candidate = None
#                             disp_x = None
#                             disp_y = None

#             if best_candidate is not None and best_candidate[1] is not None:
#                 pred_x = best_candidate[1]
#                 pred_y = best_candidate[3]
#                 # Calculate predicted position accuracy percentage
#                 delta_x = center_x - pred_x
#                 delta_y = center_y - pred_y
#                 distance = math.sqrt(delta_x**2 + delta_y**2)

#                 position_accuracy = max(0, min(100, (1 - (distance / max_error)) * 100))
#                 logger.info(
#                     "Label %s: Predicted vehicle position: (%.4f, %.4f), True center: (%.4f, %.4f), "
#                     "Minimum disp_x: %.4f, Minimum disp_y: %.4f, Distance: %.4f, 預測位置精確度: %.2f%%", 
#                     label_path, pred_x, pred_y, center_x, center_y, disp_x, disp_y, distance, position_accuracy
#                 )
#             if true_kf is not None:
#                 true_kf.update((center_x, center_y))

#             if position_accuracy is not None:
#                 total_accuracy_sum += position_accuracy
#                 accuracy_count += 1

#             temp_assignments.append((feature, assigned_id, center_x, center_y, disp_x, disp_y))

#         temp_assignments, id_counter = resolve_duplicates(
#             temp_assignments,
#             buffer,
#             single_thresholds[time_range][cam_id],
#             time_range,
#             cam_id,
#             id_counter
#         )
#         first_frame = False
#         buffer[label_path].extend(temp_assignments)

#     while buffer:
#         save_buffer_to_storage(buffer, storage)
#     if accuracy_count > 0:
#         avg_accuracy = total_accuracy_sum / accuracy_count
#         logger.info("Overall average position accuracy: %.2f%% (based on %d predictions)", avg_accuracy, accuracy_count)
#     else:
#         logger.info("No valid accuracy measurements were computed.")

#     return storage

def multi_camera_mapping(merge_storage, thresholds):
    """
    Merge multi-camera tracking results by mapping IDs across cameras per time.
    
    Args:
        merge_storage (dict): Dictionary with file_path keys and values as lists of tuples 
                              (feature, id, center_x, center_y).
        all_camera_thresholds (float): Similarity threshold for mapping IDs across cameras.
        
    Returns:
        final_multi_camera_storage (defaultdict): Nested dict keyed by time and file_path with 
                                                    tuples (feature, mapping_id, center_x, center_y).
    """
    from collections import defaultdict
    from tqdm import tqdm
    import torch.nn.functional as F

    # Initialize data structures:
    # multi_camera_storage: {time: {file_path: [(feature, mapping_id, center_x, center_y), ...]}}
    # Clusters per time: {time: {camera: {id: [(feature, file_path, center_x, center_y), ...]}}}
    cam_id_cluster_per_time = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # For non-camera 0, record the set of camera ids encountered (camera 0 is our base and is treated separately)
    camera_set_per_time = defaultdict(set)
    # For assigning new mapping IDs per time
    id_cont_per_time = defaultdict(lambda: 1)
    # For camera 0, we create a mapping from original id to new mapping id
    new_id_mapping = defaultdict(dict)
    # Keep track of all times encountered
    time_set = set()

    # -------------------------------------------------------------------------
    # STEP 1: Build initial clusters from merge_storage
    # -------------------------------------------------------------------------
    for file_path, entries in tqdm(merge_storage.items(), desc="Building clusters for multi-camera mapping"):
        if entries[0] == (None, None, None, None):
            continue

        parts = file_path.split(os.sep)
        if len(parts) < 2:
            continue
        time = parts[-2]  # Extract time from directory name
        time_set.add(time)
        try:
            # Assume the file name starts with the camera id (e.g., "0_label.txt")
            cam = int(parts[-1].split('_')[0])
        except ValueError:
            continue  # Skip if camera id is not an integer
        
        # For non-camera 0, record the camera id
        if cam != 0:
            camera_set_per_time[time].add(cam)
        
        for (feature, orig_id, center_x, center_y) in entries:
            if cam == 0:
                # For camera 0, assign a new mapping id if not already done
                if orig_id not in new_id_mapping[time]:
                    new_id_mapping[time][orig_id] = id_cont_per_time[time]
                    id_cont_per_time[time] += 1
                mapping_id = new_id_mapping[time][orig_id]
                # Update multi_camera_storage and clusters for camera 0
                cam_id_cluster_per_time[time][0][mapping_id].append((feature, file_path, center_x, center_y))
            else:
                # For non-camera 0, initially keep the original id (to be remapped later)
                if feature is None:
                    cam_id_cluster_per_time[time][cam][orig_id].append((feature, file_path, center_x, center_y))
                else:

                    # orig_id+=10000
                    cam_id_cluster_per_time[time][cam][orig_id].append((feature, file_path, center_x, center_y))
    
    # -------------------------------------------------------------------------
    # STEP 2: Remap IDs for non-camera 0 clusters using reference clusters 
    #         (camera 0 and lower-index non-zero cameras)
    # -------------------------------------------------------------------------

    logger.info("STEP 2: Remapping IDs for non-camera 0 clusters")
    for time in tqdm(time_set, desc="Mapping IDs across cameras per time"):
        # Get sorted non-zero cameras for this time (lowest first)
        sorted_nonzero = sorted(camera_set_per_time[time])
        time_range = "_".join(time.split('_')[1:])  # Extracts "150000_151900"

        # For each non-zero camera
        for cam in sorted_nonzero:
            # Iterate over a copy of the clusters to allow key modifications
            for cluster_id, cluster_entries in list(cam_id_cluster_per_time[time][cam].items()):
                if cluster_entries[0] == (None, None, None, None):
                    continue
                best_similarity = -1
                # assign = False
                best_mapping_id = None
                # Define reference cameras: include camera 0 and any non-zero camera with an id lower than current cam
                ref_cams = [0] + [c for c in sorted_nonzero if c < cam]
                # Compare each feature in the current cluster with each feature in all reference clusters
                for ref_cam in ref_cams:
                    for ref_cluster_id, ref_entries in cam_id_cluster_per_time[time].get(ref_cam, {}).items():
                        # worst_similarity = 1
                        for (feature, _, _, _) in cluster_entries:
                            if feature is None:
                                continue  # Skip if the feature is None
                            for (ref_feature, _, _, _) in ref_entries:
                                if ref_feature is None:
                                    continue  # Skip if the reference feature is None
                                sim = cosine_similarity(feature, ref_feature).squeeze()
                                # if sim < worst_similarity:
                                #     worst_similarity = sim
                                    # best_mapping_id = ref_cluster_id
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_mapping_id = ref_cluster_id

                        # if worst_similarity > all_camera_thresholds[time_range] and worst_similarity>best_similarity:
                        #     if worst_similarity>best_similarity:
                        #         best_mapping_id = ref_cluster_id
                        #         best_similarity = worst_similarity
                        #         assign = True
                            
                # Decide the mapping id based on the similarity threshold
                if best_similarity > thresholds:
                    # logger.debug("At time %s, comparing cluster %s and ref_cluster %s: sim=%.4f", time, cluster_id, best_mapping_id, best_similarity)

                    mapping_id = best_mapping_id
                else:
                    mapping_id = id_cont_per_time[time]
                    id_cont_per_time[time] += 1
                # Update the cluster's mapping id if it differs from its original cluster_id
                if mapping_id != cluster_id:
                    if mapping_id in cam_id_cluster_per_time[time][cam]:
                        cam_id_cluster_per_time[time][cam][mapping_id].extend(cluster_entries)
                    else:
                        cam_id_cluster_per_time[time][cam][mapping_id] = cluster_entries
                    del cam_id_cluster_per_time[time][cam][cluster_id]
    
    # logger.info("STEP 2: Remapping IDs for non-camera 0 clusters")
    # for time in tqdm(time_set, desc="Mapping IDs across cameras per time"):
    #     # Get sorted non-zero cameras for this time (lowest first)
    #     sorted_nonzero = sorted(camera_set_per_time[time])
    #     time_range = "_".join(time.split('_')[1:])  # e.g., "150000_151900"

    #     # For each non-zero camera
    #     for cam in sorted_nonzero:
    #         # Iterate over a copy of the clusters to allow key modifications
    #         for cluster_id, cluster_entries in list(cam_id_cluster_per_time[time][cam].items()):
    #             if cluster_entries[0] == (None, None, None, None):
    #                 continue

    #             best_worst_similarity = -1  # Highest among worst-case similarities
    #             best_mapping_id = None

    #             # Define reference cameras: include camera 0 and any non-zero camera with id lower than current cam
    #             ref_cams = [0] + [c for c in sorted_nonzero if c < cam]
    #             # Compare current cluster with each reference cluster
    #             for ref_cam in ref_cams:
    #                 for ref_cluster_id, ref_entries in cam_id_cluster_per_time[time].get(ref_cam, {}).items():
    #                     worst_similarity = None  # Will hold the minimum similarity for this reference cluster

    #                     # Compare every feature in current cluster with every feature in the reference cluster
    #                     for (feature, _, _, _) in cluster_entries:
    #                         if feature is None:
    #                             continue  # Skip if feature is missing
    #                         for (ref_feature, _, _, _) in ref_entries:
    #                             if ref_feature is None:
    #                                 continue  # Skip if reference feature is missing
    #                             sim = cosine_similarity(feature, ref_feature).squeeze()
    #                             # For the current reference cluster, record the minimum similarity across all pairs
    #                             if worst_similarity is None:
    #                                 worst_similarity = sim
    #                             else:
    #                                 worst_similarity = min(worst_similarity, sim)
                        
    #                     # If we obtained a valid worst similarity, check if it is the best so far
    #                     if worst_similarity is not None and worst_similarity > best_worst_similarity:
    #                         best_worst_similarity = worst_similarity
    #                         best_mapping_id = ref_cluster_id

    #             # Use the best worst similarity to decide mapping id
    #             if best_worst_similarity > all_camera_thresholds[time_range] * 1:
    #                 mapping_id = best_mapping_id
    #             else:
    #                 mapping_id = id_cont_per_time[time]
    #                 id_cont_per_time[time] += 1

    #             # Update the cluster's mapping id if it differs from its original cluster_id
    #             if mapping_id != cluster_id:
    #                 if mapping_id in cam_id_cluster_per_time[time][cam]:
    #                     cam_id_cluster_per_time[time][cam][mapping_id].extend(cluster_entries)
    #                 else:
    #                     cam_id_cluster_per_time[time][cam][mapping_id] = cluster_entries
    #                 del cam_id_cluster_per_time[time][cam][cluster_id]

    # -------------------------------------------------------------------------
    # STEP 3: Build a lookup table for final mapping IDs and reconstruct final storage
    # -------------------------------------------------------------------------
    # Create a mapping from file_path and a key (e.g., rounded coordinates) to final mapping id.
    final_id_mapping = defaultdict(dict)
    for time, cam_dict in cam_id_cluster_per_time.items():
        for cam, cluster_dict in cam_dict.items():
            for mapping_id, cluster_entries in cluster_dict.items():
                for (feature, file_path, center_x, center_y) in cluster_entries:
                    # Check if center_x or center_y is None
                    if center_x is None or center_y is None:
                        key = (None, None)
                    else:
                        key = (round(center_x, 4), round(center_y, 4))
                    final_id_mapping[file_path][key] = mapping_id


    # Now, iterate over merge_storage in its original order and update mapping IDs
    final_multi_camera_storage = defaultdict(list)
    for file_path, entries in merge_storage.items():
        if entries[0] == (None, None, None, None):
            final_multi_camera_storage[file_path].append((None, None, None, None))
            continue

        for (feature, orig_id, center_x, center_y) in entries:
            if center_x is None or center_y is None:
                key = (None, None)
            else:
                key = (round(center_x, 4), round(center_y, 4))
            # If a final mapping exists, use it; otherwise fallback to the original id.
            mapping_id = final_id_mapping[file_path].get(key, orig_id)
            final_multi_camera_storage[file_path].append((feature, mapping_id, center_x, center_y))
            # logger.debug("Final mapping for %s: orig_id %s -> mapping_id %s", file_path, orig_id, mapping_id)

    return final_multi_camera_storage




# last similarity methold
# def multi_camera_mapping(merge_storage, all_camera_thresholds):
#     """
#     Merge multi-camera tracking results by mapping IDs across cameras per time.
#     """
#     from collections import defaultdict
#     from tqdm import tqdm
#     import torch.nn.functional as F

#     # Initialize data structures
#     cam_id_cluster_per_time = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     camera_set_per_time = defaultdict(set)
#     id_cont_per_time = defaultdict(lambda: 1)
#     new_id_mapping = defaultdict(dict)
#     time_set = set()

#     logger.info("STEP 1: Building initial clusters for multi-camera mapping")
#     for file_path, entries in tqdm(merge_storage.items(), desc="Building clusters"):
#         parts = file_path.split(os.sep)
#         if len(parts) < 2:
#             logger.warning("Skipping file with unexpected path format: %s", file_path)
#             continue
#         time = parts[-2]
#         time_set.add(time)
#         try:
#             cam = int(parts[-1].split('_')[0])
#         except ValueError:
#             logger.error("Camera ID not found in file name: %s", parts[-1])
#             continue

#         if cam != 0:
#             camera_set_per_time[time].add(cam)
        
#         for (feature, orig_id, center_x, center_y) in entries:
#             if cam == 0:
#                 if orig_id not in new_id_mapping[time]:
#                     new_id_mapping[time][orig_id] = id_cont_per_time[time]
#                     id_cont_per_time[time] += 1
#                 mapping_id = new_id_mapping[time][orig_id]
#                 cam_id_cluster_per_time[time][0][mapping_id].append((feature, file_path, center_x, center_y))
#                 logger.debug("Camera 0: Mapped orig_id %s to mapping_id %s at time %s", orig_id, mapping_id, time)
#             else:
#                 cam_id_cluster_per_time[time][cam][orig_id].append((feature, file_path, center_x, center_y))
    
#     logger.info("STEP 2: Remapping IDs for non-camera 0 clusters")
#     for time in tqdm(time_set, desc="Mapping IDs per time"):
#         sorted_nonzero = sorted(camera_set_per_time[time])
#         time_range = "_".join(time.split('_')[1:])  
#         for cam in sorted_nonzero:
#             for cluster_id, cluster_entries in list(cam_id_cluster_per_time[time][cam].items()):
#                 best_similarity = -1
#                 assign = False
#                 best_mapping_id = None
#                 # Initialize worst_similarity properly (e.g., assuming cosine similarity in [0,1])
#                 worst_similarity = 1.0

#                 ref_cams = [0] + [c for c in sorted_nonzero if c < cam]
#                 for ref_cam in ref_cams:
#                     for ref_cluster_id, ref_entries in cam_id_cluster_per_time[time].get(ref_cam, {}).items():
#                         for (feature, _, _, _) in cluster_entries:
#                             if feature is None:
#                                 continue
#                             for (ref_feature, _, _, _) in ref_entries:
#                                 if ref_feature is None:
#                                     continue
#                                 sim = cosine_similarity(feature, ref_feature).squeeze()
#                                 # Update similarity metrics as needed
#                                 if sim < worst_similarity :
#                                     worst_similarity = sim
#                         if worst_similarity > all_camera_thresholds[time_range] and worst_similarity > best_similarity:
#                             best_similarity = worst_similarity
#                             best_mapping_id = ref_cluster_id
#                             assign = True
                            
#                             logger.debug("At time %s, comparing cluster %s and ref_cluster %s: sim=%.4f", time, cluster_id, ref_cluster_id, worst_similarity)
#                             worst_similarity = 1.0

#                 if assign:
#                     mapping_id = best_mapping_id
#                 else:
#                     mapping_id = id_cont_per_time[time]
#                     id_cont_per_time[time] += 1
#                 if mapping_id != cluster_id:
#                     if mapping_id in cam_id_cluster_per_time[time][cam]:
#                         cam_id_cluster_per_time[time][cam][mapping_id].extend(cluster_entries)
#                     else:
#                         cam_id_cluster_per_time[time][cam][mapping_id] = cluster_entries
#                     del cam_id_cluster_per_time[time][cam][cluster_id]
#                     logger.info("Reassigned cluster %s to mapping_id %s at time %s", cluster_id, mapping_id, time)

#     # STEP 3: Build final mapping lookup table
#     final_id_mapping = defaultdict(dict)
#     for time, cam_dict in cam_id_cluster_per_time.items():
#         for cam, cluster_dict in cam_dict.items():
#             for mapping_id, cluster_entries in cluster_dict.items():
#                 for (feature, file_path, center_x, center_y) in cluster_entries:
#                     key = (None, None) if center_x is None or center_y is None else (round(center_x, 4), round(center_y, 4))
#                     final_id_mapping[file_path][key] = mapping_id

#     final_multi_camera_storage = defaultdict(list)
#     for file_path, entries in merge_storage.items():
#         for (feature, orig_id, center_x, center_y) in entries:
#             key = (None, None) if center_x is None or center_y is None else (round(center_x, 4), round(center_y, 4))
#             mapping_id = final_id_mapping[file_path].get(key, orig_id)
#             final_multi_camera_storage[file_path].append((feature, mapping_id, center_x, center_y))
#             logger.debug("Final mapping for %s: orig_id %s -> mapping_id %s", file_path, orig_id, mapping_id)

#     logger.info("Completed multi-camera mapping")
#     return final_multi_camera_storage


                        



        





 

    
    
        



# =============================================================================
# Main Pipeline
# =============================================================================
if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # Setup: paths, models, and thresholds
    # -----------------------------------------------------------------------------
    # dataset_path = '/home/eddy/Desktop/cropped_image'
    backbone_model = 'swin'
    # weights_path = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets_fine_tune/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_center_loss_best.pth'
    
    # backbone_model = 'swin'
    # weights_path = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets_fine_tune/swin_center_lr_0.5_loss_3e-4_smoothing_0.1/swin_center_loss_best.pth'
    weights_path = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets_fine_tune/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_center_loss_best.pth'
    # weights_path = '/home/eddy/Desktop/MasterThesis/mainProgram/Veri776_datasets_train/swin_center_lr_0.5_loss_3e-4_smmothing_0.1/swin_centerloss_best.pth'

    # weights_path ="/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/model_weight/swin_use_test.pth"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for each camera ,calculate the threshold 


    # -----------------------------------------------------------------------------
    # Load image transformations and model
    # -----------------------------------------------------------------------------
    image_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/images'
    label_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/labels'
    # label_root_directory = '/home/eddy/Desktop/MasterThesis/dcslab-ai-cup2024/yolov9/runs/detect'
    # image_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/valid/images'
    # label_root_directory = '/home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/valid/labels'

    
    save_path = os.path.join('/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking', 'labels')
    os.makedirs(save_path, exist_ok=True)
    batch_size = 16
    num_workers = 16


    image_transform = Transforms.get_valid_transform()
    # my method 
    net = make_model(backbone=backbone_model, num_classes=3441)
    # 學長方法
    # net = make_model(backbone=backbone_model, num_classes=576)
    state_dict = torch.load(weights_path, map_location='cpu')
    net.load_state_dict(state_dict)
    net.eval()


    os.makedirs(save_path, exist_ok=True)

    # my method 
    # net = make_model(backbone=backbone_model, num_classes=3441)
    # 學長方法
    # net = make_model(backbone=backbone_model, num_classes=576)
    # state_dict = torch.load(weights_path, map_location='cpu')
    # net.load_state_dict(state_dict)
    # net.eval()

    # Create a label-to-feature mapping
    label_to_feature_map = create_label_feature_map(net, image_root_directory, label_root_directory, image_transform,batch_size,num_workers)

    # -----------------------------------------------------------------------------
    # Forward tracking
    # -----------------------------------------------------------------------------
    buffer_size = 3
    thresholds = 0.5
    storage_forward = process_label_features(label_to_feature_map, thresholds, buffer_size)

    # -----------------------------------------------------------------------------
    # Reverse tracking: sort labels in reverse (by time and frame)
    # -----------------------------------------------------------------------------
    reversed_label_to_feature_map = dict(
        sorted(
            label_to_feature_map.items(),
            key=lambda item: (
                os.path.basename(os.path.dirname(item[0])),  # time_range directory name
                -int(os.path.basename(item[0]).split('.')[0])  # descending frame number
            )
        )
    )
    storage_reverse = process_label_features(reversed_label_to_feature_map, thresholds, buffer_size)

    # -----------------------------------------------------------------------------
    # Merge storages and write results
    # -----------------------------------------------------------------------------
    merged_storage = merge_storages(storage_forward, storage_reverse)
    

    # final_multi_camera_storage = multi_camera_mapping(merged_storage,thresholds)
    write_storage(merged_storage, storage_forward, storage_reverse)

    # target_labels = '/home/eddy/Desktop/vehicle_reid_itsc2023/trackingResult/multi_camera_labels'
    source_labels = label_root_directory
    # update_labels(target_labels,source_labels)
    target_labels = '/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/forward_labels'
    update_labels(target_labels,source_labels)

    target_labels = '/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/reverse_labels'
    update_labels(target_labels,source_labels)

    target_labels = '/home/eddy/Desktop/MasterThesis/mainProgram/reid_tracking/merge_labels'
    update_labels(target_labels,source_labels)
logger.info('Tracking pipeline finished.')

    





