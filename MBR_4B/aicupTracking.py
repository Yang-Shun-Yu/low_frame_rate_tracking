import os
from PIL import Image
from tqdm import tqdm
import argparse
import torch
import yaml
from torchvision import transforms
import torch.nn.functional as F
from processor import get_model
from typing import OrderedDict
import math
import copy

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
    folder_path = os.path.join('/home/eddy/Desktop/vehicle_reid_itsc2023/trackingResult/labels', folder)
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
    folder_path = os.path.join('/home/eddy/Desktop/vehicle_reid_itsc2023/trackingResult/labels', folder)
    file_path = os.path.join(folder_path, file_name)
    for buffer_feature, buffer_id, center_x_ratio, center_y_ratio, _, _ in buffer_feature_id_list:
        storage[file_path].append((buffer_feature,buffer_id, center_x_ratio, center_y_ratio))

def write_storage(merge_storage, storage_forward, storage_reverse, save_path):
    """
    Write all storage dictionaries to disk under save_path.
    """
    for storage, label_folder in zip(
        [merge_storage, storage_forward, storage_reverse],
        ['merge_labels', 'forward_labels', 'reverse_labels']
    ):
        for file_path, entries in storage.items():
            # pull out just the time‐range folder and the filename
            time_range = os.path.basename(os.path.dirname(file_path))
            file_name  = os.path.basename(file_path)

            # build exactly: <save_path>/<label_folder>/<time_range>/<file_name>
            out_folder = os.path.join(save_path, label_folder, time_range)
            os.makedirs(out_folder, exist_ok=True)
            out_path = os.path.join(out_folder, file_name)

            # debug print—uncomment to verify where it's writing:
            # print("Writing tracking labels to:", out_path)

            with open(out_path, 'w') as f:
                for _, obj_id, _, _ in entries:
                    f.write(f"{obj_id}\n")


# def write_storage(merge_storage, storage_forward, storage_reverse,multi_cam_storage):
#     """
#     Write all storage dictionaries to disk.
#     The folder name is changed based on label type.
#     """

#     for storage, label_folder in zip(
#         [merge_storage, storage_forward, storage_reverse,multi_cam_storage],
#         ['merge_labels', 'forward_labels', 'reverse_labels','multi_cam_labels']
#     ):
#         for file_path, entries in storage.items():

#             parts = file_path.split(os.sep)
#             parts[-3] = label_folder  # Change folder name to match label type
#             folder_path = os.sep.join(parts[:-1])
#             os.makedirs(folder_path, exist_ok=True)
#             final_file_path = os.sep.join(parts)

#             with open(final_file_path, 'w') as f:
#                 for _,obj_id, _, _ in entries:
#                     f.write(f"{obj_id}\n")

# def write_storage(storage):
#     """
#     Write all storage dictionaries to disk.
#     The folder name is changed based on label type.
#     """

#     for storage, label_folder in zip(
#         [storage],
#         ['forward_without_labels']
#     ):
#         for file_path, entries in storage.items():

#             parts = file_path.split(os.sep)
#             parts[-3] = label_folder  # Change folder name to match label type
#             folder_path = os.sep.join(parts[:-1])
#             os.makedirs(folder_path, exist_ok=True)
#             final_file_path = os.sep.join(parts)

#             with open(final_file_path, 'w') as f:
#                 for _,obj_id, _, _ in entries:
#                     f.write(f"{obj_id}\n")



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
    
    The function does the following in a loop:
      1. Group assignments by their assigned_id.
      2. For any group that contains duplicates, choose one candidate to keep (the one
         that best matches the expected position from the latest buffer entry).
      3. For the other duplicates, compute a score based on similarity to the latest buffer
         entries (ignoring those with the duplicate id) and if the score exceeds the given
         threshold, reassign using the candidate's id; otherwise, assign a new unique id.
    
    Args:
        assignments (list): List of tuples as described above.
        buffer (OrderedDefaultdict): The buffer used to track recent frames.
        threshold (float): Similarity threshold used for duplicate resolution.
        time_range (str): A time range identifier (used in logging or further logic).
        cam_id (str): The camera id identifier.
        id_counter (int): Starting unique id counter to use for new assignments.
    
    Returns:
        (assignments, id_counter) (tuple):
            assignments (list): Updated list of assignments with resolved IDs.
            id_counter (int): Updated id_counter after new assignments.
    """
    iteration = 0
    while True:
        iteration += 1
        # Group indices in assignments by their assigned ID.
        id_groups = defaultdict(list)
        for idx, (_, assigned_id, _, _, _, _) in enumerate(assignments):
            id_groups[assigned_id].append(idx)

        # Identify groups where duplicates exist.
        duplicates = {aid: idxs for aid, idxs in id_groups.items() if len(idxs) > 1}
        if not duplicates:
            break

        # Get the most recent buffer entry (assuming this is most relevant).
        _, latest_buffer_list = next(reversed(buffer.items()))

        # Process each duplicate group.
        for dup_id, indices in duplicates.items():
            best_idx = None
            best_cost = float('inf')
            # For each candidate in the duplicate group, compute a cost with respect to the latest
            # buffer entry. Lower cost means a better match.
            for idx in indices:
                feat, a_id, cx, cy, dpx, dpy = assignments[idx]
                cost_candidate = float('inf')
                # Compare against each candidate in the latest buffer with the same duplicate id.
                for buf_feat, buf_id, buf_cx, buf_cy, buf_dpx, buf_dpy in latest_buffer_list:
                    if buf_id != dup_id:
                        continue
                    if buf_dpx is not None and buf_dpy is not None:
                        # Compute a predicted position based on the buffer.
                        pred_x = buf_cx + buf_dpx
                        pred_y = buf_cy + buf_dpy
                        error = math.sqrt((cx - pred_x) ** 2 + (cy - pred_y) ** 2)
                        cost_candidate = min(cost_candidate, error)
                    else:
                        # If no displacement info is available, use (1 - similarity) as error.
                        if buf_feat is not None and feat is not None:
                            sim = cosine_similarity(feat, buf_feat).squeeze().item()
                            cost_candidate = min(cost_candidate, 1 - sim)
                if cost_candidate < best_cost:
                    best_cost = cost_candidate
                    best_idx = idx

            # Remove the index to keep from the list of duplicate indices.
            if best_idx in indices:
                indices.remove(best_idx)

            # For the remaining indices in this duplicate group, assign new unique IDs.
            for idx in indices:
                feat, a_id, cx, cy, dpx, dpy = assignments[idx]
                sim_scores = []
                # Build a similarity matrix using all buffer entries (ignore those with the duplicate id).
                for buf_feat, buf_id, buf_cx, buf_cy, _, _ in latest_buffer_list:
                    if buf_id == dup_id:
                        continue
                    if buf_feat is None or feat is None:
                        continue
                    sim = cosine_similarity(feat, buf_feat).squeeze().item()
                    sim_scores.append((sim, buf_id, buf_cx, buf_cy))
                # Sort in descending order so that highest similarity comes first.
                sim_scores.sort(key=lambda x: x[0], reverse=True)
                # Choose a candidate from sim_scores based on the current iteration.
                if len(sim_scores) >= iteration:
                    candidate_sim, candidate_id, ref_cx, ref_cy = sim_scores[iteration - 1]
                    if candidate_sim > threshold:
                        new_id = candidate_id
                        # Update displacement relative to the reference from the buffer.
                        dpx = cx - ref_cx
                        dpy = cy - ref_cy
                    else:
                        new_id = id_counter
                        id_counter += 1
                else:
                    new_id = id_counter
                    id_counter += 1
                # Update the assignment with the newly assigned ID.
                assignments[idx] = (feat, new_id, cx, cy, dpx, dpy)

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
    cand_f, cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y, cand_sim = candidate

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
    if angle_penalty > angle_threshold and distance > 0.1:
        return float('inf'), cand_f

    alpha = 0.0  # Weight for angular penalty; adjust as needed.
    cost = distance + alpha * angle_penalty - cand_sim * 0.5

    return cost, cand_f

def remove_candidate_from_buffer(buffer, candidate_id):
    """
    Remove every item from buffer_copy that has candidate ID == candidate_id.
    This ensures that once a candidate mapping is chosen, that candidate is not re-used.
    """
    for key in buffer:
        buffer[key] = [item for item in buffer[key] if item[1] != candidate_id]

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
        # buffer_copy = copy.deepcopy(buffer)
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
                                candidates.append((buf_feature,buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y, sim))
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
                        cand_f,cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y ,_ = tmp_candidate
                        assigned_id = cand_id
                        disp_x = center_x - cand_center_x
                        disp_y = center_y - cand_center_y
                        best_candidate = (cand_center_x, cand_disp_x, cand_center_y, cand_disp_y)
                        # remove_candidate_from_buffer(buffer_copy, cand_id)

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
                        sim = cosine_similarity(feature, buf_feature).squeeze()
                        candidates.append((buf_feature,buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y, 0))
                    
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
                            cand_f,cand_id, cand_center_x, cand_center_y, cand_disp_x, cand_disp_y ,_ = tmp_candidate
                            assigned_id = cand_id
                            disp_x = center_x - cand_center_x
                            disp_y = center_y - cand_center_y
                            best_candidate = (cand_center_x, cand_disp_x, cand_center_y, cand_disp_y)
                            # remove_candidate_from_buffer(buffer_copy, cand_id)
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

def single_cam_without_predict(label_to_feature_map, thresholds, buffer_size=5, id_counter_start=1):
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
        # buffer_copy = copy.deepcopy(buffer)
        for obj in objects:
            feature = obj["feature"]

            assigned_id = None

            position_accuracy = None  # We'll calculate this if/when relevant

            if first_frame:
                assigned_id = id_counter
                id_counter += 1
            else:
                best_sim = -1
                candidates = None
                # Iterate over buffered frames in reverse order
                for _, buf_feature_list in reversed(buffer.items()):
                    for buf_feature, buf_id, buf_center_x, buf_center_y, buf_disp_x, buf_disp_y in buf_feature_list:
                        if buf_feature is not None:
                            sim = cosine_similarity(feature, buf_feature).squeeze()
                            if sim > best_sim:
                                best_sim = sim
                                candidates_id =buf_id
                if best_sim > thresholds:
                    assigned_id = candidates_id
                else:
                    assigned_id = id_counter
                    id_counter += 1                              
            temp_assignments.append((feature, assigned_id, None, None, None, None))

        first_frame = False
        buffer[label_path].extend(temp_assignments)

    # Flush remaining buffer entries to storage
    while buffer:
        save_buffer_to_storage(buffer, storage)

    # ---- NEW: Log the average accuracy across all predicted frames ----
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


# def multi_camera_mapping(merge_storage, thresholds):
#     """
#     Merge multi-camera tracking results by mapping IDs across cameras per time.
    
#     Args:
#         merge_storage (dict): Dictionary with file_path keys and values as lists of tuples 
#                               (feature, id, center_x, center_y).
#         all_camera_thresholds (float): Similarity threshold for mapping IDs across cameras.
        
#     Returns:
#         final_multi_camera_storage (defaultdict): Nested dict keyed by time and file_path with 
#                                                     tuples (feature, mapping_id, center_x, center_y).
#     """
#     from collections import defaultdict
#     from tqdm import tqdm
#     import torch.nn.functional as F

#     # Initialize data structures:
#     # multi_camera_storage: {time: {file_path: [(feature, mapping_id, center_x, center_y), ...]}}
#     # Clusters per time: {time: {camera: {id: [(feature, file_path, center_x, center_y), ...]}}}
#     cam_id_cluster_per_time = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     # For non-camera 0, record the set of camera ids encountered (camera 0 is our base and is treated separately)
#     camera_set_per_time = defaultdict(set)
#     # For assigning new mapping IDs per time
#     id_cont_per_time = defaultdict(lambda: 1)
#     # For camera 0, we create a mapping from original id to new mapping id
#     new_id_mapping = defaultdict(dict)
#     # Keep track of all times encountered
#     time_set = set()

#     # -------------------------------------------------------------------------
#     # STEP 1: Build initial clusters from merge_storage
#     # -------------------------------------------------------------------------
#     for file_path, entries in tqdm(merge_storage.items(), desc="Building clusters for multi-camera mapping"):
#         if entries[0] == (None, None, None, None):
#             continue

#         parts = file_path.split(os.sep)
#         if len(parts) < 2:
#             continue
#         time = parts[-2]  # Extract time from directory name
#         time_set.add(time)
#         try:
#             # Assume the file name starts with the camera id (e.g., "0_label.txt")
#             cam = int(parts[-1].split('_')[0])
#         except ValueError:
#             continue  # Skip if camera id is not an integer
        
#         # For non-camera 0, record the camera id
#         if cam != 0:
#             camera_set_per_time[time].add(cam)
        
#         for (feature, orig_id, center_x, center_y) in entries:
#             if cam == 0:
#                 # For camera 0, assign a new mapping id if not already done
#                 if orig_id not in new_id_mapping[time]:
#                     new_id_mapping[time][orig_id] = id_cont_per_time[time]
#                     id_cont_per_time[time] += 1
#                 mapping_id = new_id_mapping[time][orig_id]
#                 # Update multi_camera_storage and clusters for camera 0
#                 cam_id_cluster_per_time[time][0][mapping_id].append((feature, file_path, center_x, center_y))
#             else:
#                 # For non-camera 0, initially keep the original id (to be remapped later)
#                 if feature is None:
#                     cam_id_cluster_per_time[time][cam][orig_id].append((feature, file_path, center_x, center_y))
#                 else:

#                     # orig_id+=10000
#                     cam_id_cluster_per_time[time][cam][orig_id].append((feature, file_path, center_x, center_y))
    
#     # -------------------------------------------------------------------------
#     # STEP 2: Remap IDs for non-camera 0 clusters using reference clusters 
#     #         (camera 0 and lower-index non-zero cameras)
#     # -------------------------------------------------------------------------

#     logger.info("STEP 2: Remapping IDs for non-camera 0 clusters")
#     for time in tqdm(time_set, desc="Mapping IDs across cameras per time"):
#         # Get sorted non-zero cameras for this time (lowest first)
#         sorted_nonzero = sorted(camera_set_per_time[time])
#         time_range = "_".join(time.split('_')[1:])  # Extracts "150000_151900"

#         # For each non-zero camera
#         for cam in sorted_nonzero:
#             # Iterate over a copy of the clusters to allow key modifications
#             for cluster_id, cluster_entries in list(cam_id_cluster_per_time[time][cam].items()):
#                 if cluster_entries[0] == (None, None, None, None):
#                     continue
#                 best_similarity = -1
#                 # assign = False
#                 best_mapping_id = None
#                 # Define reference cameras: include camera 0 and any non-zero camera with an id lower than current cam
#                 ref_cams = [0] + [c for c in sorted_nonzero if c < cam]
#                 # Compare each feature in the current cluster with each feature in all reference clusters
#                 for ref_cam in ref_cams:
#                     for ref_cluster_id, ref_entries in cam_id_cluster_per_time[time].get(ref_cam, {}).items():
#                         # worst_similarity = 1
#                         for (feature, _, _, _) in cluster_entries:
#                             if feature is None:
#                                 continue  # Skip if the feature is None
#                             for (ref_feature, _, _, _) in ref_entries:
#                                 if ref_feature is None:
#                                     continue  # Skip if the reference feature is None
#                                 sim = cosine_similarity(feature, ref_feature).squeeze()
#                                 # if sim < worst_similarity:
#                                 #     worst_similarity = sim
#                                     # best_mapping_id = ref_cluster_id
#                                 if sim > best_similarity:
#                                     best_similarity = sim
#                                     best_mapping_id = ref_cluster_id

#                         # if worst_similarity > all_camera_thresholds[time_range] and worst_similarity>best_similarity:
#                         #     if worst_similarity>best_similarity:
#                         #         best_mapping_id = ref_cluster_id
#                         #         best_similarity = worst_similarity
#                         #         assign = True
                            
#                 # Decide the mapping id based on the similarity threshold
#                 if best_similarity > thresholds:
#                     # logger.debug("At time %s, comparing cluster %s and ref_cluster %s: sim=%.4f", time, cluster_id, best_mapping_id, best_similarity)

#                     mapping_id = best_mapping_id
#                 else:
#                     mapping_id = id_cont_per_time[time]
#                     id_cont_per_time[time] += 1
#                 # Update the cluster's mapping id if it differs from its original cluster_id
#                 if mapping_id != cluster_id:
#                     if mapping_id in cam_id_cluster_per_time[time][cam]:
#                         cam_id_cluster_per_time[time][cam][mapping_id].extend(cluster_entries)
#                     else:
#                         cam_id_cluster_per_time[time][cam][mapping_id] = cluster_entries
#                     del cam_id_cluster_per_time[time][cam][cluster_id]
    
#     # logger.info("STEP 2: Remapping IDs for non-camera 0 clusters")
#     # for time in tqdm(time_set, desc="Mapping IDs across cameras per time"):
#     #     # Get sorted non-zero cameras for this time (lowest first)
#     #     sorted_nonzero = sorted(camera_set_per_time[time])
#     #     time_range = "_".join(time.split('_')[1:])  # e.g., "150000_151900"

#     #     # For each non-zero camera
#     #     for cam in sorted_nonzero:
#     #         # Iterate over a copy of the clusters to allow key modifications
#     #         for cluster_id, cluster_entries in list(cam_id_cluster_per_time[time][cam].items()):
#     #             if cluster_entries[0] == (None, None, None, None):
#     #                 continue

#     #             best_worst_similarity = -1  # Highest among worst-case similarities
#     #             best_mapping_id = None

#     #             # Define reference cameras: include camera 0 and any non-zero camera with id lower than current cam
#     #             ref_cams = [0] + [c for c in sorted_nonzero if c < cam]
#     #             # Compare current cluster with each reference cluster
#     #             for ref_cam in ref_cams:
#     #                 for ref_cluster_id, ref_entries in cam_id_cluster_per_time[time].get(ref_cam, {}).items():
#     #                     worst_similarity = None  # Will hold the minimum similarity for this reference cluster

#     #                     # Compare every feature in current cluster with every feature in the reference cluster
#     #                     for (feature, _, _, _) in cluster_entries:
#     #                         if feature is None:
#     #                             continue  # Skip if feature is missing
#     #                         for (ref_feature, _, _, _) in ref_entries:
#     #                             if ref_feature is None:
#     #                                 continue  # Skip if reference feature is missing
#     #                             sim = cosine_similarity(feature, ref_feature).squeeze()
#     #                             # For the current reference cluster, record the minimum similarity across all pairs
#     #                             if worst_similarity is None:
#     #                                 worst_similarity = sim
#     #                             else:
#     #                                 worst_similarity = min(worst_similarity, sim)
                        
#     #                     # If we obtained a valid worst similarity, check if it is the best so far
#     #                     if worst_similarity is not None and worst_similarity > best_worst_similarity:
#     #                         best_worst_similarity = worst_similarity
#     #                         best_mapping_id = ref_cluster_id

#     #             # Use the best worst similarity to decide mapping id
#     #             if best_worst_similarity > all_camera_thresholds[time_range] * 1:
#     #                 mapping_id = best_mapping_id
#     #             else:
#     #                 mapping_id = id_cont_per_time[time]
#     #                 id_cont_per_time[time] += 1

#     #             # Update the cluster's mapping id if it differs from its original cluster_id
#     #             if mapping_id != cluster_id:
#     #                 if mapping_id in cam_id_cluster_per_time[time][cam]:
#     #                     cam_id_cluster_per_time[time][cam][mapping_id].extend(cluster_entries)
#     #                 else:
#     #                     cam_id_cluster_per_time[time][cam][mapping_id] = cluster_entries
#     #                 del cam_id_cluster_per_time[time][cam][cluster_id]

#     # -------------------------------------------------------------------------
#     # STEP 3: Build a lookup table for final mapping IDs and reconstruct final storage
#     # -------------------------------------------------------------------------
#     # Create a mapping from file_path and a key (e.g., rounded coordinates) to final mapping id.
#     final_id_mapping = defaultdict(dict)
#     for time, cam_dict in cam_id_cluster_per_time.items():
#         for cam, cluster_dict in cam_dict.items():
#             for mapping_id, cluster_entries in cluster_dict.items():
#                 for (feature, file_path, center_x, center_y) in cluster_entries:
#                     # Check if center_x or center_y is None
#                     if center_x is None or center_y is None:
#                         key = (None, None)
#                     else:
#                         key = (round(center_x, 4), round(center_y, 4))
#                     final_id_mapping[file_path][key] = mapping_id


#     # Now, iterate over merge_storage in its original order and update mapping IDs
#     final_multi_camera_storage = defaultdict(list)
#     for file_path, entries in merge_storage.items():
#         if entries[0] == (None, None, None, None):
#             final_multi_camera_storage[file_path].append((None, None, None, None))
#             continue

#         for (feature, orig_id, center_x, center_y) in entries:
#             if center_x is None or center_y is None:
#                 key = (None, None)
#             else:
#                 key = (round(center_x, 4), round(center_y, 4))
#             # If a final mapping exists, use it; otherwise fallback to the original id.
#             mapping_id = final_id_mapping[file_path].get(key, orig_id)
#             final_multi_camera_storage[file_path].append((feature, mapping_id, center_x, center_y))
#             # logger.debug("Final mapping for %s: orig_id %s -> mapping_id %s", file_path, orig_id, mapping_id)

#     return final_multi_camera_storage

from collections import defaultdict
import torch.nn.functional as F

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY  # Union by arbitrary choice

    def get_groups(self):
        groups = {}
        for k in self.parent:
            root = self.find(k)
            if root not in groups:
                groups[root] = []
            groups[root].append(k)
        return list(groups.values())

def multi_camera_mapping(merge_storage, threshold):
    # Step 1: Group objects by time range
    time_to_objects = defaultdict(list)
    for file_path, entries in merge_storage.items():
        parts = file_path.split(os.sep)
        time_range = parts[-2]  # e.g., 'time_150000_151900'
        cam_str = parts[-1].split('_')[0]
        cam = int(cam_str)
        for feature, local_id, center_x, center_y in entries:
            if feature is not None:
                time_to_objects[time_range].append((cam, local_id, feature, center_x, center_y))

    # Step 2: Assign global IDs using union-find
    global_id_counter = 1
    global_id_mapping = defaultdict(dict)  # (time_range, cam, local_id) -> global_id

    for time_range, objects_list in time_to_objects.items():
        if not objects_list:
            continue
        union_find = UnionFind()
        # Add all (cam, local_id) to union-find
        for cam, local_id, _, _, _ in objects_list:
            key = (cam, local_id)
            if key not in union_find.parent:
                union_find.parent[key] = key

        # Compute pairwise similarities and union
        for i in range(len(objects_list)):
            for j in range(i + 1, len(objects_list)):
                cam1, id1, feat1, _, _ = objects_list[i]
                cam2, id2, feat2, _, _ = objects_list[j]
                if cam1 != cam2 and feat1 is not None and feat2 is not None:
                    sim = F.cosine_similarity(feat1, feat2).item()
                    if sim > threshold:
                        union_find.union((cam1, id1), (cam2, id2))

        # Get groups and assign global IDs
        groups = union_find.get_groups()
        for group in groups:
            global_id = global_id_counter
            for member in group:
                cam, local_id = member
                global_id_mapping[(time_range, cam, local_id)] = global_id
            global_id_counter += 1

    # Step 3: Build final_multi_camera_storage
    final_multi_camera_storage = defaultdict(list)
    for file_path, entries in merge_storage.items():
        parts = file_path.split(os.sep)
        time_range = parts[-2]
        cam_str = parts[-1].split('_')[0]
        cam = int(cam_str)
        for feature, local_id, center_x, center_y in entries:
            key = (time_range, cam, local_id)
            if key in global_id_mapping:
                global_id = global_id_mapping[key]
            else:
                # Handle unmapped objects (e.g., due to missing features)
                continue
            final_multi_camera_storage[file_path].append((feature, global_id, center_x, center_y))

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
    import argparse

    # -----------------------------------------------------------------------------
    # Parse command-line arguments
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Multi-camera vehicle re-ID tracking pipeline")
    parser.add_argument('--image_root', '-i',type=str, required=True,help='Root directory of input images')
    parser.add_argument('--label_root', '-l',type=str, required=True,help='Root directory of input label files')
    parser.add_argument('--save_path', '-s',type=str, required=True,help='Directory to save tracking results')
    parser.add_argument('--batch_size', '-b',type=int, default=16,help='Batch size for feature extraction')
    parser.add_argument('--num_workers', '-w',type=int, default=8,help='Number of DataLoader worker processes')
    parser.add_argument('--weights', '-p',type=str, required=True,help='Path to model weights directory (containing config.yaml and checkpoint)')
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Device setup
    # -----------------------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------------
    # Load model config & weights
    # -----------------------------------------------------------------------------
    cfg_path = os.path.join(args.weights, 'config.yaml')
    ckpt_path = os.path.join(args.weights, 'best_mAP.pt')
    with open(cfg_path, 'r') as f:
        data = yaml.safe_load(f)

    model = get_model(data, torch.device('cpu'))
    try:
        state = torch.load(ckpt_path, map_location='cpu')
        # strip 'module.' if needed
        if any(k.startswith('module.') for k in state.keys()):
            state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())
        model.load_state_dict(state)
    except Exception as e:
        logger.error("Failed to load weights: %s", e)
        raise

    net = model.to(device).eval()

    # -----------------------------------------------------------------------------
    # Prepare transforms
    # -----------------------------------------------------------------------------
    image_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(data['n_mean'], data['n_std']),
    ])

    # -----------------------------------------------------------------------------
    # Build feature map
    # -----------------------------------------------------------------------------
    os.makedirs(args.save_path, exist_ok=True)
    label_to_feature = create_label_feature_map(
        net,
        args.image_root,
        args.label_root,
        image_transform,
        args.batch_size,
        args.num_workers
    )

    # -----------------------------------------------------------------------------
    # Forward & reverse tracking
    # -----------------------------------------------------------------------------
    buffer_size = 3
    threshold   = 0.5

    storage_fwd = process_label_features(label_to_feature, threshold, buffer_size)
    # reverse order by time & frame
    rev_label_to_feature = dict(
        sorted(
            label_to_feature.items(),
            key=lambda kv: (
                os.path.basename(os.path.dirname(kv[0])),
                -int(os.path.basename(kv[0]).split('.')[0])
            )
        )
    )
    storage_rev = process_label_features(rev_label_to_feature, threshold, buffer_size)

    # -----------------------------------------------------------------------------
    # Merge, write results & update labels
    # -----------------------------------------------------------------------------
    merged = merge_storages(storage_fwd, storage_rev)
    
    # multi_cam = multi_camera_mapping(merged,0.5)
    write_storage(merged, storage_fwd, storage_rev, args.save_path)
    for folder in ['forward_labels','reverse_labels','merge_labels']:
        update_labels(
            os.path.join(args.save_path, folder),
            args.label_root
        )

    # for folder in ['forward_labels', 'reverse_labels', 'merge_labels']:
    #     target = os.path.join(args.save_path, folder)
    #     update_labels(target, args.label_root)

    logger.info('Tracking pipeline finished.')

logger.info('Tracking pipeline finished.')




# Ex : python aicupTracking.py -i /home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/images -l /home/eddy/Desktop/MasterThesis/mainProgram/AICUP_datasets/test/labels/ -s /home/eddy/Desktop/vehicle_reid_itsc2023/trackingResult -p /home/eddy/Desktop/vehicle_reid_itsc2023/logs/AICUP/MBR_4B/7/
