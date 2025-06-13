import os
import shutil
import xml.etree.ElementTree as ET
import argparse
from PIL import Image
import random  # For random sampling
def process_partition_and_crop(root_dir, partition, target_img_dir):
    """
    Processes one partition (train or test). It traverses the timestamp folders in 
    partition/images, reads the corresponding label file in partition/labels, crops 
    the image according to each bounding box in the label file, generates a new image 
    name in VeRi-776 style, saves the cropped image to target_img_dir, and returns:
      - xml_items: a list of dictionaries for XML annotation.
      - names_list: a list of new image names.
      - mapping: a list of tuples (new_name, src_path) for further use (e.g., query selection).
    
    Assumes each label file contains one or more lines, with each line formatted as:
      class center_x center_y width height track_ID
    with normalized coordinates.
    """
    images_dir = os.path.join(root_dir, partition, "images")
    labels_dir = os.path.join(root_dir, partition, "labels")
    
    # List all timestamp folders (e.g. "0902_150000_151900")
    timestamp_folders = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    
    xml_items = []  # For XML annotations
    names_list = []  # For name_*.txt files
    mapping = []     # List of tuples (new_name, src_img_path)
    
    for ts in timestamp_folders:
        ts_images_dir = os.path.join(images_dir, ts)
        ts_labels_dir = os.path.join(labels_dir, ts)
        image_files = [f for f in os.listdir(ts_images_dir) if f.lower().endswith(".jpg")]
        
        for img in image_files:
            src_img_path = os.path.join(ts_images_dir, img)
            # Corresponding label file (assumes same basename with .txt extension)
            label_file = os.path.splitext(img)[0] + ".txt"
            label_path = os.path.join(ts_labels_dir, label_file)
            if not os.path.exists(label_path):
                print(f"Label file not found for {src_img_path}")
                continue

            # Open image with PIL
            try:
                image = Image.open(src_img_path)
            except Exception as e:
                print(f"Error opening {src_img_path}: {e}")
                continue
            W, H = image.size  # Get image dimensions

            # Read all bounding box lines
            with open(label_path, "r") as f:
                lines = f.readlines()
            if not lines:
                print(f"No labels found in {label_path}")
                continue

            # Extract camera and frame info from original filename.
            # Expecting format "CamID_FrameNum.jpg", e.g., "0_00001.jpg"
            parts = img.split("_")
            if len(parts) < 2:
                print(f"Unexpected filename format: {img}")
                continue
            cam = parts[0]    # e.g., "0"
            frame = os.path.splitext(parts[1])[0]  # e.g., "00001"
            try:
                cam_int = int(cam)
            except ValueError:
                cam_int = 0
            # Format cameraID as "cXXX" (e.g., "c001")
            cameraID_str = f"c{cam_int+1:03d}"
            
            # Process each bounding box (each line) in the label file.
            for idx, line in enumerate(lines):
                tokens = line.strip().split()
                if len(tokens) < 6:
                    print(f"Invalid label line in {label_path}: {line}")
                    continue
                
                # Parse values; use track_ID as vehicleID
                # Format: class center_x center_y width height track_ID
                try:
                    center_x = float(tokens[1])
                    center_y = float(tokens[2])
                    width_norm = float(tokens[3])
                    height_norm = float(tokens[4])
                    track_ID = tokens[5]
                    vehicleID = int(track_ID)
                except Exception as e:
                    print(f"Error parsing label in {label_path}: {e}")
                    continue
                
                vehicleID_str = f"{vehicleID:04d}"  # e.g., "0001"
                
                # Compute absolute bounding box coordinates
                # Convert normalized center and size to pixel coordinates
                x1 = int((center_x - width_norm / 2) * W)
                y1 = int((center_y - height_norm / 2) * H)
                x2 = int((center_x + width_norm / 2) * W)
                y2 = int((center_y + height_norm / 2) * H)
                # Clamp coordinates to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                
                # Crop the image region
                cropped_img = image.crop((x1, y1, x2, y2))
                
                # Construct new image name in VeRi-776 style.
                # Append an index if multiple boxes exist in the same image.
                new_name = f"{vehicleID_str}_{cameraID_str}_{ts}_{frame}_{idx+1}.jpg"
                
                # Create XML item attributes (colorID and typeID default to "1")
                item_attrib = {
                    "imageName": new_name,
                    "vehicleID": vehicleID_str,
                    "cameraID": cameraID_str,
                    "colorID": "1",
                    "typeID": "1"
                }
                xml_items.append(item_attrib)
                names_list.append(new_name)
                mapping.append((new_name, src_img_path))
                
                # Ensure target directory exists and save the cropped image
                os.makedirs(target_img_dir, exist_ok=True)
                dst_path = os.path.join(target_img_dir, new_name)
                try:
                    cropped_img.save(dst_path)
                except Exception as e:
                    print(f"Error saving cropped image {dst_path}: {e}")
    
    return xml_items, names_list, mapping

# def write_xml(xml_items, xml_file):
#     """
#     Writes XML annotation file from a list of item dictionaries.
#     """
#     root = ET.Element("Items")
#     for item in xml_items:
#         ET.SubElement(root, "Item", attrib=item)
#     tree = ET.ElementTree(root)
#     tree.write(xml_file, encoding="utf-8", xml_declaration=True)

import xml.etree.ElementTree as ET
import xml.dom.minidom

def write_xml(xml_items, xml_file):
    """
    Writes XML annotation file with the following structure:
    
    <?xml version="1.0" encoding="gb2312"?>
    <TrainingImages Version="1.0">
      <Items number="37778">
        <Item imageName="0001_c001_00016450_0.jpg" vehicleID="0001" cameraID="c001" colorID="1" typeID="4"/>
        <Item imageName="4317_c001_1016_150000_151900_00111_2.jpg" vehicleID="4317" cameraID="c001" colorID="1" typeID="1"/>
        ...
      </Items>
    </TrainingImages>
    
    The Items element's 'number' attribute is set to the number of items in xml_items.
    """
    # Create root element <TrainingImages Version="1.0">
    root = ET.Element("TrainingImages", attrib={"Version": "1.0"})
    
    # Create <Items> element with 'number' attribute
    items_elem = ET.SubElement(root, "Items", attrib={"number": str(len(xml_items))})
    
    # Append each item to the Items element
    for item in xml_items:
        ET.SubElement(items_elem, "Item", attrib=item)
    
    # Convert to string and pretty print using minidom with encoding "gb2312"
    xml_str = ET.tostring(root, encoding="utf-8")
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="utf-8")
    
    # Write the pretty-printed XML (as bytes) to file
    with open(xml_file, "wb") as f:
        f.write(pretty_xml)

def write_names_txt(names_list, txt_file):
    """
    Writes image names (one per line) into a text file.
    """
    with open(txt_file, "w") as f:
        for name in names_list:
            f.write(name + "\n")




def write_gt_index(query_folder, test_folder, output_file):
    # Get all jpg files from each folder and sort them
    query_files = sorted([f for f in os.listdir(query_folder) if f.lower().endswith('.jpg')])
    test_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')])
    
    # Open the output file for writing
    with open(output_file, 'w') as out_f:
        # Process each query image
        for query in query_files:
            # Extract the prefix from the query file (everything before the first underscore)
            query_prefix = query.split('_')[0]
            
            # Initialize a list to store matching indices (using 1-indexing)
            matching_indices = []
            
            # Loop through test images, with index
            for idx, test_file in enumerate(test_files):
                # Extract prefix of test image
                test_prefix = test_file.split('_')[0]
                # Check if the prefixes match and the filenames are not exactly the same
                if test_prefix == query_prefix and test_file != query:
                    # Append the 1-indexed position (idx+1) to the list
                    matching_indices.append(str(idx + 1))
            
            # Write the matching indices to the output file on one line
            out_f.write(" ".join(matching_indices) + "\n")


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert dataset to VeRi-776 format with cropping.")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to the original dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to output the converted dataset")
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    
    # Define target directories for images
    image_train_dir = os.path.join(output_dir, "image_train")
    image_test_dir  = os.path.join(output_dir, "image_test")
    image_query_dir = os.path.join(output_dir, "image_query")
    
    # Process training data
    print("Processing training data with cropping...")
    train_xml_items, train_names, _ = process_partition_and_crop(dataset_dir, "train", image_train_dir)
    write_xml(train_xml_items, os.path.join(output_dir, "train_label.xml"))
    write_names_txt(train_names, os.path.join(output_dir, "name_train.txt"))
    
    # Process test data
    print("Processing test data with cropping...")
    test_xml_items, test_names, test_mapping = process_partition_and_crop(dataset_dir, "valid", image_test_dir)
    write_xml(test_xml_items, os.path.join(output_dir, "test_label.xml"))
    write_names_txt(test_names, os.path.join(output_dir, "name_test.txt"))
    
    # Create query images from test data:
    # For each vehicleID, group the images and randomly select one third (rounded; at least one image).
    print("Selecting query images from test data (one third per vehicleID)...")
    vehicle_dict = {}
    for new_name, src_path in test_mapping:
        # Extract vehicleID from new image name (format: vehicleID_cameraID_timestamp_frame_index.jpg)
        vehicleID = new_name.split("_")[0]
        vehicle_dict.setdefault(vehicleID, []).append((new_name, src_path))
    
    query_items = []
    for vehicleID, items in vehicle_dict.items():
        count = len(items)
        # Calculate one third, rounding to nearest integer and ensuring at least one image is selected.
        query_count = max(1, round(count / 3))
        # Randomly sample query_count images from this vehicle's images.
        sampled_items = random.sample(items, query_count)
        query_items.extend(sampled_items)
    
    query_names = [item[0] for item in query_items]
    
    os.makedirs(image_query_dir, exist_ok=True)
    for new_name, src_path in query_items:
        # Copy the already cropped image from test partition to query folder.
        src_cropped_path = os.path.join(image_test_dir, new_name)
        dst_path = os.path.join(image_query_dir, new_name)
        if os.path.exists(src_cropped_path):
            shutil.copy2(src_cropped_path, dst_path)
        else:
            print(f"Query image {src_cropped_path} not found.")
    write_names_txt(query_names, os.path.join(output_dir, "name_query.txt"))

    # Create the gallery set by excluding the query images from the test mapping.
    # all_test_names = set([new_name for new_name, _ in test_mapping])
    # query_set = set(query_names)
    # gallery_names = list(all_test_names - query_set)
    
    # Write the gt_index.txt file.
    gt_index_file = os.path.join(output_dir, "gt_index.txt")
    jk_index_file = os.path.join(output_dir, "jk_index.txt")

    with open(jk_index_file, 'w'):
        pass

    write_gt_index(image_query_dir,image_test_dir,gt_index_file)


    # write_gt_index(query_names, gallery_names, gt_index_file)
    
    print("Conversion, cropping, and gt_index.txt creation completed.")