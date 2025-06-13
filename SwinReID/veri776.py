from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from collections import defaultdict
import torch
import os


class Veri776Train(Dataset):
    """
    Dataset for training on the Veri776 dataset.
    Each sample contains an anchor, a positive, and a negative image.
    """

    def __init__(
        self,
        img_paths: list,
        vehicle_ids: list,
        class_map: dict,
        transform,
        veri776_root: str,
    ):
        """
        Initialize the training dataset.

        Args:
            img_paths (list): List of image file paths.
            vehicle_ids (list): List of vehicle IDs corresponding to the images.
            class_map (dict): Mapping from vehicle ID to class index.
            transform: Transformations to be applied on the images.
            veri776_root (str): Root directory of the Veri776 dataset.
        """
        self.img_paths = np.array(img_paths)
        self.vehicle_ids = np.array(vehicle_ids)
        self.class_map = class_map
        self.transform = transform
        self.veri776_root = veri776_root
        self.class_tree = self.build_class_tree(img_paths, vehicle_ids, class_map)

    def build_class_tree(self, img_paths: list, vehicle_ids: list, class_map: dict) -> dict:
        """
        Build a mapping from each class to a list of image paths.

        Args:
            img_paths (list): List of image file paths.
            vehicle_ids (list): List of vehicle IDs.
            class_map (dict): Mapping from vehicle ID to class index.

        Returns:
            dict: A dictionary mapping each class index to its image paths.
        """
        class_tree = defaultdict(list)
        for vid, path in zip(vehicle_ids, img_paths):
            class_index = class_map[vid]
            class_tree[class_index].append(path)
        return class_tree

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        """
        Retrieve a triplet sample: anchor, positive, and negative images.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: A tuple containing a stacked tensor of images and a tensor of labels.
        """
        # Load anchor image
        anchor_path = os.path.join(self.veri776_root, "image_train", self.img_paths[index])
        anchor_img = Image.open(anchor_path)

        # Get the positive image class and candidate images (excluding the anchor)
        positive_class = self.class_map[self.vehicle_ids[index]]
        positive_candidates = [
            p for p in self.class_tree[positive_class] if p != self.img_paths[index]
        ]
        # Fallback: if no candidate exists, use the anchor image
        if not positive_candidates:
            positive_path = self.img_paths[index]
        else:
            positive_path = np.random.choice(positive_candidates)

        positive_img = Image.open(os.path.join(self.veri776_root, "image_train", positive_path))

        # Select a negative image from a different class
        negative_class = self.random_number_except(0, len(self.class_map), positive_class)
        negative_path = np.random.choice(self.class_tree[negative_class])
        negative_img = Image.open(os.path.join(self.veri776_root, "image_train", negative_path))

        # Apply transformations if provided
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        images = torch.stack((anchor_img, positive_img, negative_img), dim=0)
        labels = torch.tensor([positive_class, positive_class, negative_class])
        return images, labels

    def random_number_except(self, start: int, end: int, exclude: int) -> int:
        """
        Generate a random number within [start, end) excluding a specified value.

        Args:
            start (int): Start of the range (inclusive).
            end (int): End of the range (exclusive).
            exclude (int): Number to exclude.

        Returns:
            int: A randomly chosen number not equal to 'exclude'.
        """
        numbers = list(range(start, end))
        numbers.remove(exclude)
        return np.random.choice(numbers)


class Veri776Test:
    """
    Dataset for testing on the Veri776 dataset.
    """

    def __init__(
        self, img_file_names: list, vehicle_ids: list, transform, veri776_root: str
    ):
        """
        Initialize the test dataset.

        Args:
            img_file_names (list): List of test image file names.
            vehicle_ids (list): List of vehicle IDs.
            transform: Transformations to be applied on the images.
            veri776_root (str): Root directory of the Veri776 dataset.
        """
        self.img_file_names = np.array(img_file_names)
        self.vehicle_ids = vehicle_ids
        self.transform = transform
        self.veri776_root = veri776_root

    def __len__(self) -> int:
        return len(self.img_file_names)

    def __getitem__(self, index: int):
        """
        Retrieve a test sample.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: A tuple containing the image name and the transformed image.
        """
        image_name = self.img_file_names[index]
        img_path = os.path.join(self.veri776_root, "image_test", image_name)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return image_name, img


def parse_xml(xml_path: str):
    """
    Parse the XML file to extract image paths, vehicle IDs, and a class mapping.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        tuple: A tuple containing lists of image paths, vehicle IDs, and a class mapping dictionary.
    """
    with open(xml_path) as f:
        xml_content = f.read()
    et = ET.fromstring(xml_content)
    image_paths = []
    vehicle_ids = []
    class_map = {}
    cur_class = 0

    for item in et.iter("Item"):
        image_paths.append(item.attrib["imageName"])
        vehicle_id = int(item.attrib["vehicleID"])
        vehicle_ids.append(vehicle_id)
        if vehicle_id not in class_map:
            class_map[vehicle_id] = cur_class
            cur_class += 1
    return image_paths, vehicle_ids, class_map


def get_veri776_test(
    veri_776_path: str, num_workers: int, batch_size: int, transform
) -> DataLoader:
    """
    Create a DataLoader for the Veri776 test dataset.

    Args:
        veri_776_path (str): Root directory of the Veri776 dataset.
        num_workers (int): Number of worker processes.
        batch_size (int): Batch size for the DataLoader.
        transform: Transformations to be applied on the images.

    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    xml_path = os.path.join(veri_776_path, "test_label.xml")
    img_file_names, vehicle_ids, _ = parse_xml(xml_path)
    test_set = Veri776Test(img_file_names, vehicle_ids, transform, veri_776_path)
    return DataLoader(test_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)


def get_veri776_train(
    veri776_path: str,
    num_workers: int,
    batch_size: int,
    transform,
    drop_last: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the Veri776 training dataset.

    Args:
        veri776_path (str): Root directory of the Veri776 dataset.
        num_workers (int): Number of worker processes.
        batch_size (int): Batch size for the DataLoader.
        transform: Transformations to be applied on the images.
        drop_last (bool): Whether to drop the last incomplete batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: DataLoader for the training dataset.
    """
    xml_path = os.path.join(veri776_path, "train_label.xml")
    img_paths, vehicle_ids, class_map = parse_xml(xml_path)
    train_set = Veri776Train(img_paths, vehicle_ids, class_map, transform, veri776_path)
    return DataLoader(
        train_set,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
    )
