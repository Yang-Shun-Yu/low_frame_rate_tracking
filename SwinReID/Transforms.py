from torchvision import transforms
from transformers import AutoImageProcessor
from PIL import Image
# Define ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard image size for resizing
IMAGE_SIZE = 224


def get_training_transform() -> transforms.Compose:
    """
    Compose a series of transformations for training images.

    The transformation pipeline includes:
      - Converting the image to a tensor.
      - Resizing the image to a fixed size.
      - Random horizontal flipping.
      - Normalization using ImageNet statistics.
      - Random cropping with padding.
      - Random erasing with a specified scale.

    Returns:
        transforms.Compose: A composition of training transformations.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE), padding=(8, 8)),
        transforms.RandomErasing(scale=(0.02, 0.4), value=IMAGENET_MEAN),
    ])
    return transform


def get_test_transform(equalized: bool = False) -> transforms.Compose:
    """
    Compose a series of transformations for test images.

    If `equalized` is True, the transformation pipeline will include random
    equalization; otherwise, it applies the standard transformations.

    Args:
        equalized (bool): Whether to apply random equalization. Default is False.

    Returns:
        transforms.Compose: A composition of test transformations.
    """
    if equalized:
        transform = transforms.Compose([
            transforms.RandomEqualize(p=1),
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transform


image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

class TrainingTransformWithAutoProcessor:
    """
    Custom transform that combines data augmentations with AutoImageProcessor.
    
    The pipeline does:
      1. Random horizontal flip and random crop (with padding) on the PIL image.
      2. Uses AutoImageProcessor to resize, normalize, and convert the image to a tensor.
      3. Applies random erasing on the resulting tensor.
    """
    def __init__(self, image_processor):
        self.image_processor = image_processor
        # Augmentations on the PIL image.
        self.augmentations = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMAGE_SIZE, padding=8)
        ])
        # RandomErasing is applied on the tensor.
        self.random_erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3))
    
    def __call__(self, image: Image.Image):
        # 1. Apply augmentations to the PIL image.
        augmented_image = self.augmentations(image)
        # 2. Use AutoImageProcessor to process the image:
        #    - Resize (if not already the correct size)
        #    - Normalize with the model's expected mean and std
        #    - Convert to a PyTorch tensor.
        processed = self.image_processor(augmented_image, return_tensors="pt")
        # The processor returns a dict with key "pixel_values" having shape (1, C, H, W)
        tensor = processed["pixel_values"].squeeze(0)  # remove batch dimension -> (C, H, W)
        # 3. Apply random erasing on the tensor.
        tensor = self.random_erasing(tensor)
        return tensor

class TestingTransformWithAutoProcessor:
    """
    Custom transform that combines data augmentations with AutoImageProcessor.
    
    The pipeline does:
      1. Random horizontal flip and random crop (with padding) on the PIL image.
      2. Uses AutoImageProcessor to resize, normalize, and convert the image to a tensor.
      3. Applies random erasing on the resulting tensor.
    """
    def __init__(self, image_processor):
        self.image_processor = image_processor
        # Augmentations on the PIL image.

    
    def __call__(self, image):

        #    - Convert to a PyTorch tensor.
        processed = self.image_processor(image, return_tensors="pt")
        # The processor returns a dict with key "pixel_values" having shape (1, C, H, W)
        tensor = processed["pixel_values"].squeeze(0)  # remove batch dimension -> (C, H, W)

        return tensor

def get_training_transform_vit():
    """
    Returns a transform function that applies the augmentations followed by
    the AutoImageProcessor pre-processing and random erasing.
    """
    return TrainingTransformWithAutoProcessor(image_processor)

def get_testing_transform_vit():

    return TestingTransformWithAutoProcessor(image_processor)