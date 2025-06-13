import torch.nn as nn
import torch

import os
import torch.nn.functional as F
from collections import OrderedDict
import DenseNet
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights,swin_v2_b,Swin_V2_B_Weights

# from yolov9.models.yolo import Model
# from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTModel
import ibnnet

__all__ = ['make_model', 'CNN_IBN', 'resnet101_ibn_a', 'resnext101_ibn_a', 'densenet169_ibn_a',
            'se_resnet101_ibn_a', 'swin_reid', 'resnet34_ibn_a']

def weights_init_kaiming(m: nn.Module) -> None:
    """
    Initialize weights using Kaiming normalization.
    
    Args:
        m (nn.Module): A module whose weights need initialization.
    """
    classname = m.__class__.__name__
    if "Linear" in classname:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif "Conv" in classname:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif "BatchNorm" in classname:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m: nn.Module) -> None:
    """
    Initialize classifier weights.
    
    Args:
        m (nn.Module): A module (typically Linear) for which weights need initialization.
    """
    classname = m.__class__.__name__
    if "Linear" in classname:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def get_backbone(backbone: str, pretrained: bool) -> nn.Module:
    """
    Retrieve the backbone model based on the specified type.
    
    Args:
        backbone (str): The backbone model identifier.
        pretrained (bool): Whether to load pretrained weights.
    
    Returns:
        nn.Module: The backbone model.
    
    Raises:
        ValueError: If the backbone type is not supported.
    """
    if backbone == "resnet_a":
        return torch.hub.load("XingangPan/IBN-Net", "resnet101_ibn_a", pretrained=pretrained)
    elif backbone == "resnet_b":
        return torch.hub.load("XingangPan/IBN-Net", "resnet101_ibn_b", pretrained=pretrained)
    elif backbone == "resnext_a":
        return torch.hub.load("XingangPan/IBN-Net", "resnext101_ibn_a", pretrained=pretrained)
    elif backbone == "seresnet_a":
        return torch.hub.load("XingangPan/IBN-Net", "se_resnet101_ibn_a", pretrained=pretrained)
    elif backbone == "resnet34":
        return torch.hub.load("XingangPan/IBN-Net", "resnet34_ibn_a", pretrained=pretrained)
    elif backbone == "resnext_b":
        return ibnnet.resnext101_ibn_b(pretrained=False)
    elif backbone == "seresnet_b":
        return ibnnet.se_resnet101_ibn_b(pretrained=False)
    elif backbone == "densenet":
        return DenseNet.densenet169_ibn_a(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
class CNN_IBN(nn.Module):
    """
    CNN model with an IBN backbone for re-identification tasks.

    Attributes:
        backbone (nn.Module): The backbone model.
        bottleneck (nn.BatchNorm1d): Batch normalization layer for embedding.
        classifier (nn.Linear): Classification layer.
    """

    def __init__(self, backbone, pretrained=True, num_classes=576, embedding_dim=2048):
        """
        Initialize the CNN_IBN model.

        Args:
            backbone (str): Identifier of the backbone model.
            pretrained (bool): Whether to load pretrained weights.
            num_classes (int): Number of classes for the classifier.
            embedding_dim (int): Dimensionality of the embedding space.
        """
        super(CNN_IBN, self).__init__()
        self.backbone = get_backbone(backbone, pretrained=pretrained)

        # Adjust the final layer based on the backbone type
        if backbone == "densenet":
            self.backbone.classifier = nn.Linear(
                self.backbone.classifier.in_features, embedding_dim
            )
        elif backbone == "resnet34":
            self.backbone.fc = nn.Linear(
                self.backbone.fc.in_features, embedding_dim
            )
        else:
            # For other backbones, remove the final fully connected layer
            self.backbone.fc = nn.Identity()

        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        # Freeze the bias of the batch norm layer
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        # Initialize layers
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the CNN_IBN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (features, normalized_features, classifier_output)
        """
        f_t = self.backbone(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    
class SwinReID(nn.Module):
    """
    Swin Transformer-based model for re-identification tasks.

    Attributes:
        swin (nn.Module): Swin Transformer backbone.
        bottleneck (nn.BatchNorm1d): Batch normalization layer for embedding.
        classifier (nn.Linear): Classification layer.
    """
    def __init__(self, num_classes, embedding_dim=2048, imagenet_weight=True):
        super().__init__()
        """
        Initialize the SwinReID model.

        Args:
            num_classes (int): Number of classes for the classifier.
            embedding_dim (int): Dimensionality of the embedding space.
            imagenet_weight (bool): Whether to load ImageNet-pretrained weights.
        """

        self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if imagenet_weight else None)

        self.swin.head = nn.Linear(self.swin.head.in_features, embedding_dim)
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.swin(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    

    
def make_model(backbone: str, num_classes: int, embedding_dim: int = 2048) -> nn.Module:
    """
    Factory function to create a model based on the backbone type.

    Args:
        backbone (str): The backbone model identifier. Options include 'swin', 'yolo', 'yolo11',
                        or IBN backbones.
        num_classes (int): Number of classes for classification.
        embedding_dim (int): Dimensionality of the embedding space.

    Returns:
        nn.Module: The constructed model.
    """
    print(f"Using {backbone} as backbone")

    if backbone == "swin":
        return SwinReID(num_classes=num_classes, embedding_dim=embedding_dim)
    # elif backbone == "yolo":
    #     # Assuming you want to use the YOLO model imported above
    #     return Model(num_classes=num_classes)
    elif backbone == "yolo11":
        # Placeholder: implement or import the Yolo11 model as needed
        raise NotImplementedError("Yolo11 model is not implemented.")
    elif backbone =='swin_v2':
        return SwinReIDV2(num_classes=num_classes, embedding_dim=embedding_dim)
    elif backbone=='vit':
        return ViTReID(num_classes=num_classes, embedding_dim=embedding_dim)
    
    # For other IBN backbones, use the CNN_IBN model
    return CNN_IBN(
        backbone=backbone, pretrained=True,
        num_classes=num_classes, embedding_dim=embedding_dim
    )


# this class is kept for compatibility issue run3, run4 and rerun use this class 
class Resnet101IbnA(nn.Module):
    def __init__(self, num_classes=576):
        from warnings import warn
        warn('Deprecated warning: You should only use this class if you want to load the model trained in older commits. You should use `make_model(backbone, num_classes)` to build the model in newer version.')

        super().__init__()
        self.resnet101_ibn_a = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
        
        embedding_dim = self.resnet101_ibn_a.fc.in_features
        
        self.resnet101_ibn_a.fc = nn.Identity() # pretend the last layer does not exist



        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.resnet101_ibn_a(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out

# class Yolo(nn.Module):
#     def __init__(self, num_classes, embedding_dim=2048, pretrained=True):
#         super().__init__()

#         cfg = 'yolov9/models/detect/yolov9-t.yaml'
#         yolo = Model(cfg, ch=3, nc=1)
#         self.backbone_layers = yolo.model[:9] 
#         self.backbone = nn.Sequential(*self.backbone_layers)
#         dummy_input = torch.randn(1, 3, 224, 224)
#         features = self.backbone(dummy_input)
#         features = features.view(features.size(0), -1)
#         backbone_output_dim = features.size(1)
#         self.fc = nn.Linear(backbone_output_dim , embedding_dim)

#         self.bottleneck = nn.BatchNorm1d(embedding_dim)
#         self.bottleneck.bias.requires_grad_(False)  # no shift

#         self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)
#     def forward(self, x):
#         f = self.backbone(x)             # Output shape: (N, C, H, W)
#         f = f.view(f.size(0), -1)        # Flatten to shape: (N, C * H * W)
#         f_t = self.fc(f)                 # Shape: (N, embedding_dim)
#         f_i = self.bottleneck(f_t)       # Shape: (N, embedding_dim)
#         out = self.classifier(f_i)       # Shape: (N, num_classes)
#         return f_t, f_i, out
    
# class Yolo11(nn.Module):
#     def __init__(self, num_classes, embedding_dim=2048):
#         super().__init__()
#         model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
#         model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
#         model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights
#         classification_model = model.model 
#         model_layers = classification_model.model  # This is an nn.Sequential
#         backbone_layers = model_layers[:10]  # Exclude the classification head at index 10
#         self.backbone = nn.Sequential(*backbone_layers)

#         dummy_input = torch.randn(1, 3, 224, 224)
#         features = self.backbone(dummy_input)
#         features = features.view(features.size(0), -1)
#         backbone_output_dim = features.size(1)
#         self.fc = nn.Linear(backbone_output_dim , embedding_dim)

#         self.bottleneck = nn.BatchNorm1d(embedding_dim)
#         self.bottleneck.bias.requires_grad_(False)  # no shift
#         self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)

#     def forward(self, x):
#         f = self.backbone(x)             # Output shape: (N, C, H, W)
#         f = f.view(f.size(0), -1)        # Flatten to shape: (N, C * H * W)
#         f_t = self.fc(f)                 # Shape: (N, embedding_dim)
#         f_i = self.bottleneck(f_t)       # Shape: (N, embedding_dim)
#         out = self.classifier(f_i)       # Shape: (N, num_classes)
#         return f_t, f_i, out
    
class SwinReIDV2(nn.Module):
    """
    Swin Transformer-based model for re-identification tasks.

    Attributes:
        swin (nn.Module): Swin Transformer backbone.
        bottleneck (nn.BatchNorm1d): Batch normalization layer for embedding.
        classifier (nn.Linear): Classification layer.
    """
    def __init__(self, num_classes, embedding_dim=2048, imagenet_weight=True):
        super().__init__()
        """
        Initialize the SwinReID model.

        Args:
            num_classes (int): Number of classes for the classifier.
            embedding_dim (int): Dimensionality of the embedding space.
            imagenet_weight (bool): Whether to load ImageNet-pretrained weights.
        """

        self.swin = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1 if imagenet_weight else None)

        self.swin.head = nn.Linear(self.swin.head.in_features, embedding_dim)
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.swin(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    
from transformers import ViTModel

class ViTReID(nn.Module):
    """
    ViT-based model for re-identification tasks.

    Attributes:
        vit (ViTModel): Pre-trained ViT backbone.
        embedding_projection (nn.Linear): Linear layer to project features to the desired embedding dimension.
        bottleneck (nn.BatchNorm1d): Batch normalization layer for embedding normalization.
        classifier (nn.Linear): Classification layer for identity loss.
    """

    def __init__(self, num_classes,embedding_dim=2048):
        super(ViTReID,self).__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") 
        self.embedding_projection = nn.Linear(self.vit.config.hidden_size,embedding_dim)
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        # Apply weight initialization.
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (Tensor): Input tensor (images).

        Returns:
            f_t (Tensor): Features used for triplet loss.
            f_i (Tensor): Normalized features for inference.
            out (Tensor): Classification logits for cross entropy loss.
        """
        # Forward pass through ViT. The model returns a dict with "last_hidden_state".
        outputs = self.vit(x)
        # Use the [CLS] token (first token) as the global image representation.
        cls_token = outputs.last_hidden_state[:, 0]
        # Project the feature to the desired embedding dimension.
        features = self.embedding_projection(cls_token)
        # f_t is used directly for triplet loss.
        f_t = features
        # f_i is passed through a bottleneck for normalized features (used for classification and inference).
        f_i = self.bottleneck(features)
        # Classifier output for cross entropy loss.
        out = self.classifier(f_i)
        return f_t, f_i, out