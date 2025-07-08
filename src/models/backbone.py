"""
Backbone models for feature extraction.

This module provides pre-trained backbones from torchvision and Hugging Face
that can be used for feature extraction in the imprinting framework.
Each backbone is configured to output feature embeddings by removing
classification layers.
"""

import torch
import torchvision.models as models

from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    Swin_B_Weights,
)
from transformers import ConvNextV2ForImageClassification

# List of supported backbone architectures
available_backbones = [
    "resnet18",
    "resnet50",
    "vit_b_16",
    "swin_b",
    "convnextv2-femto-1k-224",
]

# Pre-trained backbone weights for ImageNet
backbone_weights = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": ResNet50_Weights.IMAGENET1K_V1,
    "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,
    "swin_b": Swin_B_Weights.IMAGENET1K_V1,
}

backbone_lambda_regs = {
    "resnet18": 0.0001,  # see https://arxiv.org/pdf/1512.03385
    "resnet50": 0.0001,  # ""
    "vit_b_16": 0.1,  # see https://arxiv.org/pdf/2010.11929
    "swin_b": 0.05,  # see https://arxiv.org/pdf/2103.14030
    "convnextv2-femto-1k-224": 0.05,  # see https://arxiv.org/pdf/2301.00808
}


class BackboneHandler:
    """
    Handler class for neural network backbone models.

    This class manages the loading and configuration of pretrained backbones
    from torchvision, sets them up for feature extraction (by removing classification
    layers), and handles device placement.
    """

    def __init__(self, backbone_name, device_name="cpu"):
        """
        Initialize a backbone handler.

        Args:
            backbone_name (str): Name of the backbone model from available_backbones
            device_name (str): Device to run the model on ('cpu', 'cuda', 'mps')

        Raises:
            ValueError: If backbone_name is not in available_backbones
        """
        if backbone_name not in available_backbones:
            raise ValueError(
                f"Backbone {backbone_name} not supported. " f"Choose from: {available_backbones}"
            )

        self.backbone_name = backbone_name
        self.device_name = device_name

        # Determine device to use
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_name == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.is_huggingface_model = self.backbone_name == "convnextv2-femto-1k-224"
        self.backbone = self.load_backbone()

    def load_backbone(self):
        """
        Load and configure the backbone model.

        Returns:
            torch.nn.Module: Configured backbone model for feature extraction
        """
        if self.backbone_name == "convnextv2-femto-1k-224":
            # Load ConvNextV2 model from Hugging Face
            backbone = ConvNextV2ForImageClassification.from_pretrained(
                "facebook/convnextv2-femto-1k-224"
            )
            # Remove classification head
            backbone.classifier = torch.nn.Identity()

            # Move backbone to device
            backbone = backbone.to(self.device)
            backbone.eval()

            return backbone

        # Load backbone from torchvision
        backbone = getattr(models, self.backbone_name)(weights=backbone_weights[self.backbone_name])

        backbone = backbone.to(self.device)
        backbone.eval()

        # Remove classification layer for non-ViT models
        if self.backbone_name.startswith("resnet"):
            backbone.fc = torch.nn.Identity()
        elif self.backbone_name == "swin_b":
            backbone.head = torch.nn.Identity()
        elif self.backbone_name == "vit_b_16":
            backbone.heads = torch.nn.Identity()

        return backbone
