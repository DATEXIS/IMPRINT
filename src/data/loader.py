"""
Dataset loading and management module.

This module provides utilities for loading, downloading, and preprocessing
various image datasets used in the weight imprinting framework. It handles
dataset acquisition from different sources.

The module supports automatic downloading, extraction, and validation of
image data, ensuring consistency across different backbone models.
"""

import os
import shutil
import zipfile
from tqdm import tqdm
from dotenv import load_dotenv


import pandas as pd
import rasterio
import requests
import torch
import torchvision.datasets as datasets
from rasterio import RasterioIOError
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from transformers import AutoImageProcessor

from src.models.backbone import backbone_weights
from src.utils.helpers import set_all_seeds
from src.data.mnist_m import MNISTM

load_dotenv()

available_datasets = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "MNIST-M",
    "USPS",
    "SVHN",
    # "CIFAR100",
    # "Places365",
    "ImageNet",
]

torch_datasets = {
    "MNIST": datasets.MNIST,
    "FashionMNIST": datasets.FashionMNIST,
    "CIFAR10": datasets.CIFAR10,
    "USPS": datasets.USPS,
    "SVHN": datasets.SVHN,
    # "CIFAR100": datasets.CIFAR100,
    # "Places365": datasets.Places365,
}


class DatasetHandler:
    """
    Handler for dataset acquisition, preprocessing and loading.

    This class manages the entire data pipeline from downloading raw datasets
    to preparing them for embedding extraction. It supports various dataset
    formats and sources.

    The handler applies appropriate transformations based on the selected
    backbone model and handles dataset splitting when needed.
    """

    def __init__(
        self,
        dataset_name: str = "CIFAR10",
        backbone_name: str = "resnet18",
        root: str = "./data",
        train: bool = False,
        batch_size: int = 64,
        seed: int = 42,
    ):
        """
        Initialize a dataset handler.

        Args:
                dataset_name: Name of the dataset to load
                backbone_name: Name of the backbone model to use for embeddings
                root: Root directory to store the raw data
                train: Whether to load training set (True) or test set (False)
                batch_size: Batch size for the DataLoader
                seed: Random seed for reproducibility
        """

        self.data_loader = None
        self.dataset_name = dataset_name
        self.backbone_name = backbone_name
        self.root = root
        self.train = train
        self.batch_size = batch_size
        self.seed = seed  # Store the seed for use in data loading

        self.train_str = "train" if self.train else "test"
        if self.dataset_name == "Places365" and self.train:
            self.train_str = "train-standard"
        if self.dataset_name in ["Places365", "ImageNet"] and not self.train:
            # For ImageNet, there is no test set. So we have to use the
            #  validation set here.
            self.train_str = "val"
        self.raw_data_path = os.path.join(self.root, "raw", self.dataset_name, self.train_str)

        # Check if the dataset is available on disk
        self.data_downloaded = os.path.exists(self.raw_data_path) and (
            len(os.listdir(self.raw_data_path)) > 0
        )
        # NOTE: This download is independent of class focus (because usually
        #  the raw data download is not selectable by class). But below,
        #  when setting up the dataloader, we use the focus_classes to sub-
        #  sample.

        self.load_dataset()

    def load_dataset(self):

        print("INFO: Setting up datasets folder.")

        if self.dataset_name in torch_datasets:
            if not self.data_downloaded:
                print(f"\tDownloading {self.dataset_name} ({self.train_str}) " "raw dataset...")
            else:
                print(
                    f"\t{self.dataset_name} ({self.train_str}) raw dataset "
                    "already exists. Loading from disk..."
                )

            if self.dataset_name == "Places365":
                dataset = torch_datasets[self.dataset_name](
                    root=self.raw_data_path,
                    split=self.train_str,
                    download=True,
                    small=True,  # 256x256 images
                )
            elif self.dataset_name == "SVHN":
                dataset = torch_datasets[self.dataset_name](
                    root=self.raw_data_path,
                    split=self.train_str,
                    download=True,
                )
            else:
                dataset = torch_datasets[self.dataset_name](
                    root=self.raw_data_path,
                    train=self.train,
                    download=True,
                )
        elif self.dataset_name == "MNIST-M":
            if self.data_downloaded:
                print(
                    f"	{self.dataset_name} ({self.train_str}) raw dataset "
                    "already exists. Loading from disk..."
                )

            dataset_path = os.path.join(self.root, "raw", self.dataset_name, self.train_str)

            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            dataset = MNISTM(
                root=dataset_path,
                train=self.train,  # True for training set, False for test set
                transform=None,
                target_transform=None,
                download=not (self.data_downloaded),
            )

        elif self.dataset_name == "ImageNet":
            dataset_path = self.download_from_imagenet(self.raw_data_path, split=self.train_str)
            dataset = datasets.ImageNet(root=dataset_path, split=self.train_str)
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} ({self.train_str}) is not available. "
                f"Choose one from {available_datasets}"
            )

        print("INFO: Setting up DataLoader...")
        transform = self.get_image_transformation()

        def collate_fn(batch):
            return (
                torch.stack([transform(img) for img, _ in batch]),
                torch.tensor([label for _, label in batch]),
            )

        # Use the provided seed
        set_all_seeds(self.seed)

        # Create a worker initialization function that uses set_all_seeds
        def seed_worker(worker_id):
            set_all_seeds(self.seed)

        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.train,
            # Set worker_init_fn to ensure deterministic behavior across workers
            worker_init_fn=seed_worker,
            # Uncomment these if needed for your setup:
            # num_workers=4,  # Number of CPU threads for loading data
            # pin_memory=True,  # Enable pinned memory for faster GPU transfers
            collate_fn=collate_fn,
        )
        print("INFO: Finished setting up DataLoader.")

    def get_image_transformation(self):
        # Initialize Weight Transforms for correct preprocessing
        # Check if we're using a HuggingFace model
        if self.backbone_name == "convnextv2-femto-1k-224":

            # Get image processor
            processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-femto-1k-224")

            # Create a transformation that uses the processor directly
            def transform_with_processor(img):
                # Convert grayscale to RGB if needed
                if self.dataset_name in ["FashionMNIST", "MNIST", "USPS"]:
                    # Convert PIL Image to RGB if it's grayscale
                    if img.mode == "L":
                        img = img.convert("RGB")

                # Process the image using the Hugging Face processor
                processed = processor(images=img, return_tensors="pt")
                # Return just the pixel_values tensor and remove the batch dimension
                return processed.pixel_values.squeeze(0)

            return transform_with_processor
        else:
            # Standard torchvision weights handling
            weights = backbone_weights[self.backbone_name]

            if self.dataset_name in [
                "FashionMNIST",
                "MNIST",
                "USPS",
            ]:  # grayscale datasets
                transform = transforms.Compose(
                    [transforms.Grayscale(num_output_channels=3), weights.transforms()]
                )
            else:
                transform = weights.transforms()

        return transform

    def download_from_imagenet(self, dataset_path, split="val"):

        if self.data_downloaded:
            print(
                f"	{self.dataset_name} ({self.train_str}) raw dataset "
                "already exists. Loading from disk..."
            )
            return dataset_path

        dataset_path = os.path.join(self.root, "raw", self.dataset_name, split)

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        print(f"	Downloading {self.dataset_name}-{split} from ImageNet...")

        imagenet_url_devkit_t_1_2 = os.getenv("IMAGENET_URL_DEVKIT_T_1_2")
        imagenet_url_train_t_1_2 = os.getenv("IMAGENET_URL_TRAIN_T_1_2")
        imagenet_url_val_t_all = os.getenv("IMAGENET_URL_VAL_T_ALL")
        urls = [
            imagenet_url_devkit_t_1_2,
        ]
        if split == "train":
            urls.append(imagenet_url_train_t_1_2)
        elif split == "val":
            urls.append(imagenet_url_val_t_all)
        else:
            raise ValueError(f"Invalid split {split} for ImageNet dataset")

        for url in urls:
            file_name = os.path.basename(url)
            save_path = os.path.join(dataset_path, file_name)
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with open(save_path, "wb") as file, tqdm(
                    desc=f"\t\tDownloading {os.path.basename(save_path)}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                        progress_bar.update(len(chunk))

        print(f"Downloaded {self.dataset_name}-{split} to {dataset_path}")

        return dataset_path
