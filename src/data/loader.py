"""
Dataset loading and management module.

This module provides utilities for loading, downloading, and preprocessing
various image datasets used in the weight imprinting framework. It handles
dataset acquisition from different sources including PyTorch datasets,
Kaggle, and custom URLs.

The module supports automatic downloading, extraction, and validation of
image data, ensuring consistency across different backbone models.
"""

import os
import shutil
import zipfile
from tqdm import tqdm
from dotenv import load_dotenv


import kaggle
import pandas as pd
import rasterio
import requests
import torch
import torchvision.datasets as datasets
import wild_time_data
from rasterio import RasterioIOError
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from src.models.backbone import backbone_weights

load_dotenv()

available_datasets = [
    "CIFAR10",
    "FashionMNIST",
    "MNIST",
    "ImageNet",
]

torch_datasets = {
    "CIFAR10": datasets.CIFAR10,
    "FashionMNIST": datasets.FashionMNIST,
    "KMNIST": datasets.KMNIST,
    "MNIST": datasets.MNIST,
}


class DatasetHandler:
    """
    Handler for dataset acquisition, preprocessing and loading.

    This class manages the entire data pipeline from downloading raw datasets
    to preparing them for embedding extraction. It supports various dataset
    formats and sources, including PyTorch datasets, Kaggle datasets, and
    custom URLs.

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
        split_ratio: float = 0.8,
    ):
        """
        Initialize a dataset handler.

        Args:
            dataset_name: Name of the dataset to load
            backbone_name: Name of the backbone model to use for embeddings
            root: Root directory to store the raw data
            train: Whether to load training set (True) or test set (False)
            batch_size: Batch size for the DataLoader
            split_ratio: Ratio to split the dataset into training and test sets
        """

        self.data_loader = None
        self.dataset_name = dataset_name
        self.backbone_name = backbone_name
        self.root = root
        self.train = train
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.train_str = "train" if self.train else "test"
        if self.dataset_name == "ImageNet" and not self.train:
            # For ImageNet, there is no test set. So we have to use the
            #  validation set here.
            self.train_str = "val"
        self.raw_data_path = os.path.join(
            self.root, "raw", self.dataset_name, self.train_str
        )

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
        transform = self.get_image_transformation()

        if self.dataset_name in torch_datasets:
            if not self.data_downloaded:
                print(
                    f"\tDownloading {self.dataset_name} ({self.train_str}) "
                    "raw dataset..."
                )
            else:
                print(
                    f"\t{self.dataset_name} ({self.train_str}) raw dataset "
                    "already exists. Loading from disk..."
                )

            dataset = torch_datasets[self.dataset_name](
                root=self.raw_data_path,
                train=self.train,
                download=True,
            )

        print("INFO: Setting up DataLoader...")

        def collate_fn(batch):
            return (
                torch.stack([transform(img) for img, _ in batch]),
                torch.tensor([label for _, label in batch]),
            )

        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.train,
            # num_workers=4,  # Number of CPU threads for loading data
            # pin_memory=True,  # Enable pinned memory for faster GPU transfers
            collate_fn=collate_fn,
        )
        print("INFO: Finished setting up DataLoader.")

    def get_image_transformation(self):
        # Initialize Weight Transforms for correct preprocessing
        weights = backbone_weights[self.backbone_name]

        if self.dataset_name in [
            "FashionMNIST",
            "MNIST",
            "ImageNet",
        ]:
            transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=3), weights.transforms()]
            )
        else:
            transform = weights.transforms()

        return transform

    def download_and_extract_zip(self, url, dataset_path, split=False):

        if self.data_downloaded:
            print(
                f"\t{self.dataset_name} ({self.train_str}) raw dataset "
                "already exists. Loading from disk..."
            )

        else:

            if self.dataset_name[:5] == "CLEAR":
                # Fixing the CLEAR dataset structure
                dataset_path = os.path.join(self.root, "raw", self.dataset_name)

            if split:
                dataset_path = os.path.join(self.root, "raw", self.dataset_name)

            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            temp_dir = os.path.join(self.root, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            zip_path = os.path.join(temp_dir, f"{self.dataset_name}.zip")
            if not os.path.exists(zip_path):
                print(
                    f"\tDownloading {self.dataset_name} ({self.train_str}) from {url}..."
                )
                with requests.get(url, stream=True) as r:
                    with open(zip_path, "wb") as f:
                        shutil.copyfileobj(r.raw, f)

            # Extract the zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                print(
                    f"Extracting {self.dataset_name} ({self.train_str}) to {dataset_path}"
                )
                zip_ref.extractall(dataset_path)

            # Remove the temp directory
            shutil.rmtree(temp_dir)

            if split:
                self.split_dataset(dataset_path)
                dataset_path = os.path.join(
                    self.root, "raw", self.dataset_name, self.train_str
                )
            else:
                # After extraction, check and clean images
                self.check_and_clean_images(self.raw_data_path)

        return dataset_path

    def download_from_kaggle(self, dataset_name, dataset_path, split=False):
        if self.data_downloaded:
            print(
                f"	{self.dataset_name} ({self.train_str}) raw dataset "
                "already exists. Loading from disk..."
            )
            return dataset_path

        if split:
            dataset_path = os.path.join(self.root, "raw", self.dataset_name)

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        kaggle.api.authenticate()
        print(f"	Downloading {self.dataset_name} from Kaggle...")
        kaggle.api.dataset_download_files(
            dataset_name, path=dataset_path, unzip=True, quiet=False
        )
        print(f"Downloaded {self.dataset_name} to {dataset_path}")

        if split:
            self.split_dataset(dataset_path)
            dataset_path = os.path.join(
                self.root, "raw", self.dataset_name, self.train_str
            )
        else:
            # After extraction, check and clean images
            self.check_and_clean_images(self.raw_data_path)

        return dataset_path

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

    def split_dataset(
        self,
        dataset_path,
    ):
        class_folders = os.listdir(dataset_path)

        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        for class_folder in class_folders:
            class_folder_path = os.path.join(dataset_path, class_folder)
            if os.path.isdir(class_folder_path):
                images = os.listdir(class_folder_path)
                num_images = len(images)
                shuffled_indices = torch.randperm(num_images).tolist()
                split_idx = int(num_images * self.split_ratio)

                train_class_path = os.path.join(train_path, class_folder)
                test_class_path = os.path.join(test_path, class_folder)
                if not os.path.exists(train_class_path):
                    os.makedirs(train_class_path)
                if not os.path.exists(test_class_path):
                    os.makedirs(test_class_path)

                for i, idx in enumerate(shuffled_indices):
                    image = images[idx]
                    image_path = os.path.join(class_folder_path, image)
                    if os.path.isfile(image_path):
                        if i < split_idx:
                            shutil.move(image_path, train_class_path)
                        else:
                            shutil.move(image_path, test_class_path)

                # recursively remove old class folder with all the remaining content
                shutil.rmtree(class_folder_path)

            else:
                # If the file is not a directory, delete it
                os.remove(class_folder_path)

        # After extraction, check and clean images
        self.check_and_clean_images(train_path)
        self.check_and_clean_images(test_path)

    def check_and_clean_images(self, path):
        """
        Traverses the extracted dataset directories, attempts to open each image,
        and deletes any image files that are corrupt.
        """
        print(f"Checking for corrupt images in {path} ...")
        num_checked = 0
        num_deleted = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                num_checked += 1
                file_path = os.path.join(root, file)
                # Only check image files
                if file.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
                    try:
                        with rasterio.open(file_path) as src:
                            src.read()
                    except (RasterioIOError, ValueError) as e:
                        print(f"Corrupt image found and deleted: {file_path}")
                        os.remove(file_path)
                        num_deleted += 1
        if num_checked == 0:
            print("No images to check and clean at all found?! :O")
        elif num_deleted == 0:
            print("No corrupt images found.")
        else:
            print(f"Deleted {num_deleted} corrupt image(s).")
