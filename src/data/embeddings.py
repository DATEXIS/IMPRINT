"""
Embedding extraction and management module.

This module provides utilities for extracting, saving, and loading
feature embeddings from datasets. It interfaces with the backbone models
to extract features and stores them efficiently in HDF5 format.
"""

import os
import time
import pandas as pd

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.data.loader import DatasetHandler
from src.data.datasets import EmbeddingDataset
from src.models.backbone import BackboneHandler


def get_embeddings_path(
    root: str = "./data",
    dataset_name: str = "CIFAR10",
    backbone_name: str = "resnet18",
    train=True,
):
    """
    Generate the path for storing or retrieving embeddings.

    Args:
        root: Root directory for embeddings storage
        dataset_name: Name of the dataset
        backbone_name: Name of the backbone model
        train: Whether to use training set (True) or test/validation set (False)

    Returns:
        tuple: (directory_path, filename) for the embeddings
    """
    train_str = "train" if train else "test"
    if dataset_name == "ImageNet" and not train:
        train_str = "val"
    embeddings_path = os.path.join(
        root, "embeddings", dataset_name, backbone_name, train_str
    )
    embeddings_filename = f"embeddings_{dataset_name}_{backbone_name}_{train_str}.h5"
    return embeddings_path, embeddings_filename


def get_embeddings(
    dataset_name,
    backbone_name,
    offset,
    root,
    splits=["train", "test"],
    label_mapping={},
):
    """
    Load embeddings from disk for specified dataset and backbone.

    Args:
        dataset_name: Name of the dataset
        backbone_name: Name of the backbone model
        offset: Label offset for multi-dataset scenarios
        root: Root directory for embeddings storage
        splits: Which splits to load ('train', 'test', 'val')
        label_mapping: Optional mapping to transform labels

    Returns:
        tuple: Combination of train/test embedding datasets and embedding size
    """
    assert "train" in splits or "test" in splits or "val" in splits
    embedding_size = None
    embeddings_train = None
    embeddings_test = None

    if "train" in splits:
        embeddings_path_train, embeddings_filename_train = get_embeddings_path(
            root, dataset_name, backbone_name, True
        )
        embeddings_train = EmbeddingDataset(
            os.path.join(embeddings_path_train, embeddings_filename_train),
            offset=offset,
            label_mapping=label_mapping,
        )
        embedding_size = embeddings_train[:][0].size(1)

    if "test" in splits or "val" in splits:
        embeddings_path_test, embeddings_filename_test = get_embeddings_path(
            root, dataset_name, backbone_name, False
        )
        embeddings_test = EmbeddingDataset(
            os.path.join(embeddings_path_test, embeddings_filename_test),
            offset=offset,
            label_mapping=label_mapping,
        )
        embedding_size = embeddings_test[:][0].size(1)

    if "train" in splits and ("test" in splits or "val" in splits):
        return embeddings_train, embeddings_test, embedding_size
    elif "train" in splits:
        return embeddings_train, embedding_size
    elif "test" in splits or "val" in splits:
        return embeddings_test, embedding_size


class EmbeddingExtractor:
    """
    Extract and manage backbone embeddings for datasets.

    This class handles the extraction of feature embeddings from datasets using
    pretrained backbones. It manages caching, loading, and saving of embeddings
    to disk in an efficient HDF5 format.
    """

    def __init__(
        self,
        device_name="cuda",
        backbone_name="resnet18",
        dataset_name="CIFAR10",
        class_focus=None,
        batch_size=64,
        raw_data_root="./data",
        embedding_root="./data",
        train=True,
    ):
        """
        Initialize an embedding extractor.

        Args:
            device_name: Computing device ("cuda", "cpu", or "mps")
            backbone_name: Name of the backbone model to use for embedding
            dataset_name: Name of the dataset to embed
            class_focus: List of class indices to focus on (None for all classes)
            batch_size: Batch size for calculating embeddings
            raw_data_root: Root directory for raw data
            embedding_root: Root directory for storing embeddings
            train: Whether to embed training (True) or test set (False)
        """
        self.device_name = device_name
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_name == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.backbone_name = backbone_name
        self.dataset_name = dataset_name
        self.class_focus = class_focus
        self.batch_size = batch_size
        self.raw_data_root = raw_data_root
        self.embedding_root = embedding_root
        self.train = train
        self.train_str = "train" if train else "test"
        if self.dataset_name == "ImageNet" and not self.train:
            # For ImageNet, there is no test set. So we have to use the
            # validation set here.
            self.train_str = "val"

        # Setup embedding file paths
        self.embeddings_path, embeddings_filename = get_embeddings_path(
            self.embedding_root, self.dataset_name, self.backbone_name, self.train
        )
        self.embeddings_file = os.path.join(self.embeddings_path, embeddings_filename)
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)

        # Check if embeddings already exist
        self.embeddings_exist = os.path.exists(self.embeddings_path) and (
            len(os.listdir(self.embeddings_path)) > 0
        )

        if self.embeddings_exist:
            # Validate that all required class embeddings exist
            existing_embeddings = get_embeddings(
                self.dataset_name,
                self.backbone_name,
                0,
                self.embedding_root,
                [self.train_str],
            )[0]
            existing_class_indices = existing_embeddings.get_all_class_indices().keys()

            if self.class_focus is not None:
                # Check if classes in class_focus are in the .h5 file, that
                #  contains already saved embeddings. If not, have to redo it
                #  (especially, need to delete the existing .h5 file)
                if not set(self.class_focus).issubset(existing_class_indices):
                    print(
                        f"\tEmbeddings for some classes in {self.dataset_name} "
                        f"({self.train_str}) X {self.backbone_name.upper()} "
                        "are missing. Need to re-extract."
                    )
                    self.embeddings_exist = False
                else:
                    print(
                        f"\tEmbeddings for {len(existing_class_indices)} "
                        f"classes in "
                        f"{self.dataset_name} "
                        f"({self.train_str}) X {self.backbone_name.upper()} "
                        f"already exist; especially those in {self.class_focus}."
                    )

        if not self.embeddings_exist:
            # Load backbone to calculate the embeddings
            self.backbone = BackboneHandler(
                self.backbone_name, device_name=self.device_name
            ).backbone

        # # Get raw data (download if necessary) by initializing the dataset handler
        self.dataset_handler = DatasetHandler(
            self.dataset_name,
            self.backbone_name,
            root=self.raw_data_root,
            train=self.train,
            batch_size=self.batch_size,
        )

    def extract_embeddings(self):
        """
        Extract embeddings from the dataset using the backbone model.

        Extracts features for all samples or only those in class_focus if specified.
        Saves embeddings incrementally to disk to manage memory efficiently.
        """
        embeddings = []
        labels = []
        total_size_in_bytes = 0
        size_in_bytes = 0
        save_threshold = 1 * 1024**3  # 1 GB

        desc = (
            f"Extracting {self.backbone_name.upper()} x "
            f"{self.dataset_name} ({self.train_str}) embeddings for "
        )
        desc += f"{len(self.class_focus)}" if self.class_focus else "all"
        desc += " classes"

        first_save = True  # Flag to initialize HDF5 file

        with torch.no_grad():
            for inputs, target in tqdm(self.dataset_handler.data_loader, desc=desc):
                if self.class_focus is not None:
                    # Filter samples based on class_focus
                    mask = torch.tensor([t.item() in self.class_focus for t in target])
                    if not mask.any():
                        continue  # Skip batch if no relevant samples

                    inputs = inputs[mask].to(self.device)  # Filter inputs
                    target = target[mask]  # Filter targets
                else:
                    inputs = inputs.to(self.device)

                # Compute embeddings based on backbone type
                outputs = self.backbone(inputs)
                outputs = outputs.view(outputs.size(0), -1)  # Flatten the outputs

                # Convert to numpy and accumulate
                filtered_embeddings = outputs.cpu().numpy()
                filtered_labels = target.cpu().numpy()

                # Accumulate filtered embeddings and labels
                embeddings.append(filtered_embeddings)
                labels.append(filtered_labels)
                size_in_bytes += filtered_embeddings.nbytes + filtered_labels.nbytes

                # Save to disk if accumulated size exceeds threshold
                if size_in_bytes >= save_threshold:
                    print("INFO: Threshold of 1GB reached. Saving embeddings.")
                    self.save_embeddings(
                        np.concatenate(embeddings, axis=0),
                        np.concatenate(labels, axis=0),
                        init=first_save,
                    )
                    embeddings.clear()
                    labels.clear()
                    total_size_in_bytes += size_in_bytes
                    size_in_bytes = 0
                    first_save = False

        # Save any remaining embeddings after the loop
        if embeddings:
            self.save_embeddings(
                np.concatenate(embeddings, axis=0),
                np.concatenate(labels, axis=0),
                init=first_save,
            )

        print(
            f"\t{total_size_in_bytes / 10**9}GB of embeddings saved to "
            f"{self.embeddings_file}"
        )

    def save_embeddings(self, embeddings, labels, init=False):
        """
        Save embeddings and labels incrementally to HDF5 format.

        Args:
            embeddings: Numpy array of embedding vectors
            labels: Numpy array of corresponding labels
            init: Whether this is the initial save (True) or an append (False)
        """
        mode = "w" if init else "a"
        with h5py.File(self.embeddings_file, mode) as f:
            if init:
                f.create_dataset(
                    "embeddings",
                    data=embeddings,
                    maxshape=(None, embeddings.shape[1]),
                    dtype=np.float32,
                )
                f.create_dataset(
                    "labels", data=labels, maxshape=(None,), dtype=np.int32
                )
            else:
                f["embeddings"].resize(
                    (f["embeddings"].shape[0] + embeddings.shape[0]), axis=0
                )
                f["embeddings"][-embeddings.shape[0] :] = embeddings
                f["labels"].resize((f["labels"].shape[0] + labels.shape[0]), axis=0)
                f["labels"][-labels.shape[0] :] = labels

    def load_embeddings(self):
        """
        Load embeddings from disk or extract them if they don't exist.

        Returns:
            EmbeddingDataset: Dataset containing the embeddings
        """
        # Check if embeddings already exist
        if not self.embeddings_exist:
            self.extract_embeddings()

        return EmbeddingDataset(self.embeddings_file)

    def run(self):
        """
        Run the embedding extraction process and return statistics.

        Returns:
            tuple: Statistics about the extraction process including dataset info,
                  embedding dimensions, and processing time
        """
        start_time = time.time()

        number_of_samples = len(self.dataset_handler.data_loader.dataset)

        # Find raw image resolution
        x = self.dataset_handler.data_loader.dataset[0][0].size[0]
        y = self.dataset_handler.data_loader.dataset[0][0].size[1]
        mode = self.dataset_handler.data_loader.dataset[0][0].mode
        # About modes: https://stackoverflow.com/questions/52307290/what-is-the-difference-between-images-in-p-and-l-mode-in-pil
        # TL;DR: grayscale vs. RGB
        resolution = f"{x}x{y} ({mode})"

        # Calculate the embeddings
        print("INFO: Starting to calculate the embeddings.")
        embeddings_dataloader = self.load_embeddings()

        # Get information about the embeddings
        embedding_dimensions = len(embeddings_dataloader[0][0])
        num_embeddings = len(embeddings_dataloader)

        # Determine data sources
        embeddings_location_str = "From Disk" if self.embeddings_exist else "Extracted"
        data_location_str = (
            "From Disk" if self.dataset_handler.data_downloaded else "Downloaded"
        )

        time_taken = time.time() - start_time

        return (
            self.dataset_name,
            self.train_str,
            number_of_samples,
            resolution,
            data_location_str,
            self.backbone_name,
            embeddings_location_str,
            num_embeddings,
            embedding_dimensions,
            time_taken,
        )
