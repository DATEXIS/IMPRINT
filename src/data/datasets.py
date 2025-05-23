"""
Dataset classes for the weight imprinting framework.

This module provides custom PyTorch Dataset implementations for handling embeddings
stored in HDF5 format.
"""

import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset


class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Dataset for loading and accessing feature embeddings stored in HDF5 format.

    This dataset handles loading feature embeddings and labels from an HDF5 file,
    with options for label offset and mapping to support multi-dataset and
    incremental learning scenarios.
    """

    def __init__(
        self,
        hdf5_path,
        offset: int = 0,
        label_mapping: dict = {},
    ):
        """
        Initialize an embedding dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing embeddings and labels
            offset: Offset added to the labels (for concatenated datasets)
            label_mapping: Dictionary to map original labels to new labels
                           Example: {0: 0, 1: 0, 2: 1, 3: 1} combines classes 0-1
                           and 2-3 into single classes. Any label not listed
                           in here will be automatically set to -1.

        Raises:
            FileNotFoundError: If the embeddings file doesn't exist

        NOTE: If both offset and label_mapping are used, e.g., as in
        ```
            "dataset_name": ["CIFAR10", "FashionMNIST"]
            "label_mapping": {
                0: 0,
                1: 1,
                10: 0,
                11: 1,
            }
            "task_splits": [[0,1]]
        ```

        the offset is always applied first. I.e., when the label_mapping is
        provided, the offset has to be considered in there (see above).
        Just like it has to be when task_splits are defined, e.g., as in
        ```
            "dataset_name": ["CIFAR10", "FashionMNIST"]
            "label_mapping": {}
            "task_splits": [[0,1,10,11]]
        ```
        """
        self.hdf5_path = hdf5_path
        self.offset = offset
        self.label_mapping = label_mapping if label_mapping else None

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"Embeddings file not found at {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            self.num_samples = f["embeddings"].shape[0]
            self.targets = f["labels"][()]
            self.number_of_classes_without_mapping = len(np.unique(self.targets))

            # Apply label mapping if provided
            if self.label_mapping:
                self.targets = np.vectorize(lambda x: self.label_mapping.get(x, -1))(
                    self.targets
                )
            self.classes = np.unique(self.targets).tolist()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get an embedding and its label by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple: (embedding, label)
        """
        with h5py.File(self.hdf5_path, "r") as f:
            embedding = torch.tensor(f["embeddings"][idx], dtype=torch.float32)
            label = torch.tensor(f["labels"][idx], dtype=torch.int32) + self.offset
            if self.label_mapping:
                label = np.vectorize(lambda x: self.label_mapping.get(x, -1))(label)
                label = torch.tensor(label, dtype=torch.int32)
        return embedding, label

    def get_indices_by_class(self, class_label):
        """
        Get indices of all samples belonging to a specific class.

        Args:
            class_label: The class label to filter by

        Returns:
            list: Indices of samples with the specified class label
        """
        with h5py.File(self.hdf5_path, "r") as f:
            labels = f["labels"][()] + self.offset
            if self.label_mapping:
                labels = np.vectorize(lambda x: self.label_mapping.get(x, -1))(labels)
            indices = np.where(labels == class_label)[0].tolist()
        return indices

    def get_all_class_indices(self):
        """
        Get indices for all classes in the dataset.

        Returns:
            dict: Mapping from class labels to lists of sample indices
        """
        class_indices = {}
        with h5py.File(self.hdf5_path, "r") as f:
            labels = f["labels"][()] + self.offset
            if self.label_mapping:
                labels = np.vectorize(lambda x: self.label_mapping.get(x, -1))(labels)
            for class_label in self.classes:
                class_indices[class_label] = np.where(labels == class_label)[
                    0
                ].tolist()
        return class_indices

    def get_subset_by_labels(self, selected_labels):
        """
        Create a subset of the dataset containing only samples with specific labels.

        Args:
            selected_labels: List of class labels to include

        Returns:
            SubsetEmbeddingDataset: A subset of this dataset
        """
        subset_indices = []
        with h5py.File(self.hdf5_path, "r") as f:
            labels = f["labels"][()] + self.offset
            if self.label_mapping:
                labels = np.vectorize(lambda x: self.label_mapping.get(x, -1))(labels)
            for label in selected_labels:
                subset_indices.extend(np.where(labels == label)[0].tolist())

        return SubsetEmbeddingDataset(
            self.hdf5_path, subset_indices, self.label_mapping
        )


class SubsetEmbeddingDataset(torch.utils.data.Dataset):
    """
    A subset of an EmbeddingDataset containing only selected indices.

    This dataset provides access to a subset of embeddings from an HDF5 file
    without loading the entire dataset into memory.
    """

    def __init__(
        self,
        hdf5_path,
        indices,
        label_mapping: dict = None,
    ):
        """
        Initialize a subset of an embedding dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing embeddings and labels
            indices: List of indices to include in this subset
            label_mapping: Optional dictionary to map labels
        """
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.label_mapping = label_mapping

    def __len__(self):
        """Return the number of samples in the subset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get an embedding and its label by subset index.

        Args:
            idx: Index within the subset

        Returns:
            tuple: (embedding, label)
        """
        real_idx = self.indices[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            embedding = torch.tensor(f["embeddings"][real_idx], dtype=torch.float32)
            label = torch.tensor(f["labels"][real_idx], dtype=torch.int32)
            if self.label_mapping:
                label = np.vectorize(self.label_mapping.get)(label)
                label = torch.tensor(label, dtype=torch.int32)
        return embedding, label


class Subset(torch.utils.data.Subset):
    """
    A subset of a dataset that preserves the classes attribute.

    This extension of torch.utils.data.Subset ensures that the classes
    attribute from the original dataset is preserved.
    """

    def __init__(self, dataset, indices):
        """
        Initialize a subset that preserves the classes attribute.

        Args:
            dataset: The original dataset
            indices: List of indices to include in this subset
        """
        super().__init__(dataset, indices)
        self.classes = dataset.classes
