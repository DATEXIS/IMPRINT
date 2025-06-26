"""
Continual learning dataset management.

This module provides classes and utilities for managing data in a continual
learning scenario, organizing data into tasks and providing efficient memory
management.

Note that in this paper, we only focus on 1-step CL, that is, only one
task is learned at a time. But this code allows for actual CL as well.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset


class ClassContinualDataset:
    """
    Manages datasets for class-incremental continual learning scenarios.

    This class organizes a dataset into task sequences based on class labels,
    handling data loading, caching, and memory management for efficient
    continual learning experiments.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        task_splits: list[list[int]] = [],
        use_cache: bool = False,
        use_shared_memory: bool = False,
    ):
        """
        Initialize the continual learning dataset manager.

        Args:
            train_dataset: Training dataset (can be a single dataset or a
                           concatenation)
            test_dataset: Test dataset corresponding to the training dataset
            task_splits: List of class indices for each task in the continual
                         learning scenario.
            use_cache: Whether to cache task data in memory
            use_shared_memory: Whether to use shared memory tensors for
                               multiprocessing
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Validate task splits configuration
        assert len(task_splits) == 1, "Must have only 1 task split in current scenario"
        for task_split in task_splits:
            assert len(task_split) > 0, "Each task split must have at least one class"
            assert all(
                isinstance(class_idx, int) and class_idx > -1 for class_idx in task_split
            ), "Class indices must be non-negative integers"

        self.task_splits = task_splits
        self.use_cache = use_cache
        self.use_shared_memory = use_shared_memory

        # Initialize cache storage if enabled
        if use_cache:
            self.shared_train_data = {}
            self.shared_train_labels = {}
            self.shared_test_data = {}
            self.shared_test_labels = {}
            self.tasks_cached = []

    def get_task(self, task_idx):
        """
        Get datasets for the specified task.

        Args:
            task_idx: Index of the task to retrieve.

        Returns:
            If use_cache is True:
                tuple: (train_data, train_labels, test_data, test_labels) as tensors
            Otherwise:
                tuple: (train_dataset, test_dataset) as PyTorch Datasets

        Raises:
            ValueError: If task_idx is invalid
        """
        # Validate task index
        if not isinstance(task_idx, int) or task_idx < 0 or task_idx >= len(self.task_splits):
            raise ValueError(
                f"Invalid task index {task_idx} "
                f"(must be an integer between 0 and {len(self.task_splits)-1})"
            )

        # Return cached data if available
        if self.use_cache and task_idx in self.tasks_cached:
            return (
                self.shared_train_data[task_idx],
                self.shared_train_labels[task_idx],
                self.shared_test_data[task_idx],
                self.shared_test_labels[task_idx],
            )

        # Get the class labels for this task
        current_classes = self.task_splits[task_idx]

        # Handle both single and concatenated datasets
        if isinstance(self.train_dataset, ConcatDataset):
            # For concatenated datasets (e.g., MNIST&MNIST-M&USPS&SVHN)
            task_train_datasets = []
            task_test_datasets = []
            for dataset in self.train_dataset.datasets:
                task_train_datasets.append(dataset.get_subset_by_labels(current_classes))
            for dataset in self.test_dataset.datasets:
                task_test_datasets.append(dataset.get_subset_by_labels(current_classes))
            task_train_dataset = ConcatDataset(task_train_datasets)
            task_test_dataset = ConcatDataset(task_test_datasets)

        else:
            # For a single dataset
            task_train_dataset = self.train_dataset.get_subset_by_labels(current_classes)
            task_test_dataset = self.test_dataset.get_subset_by_labels(current_classes)

        # Handle caching if enabled
        if self.use_cache:
            # Prepare cache storage
            memory_type = "shared memory" if self.use_shared_memory else "memory"
            print(
                f"\t[INFO] Loading data for task {task_idx} "
                f"({self.task_splits[task_idx]}) into {memory_type}."
            )

            # Extract data and labels
            train_data = torch.vstack([data for data, _ in task_train_dataset])
            train_labels = torch.hstack([label for _, label in task_train_dataset])
            test_data = torch.vstack([data for data, _ in task_test_dataset])
            test_labels = torch.hstack([label for _, label in task_test_dataset])

            # Use shared memory if requested
            if self.use_shared_memory:
                self.shared_train_data[task_idx] = train_data.share_memory_()
                self.shared_train_labels[task_idx] = train_labels.share_memory_()
                self.shared_test_data[task_idx] = test_data.share_memory_()
                self.shared_test_labels[task_idx] = test_labels.share_memory_()
            else:
                self.shared_train_data[task_idx] = train_data
                self.shared_train_labels[task_idx] = train_labels
                self.shared_test_data[task_idx] = test_data
                self.shared_test_labels[task_idx] = test_labels

            # Mark task as cached
            self.tasks_cached.append(task_idx)

            return (
                self.shared_train_data[task_idx],
                self.shared_train_labels[task_idx],
                self.shared_test_data[task_idx],
                self.shared_test_labels[task_idx],
            )
        else:
            # Return datasets without caching
            return task_train_dataset, task_test_dataset

    def num_tasks(self):
        """
        Get the number of tasks in this continual learning scenario.

        Returns:
            int: Total number of tasks
        """
        return len(self.task_splits)

    def to(self, device):
        """
        Move cached data to the specified device.

        Args:
            device: Target device (e.g., 'cpu', 'cuda', 'mps')

        Raises:
            AssertionError: If no data is cached
        """
        # Verify cache is available
        assert (
            self.use_cache and len(self.tasks_cached) > 0
        ), "No data cached to move. Enable caching and load tasks first."

        # Move each cached task's data to the specified device
        for task_idx in self.tasks_cached:
            self.shared_train_data[task_idx] = self.shared_train_data[task_idx].to(device)
            self.shared_train_labels[task_idx] = self.shared_train_labels[task_idx].to(device)
            self.shared_test_data[task_idx] = self.shared_test_data[task_idx].to(device)
            self.shared_test_labels[task_idx] = self.shared_test_labels[task_idx].to(device)
