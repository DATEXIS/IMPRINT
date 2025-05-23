"""
Neural Collapse metrics implementation.

This module provides the tools to calculate Neural Collapse (NC) metrics
for embedding spaces. Neural Collapse refers to the phenomenon where deep
neural networks exhibit a collapse of the feature space during training,
especially in the later stages.

The metrics implemented here include:
- NC1: Within-class variability relative to between-class variability
"""

import numpy as np
import torch


class NeuralCollapse:
    """
    Calculate Neural Collapse metrics for embedding spaces.

    Neural Collapse refers to the phenomenon where deep neural networks exhibit
    specific structural properties in their feature space during training,
    especially in the later stages of training.

    Attributes:
        embeddings: Tensor of shape (n_samples, embedding_dim) containing
                    feature embeddings
        labels: Tensor of shape (n_samples,) containing class labels
        unique_labels: Array of unique class labels
        data_mean: Global mean of all embeddings
        centered_class_mean: Class means centered by the global mean
        centered_class_mean_norm: Normalized centered class means
    """

    def __init__(self, embeddings, labels):
        """
        Initialize the Neural Collapse calculator with embeddings and their
        labels.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim) containing
                        feature embeddings
            labels: Tensor of shape (n_samples,) containing class labels
        """
        self.embeddings = embeddings
        self.labels = labels
        self.unique_labels = np.unique(self.labels)
        self.data_mean = self.embeddings.mean(dim=0)

        # Initialize tensors for class means
        n_classes = len(self.unique_labels)
        embedding_dim = self.embeddings.size(1)
        self.centered_class_mean = torch.zeros((n_classes, embedding_dim))
        self.centered_class_mean_norm = torch.zeros((n_classes, embedding_dim))

        # Calculate centered class means and their normalized versions
        for idx, label in enumerate(self.unique_labels):
            class_samples = self.embeddings[self.labels == label]

            # Skip empty classes if any
            if len(class_samples) == 0:
                continue

            # Calculate class mean
            class_mean = class_samples.mean(dim=0)

            # Center the class mean by the global mean
            self.centered_class_mean[idx, :] = class_mean - self.data_mean

            # Calculate normalized centered class mean
            norm = torch.norm(self.centered_class_mean[idx, :], p=2, keepdim=True)
            eps = torch.finfo(
                self.centered_class_mean.dtype
            ).eps  # For numerical stability
            self.centered_class_mean_norm[idx, :] = self.centered_class_mean[
                idx, :
            ] / (norm + eps)

    def covariance(self):
        """
        Calculate intra-class and inter-class covariance matrices.

        Intra-class covariance represents the variability within each class,
        while inter-class covariance represents the variability between class means.

        Returns:
            tuple: (intra_class_covariance, inter_class_covariance)
                - intra_class_covariance: Tensor of shape (n_classes, embedding_dim, embedding_dim)
                - inter_class_covariance: Tensor of shape (embedding_dim, embedding_dim)
        """
        n_classes = len(self.unique_labels)
        embedding_dim = self.embeddings.size(1)

        # Initialize tensor for intra-class covariance matrices (one per class)
        intra_class_covariance = torch.zeros((n_classes, embedding_dim, embedding_dim))

        # Calculate intra-class covariance for each class
        for idx, label in enumerate(self.unique_labels):
            class_samples = self.embeddings[self.labels == label]

            # Skip empty classes if any
            if len(class_samples) == 0:
                continue

            # Center the samples by their class mean
            centered_class_data = class_samples - self.centered_class_mean[idx, :]

            # Calculate covariance matrix for this class
            intra_class_covariance[idx] = (
                centered_class_data.T @ centered_class_data
            ) / max(
                1, len(centered_class_data)
            )  # Avoid division by zero

        # Calculate inter-class covariance
        inter_class_covariance = (
            self.centered_class_mean.T @ self.centered_class_mean
        ) / max(
            1, n_classes
        )  # Avoid division by zero

        return intra_class_covariance, inter_class_covariance

    def nc_1(self, intra_m=None, inter_m=None):
        """
        Calculate Neural Collapse metric NC1.

        NC1 measures the ratio of within-class variability to between-class
        variability, calculated as:
        NC1 = Tr(Σ_W × Σ_B⁻¹)/C

        where Σ_W is the average within-class covariance, Σ_B is the
        between-class covariance, and C is the number of classes.

        Lower values indicate stronger Neural Collapse (more separation
        between classes relative to within-class variance).

        Args:
            intra_m: Pre-computed intra-class covariance matrix. If None, it
                     will be calculated.
            inter_m: Pre-computed inter-class covariance matrix. If None, it
                     will be calculated.

        Returns:
            float: The NC1 metric value
        """
        # Calculate covariance matrices if not provided
        if intra_m is None or inter_m is None:
            intra_m, inter_m = self.covariance()

        # Handle case of empty classes or singular matrices
        if torch.all(torch.isclose(inter_m, torch.zeros_like(inter_m))):
            return float("inf")

        # Calculate average intra-class covariance
        avg_intra_m = intra_m.mean(dim=0)

        # Use pseudoinverse for numerical stability
        pseudo_inv_inter = torch.linalg.pinv(inter_m)

        # Calculate NC1 metric
        nc_1_matrix = pseudo_inv_inter @ avg_intra_m
        nc_1_value = torch.trace(nc_1_matrix).item() / len(self.unique_labels)

        return nc_1_value
