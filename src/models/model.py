"""
Weight imprinting model implementation.

This module provides the core model implementation for weight imprinting,
including normalization strategies, aggregation methods, and evaluation routines.
The ImprintedModel class supports different weight imprinting strategies and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

from src.utils.helpers import quantile_norm, safe_quantile_with_retry


class ImprintedModel(nn.Module):
    """
    Neural network model with weight imprinting capabilities.

    This model allows for weight imprinting, where class weights are directly set
    from feature embeddings rather than learned through backpropagation. It supports
    various normalization strategies and aggregation methods for combining feature
    embeddings into class representations for final classifications.
    """

    def __init__(
        self,
        normalize_input_data: str = "l2",
        normalize_layer_activations: str = "topk",
        normalize_weights: str = "l2",
        aggregation_method: str = "mean",
        m: int = 5,  # for m-nearest neighbors
        embedding_size: int = 512,
    ):
        """
        Initialize the imprinted model.

        Args:
            normalize_input_data: Normalization method for input data ("none", "l2")
            normalize_layer_activations: Normalization method for layer activations
                                         ("none", "l2", "topk")
            normalize_weights: Normalization method for weights
                              ("none", "l2", "quantile")
            aggregation_method: Method for aggregating activations
                               ("max", "mnn")
            m: Number of nearest neighbors for mNN aggregation
            embedding_size: Size of the feature embeddings
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = 0  # model starts off empty with 0 classes
        self.normalize_input_data = normalize_input_data
        self.normalize_layer_activations = normalize_layer_activations
        self.normalize_weights = normalize_weights
        self.aggregation_method = aggregation_method
        self.m = m

        self.w1s = nn.ParameterList()

    @torch.no_grad()
    def extend_num_classes(self, num_class_extension: int):
        """
        Extend the model by adding new classes.

        Args:
            num_class_extension: Number of new classes to add
        """
        self.num_classes += num_class_extension
        # Initialize empty weight parameters for new classes
        self.w1s.extend(
            num_class_extension
            * [
                torch.zeros(
                    (0, self.embedding_size), dtype=torch.float32, requires_grad=False
                )
            ]
        )

    @torch.no_grad()
    def extend_ws(self, data, class_index):
        """
        Extend weight matrices with new data (imprinting process).

        Args:
            data: Feature embeddings to imprint
            class_index: Target class index for imprinting
        """
        data = self.normalize(data, "weights", ref_dist=[*self.w1s])
        self.w1s[class_index] = torch.vstack((self.w1s[class_index], data))

    def forward(self, data):
        """
        Perform forward pass through the model.

        Args:
            data: Input feature embeddings

        Returns:
            torch.Tensor: Output class scores
        """
        data = self.normalize(data, "input_data")

        # Apply layer
        w1 = torch.vstack([*self.w1s])
        x = w1 @ data.T

        x = self.normalize(x, "layer_activations")

        # Aggregate activations using the specified method
        y = self.aggregate_activations(x, data)

        return y

    def normalize(self, data, origin="input_data", ref_dist=None):
        """
        Normalize data using the specified method.

        Args:
            data: Data to normalize
            origin: Data origin ("input_data", "layer_activations", "weights")
            ref_dist: Reference distribution for quantile normalization

        Returns:
            torch.Tensor: Normalized data

        Raises:
            ValueError: If unknown normalization method is specified
        """
        assert origin in ["input_data", "layer_activations", "weights"]

        if (normalization_type := getattr(self, f"normalize_{origin}")) == "none":
            return data

        if origin == "input_data":
            if normalization_type == "l2":
                data = F.normalize(data, p=2, dim=1)
            else:
                raise ValueError(
                    f"Unknown normalization type: {normalization_type} for normalizing input data."
                )
        elif origin == "layer_activations":
            if normalization_type == "l2":
                data = F.normalize(data, p=2, dim=0)
            elif normalization_type == "topk":
                threshold = safe_quantile_with_retry(data, 0.95, dim=0)
                data[data < threshold] = 0
            else:
                raise ValueError(
                    f"Unknown normalization type: {normalization_type} for normalizing layer activations."
                )
        elif origin == "weights":
            if normalization_type == "l2":
                data = F.normalize(data, p=2, dim=1)
            elif normalization_type == "quantile":
                assert ref_dist is not None, "No reference distribution provided."
                ref_dist = torch.vstack(ref_dist)
                data = quantile_norm(
                    x=data,
                    ref_dist=ref_dist,
                )
            else:
                raise ValueError(
                    f"Unknown normalization type: {normalization_type} for normalizing weights."
                )

        return data

    def aggregate_activations(self, activations, data):
        """
        Aggregate activations using the specified method.
        Note that, if the aggregation method is "mnn", only the provided
        data is used; the layer activations are not used. This also
        means the layer activations are irrelevant to the mnn aggregation.

        Args:
            activations: Layer activations to aggregate
            data: Original input data (used for mNN aggregation)

        Returns:
            torch.Tensor: Aggregated class scores

        Raises:
            ValueError: If unknown aggregation method is specified
        """
        y = torch.zeros((self.num_classes, activations.shape[-1]), device=data.device)

        if self.aggregation_method == "mnn":
            y = self.mnn_aggregation(data)
        else:
            lens = [len(w1) for w1 in self.w1s]
            start = 0

            for class_idx in range(self.num_classes):
                end = start + lens[class_idx]
                class_activations = activations[start:end]

                if self.aggregation_method == "max":
                    y[class_idx, :] = class_activations.max(dim=0).values
                else:
                    raise ValueError(
                        f"Unknown aggregation method: {self.aggregation_method}"
                    )

                start = end

        return y

    def mnn_aggregation(self, data):
        """
        Aggregate activations using m-Nearest Neighbors.

        This method uses the mNN algorithm to classify input data directly based on
        the imprinted weight vectors, bypassing the intermediate layer activations.

        Args:
            data: Input feature embeddings

        Returns:
            torch.Tensor: Classification scores based on mNN voting
        """
        # Convert to numpy
        w1 = torch.vstack([*self.w1s]).cpu()
        all_class_weights_np = w1.numpy()
        data_np = data.cpu().numpy()

        # Ensure k is not greater than the number of class weights
        m = min(all_class_weights_np.shape[0], self.m)
        if m != self.m:
            print(
                f"Warning: m was set to {self.m}, but only {m} "
                f"class weights are available. Using k={m} for mnn "
                "aggregation instead."
            )

        # Fit mNN on the class weights
        mnn = NearestNeighbors(n_neighbors=m, algorithm="auto").fit(
            all_class_weights_np
        )
        distances, indices = mnn.kneighbors(data_np)

        # Get the number of proxies per class
        num_proxies_per_class = [
            len(self.w1s[class_idx]) for class_idx in range(self.num_classes)
        ]
        cumulative_num_proxies = [0] + list(
            torch.cumsum(torch.tensor(num_proxies_per_class), dim=0).numpy()
        )

        # Map indices to class labels based on the proxy ranges
        mnn_labels = torch.tensor(
            [
                [
                    next(
                        class_idx
                        for class_idx in range(self.num_classes)
                        if cumulative_num_proxies[class_idx]
                        <= idx
                        < cumulative_num_proxies[class_idx + 1]
                    )
                    for idx in sample_indices
                ]
                for sample_indices in indices
            ],
            dtype=torch.int64,
        )

        # Weighted majority voting using inverse distances
        distances = torch.tensor(distances)
        predicted_labels = torch.tensor(
            [
                mnn_labels[_i]
                .bincount(weights=1 / (distances[_i] + 1e-10))
                .argmax()
                .item()
                for _i in range(len(data_np))
            ]
        )

        # Create one-hot encoding for predictions
        y = torch.zeros((self.num_classes, data.shape[0]), device=data.device)
        y[predicted_labels, torch.arange(data.shape[0])] = 1

        return y

    @torch.no_grad()
    def accuracy(self, data, labels):
        """
        Calculate accuracy of the model on given data.

        Args:
            data: Input features
            labels: Ground truth labels

        Returns:
            float: Accuracy percentage
        """
        # (TODO): It might make sense to do the following to utilize GPU;
        #  depending on the dataset size
        # f_accuracy = torchmetrics.Accuracy(
        #     task="multiclass",
        #     num_classes=num_classes,
        #     top_k=1,
        # ).to(device)
        pred = torch.argmax(self.forward(data), dim=0).to(data.device)
        correct = sum(pred == labels)
        return float(100 * correct / len(data))

    @torch.no_grad()
    def f1_scores(self, data, labels):
        """
        Calculate F1 scores for each class (one-vs-rest).

        Args:
            data: Input features
            labels: Ground truth labels

        Returns:
            dict: Dictionary with F1 scores and class sizes
        """
        pred = torch.argmax(self.forward(data), dim=0).to(data.device)
        f1_scores = {
            "remapped_class_index": [],
            "one-vs-rest-f1": [],
            "class_size": [],
        }

        for remppd_cl_idx in range(self.num_classes):
            # Treat current class as positive and others as negative
            tp = ((labels == remppd_cl_idx) & (pred == remppd_cl_idx)).sum().item()
            fp = ((labels != remppd_cl_idx) & (pred == remppd_cl_idx)).sum().item()
            fn = ((labels == remppd_cl_idx) & (pred != remppd_cl_idx)).sum().item()
            class_size = (labels == remppd_cl_idx).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores["remapped_class_index"].append(remppd_cl_idx)
            f1_scores["one-vs-rest-f1"].append(f1 * 100)
            f1_scores["class_size"].append(class_size)

        return f1_scores
