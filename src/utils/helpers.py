"""
Helper utilities for the weight imprinting framework.

This module provides utility functions for tasks such as:
- Setting random seeds for reproducibility
- Quantile normalization of distributions
- Memory management for large tensor operations
- Metrics calculation for evaluation
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility.

    This function sets the seeds for Python's random module, NumPy, PyTorch,
    PyTorch CUDA, and Python's hash seed to ensure reproducible results.
    It also sets deterministic mode for cudnn which may affect performance
    but ensures reproducibility across hardware platforms.

    Args:
        seed: Integer seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Additional settings for reproducibility across hardware
    # These may impact performance but are necessary for consistent results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quantile_norm(x, ref_dist, regularity="discrete"):
    """
    Normalize vectors using quantile normalization.

    This function normalizes the values in tensor x to match the distribution
    of values in the reference distribution ref_dist.

    Args:
        x: Tensor of vectors to be normalized
        ref_dist: Reference distribution to normalize to
        regularity: Type of regularity, either "discrete" or "continuous"

    Returns:
        Normalized tensor with the same shape as x

    Raises:
        ValueError: If an unknown regularity type is provided
    """
    if ref_dist.numel() == 0:
        # No reference distribution to normalize to
        return x

    if regularity == "discrete":
        N = x.shape[-1]
        y = safe_quantile_with_retry(
            ref_dist.flatten(),
            torch.arange(0, 1, 1 / N, dtype=ref_dist.dtype).to(ref_dist.device)
            + 1 / (2 * N),
        )
    elif regularity == "continuous":
        y = ref_dist.sort(dim=-1).values.mean(axis=0)
    else:
        raise ValueError(f"Unknown regularity {regularity}.")

    sorted_indices = torch.argsort(x)
    for _i in range(x.shape[0]):
        x[_i][sorted_indices[_i]] = y

    return x


def safe_quantile_with_retry(tensor, q, dim=None, min_size=100, max_retries=10):
    """
    Compute quantile with automatic downsampling for memory efficiency.

    This function attempts to compute quantiles, and if it encounters memory
    issues, it progressively downsamples the input tensor until computation succeeds.

    Args:
        tensor: Input tensor
        q: Quantile level(s) to compute (0 ≤ q ≤ 1)
        dim: Dimension along which to compute quantiles (None for entire tensor)
        min_size: Minimum tensor size before stopping downsampling attempts
        max_retries: Maximum number of downsampling attempts

    Returns:
        Tensor of computed quantile values

    Raises:
        RuntimeError: If computation fails after max_retries attempts or
                      if tensor becomes too small to downsample further
    """
    retries = 0
    factor = 2  # Start with a factor of 2 for downsampling

    while retries < max_retries:
        try:
            # Check if tensor can fit into available memory
            required_memory = estimate_memory_requirement(tensor.numel(), tensor.dtype)
            free_memory = get_free_gpu_memory(tensor.device)

            if required_memory > free_memory:
                if tensor.numel() <= min_size:
                    raise RuntimeError(
                        "Tensor size is too small to downsample further."
                    )

                # Downsample the tensor
                print(
                    f"\t[WARN] Downsampling tensor by ::{factor} due to memory constraints."
                )
                tensor = tensor[::factor]
                factor *= 2  # Increase the downsampling factor for the next retry
                retries += 1
                continue

            # If memory is sufficient, attempt to compute quantile
            return torch.quantile(tensor, q, dim=dim)
        except RuntimeError as e:
            # Handle unrelated errors
            if "CUDA out of memory" not in str(
                e
            ) and "input tensor is too large" not in str(e):
                raise
            if tensor.numel() <= min_size:
                raise RuntimeError(
                    "Tensor size is too small to downsample further."
                ) from e

            # Downsample the tensor
            tensor = tensor[::factor]
            factor *= 2  # Increase the downsampling factor for the next retry
            retries += 1

    raise RuntimeError(f"Failed to compute quantile after {max_retries} retries.")


def get_free_gpu_memory(device):
    """
    Get the available GPU memory in bytes.

    Args:
        device: The torch.device to check

    Returns:
        Free memory in bytes, or infinity if CUDA is not available
    """
    if torch.cuda.is_available():
        stats = torch.cuda.mem_get_info(device)
        return stats[0]  # Free memory in bytes
    return float("inf")  # If CUDA is not available, assume infinite memory (CPU)


def estimate_memory_requirement(tensor_size, dtype):
    """
    Estimate the memory requirement for a tensor in bytes.

    Args:
        tensor_size: Number of elements in the tensor
        dtype: Data type of the tensor

    Returns:
        Estimated memory requirement in bytes
    """
    return tensor_size * torch.tensor([], dtype=dtype).element_size()


def calc_weighted_f1_score(f1s):
    """
    Calculate weighted F1 score based on class sizes.

    This function computes the weighted average of F1 scores,
    where weights are proportional to class sizes.

    Args:
        f1s: Dictionary containing class information with keys:
             'remapped_class_index', 'one-vs-rest-f1', and 'class_size'

    Returns:
        Weighted F1 score or NaN if f1s is empty
    """
    # If f1s dict is empty, return nan
    if not f1s:
        return np.nan
    if sum(f1s["class_size"]) > 0:
        return sum(
            f * w for f, w in zip(f1s["one-vs-rest-f1"], f1s["class_size"])
        ) / sum(f1s["class_size"])
    else:
        return 0


def load_config(config_path="src/config/config.yaml"):
    """
    Load configuration from YAML file.

    This function loads the YAML configuration and replaces special references
    like RANDOM_CLASS_REMAPPINGS with the actual mappings.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration dictionary with resolved references
    """
    import yaml

    try:
        with open(config_path, "r") as f:
            # Handle comments in YAML (which are not part of standard YAML)
            content = ""
            for line in f:
                if "#" in line:
                    line = line.split("#")[0]
                content += line

        config = yaml.safe_load(content)

        # Check if we need to process RANDOM_CLASS_REMAPPINGS
        if (
            "label_remappings" in config
            and "RANDOM_CLASS_REMAPPINGS" in config["label_remappings"]
        ):
            from src.experiments.imagenet.prep import RANDOM_CLASS_REMAPPINGS

            # Create a new label_remappings dictionary
            new_label_remappings = {}

            # Copy existing mappings except RANDOM_CLASS_REMAPPINGS placeholder
            for key, value in config["label_remappings"].items():
                if key != "RANDOM_CLASS_REMAPPINGS":
                    new_label_remappings[key] = value

            # Add all the map{i}-{j} keys directly to the dictionary
            for i, mappings in RANDOM_CLASS_REMAPPINGS.items():
                for j, mapping in enumerate(mappings):
                    new_label_remappings[f"map{i}-{j}"] = mapping

            # Replace the original label_remappings with the expanded version
            config["label_remappings"] = new_label_remappings

        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return {}
