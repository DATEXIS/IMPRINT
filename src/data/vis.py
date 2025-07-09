"""
Visualization utilities for dataset samples.

This module provides functions to create visualizations of dataset samples,
particularly for showcasing the visual diversity in multi-modal datasets like CombiDigits.

The CombiDigits dataset is a synthetic multi-modal dataset constructed by merging
all classes from MNIST, MNIST-M, SVHN, and USPS. Each class label corresponds to
a digit (0–9), but the dataset includes significant visual heterogeneity across
sources, simulating multi-modal, non-collapsed class distributions.

Example:
    Create a visualization of class 7 samples from CombiDigits:

    >>> from src.data.vis import create_combidigits_class_grid
    >>> fig = create_combidigits_class_grid(
    ...     target_class=7,
    ...     raw_data_root="./imprinting-reproduce",
    ...     save_path="combidigits_class7_samples.pdf"
    ... )
"""

import random
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from src.data.loader import DatasetHandler


def create_combidigits_class_grid(
    target_class: int = 7,
    raw_data_root: str = "./imprinting-reproduce",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a grid visualization showing samples of a specific class from
    the CombiDigits dataset (MNIST, MNIST-M, SVHN, USPS).

    This function creates a 4×5 grid showing 5 random samples of the specified
    class from each of the four datasets that compose CombiDigits. The visualization
    demonstrates the visual heterogeneity across different data sources for the
    same semantic class.

    Args:
        target_class: Class label to visualize (0-9). Default is 7.
        raw_data_root: Root directory for raw data. Default is "./imprinting-reproduce".
        save_path: Optional path to save the figure. If None, figure is not saved.

    Returns:
        matplotlib.pyplot.Figure: The created figure object.

    Raises:
        ValueError: If target_class is not in range 0-9.

    Example:
        >>> fig = create_combidigits_class_grid(target_class=7, save_path="class7.pdf")
        Loading samples from MNIST...
        Loading samples from MNIST-M...
        Loading samples from SVHN...
        Loading samples from USPS...
        Figure saved to class7.pdf
    """
    # Validate input
    if not 0 <= target_class <= 9:
        raise ValueError(f"target_class must be between 0 and 9, got {target_class}")

    # Set figure parameters for publication
    plt.rcParams["font.size"] = 10
    text_width_inches = 160 * 0.0393701  # Convert mm to inches (approx. 6.3 inches)

    # Dataset names that make up CombiDigits
    datasets = ["MNIST", "MNIST-M", "SVHN", "USPS"]
    samples_per_dataset = 5

    # Initialize the figure with no spacing
    fig, axes = plt.subplots(
        len(datasets),
        samples_per_dataset,
        figsize=(text_width_inches, text_width_inches * 0.6),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    # Remove all margins and padding for seamless grid
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    for dataset_idx, dataset_name in enumerate(datasets):
        print(f"Loading samples from {dataset_name}...")

        # Create dataset handler
        dataset_handler = DatasetHandler(
            dataset_name=dataset_name,
            backbone_name="resnet18",  # Backbone doesn't matter for visualization of raw data
            root=raw_data_root,
            train=True,  # Use training data
            batch_size=64,
            seed=42,
        )

        # Collect samples of target class directly from the dataset
        class_samples = []

        # Get the underlying dataset from the dataloader
        if dataset_handler.data_loader is not None:
            dataset = dataset_handler.data_loader.dataset

            # First, find all indices with the target class
            target_indices = []
            max_search = 10000  # Use a reasonable upper limit

            for i in range(max_search):
                try:
                    _, label = dataset[i]
                    if label == target_class:
                        target_indices.append(i)
                except IndexError:
                    break

            # Randomly sample from the found indices
            if len(target_indices) >= samples_per_dataset:
                # Use dataset-specific random seed to ensure reproducibility but different patterns
                dataset_seed = 42 + hash(dataset_name) % 1000
                random.seed(dataset_seed)
                selected_indices = random.sample(target_indices, samples_per_dataset)

                # Get the actual images
                for idx in selected_indices:
                    raw_img, _ = dataset[idx]
                    class_samples.append(raw_img)
            else:
                # If not enough samples, take what we have
                for idx in target_indices:
                    raw_img, _ = dataset[idx]
                    class_samples.append(raw_img)

        # Plot the samples for this dataset
        for sample_idx in range(min(samples_per_dataset, len(class_samples))):
            ax = axes[dataset_idx, sample_idx]

            # Convert PIL image to numpy array for display
            img_array = np.array(class_samples[sample_idx])

            # Handle different image formats
            if len(img_array.shape) == 2:  # Grayscale
                ax.imshow(img_array, cmap="gray", aspect="equal")
            else:  # RGB
                ax.imshow(img_array, aspect="equal")

            # Remove all axes, ticks, and padding
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

            # Remove any padding around the image
            ax.margins(0)
            ax.set_xlim([0, img_array.shape[1]])
            ax.set_ylim([img_array.shape[0], 0])  # Flip y-axis for correct image orientation

            # Add dataset label on the leftmost column
            if sample_idx == 0:
                ax.set_ylabel(dataset_name, rotation=90, va="center", ha="right")

        # Fill any remaining subplots with empty space if we don't have enough samples
        for sample_idx in range(len(class_samples), samples_per_dataset):
            axes[dataset_idx, sample_idx].axis("off")

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, pad_inches=0)  # bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    # Create the visualization
    figure = create_combidigits_class_grid(
        target_class=6,
        raw_data_root="./imprinting-reproduce",
        save_path="combidigits_class6_samples.pdf",
    )
