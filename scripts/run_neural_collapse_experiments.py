import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader

from src.data.datasets import EmbeddingDataset
from src.data.embeddings import get_embeddings_path
from src.experiments.runner import get_data
from src.nc.neural_collapse import NeuralCollapse
from src.experiments.imagenet.prep import RANDOM_CLASS_REMAPPINGS


def main():
    """
    Run neural collapse experiments focusing only on NC1 metric.
    Calculates and saves NC1 values for different backbone models and datasets,
    including random label remappings for ImageNet.
    """
    # Set up the paths and configuration
    print("Setting up paths and configuration")
    root = "./imprinting-reproduce"
    nc_dir = "./analysis/nc_results"
    torch.set_num_threads(4)

    # Create output directory if it doesn't exist
    if not os.path.exists(nc_dir):
        os.makedirs(nc_dir)

    # Define parameter combinations for "standard" datasets
    # For MNIST, FashionMNIST, and CIFAR10, we use all classes (0-9) as is.
    standard_parameters = {
        "dataset_names_with_label_mappings": [
            (["MNIST"], {}),
            (["FashionMNIST"], {}),
            (["CIFAR10"], {}),
            (
                ["MNIST", "MNIST-M", "USPS", "SVHN"],
                {i + j * 10: i for i in range(10) for j in range(4)},
            ),  # the MNIST&MNIST-M&USPS&SVHN ("CombiDigits") dataset
        ],
        "backbone_name": ["resnet18", "resnet50", "vit_b_16", "swin_b"],
    }

    # Process standard datasets
    process_standard_datasets(standard_parameters, root, nc_dir)

    # Process ImageNet with label remappings
    process_imagenet_with_remapping(root, nc_dir)

    print("\nAll neural collapse experiments completed!")


def process_standard_datasets(parameters, root, nc_dir):
    """
    Process "standard" datasets.
    """
    datasets = [str(l[0]) for l in parameters["dataset_names_with_label_mappings"]]
    print(f"\n=== Processing standard datasets {datasets} ===")

    # Generate all combinations of parameters
    combinations = list(ParameterGrid(parameters))

    # Store all results
    results = {
        "dataset": [],
        "backbone": [],
        "nc_1": [],
        "vci": [],
        "intra_trace": [],
        "inter_trace": [],
        "intra_rank": [],
        "inter_rank": [],
    }

    # Process each combination
    for idx, combination in enumerate(combinations):
        print(f"Running combination {idx+1}/{len(combinations)}")

        dataset, label_mapping = combination["dataset_names_with_label_mappings"]
        dataset_name = "&".join(dataset)
        backbone = combination["backbone_name"]

        try:
            embedding_size, embeddings_train, _ = get_data(
                backbone_name=backbone,
                dataset_name=dataset,
                data_dir=root,
                label_mapping=label_mapping,
            )

            # Extract embeddings and labels
            embeddings, labels = next(
                iter(DataLoader(embeddings_train, batch_size=len(embeddings_train)))
            )

            # Filter out invalid labels if any
            if -1 in labels:
                wanted_labels = ~torch.isin(labels, torch.tensor([-1]))
                embeddings = embeddings[wanted_labels]
                labels = labels[wanted_labels]

            # Apply normalization
            embeddings = embeddings / (
                torch.norm(embeddings, dim=1, keepdim=True) + torch.finfo(embeddings.dtype).eps
            )

            # Calculate neural collapse metrics
            print(f"Calculating NC1s for {dataset_name} with {backbone}")
            nc = NeuralCollapse(embeddings, labels)
            nc_1, vci, intra_trace, inter_trace, intra_rank, inter_rank = (
                nc.nc_1()
            )  # Let nc_1 calculate covariance internally

            # Store results
            results["dataset"].append(dataset_name)
            results["backbone"].append(backbone)
            results["nc_1"].append(nc_1)
            results["vci"].append(vci)
            results["intra_trace"].append(intra_trace)
            results["inter_trace"].append(inter_trace)
            results["intra_rank"].append(intra_rank)
            results["inter_rank"].append(inter_rank)

            print(
                f"Completed {dataset_name} with {backbone}: NC1 = {nc_1:.4f}, "
                f"VCI = {vci:.4f}, Intra-trace = {intra_trace:.4f}, "
                f"Inter-trace = {inter_trace:.4f}, Intra-rank = {intra_rank}, "
                f"Inter-rank = {inter_rank}"
            )

        except Exception as e:
            print(f"Error processing {dataset_name} with {backbone}: {str(e)}")

    # Create and save a DataFrame for standard dataset results
    print("Saving summary of standard dataset results")
    standard_df = pd.DataFrame(results)
    standard_csv_path = os.path.join(nc_dir, "standard_nc1_results.csv")
    standard_df.to_csv(standard_csv_path, index=False)
    print(f"Standard dataset results saved to {standard_csv_path}")

    # Print a summary table
    print("\nNeural Collapse (NC1) Results Summary for Standard Datasets:")
    print("=" * 80)
    print(
        f"{'Dataset':<15} {'Backbone':<15} {'NC1':<10} {'VCI':<10} {'Intra-trace':<15} {'Inter-trace':<15} {'Intra-rank':<12} {'Inter-rank':<12}"
    )
    print("-" * 80)
    for i in range(len(results["dataset"])):
        print(
            f"{results['dataset'][i]:<15} {results['backbone'][i]:<15} {results['nc_1'][i]:<10.4f}"
            f" {results['vci'][i]:<10.4f} {results['intra_trace'][i]:<15.4f} {results['inter_trace'][i]:<15.4f}"
            f" {results['intra_rank'][i]:<12} {results['inter_rank'][i]:<12}"
        )


def process_imagenet_with_remapping(root, nc_dir):
    """
    Process ImageNet with various label remappings.
    ImageNet uses 100 different random label remappings (10 configurations × 10 variations).
    """
    print("\n=== Processing ImageNet with 100 different label remappings ===")
    print("Processing 10 configurations (1-10 classes per label) × 10 random variations each")

    # Define backbones to test with ImageNet
    backbones = ["resnet18", "resnet50", "vit_b_16", "swin_b"]

    # Store all results for ImageNet
    imagenet_results = {
        "dataset": [],
        "backbone": [],
        "remapping": [],
        "n_classes_per_label": [],
        "remapping_index": [],
        "nc_1": [],
        "vci": [],
        "intra_trace": [],
        "inter_trace": [],
        "intra_rank": [],
        "inter_rank": [],
    }

    # Process each backbone
    for backbone in backbones:
        # Get embeddings path for ImageNet
        embeddings_path, embeddings_filename = get_embeddings_path(
            root, "ImageNet", backbone, train=False  # Use validation set for ImageNet
        )

        # Load embeddings dataset
        embeddings_file = os.path.join(embeddings_path, embeddings_filename)
        if not os.path.exists(embeddings_file):
            print(
                f"Skipping ImageNet with {backbone} - embeddings file not found at {embeddings_file}"
            )
            continue

        # Process each mapping configuration (1-10 classes per label)
        for n_classes_per_label in range(1, 11):
            print(f"Processing ImageNet with {backbone}, {n_classes_per_label} classes per label")

            # Process each random remapping variation (10 for each configuration)
            for remapping_idx, label_mapping in enumerate(
                RANDOM_CLASS_REMAPPINGS[n_classes_per_label]
            ):
                try:
                    # Create remapping name in the format "d_in_1-k"
                    remapping_name = f"{n_classes_per_label}_in_1-{remapping_idx}"

                    # Load the embeddings with the current label mapping
                    embeddings_dataset = EmbeddingDataset(
                        embeddings_file, label_mapping=label_mapping
                    )

                    # Extract embeddings and labels
                    embeddings = embeddings_dataset[:][0]
                    labels = embeddings_dataset[:][1]

                    # Filter out invalid labels
                    wanted_labels = ~torch.isin(labels, torch.tensor([-1]))
                    embeddings = embeddings[wanted_labels]
                    labels = labels[wanted_labels]

                    # Apply normalization
                    embeddings = embeddings / (
                        torch.norm(embeddings, dim=1, keepdim=True)
                        + torch.finfo(embeddings.dtype).eps
                    )

                    # Calculate neural collapse metrics
                    print(
                        f"  Calculating NC1s for ImageNet with {backbone}, remapping {remapping_name}"
                    )
                    nc = NeuralCollapse(embeddings, labels)
                    nc_1, vci, intra_trace, inter_trace, intra_rank, inter_rank = (
                        nc.nc_1()
                    )  # Let nc_1 calculate covariance internally

                    # Store results
                    imagenet_results["dataset"].append("ImageNet")
                    imagenet_results["backbone"].append(backbone)
                    imagenet_results["remapping"].append(remapping_name)
                    imagenet_results["n_classes_per_label"].append(n_classes_per_label)
                    imagenet_results["remapping_index"].append(remapping_idx)
                    imagenet_results["nc_1"].append(nc_1)
                    imagenet_results["vci"].append(vci)
                    imagenet_results["intra_trace"].append(intra_trace)
                    imagenet_results["inter_trace"].append(inter_trace)
                    imagenet_results["intra_rank"].append(intra_rank)
                    imagenet_results["inter_rank"].append(inter_rank)

                    print(
                        f"  Completed remapping {remapping_name}: NC1 = {nc_1:.4f}"
                        f", VCI = {vci:.4f}, Intra-trace = {intra_trace:.4f}, "
                        f"Inter-trace = {inter_trace:.4f}, Intra-rank = {intra_rank}, "
                        f"Inter-rank = {inter_rank}"
                    )

                except Exception as e:
                    print(
                        f"  Error processing ImageNet with {backbone}, remapping {n_classes_per_label}_in_1-{remapping_idx}: {str(e)}"
                    )

    # Save detailed ImageNet results as CSV
    print("Saving ImageNet remapping results to CSV")
    imagenet_df = pd.DataFrame(imagenet_results)
    imagenet_csv_path = os.path.join(nc_dir, "imagenet_nc1_results.csv")
    imagenet_df.to_csv(imagenet_csv_path, index=False)
    print(f"ImageNet detailed results saved to {imagenet_csv_path}")


if __name__ == "__main__":
    main()
