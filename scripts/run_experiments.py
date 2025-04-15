"""
Main entry point for weight imprinting experiments.

This module serves as the primary entry point for running weight imprinting
experiments. It handles command line arguments, experiment configuration loading,
and experiment execution. It supports running experiments in standalone mode
or as part of a Kubernetes job execution.

The module can be invoked directly with CLI arguments or used programmatically
by supplying configuration parameters.
"""

import argparse
import os
import sys
import json
import yaml
from warnings import simplefilter
from ast import literal_eval

# Ignore deprecation warnings
simplefilter(action="ignore", category=DeprecationWarning)

# Import project modules
from src.experiments.runner import run_combinations
from src.config.experiments import generate_combinations
from src.utils.helpers import load_config
from src.experiments.imagenet.prep import RANDOM_TASKS, RANDOM_CLASS_REMAPPINGS


def main():
    """
    Main function to manage and execute experiments.

    Handles argument parsing, configuration loading, and experiment execution.
    Can run experiments from command line arguments or from a configuration file.
    """
    print("Read arguments: ", sys.argv)

    if len(sys.argv) == 1:
        print("Using default parameters.")
        args = {
            "data_and_res_dir": "data",
            "backbone_name": "resnet18",
            "dataset_name": ["MNIST"],
            "label_mapping": {},
            "task_splits": [[0, 1, 2]],
            "combinations_slice": [0, 99999999],
            "use_wandb": False,
            "vis": False,
            "parallel_threads": 1,
            "torch_threads": 4,
            "use_cache": True,
            "device_name": "cpu",
            "overwrite": True,
            "config_path": "src/config/config.yaml",
        }
        # Using temp results dir for testing
        results_dir = "results_temp"
    else:
        # Read arguments from command line
        args = parse_input()

        results_dir = args["results_dir"]

    print("\n #### Arguments set: #####\n")
    for key, value in args.items():
        print(f"\t{key}={value}")
    print("\n #########################\n")

    # Set up folder for saving results
    res_dir = os.path.join(args["data_and_res_dir"], results_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Check if embeddings exist for the datasets
    for dataset in args["dataset_name"]:
        embeddings_path = os.path.join(
            args["data_and_res_dir"], "embeddings", dataset, args["backbone_name"]
        )
        if not os.path.exists(embeddings_path):
            print(
                f"WARNING: Embeddings for {dataset} with {args['backbone_name']} "
                f"backbone not found at {embeddings_path}"
            )

    # Get actual experiment configuration
    combinations = generate_combinations(load_config(args["config_path"]))

    # Apply combination slice if specified
    slice_start, slice_end = args["combinations_slice"]
    combinations_to_run = combinations[slice_start:slice_end]

    print(f"Running {len(combinations_to_run)} of {len(combinations)} combinations")

    # Execute experiment combinations
    run_combinations(
        combinations_to_run,
        data_dir=args["data_and_res_dir"],
        res_dir=res_dir,
        use_wandb=args["use_wandb"],
        parallel_threads=args["parallel_threads"],
        torch_threads=args["torch_threads"],
        use_cache=args["use_cache"],
        device_name=args["device_name"],
        overwrite=args["overwrite"],
    )


def parse_input():
    """
    Parse command line arguments for experiment configuration.

    This function handles the CLI arguments, performs validation,
    and returns a structured configuration dictionary for experiments.

    Returns:
        dict: Dictionary of parsed and validated arguments
    """
    parser = argparse.ArgumentParser(description="Weight Imprinting Experiment Runner")

    # Data and results directory arguments
    parser.add_argument(
        "--d",
        "--data_and_res_dir",
        default="data",
        type=str,
        help="Data directory (results are also stored in there)",
    )
    parser.add_argument(
        "--r",
        "--results_dir",
        default="results",
        type=str,
        help="Results directory (subfolder in data_and_res_dir for storing results)",
    )

    # Experiment configuration arguments
    parser.add_argument(
        "--b",
        "--backbone_name",
        default="resnet18",
        type=str,
        help="Backbone model to use (e.g., 'resnet18', 'vit_b_16', 'swin_b')",
    )
    parser.add_argument(
        "--ds",
        "--dataset_name",
        default=["CIFAR10"],
        type=str,
        nargs="+",
        help="Dataset(s) to use (single or pair). Options: MNIST, FashionMNIST, CIFAR10, ImageNet",
    )
    parser.add_argument(
        "--lm",
        "--label_mapping",
        default="{}",
        type=str,
        help="Label mapping as dictionary string (e.g., '{0:0, 1:0, 2:1, 3:1}')",
    )
    parser.add_argument(
        "--t",
        "--task_splits",
        default=[],
        nargs="+",
        action="append",
        help="Task splits as integer lists. Example: --t 0 1 --t 2 3",
    )
    parser.add_argument(
        "--c",
        "--combinations_slice",
        default=[0, 999999999],
        type=str,
        nargs="+",
        help="Range of configurations to run as start and end indices",
    )

    # Execution configuration arguments
    parser.add_argument(
        "--w",
        "--use_wandb",
        default=False,
        type=lambda x: x.lower() == "true",
        help="Whether to use Weights and Biases for logging",
    )
    parser.add_argument(
        "--vis",
        default=False,
        type=lambda x: x.lower() == "true",
        help="Whether to generate and save visualizations",
    )
    parser.add_argument(
        "--pt",
        "--parallel_threads",
        default=1,
        type=int,
        help="Number of threads to use for parallel processing",
    )
    parser.add_argument(
        "--tt",
        "--torch_threads",
        default=1,
        type=int,
        help="Number of threads for torch operations",
    )
    parser.add_argument(
        "--uc",
        "--use_cache",
        default=False,
        type=lambda x: x.lower() == "true",
        help="Whether to use shared memory cache for datasets",
    )
    parser.add_argument(
        "--dn",
        "--device_name",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        type=str,
        help="Device to use for computations (cpu, cuda, or mps)",
    )
    parser.add_argument(
        "--o",
        "--overwrite",
        default=False,
        type=lambda x: x.lower() == "true",
        help="Whether to overwrite existing result files",
    )
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        type=str,
        help="Path to YAML configuration file for actual experiment configuration",
    )

    args = parser.parse_args()

    # Process and validate arguments
    dataset_name = args.ds
    if not 1 <= len(dataset_name) <= 2:
        raise ValueError("Only single dataset or pair of datasets is supported")

    # Parse label mapping from string to dictionary
    try:
        label_mapping = literal_eval(args.lm)
        if not isinstance(label_mapping, dict):
            raise ValueError
    except:
        raise ValueError("Label mapping must be a valid Python dictionary string")

    # Process task splits
    task_splits = args.t
    if task_splits:
        for i, task in enumerate(task_splits):
            task_splits[i] = [int(label) for label in task]

    # Process combinations slice
    combinations_slice = args.c
    if len(combinations_slice) != 2:
        raise ValueError("Combinations slice must be a pair of integers [start, end]")
    combinations_slice = [int(combinations_slice[0]), int(combinations_slice[1])]

    return {
        "data_and_res_dir": args.d,
        "results_dir": args.r,
        "backbone_name": args.b,
        "dataset_name": dataset_name,
        "label_mapping": label_mapping,
        "task_splits": task_splits,
        "calc_and_save_cl_accuracies": args.cl,
        "combinations_slice": combinations_slice,
        "use_wandb": args.w,
        "vis": args.vis,
        "parallel_threads": args.pt,
        "torch_threads": args.tt,
        "use_cache": args.uc,
        "device_name": args.dn,
        "overwrite": args.o,
        "config_path": args.config,
    }


if __name__ == "__main__":
    main()
