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
            "data_and_res_dir": "imprinting-reproduce",  # "data",
            "backbone_name": "resnet18",
            "dataset_name": ["MNIST"],  # ["ImageNet"],
            "mapping_name": "none",  # "map1-0",
            "mapping": {},  # { 214: 0, 47: 1, 528: 2, 496: 3, 723: 4, 97: 5, 532: 6, 782: 7, 412: 8, 992: 9},
            "task_name": "short",
            "task_splits": [[0, 1, 2]],
            "combinations_slice": [0, 100],
            "use_wandb": False,
            "parallel_threads": 1,
            "torch_threads": 4,
            "use_cache": True,
            "device_name": "cpu",
            "overwrite": True,
            "save_train_acc": True,
            "config_path": "src/config/config.yaml",  # "src/config/config_reprod_sec6.3_imagenet.yaml",
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

    # Load the configuration
    config = load_config(args["config_path"])

    # Extract configuration elements
    backbones = config.get("backbones", ["resnet18"])
    datasets = config.get("datasets", [["MNIST"]])
    remappings_dict = config.get("label_remappings", {"none": {}})
    task_splits_dict = config.get(
        "task_splits", {"all": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}
    )

    # Overwrite with CLI arguments
    if "backbone_name" in args:
        backbones = [args["backbone_name"]]
    if "dataset_name" in args:
        datasets = [args["dataset_name"]]
    if "mapping_name" in args and "mapping" in args:
        remappings_dict = {args["mapping_name"]: args["mapping"]}
    elif "mapping_name" in args or "mapping" in args:
        raise ValueError("Both mapping_name and mapping must be provided together.")
    if "task_name" in args and "task_splits" in args:
        task_splits_dict = {args["task_name"]: args["task_splits"]}
    elif "task_name" in args or "task_splits" in args:
        raise ValueError("Both task_name and task_splits must be provided together.")

    # Create all combinations of backbone, dataset, label_remapping, and task_splits
    data_combinations = []
    for backbone in backbones:
        for dataset in datasets:
            for mapping_name, mapping in remappings_dict.items():
                for ts_name, ts_list in task_splits_dict.items():
                    # Skip map-style remappings for non-ImageNet datasets
                    if dataset != ["ImageNet"] and mapping_name.startswith("map"):
                        continue

                    data_combinations.append(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "task_name": ts_name,
                            "task_splits": ts_list,
                            "mapping_name": mapping_name,
                            "mapping": mapping,
                        }
                    )

    print(f"Created {len(data_combinations)} data combinations")

    # Run each data combination
    for idx, exp_combo in enumerate(data_combinations):
        backbone = exp_combo["backbone"]
        dataset = exp_combo["dataset"]
        task_name = exp_combo["task_name"]
        task_splits = exp_combo["task_splits"]
        mapping_name = exp_combo["mapping_name"]
        mapping = exp_combo["mapping"]

        print(
            f"Running experiment {idx+1}/{len(data_combinations)}: "
            f"backbone={backbone}, dataset={dataset}, "
            f"task_split={task_name}, label_mapping={mapping_name}"
        )

        # Check if embeddings exist for the desired data
        # Remember that dataset is a list which could also contain multiple
        #  dataset building blocks (e.g., MNIST and FashionMNIST could be
        #  combined into one new dataset using mappings and/or task_splits).
        for name in dataset:
            embeddings_path = os.path.join(
                args["data_and_res_dir"], "embeddings", name, backbone
            )
            if not os.path.exists(embeddings_path):
                print(
                    f"WARNING: Embeddings for {name} with {backbone} "
                    f"backbone not found at {embeddings_path}"
                )

        # Generate combinations for this specific configuration
        combinations = generate_combinations(
            config,
            backbone_name=backbone,
            dataset_name=dataset,
            mapping_name=mapping_name,
            mapping=mapping,
            task_name=task_name,
            task_splits=task_splits,
        )

        # Apply combination slice if specified
        slice_start, slice_end = args["combinations_slice"]
        combinations_to_run = combinations[slice_start:slice_end]

        print(
            f"Running {len(combinations_to_run)} of {len(combinations)} "
            "combinations."
        )

        # Execute experiment combinations for this specific configuration
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
            save_train_acc=args["save_train_acc"],
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
        default=["MNIST"],
        type=str,
        nargs="+",
        help="Dataset(s) to use (single or pair). Options: MNIST, FashionMNIST, CIFAR10, ImageNet "
        "(or combinations of those).",
    )
    parser.add_argument(
        "--mn",
        "--mapping_name",
        default="none",
        type=str,
        help="Label mapping name (e.g., 'none', 'map0-1', ...)",
    )
    parser.add_argument(
        "--m",
        "--mapping",
        default="{}",
        type=str,
        help="Label mapping as dictionary string (e.g., '{0:0, 1:0, 2:1, 3:1}')",
    )
    parser.add_argument(
        "--tn",
        "--task_name",
        default="short",
        type=str,
        help="Task name for task splits (e.g., 'short', 'all')",
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
        "--save_train_acc",
        default=True,
        type=lambda x: x.lower() == "true",
        help="Whether to save training accuracy during experiments",
    )
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        type=str,
        help="Path to YAML configuration file for actual experiment configuration",
    )

    args = parser.parse_args()

    # Process and validate arguments
    # Parse dataset name from string to list
    dataset_name = args.ds
    if not isinstance(dataset_name, list) and not 1 <= len(dataset_name) <= 2:
        raise ValueError("Only single dataset or pair of datasets is supported")

    # Parse label mapping from string to dictionary
    try:
        mapping = literal_eval(args.m)
        if not isinstance(mapping, dict):
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
        "mapping_name": args.mn,
        "mapping": mapping,
        "task_name": args.tn,
        "task_splits": task_splits,
        "combinations_slice": combinations_slice,
        "use_wandb": args.w,
        "parallel_threads": args.pt,
        "torch_threads": args.tt,
        "use_cache": args.uc,
        "device_name": args.dn,
        "overwrite": args.o,
        "save_train_acc": args.save_train_acc,
        "config_path": args.config,
    }


if __name__ == "__main__":
    main()
