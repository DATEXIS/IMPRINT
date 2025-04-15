"""
Experiment runner implementation.

This module provides functions to run weight imprinting experiments with various
configurations, handling data loading, model training, evaluation, and result saving.
"""

import ast
import os
import time
import json
from datetime import datetime
import torch
import numpy as np
import wandb
from functools import partial
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from torch.multiprocessing import Pool, Value, Lock

from src.data.embeddings import get_embeddings
from src.utils.hashing import consistent_id
from src.models.model import ImprintedModel
from src.proxy.selection import select_proxies
from src.utils.helpers import set_all_seeds, calc_weighted_f1_score
from src.data.prefilter import prefilter_data
from src.data.continual import ClassContinualDataset

# Set a longer timeout for wandb initialization
os.environ["WANDB_INIT_TIMEOUT"] = "600"  # default=90s

# Create shared objects for progress tracking in parallel execution
_counter = None
_lock = None


def init_worker(counter, lock):
    """
    Initialize each worker process with shared objects for progress tracking.

    Args:
        counter: Shared counter for tracking progress
        lock: Shared lock for synchronizing counter updates
    """
    global _counter
    global _lock
    _counter = counter
    _lock = lock


def load_data(
    backbone_name: str,
    dataset_name: str,
    label_mapping: dict,
    task_splits: list[list[int]],
    data_dir: str,
    use_cache: bool,
    device: str,
    share_memory: bool,
):
    """
    Load dataset embeddings and prepare for continual learning.

    This function loads pre-computed embeddings for the specified datasets and
    organizes them into a continual learning structure according to task splits.

    Args:
        backbone_name: Name of the backbone model used for embeddings
        dataset_name: Name(s) of the dataset(s) to load
        label_mapping: Dictionary mapping original labels to new labels
        task_splits: List of task splits defining the continual learning scenario
        data_dir: Directory containing the data
        use_cache: Whether to cache data in memory
        device: Device to use for computation
        share_memory: Whether to use shared memory tensors (for multiprocessing)

    Returns:
        tuple: (continual_loader, embedding_size) - loader for continual learning tasks
               and the size of feature embeddings
    """
    # Get embeddings for the dataset(s)
    # NOTE: Actually, this is not loading any data into memory; it only sets
    #  up EmbeddingDataset objects, which could then be used to load data on
    #  the fly.
    embedding_size, embeddings_train, embeddings_test = get_data(
        backbone_name, dataset_name, data_dir, label_mapping=label_mapping
    )

    # Create a continual loader for the task sequence
    continual_loader = ClassContinualDataset(
        train_dataset=embeddings_train,
        test_dataset=embeddings_test,
        task_splits=task_splits,
        use_cache=use_cache,
        use_shared_memory=share_memory,
    )

    n_task = continual_loader.num_tasks()

    if use_cache:
        # Pre-cache all task data
        _start_time = time.time()
        for _i in range(n_task):
            continual_loader.get_task(_i)
        print(f"\t[INFO] Caching all data took {time.time() - _start_time:.2f}s.")

        # Move cached data to the target device
        continual_loader.to(device)

    return continual_loader, embedding_size


def run_combinations(
    combinations: list,
    data_dir: str = "data",
    res_dir: str = "data/results",
    use_wandb: bool = True,
    parallel_threads: int = 1,
    torch_threads: int = 2,
    use_cache: bool = False,
    device_name: str = "cpu",
    overwrite: bool = False,
):
    """
    Run multiple experiment configurations.

    This function executes a set of experiment configurations, either in parallel
    or sequentially, handling data loading, model training, evaluation, and result saving.

    Args:
        combinations: List of experiment configurations to run
        data_dir: Directory containing the data
        res_dir: Directory to save results
        use_wandb: Whether to log results to Weights & Biases
        parallel_threads: Number of parallel processes to use (CPU only)
        torch_threads: Number of threads per process for torch operations
        use_cache: Whether to cache data in memory
        device_name: Device name ("cpu", "cuda", or "mps")
        overwrite: Whether to overwrite existing results
    """
    # Device management
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[INFO] Using device '{device}'.")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using device '{device}'.")
    else:
        device_name = "cpu"
        device = torch.device("cpu")
        print(f"[INFO] Using device '{device}'. CPUs available: {os.cpu_count()}")

    # Silence wandb
    os.environ["WANDB_SILENT"] = "true"

    # Check combinations for validity (backbone, dataset, label_mapping and
    #  task_splits must be constant, because otherwise the previous data
    #  loading/caching would work out here/improve anything).
    backbone_name, dataset_name, label_mapping, task_splits = check_combinations(
        combinations
    )

    # Print run configuration
    parallel_doc = (
        f" in parallel on {parallel_threads} threads, with {torch_threads} "
        "threads each for torch, "
        if device_name == "cpu" and parallel_threads > 1
        else " "
    )

    print(
        f"[INFO] Running {(len(combinations))} combinations on '{device}'"
        f"{parallel_doc}"
        f"for {backbone_name} x {'&'.join(dataset_name)} and "
        f"{len(task_splits)} tasks {'-'.join(map(str,[str(t) for t in task_splits]))}."
    )

    # Load data in the main process
    # If use_cache is True, data is loaded into memory and shared via multiprocessing
    continual_loader, embedding_size = load_data(
        backbone_name,
        dataset_name,
        label_mapping,
        task_splits,
        data_dir,
        use_cache,
        device,
        share_memory=(device_name == "cpu"),
    )

    # Determine if we should run in serial or parallel
    serial = not (device_name == "cpu" and parallel_threads > 1)
    # i.e., whenever cuda or mps is used, we are not doing multiprocessing
    if not serial:
        print(f"\t[INFO] Starting pool of {parallel_threads} threads.")

        # Initialize shared counter and lock for progress tracking
        counter = Value("i", 0)
        lock = Lock()

        # Run combinations in parallel using process pool
        with Pool(
            processes=parallel_threads,
            initializer=init_worker,
            initargs=(counter, lock),
        ) as pool:
            pool.map(
                partial(
                    run_combination_with_progress,
                    continual_loader,
                    dataset_name,
                    embedding_size,
                    use_wandb,
                    res_dir,
                    serial,
                    torch_threads,
                    device,
                    overwrite,
                ),
                combinations,
            )
    else:
        print(
            f"[INFO] Running all {len(combinations)} combinations in serial "
            "via a for-loop."
        )
        _counter = 0
        for _comb in combinations:
            run_combination(
                continual_loader,
                dataset_name,
                embedding_size,
                use_wandb,
                res_dir,
                serial,
                torch_threads,
                device,
                overwrite,
                _comb,
            )
            _counter += 1
            if _counter % 5 == 0:
                print(
                    f"[INFO] Finished running {_counter} of "
                    f"{len(combinations)} combinations."
                )

    print(f"[INFO] Finished running all {len(combinations)} combinations.")


def run_combination_with_progress(
    continual_loader,
    dataset_name,
    embedding_size,
    use_wandb,
    res_dir,
    serial,
    torch_threads,
    device,
    overwrite,
    combination,
):
    """
    Wrapper function to run a single combination with progress tracking in
    parallel execution.

    Args:
        See run_combination for parameter details
    """
    # Run the combination
    run_combination(
        continual_loader,
        dataset_name,
        embedding_size,
        use_wandb,
        res_dir,
        serial,
        torch_threads,
        device,
        overwrite,
        combination,
    )

    # Update progress counter
    global _counter
    global _lock
    with _lock:
        _counter.value += 1
        if _counter.value % 10 == 0:
            print(f"[INFO] Finished running {_counter.value} combinations.")


def run_combination(
    continual_loader,
    dataset_name,
    embedding_size,
    use_wandb,
    res_dir,
    serial,
    torch_threads,
    device,
    overwrite,
    combination,
):
    """
    Run a single experiment configuration.

    This function executes a single experiment configuration, handling model creation,
    proxy selection, training, evaluation, and result saving.

    Args:
        continual_loader: Loader for continual learning tasks
        dataset_name: Name(s) of the dataset(s) being used
        embedding_size: Size of feature embeddings
        use_wandb: Whether to log results to Weights & Biases
        res_dir: Directory to save results
        serial: Whether execution is in serial mode
        torch_threads: Number of threads for torch operations
        device: Device to use for computation
        overwrite: Whether to overwrite existing results
        combination: The experiment configuration to run
    """
    # Check for kNN aggregation and adjust device if needed
    if combination["aggregation_method"] == "knn" and device != torch.device("cpu"):
        print("\t[WARN] Falling back to CPU because of kNN aggregation.")
        device = torch.device("cpu")
        if continual_loader.use_cache:
            continual_loader.to(device)

    # Set random seed for reproducibility
    set_all_seeds(combination["seed"])

    # Generate unique ID for this combination
    _id = consistent_id(combination)

    # Check if result already exists
    if os.path.exists(os.path.join(res_dir, f"{_id}.json")):
        if not overwrite:
            print(f"\t[INFO] Combination id {_id} already exists. Skipping.")
            return
        else:
            print(f"\t[WARN] Overwriting existing combination id {_id}.")

    # Generate seed-independent ID for grouping in wandb
    _id_seed_independent = consistent_id({**combination, "seed": None})

    print(
        f"\t[INFO] Running combination id {_id} "
        f"(presampling_fewshot_value={combination['presampling_fewshot_value']}, "
        f"proxy_method={combination['proxy_method']}, "
        f"aggreg_method={combination['aggregation_method']})"
    )

    # Configure thread usage for torch operations
    if not serial:
        torch.set_num_threads(torch_threads)

    # Create a readable description of the label mapping
    label_mapping_desc = generate_label_mapping_desc(combination["label_mapping"])

    # Initialize wandb logging if enabled
    if use_wandb:
        wandb.init(
            entity="bht",
            project="imprinting",
            group=str(_id_seed_independent),
            name=str(_id),
            job_type="train_eval",
        )
        wandb.config.update(combination)
        wandb.config.update(
            {
                "dataset_name": " & ".join(dataset_name),
                "label_mapping_desc": label_mapping_desc,
            },
            allow_val_change=True,
        )

    # Start timing for the current combination
    start_time = time.time()

    # Gradient tracking is not needed since we're not using SGD
    torch.set_grad_enabled(False)

    # Initialize the model
    model = ImprintedModel(
        normalize_input_data=combination["normalize_input_data"],
        normalize_layer_activations=combination["normalize_layer_activations"],
        normalize_weights=combination["normalize_weights"],
        aggregation_method=combination["aggregation_method"],
        k_value=combination["k_value"],
        embedding_size=embedding_size,
    ).to(device)

    # Initialize containers for metrics
    accs = []
    f1s = []

    # Process each task in the continual learning scenario
    for task_idx in range(continual_loader.num_tasks()):
        print(f"\t\t[INFO] Imprinting task {task_idx + 1}")

        task_desc = " & ".join(map(str, continual_loader.task_splits[task_idx]))

        # Get data for the current task
        train_data, train_labels, test_data, test_labels = get_data_from_loader(
            continual_loader, task_idx, device
        )

        # Get task labels and mapping
        distinct_task_labels, mapping = get_mapping(
            continual_loader, task_idx, num_previous_classes=model.num_classes
        )

        # Extend model to accommodate new classes
        model.extend_num_classes(len(distinct_task_labels))
        model.to(device)

        # Process each class in the current task
        for label in distinct_task_labels:
            # Apply normalization if specified for proxy selection
            if combination["normalize_for_proxy_selection"] == "none":
                train_data_label = train_data[train_labels == label]
            elif combination["normalize_for_proxy_selection"] == "l2":
                train_data_label = train_data[train_labels == label]
                train_data_label = train_data_label / torch.norm(
                    train_data_label, p=2, dim=1, keepdim=True
                )
            else:
                raise ValueError(
                    "Invalid value for 'normalize_for_proxy_selection'. "
                    "Must be 'none' or 'l2'."
                )

            # Prefilter the data according to configuration
            filtered_task_data = prefilter_data(
                train_data_label,
                method=combination["presampling_method"],
                quantile=combination["presampling_quantiles_value"],
                fewshot=combination["presampling_fewshot_value"],
            )

            # Time proxy selection
            _start_time = time.time()

            # Select proxies from the filtered data
            task_proxies = select_proxies(
                filtered_task_data,
                num_proxies=combination["num_proxies"],
                method=combination["proxy_method"],
                seed=combination["seed"],
            ).to(device)

            print(
                f"\t\t[INFO] Proxy selection '{combination['proxy_method']}' "
                f"for label {label} in task {task_idx+1} took "
                f"{time.time() - _start_time:.2f}s."
            )

            # Extend model weights with the selected proxies
            model.extend_ws(data=task_proxies, class_index=mapping[label])

        # Prepare wandb logging dictionary
        wandb_log_dict = {
            "tasks_seen": task_idx + 1,
            "task_desc": task_desc,
            "num_proxies": combination["num_proxies"],
            "k_value": combination["k_value"],
        }

        # Map test labels to model's internal class indices
        test_labels_mapped = map_labels(test_labels, mapping, device)

        # Accumulate test data for final evaluation across all tasks
        if task_idx == 0 and continual_loader.num_tasks() > 1:
            test_data_accum = test_data
            test_labels_mapped_accum = test_labels_mapped
        elif task_idx > 0:
            test_data_accum = torch.vstack([test_data_accum, test_data])
            test_labels_mapped_accum = torch.hstack(
                [test_labels_mapped_accum, test_labels_mapped]
            )

        # Evaluate model performance - model is always in eval mode since we removed SGD
        model.eval()  # ...but let's be sure
        accs.append(model.accuracy(test_data, test_labels_mapped))
        f1s.append(model.f1_scores(test_data, test_labels_mapped))

        elapsed_time = time.time() - start_time
        print(
            f"\t\t[INFO] Accuracy on task {task_idx+1} ({task_desc}): "
            f"{accs[task_idx]:.2f}%; {elapsed_time:.3f}s runtime"
        )

        # Log task results to wandb
        if use_wandb:
            weighted_f1 = calc_weighted_f1_score(f1s[task_idx])
            wandb.log(
                {
                    **wandb_log_dict,
                    "time_elapsed": elapsed_time,
                    "task_acc": accs[task_idx],
                    "task_f1s": f1s[task_idx],
                    "task_f1_weight_avg": weighted_f1,
                }
            )

    # Prepare results for saving
    assert task_idx == 0, "Currently only 1 task is supported for proper saving."
    # Because for this paper, we are only interested in that...

    # Store results for first task
    combination["task_acc"] = accs[0]
    combination["task_f1s"] = f1s[0]
    combination["label_mapping_desc"] = label_mapping_desc
    combination["task_split"] = continual_loader.task_splits[0]
    combination["task_desc"] = " & ".join(map(str, continual_loader.task_splits[0]))
    combination["runtime"] = elapsed_time
    combination["created_at"] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Calculate metrics across all tasks if multiple tasks were processed
    if task_idx > 0:
        final_tasks_acc = model.accuracy(test_data_accum, test_labels_mapped_accum)
        final_tasks_f1s = model.f1_scores(test_data_accum, test_labels_mapped_accum)
        combination["final_tasks_acc"] = final_tasks_acc
        combination["final_tasks_f1s"] = final_tasks_f1s

        # Log final results to wandb
        if use_wandb:
            final_weighted_f1 = calc_weighted_f1_score(final_tasks_f1s)
            wandb_log_dict.pop("task_desc")
            wandb.log(
                {
                    **wandb_log_dict,
                    "final_tasks_acc": final_tasks_acc,
                    "final_tasks_f1s": final_tasks_f1s,
                    "final_tasks_f1_weight_avg": final_weighted_f1,
                }
            )

        # Print final results
        elapsed_time = time.time() - start_time
        print(
            f"\t\t[INFO] Final accuracy on data of all tasks: "
            f"{final_tasks_acc:.2f}%; {elapsed_time:.2f}s total runtime"
        )

    # Save results to JSON file
    with open(os.path.join(res_dir, f"{_id}.json"), "w") as json_file:
        json.dump(combination, json_file, indent=4)

    # Finish wandb logging session
    if use_wandb:
        wandb.finish()


def check_combinations(combinations: list):
    """
    Check that all combinations have consistent dataset-related parameters.

    Args:
        combinations: List of experiment configurations

    Returns:
        tuple: (backbone_name, dataset_name, label_mapping, task_splits)

    Raises:
        AssertionError: If combinations have inconsistent parameters
    """
    backbone_name = combinations[0]["backbone_name"]
    dataset_name = combinations[0]["dataset_name"]
    label_mapping = combinations[0]["label_mapping"]
    task_splits = combinations[0]["task_splits"]

    for _comb in combinations:
        assert (
            _comb["backbone_name"] == backbone_name
        ), "All combinations must be on the same backbone!"
        assert (
            _comb["dataset_name"] == dataset_name
        ), "All combinations must be on the same dataset!"
        assert (
            _comb["label_mapping"] == label_mapping
        ), "All combinations must have the same label mapping!"
        assert (
            _comb["task_splits"] == task_splits
        ), "All combinations must have the same task splits!"

    return backbone_name, dataset_name, label_mapping, task_splits


def get_data(backbone_name, dataset_name, data_dir, label_mapping=None):
    """
    Load embeddings for the specified dataset(s).

    Args:
        backbone_name: Name of the backbone model used for embeddings
        dataset_name: Name(s) of the dataset(s) to load
        data_dir: Directory containing the data
        label_mapping: Dictionary mapping original labels to new labels

    Returns:
        tuple: (embedding_size, embeddings_train, embeddings_test)

    Raises:
        ValueError: If more than two datasets are specified
    """
    if len(dataset_name) == 1:
        # Single dataset case
        embeddings_train, embeddings_test, embedding_size = get_embeddings(
            dataset_name[0],
            backbone_name,
            offset=0,
            root=data_dir,
            splits=["train", "test"],
            label_mapping=label_mapping,
        )

    elif len(dataset_name) == 2:
        # Two datasets case
        embeddings_train_1, embeddings_test_1, embedding_size_1 = get_embeddings(
            dataset_name[0],
            backbone_name,
            offset=0,
            root=data_dir,
            splits=["train", "test"],
            label_mapping=label_mapping,
        )

        # Apply offset based on the first dataset
        offset = embeddings_train_1.number_of_classes_without_mapping
        # TODO: I think the offset should be applied after the mapping, because
        #  e.g., for ImageNet, where we do not have classes 0-999, but only
        #  our selected few (focused ones), it will not work out this way.
        embeddings_train_2, embeddings_test_2, embedding_size_2 = get_embeddings(
            dataset_name[1],
            backbone_name,
            offset=offset,
            root=data_dir,
            splits=["train", "test"],
            label_mapping=label_mapping,
        )

        # Verify embedding sizes match
        assert (
            embedding_size_1 == embedding_size_2
        ), "Embedding sizes must be the same!"

        # Combine datasets
        embedding_size = embedding_size_1
        embeddings_train = ConcatDataset([embeddings_train_1, embeddings_train_2])
        embeddings_test = ConcatDataset([embeddings_test_1, embeddings_test_2])

    else:
        raise ValueError("Only one or two datasets are allowed!")

    return embedding_size, embeddings_train, embeddings_test


def get_data_from_loader(continual_loader, task_idx, device):
    """
    Get data for a specific task from the continual loader.

    Args:
        continual_loader: Loader for continual learning tasks
        task_idx: Index of the task to load
        device: Device to load data to

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """
    if continual_loader.use_cache:
        # Data is already cached and ready to use
        return continual_loader.get_task(task_idx)
    else:
        # Load data on demand
        train_dataset, test_dataset = continual_loader.get_task(task_idx)

        _loading_time = time.time()

        # Process training data
        train_data = torch.vstack([data for data, _ in train_dataset]).to(device)
        train_labels = torch.hstack([label for _, label in train_dataset]).to(device)

        # Process testing data
        test_data = torch.vstack([data for data, _ in test_dataset]).to(device)
        test_labels = torch.hstack([label for _, label in test_dataset]).to(device)

        print(
            f"\t\t[INFO] Loading data for task {task_idx} "
            f"took {time.time() - _loading_time:.2f}s."
        )

        return train_data, train_labels, test_data, test_labels


def get_mapping(continual_loader, task_idx, num_previous_classes):
    """
    Create a mapping from task labels to model class indices.

    Args:
        continual_loader: Loader for continual learning tasks
        task_idx: Index of the current task
        num_previous_classes: Number of classes already in the model

    Returns:
        tuple: (distinct_task_labels, mapping) - list of unique labels in the task
               and dictionary mapping task labels to model indices
    """
    # Get the unique labels for this task
    distinct_task_labels = continual_loader.task_splits[task_idx]

    # Create new consecutive indices for these labels
    new_labels = np.arange(
        num_previous_classes, num_previous_classes + len(distinct_task_labels)
    )

    # Create the mapping
    mapping = {
        distinct_task_labels[i]: new_labels[i]
        for i in range(len(distinct_task_labels))
    }

    return distinct_task_labels, mapping


def map_labels(labels, mapping, device):
    """
    Map original labels to model class indices.

    Args:
        labels: Original labels tensor
        mapping: Dictionary mapping original labels to model indices
        device: Device for the output tensor

    Returns:
        torch.Tensor: Tensor of mapped labels
    """
    labels_mapped = torch.tensor([mapping[_label.item()] for _label in labels]).to(
        device
    )
    return labels_mapped


def generate_label_mapping_desc(label_mapping_str):
    """
    Generate a readable description of the label mapping.

    Args:
        label_mapping: Dictionary mapping original labels to new labels

    Returns:
        str: Description of the label mapping
    """
    if isinstance(label_mapping_str, str):
        label_mapping_dict = ast.literal_eval(label_mapping_str)
    else:
        label_mapping_dict = label_mapping_str
    assert isinstance(label_mapping_dict, dict)
    return " & ".join([f"{k}->{v}" for k, v in label_mapping_dict.items() if v != -1])
