"""
Kubernetes job generator for imprinting experiments.

This script generates Kubernetes job configurations for running imprinting
experiments. It uses the project's configuration system to determine the
experiment parameters and creates job YAML files accordingly.
"""

import os
import shutil
from jinja2 import Environment, FileSystemLoader

# Import necessary modules from src package
from src.config.experiments import generate_combinations
from src.utils.helpers import load_config


def generate_jobs(
    app_name_suffix,
    results_dir,
    config_path,
    overwrite_json_files,
    use_wandb,
    device_name,
    gpu_node_selector,
    cpu_request,
    cpu_limit,
    parallel_threads,
    torch_threads,
    memory_request,
    memory_limit,
    use_cache,
    shared_memory_limit,
    max_number_of_jobs,
    clear_existing_jobs,
):
    """
    Generate Kubernetes job configurations for imprinting experiments.

    This function creates Kubernetes job YAML files based on the specified
    configuration parameters. It generates combinations of backbone models,
    datasets, label mappings, and task splits, then distributes these across
    the requested number of jobs.

    Args:
        app_name_suffix: Suffix for the job name
        results_dir: Directory name for storing experiment results
        config_path: Path to the YAML configuration file
        overwrite_json_files: Whether to overwrite existing result JSON files
        use_wandb: Whether to use Weights & Biases logging
        device_name: Computing device to use ('cpu', 'cuda', or 'mps')
        docker_digest: Docker image digest/tag to use
        gpu_node_selector: Optional GPU node selector for Kubernetes
        cpu_request: CPU cores to request for each job
        cpu_limit: Maximum CPU cores to allow for each job
        parallel_threads: Number of parallel threads for processing
        torch_threads: Number of threads to allocate for PyTorch
        memory_request: Memory to request for each job (e.g., '8Gi')
        memory_limit: Maximum memory to allow for each job (e.g., '16Gi')
        use_cache: Whether to use shared memory caching
        shared_memory_limit: Size limit for shared memory (e.g., '2Gi')
        max_number_of_jobs: Maximum number of jobs to generate
        clear_existing_jobs: Whether to clear existing job files in the output directory
    """
    # Directory for generated files
    output_dir = f"./k8s/generated_imprinting_jobs_{results_dir}"

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader("./k8s"))

    # Load the template
    template = env.get_template("imprinting_job_template.yaml.j2")

    # Load config once
    config = load_config(config_path)
    # NOTE that at this point, the config might include several data combinations.
    #  But eventually, per machine (=job), we want to have fixed data for
    #  proper parallelization.

    # Extract data configuration elements
    backbones = config.get("backbones", ["resnet18"])
    datasets = config.get("datasets", ["MNIST"])
    mappings_dict = config.get("label_remappings", {"none": {}})
    task_splits_dict = config.get(
        "task_splits", {"all": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]}
    )

    # Create all combinations of backbone, dataset, mapping, and task splits
    data_combinations = []
    for backbone in backbones:
        for dataset in datasets:
            for mapping_name, mapping in mappings_dict.items():
                for ts_name, ts_list in task_splits_dict.items():
                    # Skip map-style remappings for non-ImageNet datasets
                    if dataset != ["ImageNet"] and mapping_name.startswith("map"):
                        continue

                    data_combinations.append(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "mapping_name": mapping_name,
                            "mapping": mapping,
                            "task_name": ts_name,
                            "task_splits": ts_list,
                        }
                    )

    # Check if we have enough job slots for all experiments
    if len(data_combinations) > max_number_of_jobs:
        raise ValueError(
            f"Number of jobs ({max_number_of_jobs}) is less than the "
            f"required number of jobs for all experiments ({len(data_combinations)}). "
            "This way, there wouldn't even be enough jobs in the case of only "
            "one per each data combination."
        )
    num_spare_jobs = max_number_of_jobs - len(data_combinations)
    # Spread jobs evenly
    split_jobs_per_data_combination = num_spare_jobs // len(data_combinations)

    # Initialize file counter and prepare output directory
    if clear_existing_jobs and os.path.exists(output_dir):
        # Remove everything inside but keep the directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        file_counter = 0
    elif os.path.exists(output_dir):
        file_counter = len(os.listdir(output_dir))
    else:
        os.makedirs(output_dir)  # Create the directory
        file_counter = 0

    new_files_counter = 0

    # Find number of configuations that we will have for each data combination
    temp_config = generate_combinations(config)
    num_configurations = len(temp_config)

    # Fixed data for every job
    data = {
        "results_dir": results_dir,
        "config_path": config_path,
        "device_name": device_name,
        "overwrite": overwrite_json_files,
        "use_wandb": use_wandb,
        "parallel_threads": parallel_threads,
        "torch_threads": torch_threads,
        "cpu_request": cpu_request,
        "memory_request": memory_request,
        "cpu_limit": cpu_limit,
        "memory_limit": memory_limit,
        "gpu_node_selector": gpu_node_selector,
        "use_cache": use_cache,
        "shared_memory_limit": shared_memory_limit,
    }

    # For each data combination, calculate imprinting combinations and
    #  create a job
    for exp_combo in data_combinations:
        backbone = exp_combo["backbone"]
        dataset = exp_combo["dataset"]
        mapping_name = exp_combo["mapping_name"]
        mapping = exp_combo["mapping"]
        task_name = exp_combo["task_name"]
        task_splits = exp_combo["task_splits"]

        # Generate app_name from components
        app_name = f"{results_dir}"
        app_name = app_name + f"-{app_name_suffix}"
        app_name = app_name + "-overw" if overwrite_json_files else app_name

        # Split this data combination into split_jobs_per_outer_combination+1 parts
        if split_jobs_per_data_combination <= 0:
            combinations_slices = [[0, num_configurations]]
        else:
            part_size = num_configurations // (split_jobs_per_data_combination + 1)
            combinations_slices = [
                [_i * part_size, (_i + 1) * part_size]
                for _i in range(split_jobs_per_data_combination + 1)
            ]

        for combinations_slice in combinations_slices:
            # Update data for job
            var_data = data.copy()
            # Update the variable data
            var_data["app_name"] = app_name
            var_data["backbone_name"] = backbone
            var_data["dataset_name"] = dataset
            var_data["dataset_name_str"] = "&".join(dataset)
            var_data["mapping_name"] = mapping_name
            var_data["mapping"] = mapping
            var_data["task_name"] = task_name
            var_data["task_splits"] = task_splits
            var_data["combinations_slice"] = combinations_slice

            # Render the template with the data
            output = template.render(var_data)

            # Save the rendered template to a YAML file in the specified directory
            output_path = os.path.join(
                output_dir,
                f"job_{file_counter}_{app_name}.yaml",
            )
            with open(output_path, "w") as f:
                f.write(output)
            file_counter += 1
            new_files_counter += 1

        print(
            f"[INFO] Generated {len(combinations_slices)} job(s) for "
            f"{backbone} - {dataset} - "
            f"label mapping '{mapping_name}' - task '{task_name}'"
        )

    print(
        f"[INFO] {new_files_counter} Kubernetes job YAMLs generated "
        f"successfully in {output_dir} folder!"
    )


### Configuration #############################################################
results_dir = (
    "reprod"  # Will be used for results_dir and in dir name of generated jobs
)
app_name_suffix = "6-3-im"
config_path = "src/config/config_reprod_sec6.3_imagenet.yaml"  # Use the YAML config file with backbones, datasets, task_splits and label_remappings

clear_existing_jobs = True
use_wandb = False
overwrite_json_files = False  # Whether existing json files should be overwritte,
#  i.e., whether runs should be redone (NOTE that wandb runs are not automatically
#  overwritten, but rather duplicated then)

device_name = "cpu"

max_number_of_jobs = (
    400  # Set this higher than the expected number of data combinations
)
# Machine requirements per job
cpu_request = 8
cpu_limit = 16
parallel_threads = 1  # >1 currently does not seem to work (at least not with all data; not even with half of it)
torch_threads = int(cpu_request / parallel_threads)
# NOTE: use htop on the pod to check the CPU utilization
memory_request = "8Gi"
memory_limit = "16Gi"
use_cache = True  # shared memory stuff
shared_memory_limit = (
    "2Gi"  # easily suffices for everything except the vgg11_bn embeddings
)
gpu_node_selector = None

### End Configuration #########################################################


### Run the generation ########################################################
generate_jobs(
    app_name_suffix=app_name_suffix,
    results_dir=results_dir,
    config_path=config_path,
    overwrite_json_files=overwrite_json_files,
    use_wandb=use_wandb,
    device_name=device_name,
    gpu_node_selector=gpu_node_selector,
    cpu_request=cpu_request,
    cpu_limit=cpu_limit,
    parallel_threads=parallel_threads,
    torch_threads=torch_threads,
    memory_request=memory_request,
    memory_limit=memory_limit,
    use_cache=use_cache,
    shared_memory_limit=shared_memory_limit,
    max_number_of_jobs=max_number_of_jobs,
    clear_existing_jobs=clear_existing_jobs,
)
