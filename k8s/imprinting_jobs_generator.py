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
from src.experiments.imagenet.prep import (
    RANDOM_TASKS,
    RANDOM_CLASS_REMAPPINGS,
)
from src.config.experiments import generate_combinations
from src.utils.helpers import load_config


def generate_jobs(
    results_dir,
    config_path,
    backbone_names,
    dataset1_names,
    dataset2_names,
    label_mappings,
    task_splits,
    overwrite_json_files,
    use_wandb,
    device_name,
    docker_digest,
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
    # Directory for generated files
    output_dir = f"./k8s/generated_imprinting_jobs_{results_dir}"

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader("./k8s"))

    # Load the template
    template = env.get_template("imprinting_job_template.yaml.j2")

    for lm_name, lm_dict in label_mappings.items():
        for ts_name, ts_list in task_splits.items():
            # Generate app_name from label mapping and task (e.g., "map1-alltask")
            app_name = f"{lm_name}-{ts_name}task"
            app_name = app_name + "-overw" if overwrite_json_files else app_name
            # NOTE: Not allowed to contain underscores (_) or the like!

            # For first loop, clear old jobs completely
            # Ensure output directory exists and is empty
            if os.path.exists(output_dir):
                if clear_existing_jobs:
                    # Remove everything inside but keep the directory
                    for filename in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    clear_existing_jobs = False
            else:
                os.makedirs(output_dir)  # Create the directory

            # Fixed data
            data = {
                "app_name": app_name,  # Keep this name short!
                "docker_tag": docker_digest,
                "results_dir": results_dir,
                "device_name": device_name,
                "overwrite": overwrite_json_files,
                "label_mapping": lm_dict,
                "task_splits": ts_list,
                "use_wandb": use_wandb,
                "vis": False,
                "parallel_threads": parallel_threads,
                "torch_threads": torch_threads,
                "cpu_request": cpu_request,
                "memory_request": memory_request,
                "cpu_limit": cpu_limit,
                "memory_limit": memory_limit,
                "gpu_node_selector": gpu_node_selector,
                "use_cache": use_cache,
                "shared_memory_limit": shared_memory_limit,
                "config_path": config_path,
            }

            num_outer_combinations = (
                len(backbone_names) * len(dataset1_names) * len(dataset2_names)
            )
            min_number_of_jobs = num_outer_combinations
            if max_number_of_jobs < min_number_of_jobs:
                raise ValueError(
                    f"Number of jobs ({max_number_of_jobs}) is less than the "
                    f"minimum number of jobs ({min_number_of_jobs})"
                )
            num_spare_jobs = max_number_of_jobs - min_number_of_jobs
            # Spread jobs evenly
            split_jobs_per_outer_combination = num_spare_jobs // num_outer_combinations

            if clear_existing_jobs:
                file_counter = 0
            else:
                file_counter = len(os.listdir(output_dir))
            # Find number of configuations that we will have for each 'outer' combination
            #  (that is, disregarding number of backbones and datasets (because these are
            #   fixed in `run_combinations` function, to not load different data a lot of times)
            # Load config to get total number of combinations
            config = load_config(config_path)
            temp_configs = generate_combinations(config)
            num_configurations = len(temp_configs)

            new_files_counter = 0
            for backbone_name in backbone_names:
                for dataset1_name in dataset1_names:
                    for dataset2_name in dataset2_names:

                        # Split this outer combination into
                        #  split_jobs_per_outer_combination+1 parts
                        if split_jobs_per_outer_combination <= 0:
                            combinations_slices = [[0, num_configurations]]
                        else:
                            part_size = num_configurations // (
                                split_jobs_per_outer_combination + 1
                            )
                            combinations_slices = [
                                [_i * part_size, (_i + 1) * part_size]
                                for _i in range(split_jobs_per_outer_combination + 1)
                            ]

                        for combinations_slice in combinations_slices:
                            var_data = data.copy()
                            # Update the variable data
                            var_data["backbone_name"] = backbone_name
                            var_data["dataset_name1"] = dataset1_name
                            var_data["dataset_name2"] = dataset2_name
                            var_data["combinations_slice"] = combinations_slice

                            # Render the template with the data
                            output = template.render(var_data)

                            # Save the rendered template to a YAML file in the specified directory
                            output_path = os.path.join(
                                output_dir,
                                f"job_{file_counter}_{app_name}_{device_name}.yaml",
                            )
                            with open(output_path, "w") as f:
                                f.write(output)
                            file_counter += 1
                            new_files_counter += 1

            print(
                f"[INFO] {new_files_counter} Kubernetes job YAMLs for label "
                f"mapping '{lm_name}' and task '{ts_name}' generated "
                f"successfully in {output_dir} folder!"
            )


### Configuration #############################################################
results_dir = "tests"  # Will be used for results_dir and in dir name of generated jobs
config_path = "src/config/config.yaml"

clear_existing_jobs = True
use_wandb = False
overwrite_json_files = False  # Whether existing json files should be overwritte,
#  i.e., whether runs should be redone (NOTE that wandb runs are not automatically
#  overwritten, but rather duplicated then)

# Set label mappings of interest
label_mappings = {
    "none": {},
    # "map1": FAVORITE_LABEL_MAPPINGS[1],  # for ImageNet
    # "map2": FAVORITE_LABEL_MAPPINGS[2],  # for ImageNet
    # "map3": FAVORITE_LABEL_MAPPINGS[3],  # for ImageNet
    # "map4": FAVORITE_LABEL_MAPPINGS[4],  # for ImageNet
    # "map5": FAVORITE_LABEL_MAPPINGS[5],  # for ImageNet
}
# label_mappings = {
#     f"map{i}-{j}": mapping
#     for i, mappings in RANDOM_CLASS_REMAPPINGS.items()
#     for j, mapping in enumerate(mappings)
# }

# Set task splits of interest
task_splits = {
    "all": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "even": [[0, 2, 4, 6, 8]],
    "odd": [[1, 3, 5, 7, 9]],
    "short": [[0, 1, 2]],
    # "rand0": RANDOM_TASKS[0],  # for ImageNet
    # "rand1": RANDOM_TASKS[1],  # for ImageNet
    # "rand2": RANDOM_TASKS[2],  # for ImageNet
    # "rand3": RANDOM_TASKS[3],  # for ImageNet
    # "rand4": RANDOM_TASKS[4],  # for ImageNet
}

backbone_names = [
    "resnet18",
    "vit_b_16",
    # "resnet50",
    # "swin_b",
]
dataset1_names = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    # "ImageNet",
]
dataset2_names = [""]

device_name = "cpu"

# Latest docker digests
docker_digest = ""

# Machine requirements
cpu_request = None
cpu_limit = None
parallel_threads = 1
torch_threads = 1
max_number_of_jobs = None
gpu_node_selector = None
max_number_of_jobs = 6
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

### End Configuration #########################################################


### Run the generation ########################################################
generate_jobs(
    results_dir=results_dir,
    config_path=config_path,
    backbone_names=backbone_names,
    dataset1_names=dataset1_names,
    dataset2_names=dataset2_names,
    label_mappings=label_mappings,
    task_splits=task_splits,
    overwrite_json_files=overwrite_json_files,
    use_wandb=use_wandb,
    docker_digest=docker_digest,
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
