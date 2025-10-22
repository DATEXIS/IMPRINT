"""Contains helper functions for analysis.ipynb notebook."""

import os
import sys
import ast
import json
from datetime import datetime
import pandas as pd
import wandb
from tqdm import tqdm
from wandb.errors import CommError

# Get the absolute path of the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path
sys.path.append(project_root)

# Now import the draw_cd_diagram function
from analysis.cd_diag import draw_cd_diagram


# List of "groups" in our experiments
GROUPPARAMS = [
    "backbone_name",
    "dataset_name",
    "mapping_name",
    "task_name",
]

# List of hyperparams of our experiments
HYPERPARAMS = [
    "normalize_for_proxy_selection",
    "normalize_weights",
    "normalize_input_data",
    ###
    # "presampling_method",  #
    # "presampling_quantile_a_value",  #
    # "presampling_quantile_b_value",  #
    "presampling_fewshot_value",  #
    ###
    "proxy_method",  #
    "k",
    ###
    "aggregation_method",
    "aggregation_distance_function",
    "aggregation_weighting",
    "m",
]

# To be able to create short, readable strings for combinations, we set up a
#  list of abbreviations which can be used on a string to shorten it
ABBREVS = {
    ### KEYS:
    "normalize_input_data": "norm_emb",
    "normalize_for_proxy_selection": "norm_prx_gen",
    "normalize_weights": "norm_weig",
    "presampling_method": "pres_meth",
    "presampling_quantile_values": "pres_q",
    "presampling_quantile_a_value": "pres_q_a",
    "presampling_quantile_b_value": "pres_q_b",
    "presampling_fewshot_value": "pres_few",
    "proxy_method": "prxy_meth",
    "k": "k",
    "aggregation_method": "agg_meth",
    "aggregation_distance_function": "agg_dist",
    "aggregation_weighting": "agg_weig",
    "m": "m",
    "sgd_finetuning": "sgd",
    ### VALUES:
    "True": "1",
    "true": "1",
    "False": "0",
    "false": "0",
    "random": "rand",
    "nearest_to_mean": "near2mean",
    "kmedoids": "kmeds",
    "cov_max": "covmax",
}


def save_wandb_to_csv(data: list[dict], start_time="", finish_time="", filename="wandb_crawled_at"):
    """
    Parameters:
        data (list[dict]): List of dictionaries.
    """
    df = pd.DataFrame(data)

    # Save to CSV in analysis folder
    filename = f"{filename}_{start_time}_{finish_time}.csv"
    df.to_csv(
        os.path.join(
            "raw_results",
            "wandb_csvs",
            filename,
        ),
        index=False,
    )
    print(f"{len(df)} runs saved to {filename} at {datetime.now().strftime('%Y-%m-%d_%H-%M')}")


def fetch_data_from_wandb(start_time, last_full_fetch_until, ignore_data_after):
    """Fetches data from wandb and saves it to CSV. NOTE: Outdated."""
    # Login to wandb if needed
    wandb.login()

    # Replace with your project name
    entity_name = "bht"
    project_name = "imprinting"

    # Fetch runs
    api = wandb.Api(timeout=350)

    # TODO: Load in parallel? for example, for each dataset_name

    filters = {
        "state": "finished",
        "created_at": {
            "$gt": last_full_fetch_until.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "$lt": ignore_data_after.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        },
        # "summary.task_desc": task_desc,
    }
    runs = api.runs(
        f"{entity_name}/{project_name}",
        filters=filters,
        order="+created_at",  # ascending (i.e., start with oldest data first)
        # per_page=5_000,  # default is 50 -- I feel like this did speed up things
    )  # newest first

    # Extract relevant data
    keys = [
        "normalize_input_data",
        "normalize_weights",
        ###
        # "presampling_method",
        # "presampling_quantiles_value",
        "presampling_fewshot_value",
        ###
        "proxy_method",
        "k",
        ###
        "aggregation_method",
        "m",
        ###
        "seed",
    ]

    data = []
    _counter = 0
    _num_saved_runs = 0
    _num_saved_sgd_runs = 0
    _created_at = ""
    for run in runs:
        _counter += 1
        if (_counter % 10 == 0 and run.created_at[:10] != _created_at) or _counter % 500 == 0:
            _created_at = run.created_at[:10]
            print(
                f"Fetching runs created at {_created_at} ({_counter} runs checked, "
                f"{_num_saved_runs} runs saved (of which {_num_saved_sgd_runs} with SGD history))"
            )
            # TODO: save latest created_at to be able to know from where to start again!

        config = run.config
        summary = run.summary

        # TODO: add mapping stuff

        data.append(
            {
                "backbone_name": config.get("backbone_name"),
                "dataset_name": config.get("dataset_name"),
                "mapping_name": config.get("mapping_name"),
                "task_name": summary.get("task_name"),
                "task_acc": summary.get("task_acc"),
                "task_f1": summary.get("task_f1"),
                "time_elapsed": summary.get("time_elapsed"),
                **{key: config.get(key) for key in keys},  # Add all other hyperparameters
                "created_at": datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ").strftime(
                    "%Y-%m-%d_%H:%M:%S"
                ),
            }
        )
        _num_saved_runs += 1

        # Every 10_000 runs, save to CSV (simply overwrite every time)
        if _num_saved_runs % 10_000 == 0:
            save_wandb_to_csv(data, start_time)

    save_wandb_to_csv(data, start_time, finish_time=datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("DONE")


def gather_data_from_jsons(raw_json_results_dir, ignore_data_before):
    """
    Gather the data from the JSON files in the specified directory.
    """
    # Define relevant data to be extracted
    keys = [
        "backbone_name",
        "dataset_name",
        "mapping_name",
        "mapping",
        "label_mapping_desc",
        "task_name",
        "task_splits",
        "task_split",
        "task_desc",
        ##
        "normalize_input_data",
        "normalize_for_proxy_selection",
        "normalize_weights",
        ###
        "presampling_method",
        "presampling_quantiles_value",
        "presampling_fewshot_value",
        ###
        "proxy_method",
        "k",
        ###
        "aggregation_method",
        "aggregation_distance_function",  # Not necessarily existing, checked below
        "aggregation_weighting",  # Not necessarily existing, checked below
        "m",
        ###
        "seed",
        ###
        "task_acc",
        "task_train_acc",  # Not necessarily existing, checked below
        "task_f1s",
        "runtime",
        "total_GEN_time",  # Not necessarily existing, checked below
        ###
        "created_at",
    ]

    data = []
    _num_ignored_runs = 0
    _num_gathered_runs = 0
    _created_at = ""

    json_files = [f for f in os.listdir(raw_json_results_dir) if f.endswith(".json")]

    for run_filename in tqdm(json_files, desc="Processing JSON files"):

        # Get the full path of the file
        file_path = os.path.join(raw_json_results_dir, run_filename)

        # Get the creation time of the file
        created_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        # NOTE: Actually, ctime returned the wrong datetime, so I used mtime
        #  instead.
        # Skip files created before the specified datetime
        if created_time < ignore_data_before:
            _num_ignored_runs += 1
            continue

        # Load the json file
        run = json.load(open(file_path))

        if datetime.strptime(run["created_at"], "%Y-%m-%d_%H:%M:%S") < ignore_data_before:
            _num_ignored_runs += 1
            continue

        if (len(data) % 10_000 == 0 and run["created_at"][:10] != _created_at) or len(
            data
        ) % 50_000 == 0:
            _created_at = run["created_at"][:10]
            tqdm.write(
                f"Gathering runs created at {_created_at} ({len(data)} runs "
                f"checked, {_num_gathered_runs} runs gathered already"
            )

        # Update dataset_name for compatibility
        run["dataset_name"] = " & ".join(run["dataset_name"])

        data.append(run)
        _num_gathered_runs += 1

    df = pd.DataFrame(data)

    # Only if "task_train_acc" is present, we can slice it out
    if "task_train_acc" not in df.columns:
        keys.remove("task_train_acc")

    # Only if "total_GEN_time" is present, we can slice it out
    if "total_GEN_time" not in df.columns:
        keys.remove("total_GEN_time")

    # Slice out the keys we are interested in
    df = df[keys]

    # Rename 'run_time' to 'time_elapsed'
    df.rename(columns={"runtime": "time_elapsed"}, inplace=True)

    # Create folder to save to
    os.makedirs(os.path.join("raw_results", "local_csvs"), exist_ok=True)

    # Save to CSV
    origin_tar_file_name = raw_json_results_dir.split("/")[-2]
    filename = f"{_num_gathered_runs}runs_gathered_from_{origin_tar_file_name}_jsons.csv"
    df.to_csv(
        os.path.join(
            "raw_results",
            "local_csvs",
            filename,
        ),
        index=False,
    )
    print(f"Data saved to {filename}")


def get_combination_name_column(df, groupparams):
    """Get "combination_name" column by concatenating all hyperparams minus
    highlighted_hyperparams. Use "ABBREV" dict to shorten the keys and
    values to form 'short' strings."""
    return df[[hp for hp in HYPERPARAMS if hp not in groupparams]].apply(
        lambda x: "  &  ".join([f"{ABBREVS.get(k, k)}={ABBREVS.get(v, v)}" for k, v in x.items()]),
        axis=1,
    )


def get_voter_name_column(df):
    """Get combined "voter_name" column by concatening the columns listed
    in GROUPPARAMS with " & " as separator"""
    return df[GROUPPARAMS].apply(lambda x: "  &  ".join(x), axis=1)


def draw_cd_diag_for_selected_runs(
    df_total_agg,
    df_seeds_agg,
    highlightparams,
    metric,
    filename_prefix,
    filename_suffix,
):

    # Get combination_name column
    df_total_agg["combination_name"] = get_combination_name_column(df_total_agg, [])
    # Same for df_seeds_agg
    df_seeds_agg["combination_name"] = get_combination_name_column(df_seeds_agg, [])

    # Filter
    df_seeds_agg_filtered = df_seeds_agg[
        df_seeds_agg["combination_name"].isin(df_total_agg["combination_name"])
    ].reset_index()

    # Add "WHAT?" column
    df_seeds_agg_filtered = df_seeds_agg_filtered.merge(
        df_total_agg[["combination_name", "WHAT?"]],
        on="combination_name",
        how="left",
    )

    # Remove "OVERALL BEST" rows
    df_seeds_agg_filtered = df_seeds_agg_filtered[df_seeds_agg_filtered["WHAT?"] != "OVERALL BEST"]

    df_seeds_agg_filtered["WHAT?"] = df_seeds_agg_filtered["WHAT?"].str.replace(
        r"[\\*]", "", regex=True
    )

    # Change some names for nicer display
    df_seeds_agg_filtered.replace("fps", "k-fps", inplace=True)
    df_seeds_agg_filtered.replace("kmedoids", "k-medoids", inplace=True)
    df_seeds_agg_filtered.replace("kmeans", "k-means", inplace=True)
    df_seeds_agg_filtered.replace("random", "k-random", inplace=True)
    df_seeds_agg_filtered.replace("cov_max", "k-cov-max", inplace=True)
    df_seeds_agg_filtered.replace("fps", "k-fps", inplace=True)
    df_seeds_agg_filtered.replace("quant", "quantile", inplace=True)
    df_seeds_agg_filtered.replace("l2", "L2", inplace=True)

    # Produce new combination_name column; simply highlight_hyperparam val (or
    #  "WHAT?", if it's not the default "BEST IN GROUP")
    df_seeds_agg_filtered["combination_name"] = df_seeds_agg_filtered[
        highlightparams + ["WHAT?"]
    ].apply(
        lambda x: ("  &  ".join([str(el) for el in x[:-1]]) if x[-1] == "BEST IN GROUP" else x[-1]),
        axis=1,
    )

    # Add k to combination_name if it's not 1
    # df_seeds_agg_filtered["combination_name"] = df_seeds_agg_filtered[
    #     ["combination_name", "k"]
    # ].apply(lambda x: f"{x[0]} ($k={x[1]}$)" if x[1] != 1 else x[0], axis=1)

    # For AGG highlighting, change mnn & m to m-nn
    # df_seeds_agg_filtered["combination_name"] = df_seeds_agg_filtered[
    #     ["combination_name", "m"]
    # ].apply(lambda x: f"{x[1]}-nn" if x[1] != -1 else "max", axis=1)

    # Get voter name
    df_seeds_agg_filtered["voter_name"] = get_voter_name_column(df_seeds_agg_filtered)

    savepath = os.path.join(
        "result_tables",
        f"{filename_prefix}_best_hps_{filename_suffix}.png",
    )

    draw_cd_diagram(
        df_perf=df_seeds_agg_filtered[["voter_name", "combination_name", metric]],
        metric=metric,
        title="Average rank",  # "Accuracy"
        labels=True,
        savepath=savepath,
    )
