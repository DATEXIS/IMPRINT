"""
Experiment configuration generation.

This module provides functions for generating experiment configurations
for weight imprinting experiments. It handles parameter grid generation,
configuration filtering, and validation.
"""

import pandas as pd
from sklearn.model_selection import ParameterGrid


def generate_combinations(
    config,
    backbone_name="resnet18",
    dataset_name=["MNIST"],
    mapping_name="none",
    mapping={},
    task_name="short",
    task_splits=[[0, 1, 2]],
):
    """
    Generate experiment configurations from a base configuration.

    This function creates a grid of experiment configurations by expanding
    the parameters specified in the input configuration. It applies filtering
    to remove invalid or redundant configurations.

    Args:
        config (dict): Base configuration dictionary containing imprinting parameters
        backbone_name (str): Name of the backbone model to use (default: "resnet18")
        dataset_name (str): Name of the dataset to use (default: ["MNIST"])
        mapping_name (str): Name of the label mapping to use (default: "none")
        mapping (dict): Dictionary defining label mapping transformations (default: {})
        task_name (str): Name identifier for the task (default: "short")
        task_splits (list): List of task splits, where each split is a list of class indices (default: [[0, 1, 2]])

    Returns:
        list: List of experiment configuration dictionaries
    """
    # Extract parameters from config
    seeds = config.get("seeds", [17121997, 123987, 12412312])
    normalize_input_data = config.get("normalize_input_data", ["l2"])
    normalize_for_proxy_selection = config.get(
        "normalize_for_proxy_selection", ["none"]
    )
    presampling_methods = config.get("presampling_methods", ["all"])
    presampling_quantiles_values = config.get(
        "presampling_quantiles_values", [(0, 1.0)]
    )
    presampling_fewshot_values = config.get("presampling_fewshot_values", [-1])
    proxy_methods = config.get("proxy_methods", ["lmeans"])
    ks = config.get("ks", [20])
    normalize_weights = config.get("normalize_weights", ["l2"])
    aggregation_methods = config.get("aggregation_methods", ["max"])
    ms = config.get("ms", [-1])

    # Set up all the configurations
    base_configurations = list(
        ParameterGrid(
            {
                "backbone_name": [backbone_name],
                "dataset_name": [dataset_name],
                "mapping_name": [mapping_name],
                "mapping": [mapping],
                "task_name": [task_name],
                "task_splits": [task_splits],
                "normalize_input_data": normalize_input_data,
                "normalize_for_proxy_selection": normalize_for_proxy_selection,
                "presampling_method": presampling_methods,
                "presampling_quantiles_value": presampling_quantiles_values,
                "presampling_fewshot_value": presampling_fewshot_values,
                "proxy_method": proxy_methods,
                "k": ks,
                "normalize_weights": normalize_weights,
                "aggregation_method": aggregation_methods,
                "m": ms,
            }
        )
    )

    # Convert to DataFrame for easier filtering
    df = pd.DataFrame(base_configurations)

    print(
        "[INFO] Number of configurations per backbone, dataset, and task "
        "(disregarding seed) before filtering:",
        len(df),
    )

    # Apply filtering to remove redundant or invalid configurations
    filtered_df = filter_combinations(df)

    print(
        "[INFO] Number of configurations per backbone, dataset, mapping and "
        "task (disregarding seed) after filtering:",
        len(filtered_df),
    )

    # Expand the filtered DataFrame for each seed
    filtered_df_expanded = filtered_df.loc[
        filtered_df.index.repeat(len(seeds))
    ].reset_index(drop=True)

    # Add the `seed` column
    seeds_repeated = seeds * len(filtered_df)
    filtered_df_expanded.insert(
        len(filtered_df.columns), "seed", seeds_repeated[: len(filtered_df_expanded)]
    )

    # Sort columns in alphabetical order for consistent hashing
    filtered_df_expanded = filtered_df_expanded[
        sorted(filtered_df_expanded.columns, reverse=False)
    ]

    # Convert to list of dictionaries
    final_configurations = filtered_df_expanded.to_dict(orient="records")

    return final_configurations


def filter_combinations(df):
    """
    Filter experiment configurations to remove redundant or invalid combinations.

    This function applies a series of rules to filter out configurations that are
    redundant, invalid, or don't make sense for the weight imprinting framework.

    Args:
        df (DataFrame): DataFrame containing the configurations to filter

    Returns:
        DataFrame: Filtered configurations
    """
    # "max" aggregation is independent of m; the m is only for mNN
    df.loc[df["aggregation_method"] == "max", "m"] = -1

    # Remove rows where m is -1 for "mnn" aggregation
    df = df[~((df["aggregation_method"] == "mnn") & (df["m"] == -1))]

    # The m cannot be greater than the number of proxies
    df.loc[(df["m"] > df["k"]) & (df["k"] > 0), "m"] = df["k"]

    # "all" presampling is independent of quantile
    df.loc[df["presampling_method"] == "all", "presampling_quantiles_value"] = df.loc[
        df["presampling_method"] == "all"
    ].apply(lambda x: (0, 1.0), axis=1)

    # Presampling only makes sense if not everything is taken
    df = df[
        ~(
            (df["presampling_method"] == "weibull")
            & (df["presampling_quantiles_value"] == (0, 1.0))
        )
    ]

    # "mean" proxy method can only choose one proxy
    df.loc[df["proxy_method"].isin(["beta", "mean"]), "k"] = 1

    # Proxy method "all" means choosing all proxies
    df.loc[df["proxy_method"] == "all", "k"] = -1

    # Remove those rows where "normalize_for_proxy_selection" is not "none"
    #  and "normalize_input_data" is not the same as "normalize_for_proxy_selection"
    df = df[
        ~(
            (df["normalize_for_proxy_selection"] != "none")
            & (df["normalize_input_data"] != df["normalize_for_proxy_selection"])
        )
    ]

    # k can not be greater than selected few shots (or rather, if that happened,
    #  then these runs are the same)
    df.loc[
        (df["presampling_fewshot_value"] != -1)
        & (df["presampling_fewshot_value"] < df["k"]),
        "k",
    ] = df["presampling_fewshot_value"]

    # Remove duplicate configurations (after applying rules)
    df = df.loc[df.astype(str).drop_duplicates().index]

    return df
