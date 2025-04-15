"""
Experiment configuration generation.

This module provides functions for generating experiment configurations
for weight imprinting experiments. It handles parameter grid generation,
configuration filtering, and validation.
"""

import pandas as pd
from sklearn.model_selection import ParameterGrid


def generate_combinations(config):
    """
    Generate experiment configurations from a base configuration.

    This function creates a grid of experiment configurations by expanding
    the parameters specified in the input configuration. It applies filtering
    to remove invalid or redundant configurations.

    Args:
        config (dict): Base configuration dictionary containing parameter ranges

    Returns:
        list: List of experiment configuration dictionaries
    """
    # Extract parameters from config
    seeds = config.get("seeds", [17121997, 123987, 12412312])
    normalize_input_data = config.get("normalize_input_data", ["l2"])
    normalize_for_proxy_selection = config.get(
        "normalize_for_proxy_selection", ["none"]
    )
    normalize_layer_activations = config.get("normalize_layer_activations", ["none"])
    presampling_methods = config.get("presampling_methods", ["all"])
    presampling_quantiles_values = config.get(
        "presampling_quantiles_values", [(0, 1.0)]
    )
    presampling_fewshot_values = config.get("presampling_fewshot_values", [-1])
    proxy_methods = config.get("proxy_methods", ["lmeans"])
    nums_proxies = config.get("nums_proxies", [20])
    normalize_weights = config.get("normalize_weights", ["l2"])
    aggregation_methods = config.get("aggregation_methods", ["max"])
    k_values = config.get("k_values", [-1])

    # Set up all the configurations
    base_configurations = list(
        ParameterGrid(
            {
                "normalize_input_data": normalize_input_data,
                "normalize_for_proxy_selection": normalize_for_proxy_selection,
                "normalize_layer_activations": normalize_layer_activations,
                "presampling_method": presampling_methods,
                "presampling_quantiles_value": presampling_quantiles_values,
                "presampling_fewshot_value": presampling_fewshot_values,
                "proxy_method": proxy_methods,
                "num_proxies": nums_proxies,
                "normalize_weights": normalize_weights,
                "aggregation_method": aggregation_methods,
                "k_value": k_values,
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
    filtered_df = filter_configurations(df)

    print(
        "[INFO] Number of configurations per backbone, dataset, and task "
        "(disregarding seed) after filtering:",
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


def filter_configurations(df):
    """
    Filter experiment configurations to remove redundant or invalid combinations.

    This function applies a series of rules to filter out configurations that are
    redundant, invalid, or don't make sense for the weight imprinting framework.

    Args:
        df (DataFrame): DataFrame containing the configurations to filter

    Returns:
        DataFrame: Filtered configurations
    """
    # "mean" and "max" aggregation is independent of k; the k is only for kNN
    df.loc[df["aggregation_method"].isin(["mean", "max"]), "k_value"] = -1

    # Remove rows where k is -1 for "knn" aggregation
    df = df[~((df["aggregation_method"] == "knn") & (df["k_value"] == -1))]

    # "knn" aggregation is independent of layer activation normalization
    df = df[
        ~(
            (df["aggregation_method"] == "knn")
            & (df["normalize_layer_activations"] != "none")
        )
    ]

    # The k_value cannot be greater than the number of proxies
    df.loc[
        (df["k_value"] > df["num_proxies"]) & (df["num_proxies"] > 0), "k_value"
    ] = df["num_proxies"]

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
    df.loc[df["proxy_method"].isin(["beta", "mean"]), "num_proxies"] = 1

    # Proxy method "all" means choosing all proxies
    df.loc[df["proxy_method"] == "all", "num_proxies"] = -1

    # Remove those rows where "normalize_for_proxy_selection" is not "none"
    #  and "normalize_input_data" is not the same as "normalize_for_proxy_selection"
    df = df[
        ~(
            (df["normalize_for_proxy_selection"] != "none")
            & (df["normalize_input_data"] != df["normalize_for_proxy_selection"])
        )
    ]

    # Remove duplicate configurations (after applying rules)
    df = df.loc[df.astype(str).drop_duplicates().index]

    return df
