"""
Data prefiltering utilities.

This module provides functions for filtering and sampling data prior to
proxy selection, including random sampling and distribution-based filtering.
"""

import torch
from scipy.stats import weibull_min


@torch.no_grad()
def prefilter_data(data, method="all", quantile=(0, 1), fewshot=-1):
    """
    Prefilter data to reduce the number of samples using various strategies.

    Filtering can be performed using statistical methods followed by optional
    random few-shot sampling.

    Args:
        data (torch.Tensor): Input data embeddings
        method (str): Filtering method, options:
            - 'all': No statistical prefiltering
            - 'weibull': Filter based on Weibull distribution
        quantile (tuple): Quantiles to use for prefiltering (min, max)
        fewshot (int): Number of samples to randomly select after prefiltering.
                      -1 means keep all samples.

    Returns:
        torch.Tensor: Filtered data
    """
    # Apply statistical prefiltering if specified
    if method == "all":
        filtered_data = data  # No filtering
    elif method == "weibull":
        filtered_data = weibull_sampling(data, quantile=quantile)
    else:
        raise ValueError(f"Unknown prefiltering method: {method}")

    # Apply random few-shot sampling if specified
    if fewshot != -1:
        # Ensure fewshot does not exceed the dataset size
        fewshot = min(fewshot, filtered_data.size(0))
        # Sample random few-shot samples by shuffling and selecting
        indices = torch.randperm(filtered_data.size(0))[:fewshot]
        return filtered_data[indices]
    else:
        return filtered_data


@torch.no_grad()
def weibull_sampling(data: torch.Tensor, quantile: tuple = (0.05, 0.95)):
    """
    Filter data to keep samples within specified Weibull distribution quantiles.

    This method fits a Weibull distribution to the distances of points from their
    mean, then filters to retain only points within specified quantiles. This is
    useful for removing outliers or focusing on specific regions of the embedding space.

    Args:
        data: Input data embeddings
        quantile: Lower and upper quantiles to retain (min, max)

    Returns:
        torch.Tensor: Filtered data within the specified quantiles
    """
    # Compute mean representation
    mean = torch.mean(data, dim=0)

    # Compute Euclidean distances from each point to the mean
    distances = torch.norm(data - mean, dim=1)

    # Fit a Weibull distribution to the distance data
    params = weibull_min.fit(distances.cpu().numpy())

    # Calculate the percentile thresholds according to the fitted distribution
    q_low = weibull_min.ppf(quantile[0], *params)
    q_high = weibull_min.ppf(quantile[1], *params)

    # Filter the data to keep values within the specified quantiles
    filtered_data = data[(distances >= q_low) & (distances <= q_high)]

    return filtered_data
