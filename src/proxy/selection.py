"""
Proxy selection strategies for weight imprinting.

This module provides various algorithms for selecting representative samples (proxies)
from a set of feature embeddings. These proxies are used as class weights in
the weight imprinting framework, representing class prototypes.

Available methods include:
- Random selection
- Mean embedding
- l-means clustering
- l-medoids clustering
- Farthest point sampling
- Covariance-based methods
"""

import torch
from torch.nn.functional import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids


@torch.no_grad()
def select_proxies(
    data: torch.Tensor, method: str = "random", num_proxies: int = 1, seed: int = 42
):
    """
    Select representative samples (proxies) from data using various methods.

    This function provides multiple strategies for selecting representative samples
    from a set of embeddings. These proxies can then be used as class weights in
    weight imprinting models.

    Args:
        data: Tensor of feature embeddings (shape: [n_samples, embedding_dim])
        method: Selection method to use (see list of available methods below)
        num_proxies: Number of representatives to select (-1 means use all data,
                     that is, all samples are returned)
        seed: Random seed for reproducibility

    Returns:
        Tensor of selected representative embeddings

    Available methods:
        "none": Generate random embeddings using Xavier initialization
        "all": Use all samples as representatives
        "random": Randomly select l samples
        "mean": Use the mean embedding of all samples
        "lmeans": Use l-means clustering centroids
        "lmedoids": Use l-medoids clustering medoids
        "fps": Farthest point sampling
        "cov_max": Samples with highest covariance column sums
    """
    if method == "all" or num_proxies == -1 or len(data) < num_proxies:
        selected_data = data
    else:
        if method == "none":
            # Sample random weights as if initializing a layer in a neural network
            selected_data = torch.empty(num_proxies, data.shape[1])
            torch.nn.init.xavier_uniform_(selected_data)
        elif len(data) == 1:
            return data
        elif method == "random":
            indices = torch.randperm(len(data))[:num_proxies]
            selected_data = data[indices]
        elif method == "mean":  # Independent of num_proxies
            selected_data = data.mean(dim=0).unsqueeze(0)
        elif method == "lmeans":
            lmeans = KMeans(n_clusters=num_proxies, random_state=seed)
            lmeans.fit(data.to("cpu"))
            selected_data = torch.tensor(lmeans.cluster_centers_, dtype=torch.float32)
        elif method == "lmedoids":
            lmedoids = KMedoids(n_clusters=num_proxies, random_state=seed)
            lmedoids.fit(data.to("cpu"))
            selected_data = data[lmedoids.medoid_indices_]
        elif method == "fps":
            selected_data = farthest_point_sampling(data, num_proxies)
        elif method == "cov_max":
            selected_data = covariance_max_selection(data, num_proxies)
        else:
            raise ValueError(f"Unknown method: {method}")

    return selected_data


def farthest_point_sampling(data: torch.Tensor, l: int):
    """
    Select samples that are farthest from each other in the embedding space.

    This is a greedy algorithm that starts with a random point and iteratively
    selects the point that is farthest from the already selected set.

    Args:
        data: Tensor of feature embeddings
        l: Number of representatives to select

    Returns:
        Tensor of selected representative embeddings
    """
    selected_indices = [
        torch.randint(len(data), (1,)).item()
    ]  # Start with a random point

    for _ in range(1, l):
        remaining_indices = list(set(range(len(data))) - set(selected_indices))
        remaining_data = data[remaining_indices]
        selected_data = data[selected_indices]

        # Compute distances from remaining points to the selected set
        distances = torch.cdist(remaining_data, selected_data).min(dim=1).values
        next_index = remaining_indices[distances.argmax()]
        selected_indices.append(next_index)

    return data[selected_indices]


def covariance_max_selection(data: torch.Tensor, l: int):
    """
    Select samples with the highest column sums in the covariance matrix.

    This method selects points that have the highest covariance with other points,
    indicating they are most representative of the data distribution.

    Args:
        data: Tensor of (potentially normalized) feature embeddings
        l: Number of representatives to select

    Returns:
        Tensor of selected representative embeddings
    """
    # Compute the covariance matrix of the data
    cov_matrix = torch.cov(data)

    # Compute the sum of each column (or row) of the covariance matrix
    column_sums = torch.sum(cov_matrix, dim=0)

    # Select the indices corresponding to the k largest column sums
    top_l_indices = torch.argsort(column_sums, descending=True)[:l]

    # Select the samples corresponding to the top l indices
    selected_data = data[top_l_indices]

    return selected_data
