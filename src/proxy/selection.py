"""
Proxy selection strategies for weight imprinting.

This module provides various algorithms for selecting representative samples (proxies)
from a set of feature embeddings. These proxies are used as class weights in
the weight imprinting framework, representing class prototypes.

Available methods include:
- Random selection
- Mean embedding
- k-means clustering
- k-medoids clustering
- Farthest point sampling
- Covariance-based methods
"""

import torch
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


@torch.no_grad()
def select_proxies(
    data: torch.Tensor, method: str = "random", k: int = 1, seed: int = 42
):
    """
    Select representative samples (proxies) from data using various methods.

    This function provides multiple strategies for selecting representative samples
    from a set of embeddings. These proxies can then be used as class weights in
    weight imprinting models.

    Args:
        data: Tensor of feature embeddings (shape: [n_samples, embedding_dim])
        method: Selection method to use (see list of available methods below)
        k: Number of representatives to select (-1 means use all data, that is,
           all samples are returned)
        seed: Random seed for reproducibility

    Returns:
        Tensor of selected representative embeddings

    Available methods:
        "none": Generate random embeddings using Xavier initialization
        "all": Use all samples as representatives
        "random": Randomly select k samples
        "mean": Use the mean embedding of all samples
        "kmeans": Use k-means clustering centroids
        "kmedoids": Use k-medoids clustering medoids
        "fps": Farthest point sampling
        "cov_max": Samples with highest covariance column sums
    """
    if method == "all" or k == -1 or len(data) < k:
        selected_data = data
    else:
        if method == "none":
            # Sample random weights as if initializing a layer in a neural network
            selected_data = torch.empty(k, data.shape[1])
            torch.nn.init.xavier_uniform_(selected_data)
        elif len(data) == 1:
            return data
        elif method == "random":
            indices = torch.randperm(len(data))[:k]
            selected_data = data[indices]
        elif method == "mean":  # Independent of k
            selected_data = data.mean(dim=0).unsqueeze(0)
        elif method == "kmeans":
            kmeans = KMeans(n_clusters=k, random_state=seed)
            kmeans.fit(data.to("cpu"))
            selected_data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        elif method == "kmedoids":
            kmedoids = KMedoids(n_clusters=k, random_state=seed)
            kmedoids.fit(data.to("cpu"))
            selected_data = data[kmedoids.medoid_indices_]
        elif method == "fps":
            selected_data = farthest_point_sampling(data, k)
        elif method == "cov_max":
            selected_data = covariance_max_selection(data, k)
        else:
            raise ValueError(f"Unknown method: {method}")

    return selected_data


def farthest_point_sampling(data: torch.Tensor, k: int):
    """
    Select samples that are farthest from each other in the embedding space.

    This is a greedy algorithm that starts with a random point and iteratively
    selects the point that is farthest from the already selected set.

    Args:
        data: Tensor of feature embeddings
        k: Number of representatives to select

    Returns:
        Tensor of selected representative embeddings
    """
    selected_indices = [
        torch.randint(len(data), (1,)).item()
    ]  # Start with a random point

    for _ in range(1, k):
        remaining_indices = list(set(range(len(data))) - set(selected_indices))
        remaining_data = data[remaining_indices]
        selected_data = data[selected_indices]

        # Compute distances from remaining points to the selected set
        distances = torch.cdist(remaining_data, selected_data).min(dim=1).values
        next_index = remaining_indices[distances.argmax()]
        selected_indices.append(next_index)

    return data[selected_indices]


def covariance_max_selection(data: torch.Tensor, k: int):
    """
    Select samples with the highest column sums in the covariance matrix.

    This method selects points that have the highest covariance with other points,
    indicating they are most representative of the data distribution.

    Args:
        data: Tensor of (potentially normalized) feature embeddings
        k: Number of representatives to select

    Returns:
        Tensor of selected representative embeddings
    """
    # Compute the covariance matrix of the data
    cov_matrix = torch.cov(data)

    # Compute the sum of each column (or row) of the covariance matrix
    column_sums = torch.sum(cov_matrix, dim=0)

    # Select the indices corresponding to the k largest column sums
    top_k_indices = torch.argsort(column_sums, descending=True)[:k]

    # Select the samples corresponding to the top k indices
    selected_data = data[top_k_indices]

    return selected_data
