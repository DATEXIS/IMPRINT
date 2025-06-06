"""
Proxy selection strategies for weight imprinting.

This module provides various algorithms for selecting representative samples (proxies)
from a set of feature embeddings. These proxies are used as class weights in
the weight imprinting framework, representing class prototypes.

Available methods include:
- Random selection
- Mean embedding
- Least-Squares embeddings (as derived in [1])
- k-means clustering
- k-medoids clustering
- Farthest point sampling
- Covariance-based methods

[1] https://arxiv.org/abs/2503.06385
"""

import torch
from typing import Dict, Tuple
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


@torch.no_grad()
def select_proxies(
    data: torch.Tensor, method: str = "random", k: int = 1, seed: int = 42
):
    """
    Select representative samples (proxies) from data using various methods.

    This function provides multiple strategies for selecting representative samples
    from a set of embeddings for a fixed class. These proxies can then be used
    as class weights in weight imprinting models.

    Note that calculating Least-squares embeddings requires data from all classes,
    so it is not callable via this method. Instead, use the
    `compute_least_squares_weights` function directly as in `runner.py`.

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


def compute_least_squares_weights(
    class_data: Dict[int, torch.Tensor], lambda_reg: float = 0.05
):
    """
    Calculate the least-squares optimal weights for classification as derived
    in https://arxiv.org/abs/2503.06385.

    Using the formula:
        W_LS = (1/C)M^T (Σ_T + μ_G μ_G^T + λI)^(-1)

    Where:
    - M is the class-means matrix
    - Σ_T is the total covariance matrix
    - μ_G is the global mean
    - C is the number of classes
    - λ is the regularization parameter

    Args:
        class_data: Dict[int, Tensor] mapping class index -> embeddings ([n_i, d]).
        lambda_reg: Regularization parameter λ (default: 0.05)

    Returns:
        Dict[int, Tensor] mapping original class index -> Tensor of shape [k, d],
        where each row is the weight vector for one proxy.
    """
    # Extract dimensions
    n_classes = len(class_data)
    if n_classes == 0:
        raise ValueError("No class data provided")

    # Get embedding dimension from the first class
    first_class = next(iter(class_data.values()))
    embed_dim = first_class.shape[1]
    device = first_class.device

    # Calculate global mean μ_G
    all_samples_list = list(class_data.values())
    total_samples = sum(samples.shape[0] for samples in all_samples_list)

    # Weighted average for global mean
    mu_G = torch.zeros(embed_dim, device=device)
    for samples in all_samples_list:
        mu_G += samples.sum(dim=0)
    mu_G /= total_samples

    # Form the class-means matrix M
    M = torch.zeros((embed_dim, n_classes), device=device)

    for c, (class_idx, samples) in enumerate(class_data.items()):
        mu_c = torch.mean(samples, dim=0)
        M[:, c] = mu_c

    # Calculate total covariance matrix Σ_T
    # This can be memory intensive, so we'll calculate it incrementally
    sigma_T = torch.zeros((embed_dim, embed_dim), device=device)
    for samples in all_samples_list:
        centered_samples = samples - mu_G.unsqueeze(0)
        sigma_T += torch.mm(centered_samples.T, centered_samples)
    sigma_T /= total_samples

    # Calculate W_LS using formula (3) from the paper
    mu_G_outer = torch.outer(mu_G, mu_G)
    eye_matrix = torch.eye(embed_dim, device=device)
    inverse_term = torch.inverse(sigma_T + mu_G_outer + lambda_reg * eye_matrix)
    W_LS = torch.mm(M.T, inverse_term) * (1.0 / n_classes)

    # Format the result as a dictionary mapping class indices to weight vectors
    weights = {}
    for c, class_idx in enumerate(class_data.keys()):
        weights[class_idx] = W_LS[c, :].unsqueeze(0)  # Shape [1, embed_dim]

    return weights


def compute_prototype_least_squares_weights(
    class_data: Dict[int, torch.Tensor],
    k: int = 2,
    lambda_reg: float = 0.05,
    seed: int = 42,
) -> Dict[int, torch.Tensor]:
    """
    For each class in class_data, cluster its embeddings into k clusters,
    then treat each cluster as a separate “pseudo‐class” to compute its
    least-squares weight vector via compute_least_squares_weights. Finally,
    regroup weights by original class.

    Args:
        class_data: Dict[int, Tensor] mapping class index -> embeddings ([n_i, d]).
        k: Number of clusters (proxies) per original class.
        lambda_reg: Regularization parameter passed to compute_least_squares_weights.
        seed: Random seed for clustering.

    Returns:
        Dict[int, Tensor] mapping original class index -> Tensor of shape [k, d],
        where each row is the weight vector for one proxy.
    """
    if k == 1:
        return compute_least_squares_weights(class_data, lambda_reg=lambda_reg)

    # Step 1: Cluster each class’s embeddings and build a “flat” proxy->samples dict
    proxy_data: Dict[int, torch.Tensor] = {}
    proxy_to_orig: Dict[int, Tuple[int, int]] = {}
    next_proxy_idx = 0
    result: Dict[int, torch.Tensor] = {}
    d = next(iter(class_data.values())).shape[1]
    # Initialize a buffer of shape [k, d] per original class
    for orig_class in class_data.keys():
        if k != -1:
            result[orig_class] = torch.zeros(
                (k, d), device=next(iter(class_data.values())).device
            )
    already_set = []

    for orig_class, samples in class_data.items():
        n_samples, _ = samples.shape

        if k == -1 or n_samples < k:
            # Then, for this class, simply use all samples as proxies
            result[orig_class] = samples
            already_set.append(orig_class)
        else:
            km = KMeans(n_clusters=k, random_state=seed)
            km.fit(samples.to("cpu").numpy())
            labels = torch.tensor(km.labels_, device=samples.device)
            for proxy_id in range(k):
                mask = labels == proxy_id
                cluster_samples = samples[mask]
                proxy_data[next_proxy_idx] = cluster_samples
                proxy_to_orig[next_proxy_idx] = (orig_class, proxy_id)
                next_proxy_idx += 1

    if not proxy_data:
        raise ValueError("No proxy data generated")

    # Step 2: Compute least-squares weights for each proxy “class”
    # compute_least_squares_weights expects Dict[int, Tensor], so we can pass proxy_data directly.
    proxy_weights = compute_least_squares_weights(proxy_data, lambda_reg=lambda_reg)
    # proxy_weights: Dict[proxy_idx, Tensor] where each Tensor is [1, d]

    # Step 3: Regroup weights by the original class index
    for proxy_idx, weight_tensor in proxy_weights.items():
        orig_class, proxy_id = proxy_to_orig[proxy_idx]
        if orig_class in already_set:
            # If this class was already set, we can skip it
            continue
        # weight_tensor has shape [1, d], squeeze to [d]
        result[orig_class][proxy_id] = weight_tensor.squeeze(0)

    return result
