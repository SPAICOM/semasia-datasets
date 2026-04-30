"""KNN graph construction and spectral clustering for point clouds."""

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .laplace import LaplacianType, compute_eigenvectors, compute_laplacian


def build_knn_graph(
    point_cloud: np.ndarray | torch.Tensor,
    k: int,
    metric: str = 'euclidean',
    weighted: bool = False,
    mutual: bool = False,
) -> sp.csr_matrix:
    """Build an undirected k-nearest-neighbour graph from a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray or torch.Tensor, shape (n_points, n_features)
        Input point cloud.
    k : int
        Number of nearest neighbours per point (self excluded).
    metric : str
        Distance metric accepted by
        :class:`sklearn.neighbors.NearestNeighbors`.
    weighted : bool
        If True, edge weights are Gaussian-kernel similarities
        ``exp(−d² / (2σ²))`` where σ is the mean distance to the k-th
        nearest neighbour.  If False, all edges have weight 1.
    mutual : bool
        If True, keep only edges where both endpoints are mutual
        nearest neighbours (intersection / mutual KNN).
        If False, symmetrize by union (maximum weight per edge pair).

    Returns
    -------
    sp.csr_matrix, shape (n_points, n_points)
        Symmetric sparse adjacency matrix of the KNN graph.
    """
    if isinstance(point_cloud, torch.Tensor):
        X = point_cloud.detach().cpu().float().numpy()
    else:
        X = np.asarray(point_cloud, dtype=np.float32)

    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # drop self (nearest neighbour at distance 0)
    distances = distances[:, 1:]  # (n, k)
    indices = indices[:, 1:]  # (n, k)

    row = np.repeat(np.arange(n), k)
    col = indices.ravel()

    if weighted:
        sigma = float(distances[:, -1].mean()) or 1.0
        data = np.exp(-(distances.ravel() ** 2) / (2.0 * sigma**2)).astype(np.float32)
    else:
        data = np.ones(n * k, dtype=np.float32)

    A = sp.csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)

    if mutual:
        # retain only edges present in both directions
        mask = (A > 0).multiply(A.T > 0)
        if weighted:
            A = (A.multiply(mask) + A.T.multiply(mask)) / 2.0
        else:
            A = mask.astype(np.float32)
    else:
        # union: keep the larger weight when both directions exist
        A = A.maximum(A.T)

    return A.tocsr()


def spectral_clustering(
    point_cloud: np.ndarray | torch.Tensor,
    n_clusters: int,
    k_neighbors: int,
    normalization: LaplacianType = 'symmetric',
    metric: str = 'euclidean',
    weighted: bool = False,
    mutual: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Spectral clustering of a point cloud via graph Laplacian eigenvectors.

    Implements the Ng-Jordan-Weiss algorithm:

    1. Build a KNN graph from *point_cloud*.
    2. Compute the graph Laplacian with the requested normalisation.
    3. Extract the *n_clusters* eigenvectors for the smallest eigenvalues.
    4. Row-normalise the eigenvector matrix to unit length.
    5. Apply k-means to the normalised rows.

    Parameters
    ----------
    point_cloud : np.ndarray or torch.Tensor, shape (n_points, n_features)
        Input point cloud.
    n_clusters : int
        Number of clusters.
    k_neighbors : int
        Number of nearest neighbours for the KNN graph.
    normalization : {'unnormalized', 'symmetric', 'random_walk'}
        Laplacian normalisation.  Defaults to ``'symmetric'``, which
        corresponds to the standard normalised spectral clustering.
    metric : str
        Distance metric for the KNN graph.
    weighted : bool
        Use Gaussian-kernel edge weights.
    mutual : bool
        Use mutual (intersection) KNN instead of union KNN.
    seed : int
        Random seed for KMeans.

    Returns
    -------
    np.ndarray, shape (n_points,)
        Integer cluster labels in ``[0, n_clusters)``.
    """
    adjacency = build_knn_graph(
        point_cloud,
        k=k_neighbors,
        metric=metric,
        weighted=weighted,
        mutual=mutual,
    )
    laplacian = compute_laplacian(adjacency, normalization=normalization)
    _, eigvecs = compute_eigenvectors(laplacian, k=n_clusters)

    # row-normalise (Ng-Jordan-Weiss)
    norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = eigvecs / norms

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
    return kmeans.fit_predict(U)


__all__ = ['build_knn_graph', 'spectral_clustering']
