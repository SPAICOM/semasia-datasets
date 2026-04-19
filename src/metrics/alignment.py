"""Metrics for comparing latent spaces and aligning prototypes."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

MetricName = Literal[
    'euclidean',
    'cosine',
    'mahalanobis',
    'wasserstein',
    'sinkhorn',
    'procrustes',
    'chamfer',
    'hausdorff',
]


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between point clouds.

    Computes mean distance from points in a to nearest in b and vice versa.
    Memory-efficient: O(n*m) but avoids full matrix for large n,m by sampling.
    """
    if a.size == 0 or b.size == 0:
        raise ValueError('Input arrays cannot be empty')

    max_points = 5000
    n_a, n_b = a.shape[0], b.shape[0]

    if n_a > max_points or n_b > max_points:
        rng = np.random.default_rng(42)
        a_sample = a[rng.choice(n_a, min(max_points, n_a), replace=False)]
        b_sample = b[rng.choice(n_b, min(max_points, n_b), replace=False)]
        dist = cdist(a_sample, b_sample, metric='euclidean')
    else:
        dist = cdist(a, b, metric='euclidean')

    result = np.nanmean(dist)
    if np.isnan(result):
        return 0.0 if np.array_equal(a, b) else float(np.inf)
    return float(result)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity) between point clouds."""
    if a.size == 0 or b.size == 0:
        raise ValueError('Input arrays cannot be empty')

    max_points = 5000
    n_a, n_b = a.shape[0], b.shape[0]
    if n_a > max_points or n_b > max_points:
        rng = np.random.default_rng(42)
        a_sample = a[rng.choice(n_a, min(max_points, n_a), replace=False)]
        b_sample = b[rng.choice(n_b, min(max_points, n_b), replace=False)]
        dist = cdist(a_sample, b_sample, metric='cosine')
    else:
        dist = cdist(a, b, metric='cosine')

    result = np.nanmean(dist)
    if np.isnan(result):
        return 0.0 if np.array_equal(a, b) else 1.0
    return float(result)


def mahalanobis_distance(
    a: np.ndarray,
    b: np.ndarray,
    cov: np.ndarray | None = None,
) -> float:
    """Mahalanobis distance between point clouds."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if cov is None:
        combined = np.vstack([a, b])
        cov = np.cov(combined, rowvar=False)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    centroids_a = a.mean(axis=0)
    centroids_b = b.mean(axis=0)
    diff = centroids_a - centroids_b

    return float(np.sqrt(diff @ cov_inv @ diff))


def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein distance (Earth Mover's Distance) between sorted 1D arrays."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    return float(np.mean(np.abs(a_sorted - b_sorted)))


def sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.1,
    max_iter: int = 1000,
) -> float:
    """Sinkhorn distance (entropic regularized optimal transport)."""
    max_points = 2000
    n_a, n_b = a.shape[0], b.shape[0]

    if n_a > max_points or n_b > max_points:
        rng = np.random.default_rng(42)
        a = a[rng.choice(n_a, min(max_points, n_a), replace=False)]
        b = b[rng.choice(n_b, min(max_points, n_b), replace=False)]

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    n, m = a.shape[0], b.shape[0]
    cost = cdist(a, b, metric='euclidean')

    a_normalized = np.ones(n) / n
    b_normalized = np.ones(m) / m

    K = np.exp(-cost / epsilon)
    u = np.ones(n)

    for _ in range(max_iter):
        v = b_normalized / (K.T @ u + 1e-10)
        u = a_normalized / (K @ v + 1e-10)
        if np.allclose(u * (K @ v), a_normalized, rtol=1e-5):
            break

    Transport = np.diag(u) @ K @ np.diag(v)
    return float(np.sum(cost * Transport))


def procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Procrustes distance between point clouds."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    n_a, n_b = a.shape[0], b.shape[0]

    if n_a < n_b:
        a = np.pad(a, ((0, n_b - n_a), (0, 0)), mode='edge')
    elif n_b < n_a:
        b = np.pad(b, ((0, n_a - n_b), (0, 0)), mode='edge')

    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    aligned_a = a @ R
    diff = aligned_a - b
    return float(np.mean(np.sum(diff**2, axis=1)))


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chamfer distance between point clouds."""
    max_points = 5000
    n_a, n_b = a.shape[0], b.shape[0]

    if n_a > max_points or n_b > max_points:
        rng = np.random.default_rng(42)
        a_sample = a[rng.choice(n_a, min(max_points, n_a), replace=False)]
        b_sample = b[rng.choice(n_b, min(max_points, n_b), replace=False)]
        a, b = a_sample, b_sample

    dist_ab = cdist(a, b, metric='euclidean')
    dist_ba = cdist(b, a, metric='euclidean')

    forward = np.mean(np.min(dist_ab, axis=1))
    backward = np.mean(np.min(dist_ba, axis=1))

    return float(forward + backward)


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hausdorff distance between point clouds."""
    max_points = 5000
    n_a, n_b = a.shape[0], b.shape[0]

    if n_a > max_points or n_b > max_points:
        rng = np.random.default_rng(42)
        a_sample = a[rng.choice(n_a, min(max_points, n_a), replace=False)]
        b_sample = b[rng.choice(n_b, min(max_points, n_b), replace=False)]
        a, b = a_sample, b_sample

    dist_ab = cdist(a, b, metric='euclidean')
    dist_ba = cdist(b, a, metric='euclidean')

    forward = np.max(np.min(dist_ab, axis=1))
    backward = np.max(np.min(dist_ba, axis=1))

    return float(max(forward, backward))


def compute_metric(
    a: np.ndarray,
    b: np.ndarray,
    method: MetricName | str,
    **kwargs,
) -> float:
    """Compute distance/similarity metric between two point clouds."""
    match method:
        case 'euclidean':
            return euclidean_distance(a, b)
        case 'cosine':
            return cosine_distance(a, b)
        case 'mahalanobis':
            return mahalanobis_distance(a, b, cov=kwargs.get('cov'))
        case 'wasserstein':
            return wasserstein_distance(a, b)
        case 'sinkhorn':
            return sinkhorn_distance(a, b, epsilon=kwargs.get('epsilon', 0.1))
        case 'procrustes':
            return procrustes_distance(a, b)
        case 'chamfer':
            return chamfer_distance(a, b)
        case 'hausdorff':
            return hausdorff_distance(a, b)
        case _:
            raise ValueError(f'Unknown metric: {method!r}')


METRIC_NAMES = [
    'euclidean',
    'cosine',
    'mahalanobis',
    'wasserstein',
    'sinkhorn',
    'procrustes',
    'chamfer',
    'hausdorff',
]


def jaccard_prototype_similarity(
    cluster_indices_a: dict[int, np.ndarray],
    cluster_indices_b: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute Jaccard similarity matrix between prototypes."""
    n_proto_a = len(cluster_indices_a)
    n_proto_b = len(cluster_indices_b)
    similarity_matrix = np.empty((n_proto_a, n_proto_b), dtype=np.float32)

    for i in range(n_proto_a):
        set_a = set(cluster_indices_a[i])
        for j in range(n_proto_b):
            set_b = set(cluster_indices_b[j])
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            if union == 0:
                similarity_matrix[i, j] = 0.0
            else:
                similarity_matrix[i, j] = intersection / union

    return similarity_matrix


def compute_jaccard_metrics(
    cluster_indices_a: dict[int, np.ndarray],
    cluster_indices_b: dict[int, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Jaccard-based metrics for prototype similarity."""
    sim_matrix = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)

    best_matches_a = sim_matrix.max(axis=1)

    jaccard_mean = float(np.mean(best_matches_a))

    n_good_matches = np.sum(best_matches_a >= threshold)
    jaccard_good_match_ratio = n_good_matches / len(best_matches_a)

    return {
        'jaccard_mean': jaccard_mean,
        'jaccard_good_match_ratio': float(jaccard_good_match_ratio),
    }


def align_prototypes(
    cluster_indices_a: dict[int, np.ndarray],
    cluster_indices_b: dict[int, np.ndarray],
) -> np.ndarray:
    """Find the permutation that aligns model B's prototypes to model A's."""
    from scipy.optimize import linear_sum_assignment

    sim_matrix = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)
    _, col_ind = linear_sum_assignment(-sim_matrix)
    return col_ind
