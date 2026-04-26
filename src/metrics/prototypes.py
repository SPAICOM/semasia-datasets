"""Prototype-based metrics for comparing latent spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MatchGroup:
    """A group of matched prototype clusters from model A and model B."""

    a_clusters: list[int]
    b_clusters: list[int]
    score: float = field(default=0.0)


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def jaccard_matrix(
    clusters_A: list[set],
    clusters_B: list[set],
) -> NDArray[np.float32]:
    """Build k×k Jaccard similarity matrix between two clusterings.

    Parameters
    ----------
    clusters_A : list[set]
        List of sets of point indices for clustering A.
    clusters_B : list[set]
        List of sets of point indices for clustering B.

    Returns
    -------
    np.ndarray, shape (k, k)
        Jaccard similarity matrix where J[i, j] = Jaccard(clusters_A[i], clusters_B[j]).
    """
    k = len(clusters_A)
    m = len(clusters_B)
    J = np.empty((k, m), dtype=np.float32)
    for i, a in enumerate(clusters_A):
        for j, b in enumerate(clusters_B):
            J[i, j] = jaccard(a, b)
    return J


def correspondence_score(J: NDArray[np.float32]) -> float:
    """Hungarian-matched mean Jaccard.

    Solves the optimal assignment problem using the Hungarian algorithm
    and returns the mean of the matched Jaccard similarities.

    Parameters
    ----------
    J : np.ndarray, shape (k, k)
        Jaccard similarity matrix.

    Returns
    -------
    float
        Mean Jaccard score after optimal Hungarian matching.
        1.0 = point clouds identical at this k, 0.0 = no correspondence.
    """
    row_ind, col_ind = linear_sum_assignment(-J)
    return float(J[row_ind, col_ind].mean())


def precision_from_jaccard(
    J: NDArray[np.float32],
    threshold: float = 0.5,
) -> float:
    """Compute precision from Jaccard similarity matrix.

    Precision = fraction of model A prototypes that have a good match in model B.

    Parameters
    ----------
    J : np.ndarray, shape (k, k)
        Jaccard similarity matrix.
    threshold : float
        Threshold for considering a match "good". Default 0.5.

    Returns
    -------
    float
        Precision value between 0 and 1.
    """
    best_matches = J.max(axis=1)
    good_matches = np.sum(best_matches >= threshold)
    return float(good_matches / len(best_matches))


def recall_from_jaccard(
    J: NDArray[np.float32],
    threshold: float = 0.5,
) -> float:
    """Compute recall from Jaccard similarity matrix.

    Recall = fraction of model B prototypes that have a good match in model A.

    Parameters
    ----------
    J : np.ndarray, shape (k, k)
        Jaccard similarity matrix.
    threshold : float
        Threshold for considering a match "good". Default 0.5.

    Returns
    -------
    float
        Recall value between 0 and 1.
    """
    best_matches = J.max(axis=0)
    good_matches = np.sum(best_matches >= threshold)
    return float(good_matches / len(best_matches))


def f1_from_jaccard(
    J: NDArray[np.float32],
    threshold: float = 0.5,
) -> float:
    """Compute F1 score from Jaccard similarity matrix.

    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    J : np.ndarray, shape (k, k)
        Jaccard similarity matrix.
    threshold : float
        Threshold for considering a match "good". Default 0.5.

    Returns
    -------
    float
        F1 score between 0 and 1.
    """
    precision = precision_from_jaccard(J, threshold)
    recall = recall_from_jaccard(J, threshold)
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def entropy_from_jaccard(J: NDArray[np.float32]) -> float:
    """Compute entropy from Jaccard similarity distribution.

    Treats each row of the Jaccard matrix as a probability distribution
    (after normalization) and computes Shannon entropy.

    Parameters
    ----------
    J : np.ndarray, shape (k, k)
        Jaccard similarity matrix.

    Returns
    -------
    float
        Entropy value. Higher = more uncertainty/uniform distribution.
        Lower = more concentrated/structured.
    """
    J_norm = J / (J.sum(axis=1, keepdims=True) + 1e-10)
    entropy = -np.sum(J_norm * np.log(J_norm + 1e-10), axis=1)
    return float(entropy.mean())


def hungarian_score(J: NDArray[np.float32]) -> float:
    """Alias for correspondence_score for clarity."""
    return correspondence_score(J)


def compute_all_metrics(
    J: NDArray[np.float32],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all prototype similarity metrics.

    Parameters
    ----------
    J : np.ndarray, shape (k, k)
        Jaccard similarity matrix.
    threshold : float
        Threshold for precision/recall. Default 0.5.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: precision, recall, f1, hungarian, entropy.
    """
    return {
        'precision': precision_from_jaccard(J, threshold),
        'recall': recall_from_jaccard(J, threshold),
        'f1': f1_from_jaccard(J, threshold),
        'hungarian': hungarian_score(J),
        'entropy': entropy_from_jaccard(J),
    }


def _hungarian_matching(J: NDArray[np.float32]) -> list[MatchGroup]:
    row_ind, col_ind = linear_sum_assignment(-J)
    groups = [
        MatchGroup(a_clusters=[int(i)], b_clusters=[int(j)], score=float(J[i, j]))
        for i, j in zip(row_ind.tolist(), col_ind.tolist())
    ]
    return sorted(groups, key=lambda g: g.score, reverse=True)


def _connected_components_matching(
    J: NDArray[np.float32], threshold: float
) -> list[MatchGroup]:
    import networkx as nx

    k_a, k_b = J.shape
    G = nx.Graph()
    for i in range(k_a):
        G.add_node(i, side='a')
    for j in range(k_b):
        G.add_node(k_a + j, side='b')
    for i in range(k_a):
        for j in range(k_b):
            w = float(J[i, j])
            if w > threshold:
                G.add_edge(i, k_a + j, weight=w)

    groups = []
    for comp in nx.connected_components(G):
        a_ids = sorted(n for n in comp if G.nodes[n]['side'] == 'a')
        b_ids = sorted(n - k_a for n in comp if G.nodes[n]['side'] == 'b')
        if a_ids and b_ids:
            score = float(np.mean([J[i, j] for i in a_ids for j in b_ids]))
            groups.append(MatchGroup(a_clusters=a_ids, b_clusters=b_ids, score=score))
    return sorted(groups, key=lambda g: g.score, reverse=True)


def _spectral_matching(J: NDArray[np.float32]) -> list[MatchGroup]:
    from sklearn.cluster import KMeans

    k_a, k_b = J.shape
    n = k_a + k_b

    A = np.zeros((n, n), dtype=float)
    A[:k_a, k_a:] = J
    A[k_a:, :k_a] = J.T

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L_sym = np.eye(n) - (deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :])

    max_k = min(k_a, k_b)
    eigvals = np.sort(np.linalg.eigvalsh(L_sym))
    gaps = eigvals[2 : max_k + 2] - eigvals[1 : max_k + 1]
    k_est = max(2, min(int(np.argmax(gaps)) + 2, max_k))

    _, eigvecs = np.linalg.eigh(L_sym)
    X = eigvecs[:, 1 : k_est + 1]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(norms > 0, X / norms, X)

    labels = KMeans(n_clusters=k_est, random_state=42, n_init=10).fit_predict(X)

    groups = []
    for c in range(k_est):
        mask = labels == c
        a_ids = sorted(i for i in range(k_a) if mask[i])
        b_ids = sorted(j for j in range(k_b) if mask[k_a + j])
        if a_ids and b_ids:
            score = float(np.mean([J[i, j] for i in a_ids for j in b_ids]))
            groups.append(MatchGroup(a_clusters=a_ids, b_clusters=b_ids, score=score))
    return sorted(groups, key=lambda g: g.score, reverse=True)


def _leiden_matching(J: NDArray[np.float32], threshold: float) -> list[MatchGroup]:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError(
            'Leiden matching requires igraph and leidenalg: '
            'pip install igraph leidenalg'
        ) from exc

    k_a, k_b = J.shape
    edges, weights = [], []
    for i in range(k_a):
        for j in range(k_b):
            w = float(J[i, j])
            if w > threshold:
                edges.append((i, k_a + j))
                weights.append(w)

    G = ig.Graph(n=k_a + k_b, edges=edges, directed=False)
    G.es['weight'] = weights
    partition = leidenalg.find_partition(
        G, leidenalg.ModularityVertexPartition, weights='weight'
    )

    groups = []
    for community in partition:
        a_ids = sorted(v for v in community if v < k_a)
        b_ids = sorted(v - k_a for v in community if v >= k_a)
        if a_ids and b_ids:
            score = float(np.mean([J[i, j] for i in a_ids for j in b_ids]))
            groups.append(MatchGroup(a_clusters=a_ids, b_clusters=b_ids, score=score))
    return sorted(groups, key=lambda g: g.score, reverse=True)


def foscttm_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of samples closer than the true match (FOSCTTM), A→B direction.

    Assumes a[i] is the true correspondent of b[i].
    Lower is better: 0.0 = perfect alignment, ~0.5 = random.

    Parameters
    ----------
    a, b : np.ndarray, shape (n, d)
        Paired point clouds in the same embedding space.
    """
    from scipy.spatial.distance import cdist

    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {a.shape} vs {b.shape}')

    n = a.shape[0]
    dist = cdist(a, b, metric='euclidean')
    sorted_indices = np.argsort(dist, axis=1)
    ranks = np.where(sorted_indices == np.arange(n)[:, None])[1]
    return float(np.mean(ranks / (n - 1)))


def foscttm_score(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """FOSCTTM in both directions and their mean.

    Returns
    -------
    dict with keys 'a2b', 'b2a', 'mean'.
    """
    a2b = foscttm_distance(a, b)
    b2a = foscttm_distance(b, a)
    return {'a2b': a2b, 'b2a': b2a, 'mean': (a2b + b2a) / 2.0}


def compute_matching(
    J: NDArray[np.float32],
    method: str = 'hungarian',
    threshold: float = 0.0,
) -> list[MatchGroup]:
    """Match prototype clusters between model A and model B.

    Parameters
    ----------
    J : np.ndarray, shape (k_a, k_b)
        Jaccard similarity matrix.
    method : str
        One of 'hungarian', 'connected_components', 'spectral', 'leiden'.
    threshold : float
        Edge weight threshold for graph-based methods (connected_components, leiden).
        Ignored by hungarian and spectral.

    Returns
    -------
    list[MatchGroup]
        Groups sorted by score descending.
    """
    if method == 'hungarian':
        return _hungarian_matching(J)
    if method == 'connected_components':
        return _connected_components_matching(J, threshold)
    if method == 'spectral':
        return _spectral_matching(J)
    if method == 'leiden':
        return _leiden_matching(J, threshold)
    raise ValueError(f'Unknown matching method: {method!r}')


__all__ = [
    'MatchGroup',
    'jaccard',
    'jaccard_matrix',
    'correspondence_score',
    'precision_from_jaccard',
    'recall_from_jaccard',
    'f1_from_jaccard',
    'entropy_from_jaccard',
    'hungarian_score',
    'compute_all_metrics',
    'compute_matching',
    'foscttm_distance',
    'foscttm_score',
]
