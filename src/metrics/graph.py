"""Graph-structural and topological signal processing metrics."""

from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

_AdjMatrix = np.ndarray | sp.spmatrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_networkx(adj: _AdjMatrix) -> nx.Graph:
    if sp.issparse(adj):
        return nx.from_scipy_sparse_array(adj)
    return nx.from_numpy_array(np.asarray(adj, dtype=float))


def _degrees(adj: _AdjMatrix) -> np.ndarray:
    if sp.issparse(adj):
        return np.asarray(adj.sum(axis=1)).ravel()
    return np.asarray(adj, dtype=float).sum(axis=1)


def _dist_matrix(adj: _AdjMatrix) -> np.ndarray:
    """All-pairs shortest-path distance matrix. Disconnected pairs yield inf."""
    A = adj if sp.issparse(adj) else sp.csr_matrix(np.asarray(adj, dtype=float))
    return csgraph.shortest_path(A, directed=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cycle_length(adj: _AdjMatrix) -> float:
    """Mean fundamental cycle length via BFS spanning tree.

    Each non-tree edge defines a fundamental cycle whose length equals the
    tree path between its endpoints plus one.  Returns 0.0 for acyclic graphs.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Mean length of fundamental cycles, or 0.0 if none exist.
    """
    G = _to_networkx(adj)
    tree = nx.minimum_spanning_tree(G)
    lengths = []
    for u, v in G.edges():
        if not tree.has_edge(u, v):
            try:
                lengths.append(nx.shortest_path_length(tree, u, v) + 1)
            except nx.NetworkXNoPath:
                pass
    return float(np.mean(lengths)) if lengths else 0.0


def number_of_cycles(adj: _AdjMatrix) -> int:
    """Cyclomatic number (circuit rank) of the graph.

    Defined as |E| - |V| + |C|, where |C| is the number of connected
    components.  This equals the dimension of the cycle space.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    int
        Number of independent cycles.
    """
    G = _to_networkx(adj)
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)


def square_clustering_coefficients(adj: _AdjMatrix) -> dict[int, float]:
    """Square clustering coefficient for each node.

    For node v, this is the fraction of paths of length 2 through v that
    close into a 4-cycle (square).

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    dict[int, float]
        Mapping node index → square clustering coefficient.
    """
    return nx.square_clustering(_to_networkx(adj))


def mean_square_clustering(adj: _AdjMatrix) -> float:
    """Mean square clustering coefficient across all nodes.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Mean square clustering coefficient in [0, 1].
    """
    return float(np.mean(list(square_clustering_coefficients(adj).values())))


def schultz_index(adj: _AdjMatrix) -> float:
    """Schultz molecular topological index.

    Defined as (1/2) * sum_{i,j} (deg_i + deg_j) * d(i,j), where d(i,j)
    is the shortest-path distance.  Disconnected pairs are excluded (d = inf).

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Schultz index value.
    """
    degs = _degrees(adj)
    dist = _dist_matrix(adj)
    connected = np.isfinite(dist)
    deg_sum = degs[:, None] + degs[None, :]
    return float(np.sum(np.where(connected, deg_sum * dist, 0.0))) / 2.0


def disorder_number(adj: _AdjMatrix) -> float:
    """Albertson irregularity index.

    Defined as sum_{(u,v) in E} |deg(u) - deg(v)|.  Zero for regular graphs.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Disorder (irregularity) value.
    """
    G = _to_networkx(adj)
    return float(sum(abs(G.degree(u) - G.degree(v)) for u, v in G.edges()))


def wiener_index(adj: _AdjMatrix) -> float:
    """Wiener index: sum of all pairwise shortest-path distances.

    Only connected pairs contribute.  For connected graphs this equals
    (1/2) * sum_{i,j} d(i,j).

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Wiener index value.
    """
    dist = _dist_matrix(adj)
    return float(np.nansum(np.where(np.isfinite(dist), dist, 0.0))) / 2.0


def girth(adj: _AdjMatrix) -> int | float:
    """Length of the shortest cycle in the graph.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    int or float
        Girth value, or ``float('inf')`` for acyclic graphs.
    """
    return nx.girth(_to_networkx(adj))


def eigengap(laplacian: _AdjMatrix, k: int = 1) -> float:
    """Gap between the k-th and (k-1)-th smallest Laplacian eigenvalues.

    With k=1 (default) returns λ₁ − λ₀.  Since λ₀ = 0 for any graph,
    this equals the Fiedler value and measures how strongly connected the
    graph is.

    Parameters
    ----------
    laplacian : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric positive semi-definite Laplacian matrix.
    k : int
        Position at which the gap is measured (1-indexed into sorted eigenvalues).

    Returns
    -------
    float
        Eigengap λ_k − λ_{k−1}.
    """
    n = laplacian.shape[0]
    k_req = min(k + 1, n - 1)

    if sp.issparse(laplacian):
        # which='SM' avoids shift-invert factorization of the singular Laplacian
        vals = eigsh(
            laplacian.tocsr(),
            k=k_req,
            which='SM',
            return_eigenvectors=False,
        )
    else:
        vals = eigh(
            np.asarray(laplacian, dtype=np.float64),
            eigvals_only=True,
            subset_by_index=[0, k_req - 1],
        )

    vals = np.sort(np.abs(vals))
    if len(vals) <= k:
        return float(vals[-1] - vals[-2]) if len(vals) >= 2 else 0.0
    return float(vals[k] - vals[k - 1])


def number_of_connected_components(adj: _AdjMatrix) -> int:
    """Number of connected components (0th Betti number β₀).

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    int
        Number of connected components.
    """
    A = adj if sp.issparse(adj) else sp.csr_matrix(np.asarray(adj, dtype=float))
    n_comp, _ = csgraph.connected_components(A, directed=False)
    return int(n_comp)


def graph_diameter(adj: _AdjMatrix) -> float:
    """Maximum shortest-path distance over all connected node pairs.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Graph diameter, or 0.0 if the graph has no edges.
    """
    dist = _dist_matrix(adj)
    finite = dist[np.isfinite(dist)]
    return float(finite.max()) if len(finite) > 0 else 0.0


def density(adj: _AdjMatrix) -> float:
    """Graph density: ratio of actual to maximum possible edges.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Density in [0, 1].
    """
    return nx.density(_to_networkx(adj))


def gutman_index(adj: _AdjMatrix) -> float:
    """Gutman index: (1/2) * sum_{i,j} deg_i * deg_j * d(i,j).

    Only connected pairs contribute.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Gutman index value.
    """
    degs = _degrees(adj)
    dist = _dist_matrix(adj)
    connected = np.isfinite(dist)
    deg_product = degs[:, None] * degs[None, :]
    return float(np.sum(np.where(connected, deg_product * dist, 0.0))) / 2.0


def degree_distribution(adj: _AdjMatrix) -> np.ndarray:
    """Normalized degree distribution (probability mass per integer degree).

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    np.ndarray, shape (max_degree + 1,)
        Probability of each degree value from 0 to max_degree.
    """
    degs = _degrees(adj).astype(int)
    counts = np.bincount(degs)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def compute_graph_metrics(
    adj: _AdjMatrix,
    laplacian: _AdjMatrix | None = None,
    eigengap_k: int = 1,
) -> dict[str, float | int | np.ndarray]:
    """Compute all graph-structural and TSP metrics.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.
    laplacian : np.ndarray or sp.spmatrix, shape (n, n), optional
        Graph Laplacian.  Required for the eigengap metric.
    eigengap_k : int
        Position at which to measure the eigengap (see :func:`eigengap`).

    Returns
    -------
    dict
        Mapping metric name → scalar or array value:

        - ``'cycle_length'``: mean fundamental cycle length
        - ``'number_of_cycles'``: cyclomatic number
        - ``'mean_square_clustering'``: mean square clustering coefficient
        - ``'schultz_index'``: Schultz molecular topological index
        - ``'disorder_number'``: Albertson irregularity index
        - ``'wiener_index'``: Wiener index
        - ``'girth'``: shortest cycle length
        - ``'eigengap'``: λ_k − λ_{k−1} (only if *laplacian* is provided)
        - ``'n_connected_components'``: number of connected components
        - ``'graph_diameter'``: maximum pairwise shortest-path distance
        - ``'density'``: edge density
        - ``'gutman_index'``: Gutman index
        - ``'degree_distribution'``: normalized degree histogram
    """
    metrics: dict = {
        'cycle_length': cycle_length(adj),
        'number_of_cycles': number_of_cycles(adj),
        'mean_square_clustering': mean_square_clustering(adj),
        'schultz_index': schultz_index(adj),
        'disorder_number': disorder_number(adj),
        'wiener_index': wiener_index(adj),
        'girth': girth(adj),
        'n_connected_components': number_of_connected_components(adj),
        'graph_diameter': graph_diameter(adj),
        'density': density(adj),
        'gutman_index': gutman_index(adj),
        'degree_distribution': degree_distribution(adj),
    }
    if laplacian is not None:
        metrics['eigengap'] = eigengap(laplacian, k=eigengap_k)
    return metrics


__all__ = [
    'cycle_length',
    'number_of_cycles',
    'square_clustering_coefficients',
    'mean_square_clustering',
    'schultz_index',
    'disorder_number',
    'wiener_index',
    'girth',
    'eigengap',
    'number_of_connected_components',
    'graph_diameter',
    'density',
    'gutman_index',
    'degree_distribution',
    'compute_graph_metrics',
]
