"""Graph-structural and topological signal processing metrics."""

from __future__ import annotations

import networkx as nx
import numpy as np
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


def _csr(adj: _AdjMatrix) -> sp.csr_matrix:
    return (
        adj.tocsr() if sp.issparse(adj) else sp.csr_matrix(np.asarray(adj, dtype=float))
    )


def _dist_matrix(adj: _AdjMatrix) -> np.ndarray:
    """All-pairs shortest-path distance matrix. Disconnected pairs yield inf."""
    return csgraph.shortest_path(_csr(adj), directed=False)


def _chunk_distance_stats(
    A: sp.csr_matrix, idx: np.ndarray, unweighted: bool
) -> tuple[float, float]:
    d = csgraph.dijkstra(A, directed=False, indices=idx, unweighted=unweighted)
    d = d.astype(np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return 0.0, 0.0
    return float(d[finite].sum(dtype=np.float64)), float(d[finite].max())


def _distance_stats(
    adj: _AdjMatrix, chunk: int = 500, n_jobs: int = 1
) -> tuple[float, float]:
    """Wiener index and diameter over connected pairs, without a dense n x n matrix.

    Shortest paths are computed from ``chunk`` source nodes at a time and stored
    as float32, so peak memory is ``chunk * n * 4`` bytes per worker.

    Parameters
    ----------
    chunk : int
        Number of source nodes per block.
    n_jobs : int
        Number of worker processes (``-1`` for all cores); the blocks are
        independent.

    Returns
    -------
    (wiener, diameter) : tuple[float, float]
        ``wiener`` is the sum of d(u, v) over unordered connected pairs; ``diameter``
        is the largest finite distance (0.0 if the graph has no edges).
    """
    A = _csr(adj)
    n = A.shape[0]
    unweighted = A.nnz == 0 or bool(np.all(A.data == 1))
    blocks = [np.arange(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    if n_jobs == 1 or len(blocks) == 1:
        parts = [_chunk_distance_stats(A, b, unweighted) for b in blocks]
    else:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=n_jobs)(
            delayed(_chunk_distance_stats)(A, b, unweighted) for b in blocks
        )
    total = sum(p[0] for p in parts)
    diam = max((p[1] for p in parts), default=0.0)
    return total / 2.0, diam


def largest_component(adj: _AdjMatrix) -> tuple[sp.csr_matrix, np.ndarray, float]:
    """Subgraph induced by the largest connected component.

    Ties are broken towards the component with the smallest label.

    Returns
    -------
    sub_adj : sp.csr_matrix
        Adjacency matrix of the component.
    nodes : np.ndarray
        Indices (into ``adj``) of the nodes of the component, ascending.
    share : float
        Fraction of the nodes of ``adj`` that belong to the component.
    """
    A = _csr(adj)
    n_comp, labels = csgraph.connected_components(A, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    nodes = np.flatnonzero(labels == int(np.argmax(sizes)))
    return A[nodes][:, nodes].tocsr(), nodes, float(len(nodes)) / A.shape[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cycle_length(adj: _AdjMatrix) -> float:
    """Mean fundamental cycle length over a BFS spanning forest.

    A breadth-first spanning tree is built in every connected component, rooted
    at its highest-degree node (ties: larger neighbour-degree sum, then smaller
    index) and visiting neighbours in index order.  Rooting at a hub keeps the
    value stable when the nodes are relabelled; rooting at the smallest index
    makes it depend on an arbitrary choice (about 10x more variable on kNN graphs).
    Each non-tree edge (u, v) closes a fundamental cycle of length
    ``d_T(u, v) + 1``, where ``d_T`` is the distance in the tree.  The value is
    the mean over all non-tree edges of the graph.  Returns 0.0 for forests.

    The value is deterministic for a given adjacency matrix.  A spanning tree is
    not label-free, so under a node permutation it can still change slightly
    (ties between equally good roots and between equidistant parents); it is
    exactly invariant on vertex-transitive graphs.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.

    Returns
    -------
    float
        Mean length of fundamental cycles, or 0.0 if none exist.
    """
    A = _csr(adj)
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    n = A.shape[0]
    n_comp, labels = csgraph.connected_components(A, directed=False)

    deg = np.diff(A.indptr).astype(np.int64)
    nb_deg = A @ deg
    parent = np.full(n, -1, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for c in range(n_comp):
        members = np.flatnonzero(labels == c)
        # highest degree, then highest neighbour-degree sum, then smallest index
        root = int(members[np.lexsort((members, -nb_deg[members], -deg[members]))[0]])
        order, pred = csgraph.breadth_first_order(
            A, root, directed=False, return_predecessors=True
        )
        parent[order[1:]] = pred[order[1:]]
        for v in order[1:]:  # BFS order: parents are processed before children
            depth[v] = depth[pred[v]] + 1

    coo = sp.triu(A, k=1).tocoo()
    u, v = coo.row.astype(np.int64), coo.col.astype(np.int64)
    in_tree = (parent[u] == v) | (parent[v] == u)
    u, v = u[~in_tree], v[~in_tree]
    if len(u) == 0:
        return 0.0

    # Tree distance via lowest common ancestor: lift the deeper endpoint first.
    a, b = u.copy(), v.copy()
    dist = np.zeros(len(u), dtype=np.int64)
    while True:
        active = a != b
        if not active.any():
            break
        lift_a = active & (depth[a] >= depth[b])
        lift_b = active & ~lift_a
        a[lift_a] = parent[a[lift_a]]
        b[lift_b] = parent[b[lift_b]]
        dist += lift_a.astype(np.int64) + lift_b.astype(np.int64)
    return float(np.mean(dist + 1))


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
    """Square clustering coefficient for each node (Lind et al., 2005).

    For node v, this is the fraction of paths of length 2 through v that
    close into a 4-cycle (square).  Same quantity as
    :func:`networkx.square_clustering` (formulation of arXiv:2007.11111 used by
    networkx >= 3.4), computed for all nodes at once from the sparse product
    ``A @ A`` instead of a Python loop over node pairs.

    With ``p2(x) = |N(v) ∩ N(x)|`` (entry ``(v, x)`` of ``A @ A``)::

        squares(v)   = 1/2 * sum_{x != v} p2(x) * (p2(x) - 1)
        triangles(v) = sum_{u in N(v)} p2(u)
        potential(v) = (deg(v) - 1) * sum_{u in N(v)} deg(u)
                       - deg(v) * (deg(v) - 1) - triangles(v) - squares(v)
        C4(v)        = squares(v) / potential(v)

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.  Edge weights and self-loops are ignored.

    Returns
    -------
    dict[int, float]
        Mapping node index → square clustering coefficient (0 where undefined).
    """
    A = _csr(adj).copy()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    P = (A @ A).tocsr()
    P.setdiag(0)
    P.eliminate_zeros()

    Q = P.copy()
    Q.data = Q.data * (Q.data - 1.0)
    squares = np.asarray(Q.sum(axis=1)).ravel() / 2.0
    triangles = np.asarray(P.multiply(A).sum(axis=1)).ravel()
    potential = (deg - 1) * (A @ deg) - deg * (deg - 1) - triangles - squares
    with np.errstate(divide='ignore', invalid='ignore'):
        c4 = np.where((deg > 1) & (potential > 0), squares / potential, 0.0)
    return {int(v): float(c) for v, c in enumerate(c4)}


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
    return _distance_stats(adj)[0]


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
        # Smallest eigenvalues of L are the largest of sigma*I - L, which Lanczos
        # finds reliably (which='SM' converges slowly on near-singular matrices).
        L = laplacian.tocsr()
        sigma = (
            2.0 * float(L.diagonal().max()) + 1e-6
        )  # Gershgorin: lambda_max <= sigma
        shifted = sigma * sp.identity(n, format='csr') - L
        top = eigsh(shifted, k=k_req, which='LA', return_eigenvectors=False)
        vals = sigma - top
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
    return _distance_stats(adj)[1]


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
    metrics: list[str] | None = None,
    use_largest_component: bool = False,
    n_jobs: int = 1,
) -> dict[str, float | int | np.ndarray]:
    """Compute graph-structural and TSP metrics.

    Parameters
    ----------
    adj : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix.
    laplacian : np.ndarray or sp.spmatrix, shape (n, n), optional
        Graph Laplacian.  Required for the eigengap metric.
    eigengap_k : int
        Position at which to measure the eigengap (see :func:`eigengap`).
    metrics : list[str], optional
        Names of the metrics to compute (keys listed below).  ``None`` computes
        all of them; ``'eigengap'`` is computed only if *laplacian* is given.
        Skipping the expensive ones (``schultz_index``, ``gutman_index``,
        ``girth``, ``disorder_number``) saves most of the run time.
    use_largest_component : bool
        If True, every metric except ``n_connected_components`` is computed on
        the subgraph induced by the largest connected component.  The Laplacian
        is sliced to that component, which is exact because no edge leaves it.
    n_jobs : int
        Worker processes for the shortest-path computation (``-1`` = all cores).

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
        - ``'n_connected_components'``: number of connected components of the
          full graph
        - ``'graph_diameter'``: maximum pairwise shortest-path distance
        - ``'density'``: edge density
        - ``'gutman_index'``: Gutman index
        - ``'degree_distribution'``: normalized degree histogram

        Always present: ``'lcc_fraction'`` (share of the nodes in the largest
        component of the full graph) and ``'n_nodes_used'`` (nodes the metrics
        were computed on).
    """
    A = _csr(adj)
    n_total = A.shape[0]
    n_comp = number_of_connected_components(A)
    lcc_share = 1.0
    if n_comp > 1:
        sub, nodes, lcc_share = largest_component(A)
        if use_largest_component:
            A = sub
            if laplacian is not None:
                laplacian = laplacian.tocsr()[nodes][:, nodes]

    funcs = {
        'cycle_length': cycle_length,
        'number_of_cycles': number_of_cycles,
        'mean_square_clustering': mean_square_clustering,
        'schultz_index': schultz_index,
        'disorder_number': disorder_number,
        'wiener_index': wiener_index,
        'girth': girth,
        'graph_diameter': graph_diameter,
        'density': density,
        'gutman_index': gutman_index,
        'degree_distribution': degree_distribution,
    }
    wanted = (
        list(funcs) + ['eigengap', 'n_connected_components']
        if metrics is None
        else list(metrics)
    )
    unknown = set(wanted) - set(funcs) - {'eigengap', 'n_connected_components'}
    if unknown:
        raise ValueError(f'Unknown graph metrics: {sorted(unknown)}')

    out: dict = {}
    dist_stats = None
    if 'wiener_index' in wanted or 'graph_diameter' in wanted:
        dist_stats = _distance_stats(A, n_jobs=n_jobs)  # one pass for both
    for name in wanted:
        if name == 'wiener_index' and dist_stats is not None:
            out[name] = dist_stats[0]
        elif name == 'graph_diameter' and dist_stats is not None:
            out[name] = dist_stats[1]
        elif name == 'n_connected_components':
            out[name] = n_comp
        elif name == 'eigengap':
            if laplacian is not None:
                out[name] = eigengap(laplacian, k=eigengap_k)
        else:
            out[name] = funcs[name](A)
    out['lcc_fraction'] = lcc_share
    out['n_nodes_used'] = A.shape[0] if use_largest_component else n_total
    return out


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
    'largest_component',
    'compute_graph_metrics',
]
