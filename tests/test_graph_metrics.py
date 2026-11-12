"""Tests for the graph signatures in src.metrics.graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from src.metrics.graph import (
    compute_graph_metrics,
    cycle_length,
    eigengap,
    graph_diameter,
    largest_component,
    mean_square_clustering,
    square_clustering_coefficients,
    wiener_index,
)


def _adj(G: nx.Graph) -> sp.csr_matrix:
    return sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=sorted(G), dtype=float))


def _two_components() -> sp.csr_matrix:
    """K4 (nodes 0-3) plus a path 4-5-6."""
    G = nx.disjoint_union(nx.complete_graph(4), nx.path_graph(3))
    return _adj(G)


def _knn_like(n: int = 120, k: int = 5, seed: int = 0, dim: int = 3) -> nx.Graph:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)) @ rng.normal(size=(dim, dim))
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in np.argsort(d[i])[1 : k + 1]:
            G.add_edge(i, int(j))
    return G


# ---------------------------------------------------------------------------
# Known values
# ---------------------------------------------------------------------------


def test_path_graph():
    A = _adj(nx.path_graph(5))
    assert wiener_index(A) == 20.0  # n(n^2-1)/6 for n = 5
    assert graph_diameter(A) == 4.0
    assert cycle_length(A) == 0.0
    assert mean_square_clustering(A) == 0.0


def test_cycle_graph():
    A = _adj(nx.cycle_graph(6))
    assert wiener_index(A) == 27.0  # n^3/8 for even n = 6
    assert graph_diameter(A) == 3.0
    assert cycle_length(A) == 6.0  # the only cycle is the whole graph
    # lambda_1 of the symmetric normalised Laplacian of C_n is 1 - cos(2 pi / n)
    L = sp.csr_matrix(nx.normalized_laplacian_matrix(nx.cycle_graph(6)))
    assert eigengap(L, k=1) == pytest.approx(1 - np.cos(2 * np.pi / 6))


def test_complete_graph():
    A = _adj(nx.complete_graph(5))
    assert wiener_index(A) == 10.0
    assert graph_diameter(A) == 1.0
    # fundamental cycles w.r.t. a BFS tree (a star) are all triangles
    assert cycle_length(A) == 3.0
    L = sp.csr_matrix(nx.normalized_laplacian_matrix(nx.complete_graph(5)))
    assert eigengap(L, k=1) == pytest.approx(5 / 4)


def test_square_clustering_known():
    assert mean_square_clustering(_adj(nx.cycle_graph(4))) == pytest.approx(1.0)
    assert mean_square_clustering(_adj(nx.complete_graph(5))) == pytest.approx(1.0)
    assert mean_square_clustering(_adj(nx.path_graph(5))) == 0.0


def test_two_components_full_graph():
    A = _two_components()
    out = compute_graph_metrics(A)
    assert out['n_connected_components'] == 2
    assert out['graph_diameter'] == 2.0  # path 4-5-6; K4 has diameter 1
    assert out['wiener_index'] == 6.0 + 4.0  # K4: 6 pairs; path P3: 1 + 1 + 2
    assert out['lcc_fraction'] == pytest.approx(4 / 7)
    assert out['n_nodes_used'] == 7


def test_two_components_largest_component():
    A = _two_components()
    L = sp.csr_matrix(nx.normalized_laplacian_matrix(nx.from_scipy_sparse_array(A)))
    out = compute_graph_metrics(A, laplacian=L, use_largest_component=True)
    assert out['n_connected_components'] == 2  # always of the full graph
    assert out['n_nodes_used'] == 4
    assert out['graph_diameter'] == 1.0
    assert out['wiener_index'] == 6.0
    assert out['lcc_fraction'] == pytest.approx(4 / 7)
    # eigengap of K4 (normalised): 4/3, instead of 0 for the disconnected graph
    assert out['eigengap'] == pytest.approx(4 / 3)
    full = compute_graph_metrics(A, laplacian=L)
    assert full['eigengap'] == pytest.approx(0.0, abs=1e-8)


def test_largest_component_slice():
    sub, nodes, share = largest_component(_two_components())
    assert nodes.tolist() == [0, 1, 2, 3]
    assert sub.shape == (4, 4)
    assert share == pytest.approx(4 / 7)


def test_metric_subset_and_unknown():
    out = compute_graph_metrics(_adj(nx.cycle_graph(6)), metrics=['wiener_index'])
    assert set(out) == {'wiener_index', 'lcc_fraction', 'n_nodes_used'}
    with pytest.raises(ValueError):
        compute_graph_metrics(_adj(nx.cycle_graph(6)), metrics=['nope'])


# ---------------------------------------------------------------------------
# Agreement with the reference implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('seed', range(4))
def test_square_clustering_matches_networkx(seed):
    G = _knn_like(seed=seed)
    mine = square_clustering_coefficients(_adj(G))
    ref = nx.square_clustering(G)
    assert set(mine) == set(ref)
    for v in ref:
        assert mine[v] == pytest.approx(ref[v], abs=1e-12)


def test_distances_match_networkx_and_chunking():
    G = _knn_like(seed=1)
    comp = max(nx.connected_components(G), key=len)
    H = nx.convert_node_labels_to_integers(G.subgraph(comp))
    A = _adj(H)
    assert wiener_index(A) == pytest.approx(nx.wiener_index(H))
    assert graph_diameter(A) == nx.diameter(H)
    from src.metrics.graph import _distance_stats

    assert _distance_stats(A, chunk=7) == _distance_stats(A, chunk=10_000)
    assert _distance_stats(A, chunk=7, n_jobs=2) == _distance_stats(A, chunk=7)


@pytest.mark.parametrize('seed', range(3))
def test_eigengap_sparse_matches_dense(seed):
    G = nx.convert_node_labels_to_integers(
        G := _knn_like(seed=seed).subgraph(
            max(nx.connected_components(_knn_like(seed=seed)), key=len)
        )
    )
    L = sp.csr_matrix(nx.normalized_laplacian_matrix(G))
    assert eigengap(L, k=1) == pytest.approx(eigengap(L.toarray(), k=1), abs=1e-6)


# ---------------------------------------------------------------------------
# Cycle length: determinism and permutation behaviour
# ---------------------------------------------------------------------------


def test_cycle_length_is_deterministic():
    A = _adj(_knn_like(seed=2))
    assert cycle_length(A) == cycle_length(A.copy())


@pytest.mark.parametrize(
    'G', [nx.cycle_graph(9), nx.complete_graph(6), nx.petersen_graph()]
)
def test_cycle_length_permutation_invariant_on_symmetric_graphs(G):
    A = _adj(G)
    ref = cycle_length(A)
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.permutation(A.shape[0])
        assert cycle_length(A[p][:, p]) == pytest.approx(ref)


def test_cycle_length_permutation_stable_on_random_graph():
    # A single spanning tree is not label-free, so only near-invariance is
    # expected: on a 20-dimensional kNN graph (like the embeddings) the spread over
    # relabellings is below 2%; on 3-D geometric graphs it is about 5%.
    A = _adj(_knn_like(n=1500, k=8, seed=3, dim=20))
    ref = cycle_length(A)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(8):
        p = rng.permutation(A.shape[0])
        vals.append(cycle_length(A[p][:, p]))
    assert (max(vals + [ref]) - min(vals + [ref])) / ref < 0.02


def test_cycle_length_forest_per_component():
    # two triangles in separate components: both fundamental cycles have length 3
    G = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
    assert cycle_length(_adj(G)) == 3.0
