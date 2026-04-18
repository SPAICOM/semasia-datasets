"""Tests for metrics module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs


@pytest.fixture
def simple_a() -> np.ndarray:
    """Simple 2D point cloud with 10 points."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 2.0],
            [3.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [2.5, 2.5],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def simple_b() -> np.ndarray:
    """Simple 2D point cloud, slightly shifted from simple_a."""
    return np.array(
        [
            [0.1, 0.1],
            [1.1, 0.1],
            [0.1, 1.1],
            [1.1, 1.1],
            [0.6, 0.6],
            [2.1, 2.1],
            [3.1, 2.1],
            [2.1, 3.1],
            [3.1, 3.1],
            [2.6, 2.6],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def identical(simple_a) -> np.ndarray:
    """Identical to simple_a for identity tests."""
    return simple_a.copy()


@pytest.fixture
def larger_a() -> np.ndarray:
    """Larger point cloud with 100 points in 5D."""
    X, _ = make_blobs(
        n_samples=100,
        n_features=5,
        centers=10,
        random_state=42,
    )
    return X.astype(np.float32)


@pytest.fixture
def larger_b(larger_a) -> np.ndarray:
    """Larger point cloud, shifted version."""
    return (larger_a + 0.5).astype(np.float32)


@pytest.fixture
def single_point_a() -> np.ndarray:
    """Single point in 2D."""
    return np.array([[0.0, 0.0]], dtype=np.float32)


@pytest.fixture
def single_point_b() -> np.ndarray:
    """Single different point in 2D."""
    return np.array([[3.0, 4.0]], dtype=np.float32)


@pytest.fixture
def mismatched_shapes_a() -> np.ndarray:
    """10 points in 2D."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 2)).astype(np.float32)


@pytest.fixture
def mismatched_shapes_b() -> np.ndarray:
    """5 points in 2D."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((5, 2)).astype(np.float32)


@pytest.fixture
def orthogonal_a() -> np.ndarray:
    """Points along x-axis."""
    return np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)


@pytest.fixture
def orthogonal_b() -> np.ndarray:
    """Points along y-axis."""
    return np.array([[0, 0], [0, 1], [0, 2]], dtype=np.float32)


@pytest.fixture
def unit_a() -> np.ndarray:
    """Unit vectors."""
    return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


@pytest.fixture
def unit_b() -> np.ndarray:
    """Normalized same direction as unit_a."""
    return np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_basic_distance(self):
        from src.metrics import euclidean_distance

        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        dist = euclidean_distance(a, b)
        np.testing.assert_almost_equal(dist, 5.0, decimal=2)

    def test_single_point_same(self):
        from src.metrics import euclidean_distance

        a = np.array([[0.0, 0.0]], dtype=np.float32)
        dist = euclidean_distance(a, a)
        np.testing.assert_almost_equal(dist, 0.0, decimal=2)

    def test_symmetry(self, simple_a, simple_b):
        from src.metrics import euclidean_distance

        dist_ab = euclidean_distance(simple_a, simple_b)
        dist_ba = euclidean_distance(simple_b, simple_a)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=5)

    def test_order_invariance(self, simple_a, simple_b):
        from src.metrics import euclidean_distance

        rng = np.random.default_rng(42)
        b_shuffled = simple_b[rng.permutation(len(simple_b))]
        dist_original = euclidean_distance(simple_a, simple_b)
        dist_shuffled = euclidean_distance(simple_a, b_shuffled)
        np.testing.assert_almost_equal(dist_original, dist_shuffled, decimal=5)

    def test_larger_point_clouds(self, larger_a, larger_b):
        from src.metrics import euclidean_distance

        dist = euclidean_distance(larger_a, larger_b)
        assert isinstance(dist, float)
        assert dist > 0

    def test_single_point(self, single_point_a, single_point_b):
        from src.metrics import euclidean_distance

        dist = euclidean_distance(single_point_a, single_point_b)
        np.testing.assert_almost_equal(dist, 5.0, decimal=5)

    def test_mismatched_shapes(self, mismatched_shapes_a, mismatched_shapes_b):
        from src.metrics import euclidean_distance

        dist = euclidean_distance(mismatched_shapes_a, mismatched_shapes_b)
        assert isinstance(dist, float)


class TestCosineDistance:
    """Tests for cosine_distance function."""

    def test_orthogonal(self, orthogonal_a, orthogonal_b):
        from src.metrics import cosine_distance

        dist = cosine_distance(orthogonal_a, orthogonal_b)
        np.testing.assert_almost_equal(dist, 1.0, decimal=5)

    def test_single_point_same(self):
        from src.metrics import cosine_distance

        a = np.array([[1.0, 0.0]], dtype=np.float32)
        dist = cosine_distance(a, a)
        np.testing.assert_almost_equal(dist, 0.0, decimal=5)

    def test_symmetry(self, simple_a, simple_b):
        from src.metrics import cosine_distance

        dist_ab = cosine_distance(simple_a, simple_b)
        dist_ba = cosine_distance(simple_b, simple_a)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=5)

    def test_normalized_vectors(self, unit_a, unit_b):
        from src.metrics import cosine_distance

        dist = cosine_distance(unit_a, unit_b)
        assert 0.0 <= dist <= 1.0


class TestMahalanobisDistance:
    """Tests for mahalanobis_distance function."""

    def test_with_cov(self, simple_a, simple_b):
        from src.metrics import mahalanobis_distance

        combined = np.vstack([simple_a, simple_b])
        cov = np.cov(combined, rowvar=False)
        dist = mahalanobis_distance(simple_a, simple_b, cov=cov)
        assert isinstance(dist, float)

    def test_without_cov(self, simple_a, simple_b):
        from src.metrics import mahalanobis_distance

        dist = mahalanobis_distance(simple_a, simple_b)
        assert isinstance(dist, float)

    def test_symmetry(self, simple_a, simple_b):
        from src.metrics import mahalanobis_distance

        dist_ab = mahalanobis_distance(simple_a, simple_b)
        dist_ba = mahalanobis_distance(simple_b, simple_a)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=5)

    def test_singular_cov(self, simple_a, simple_b):
        from src.metrics import mahalanobis_distance

        cov = np.eye(2)
        dist = mahalanobis_distance(simple_a, simple_b, cov=cov)
        assert isinstance(dist, float)

    def test_equals_euclidean_for_identity(self):
        from src.metrics import euclidean_distance, mahalanobis_distance

        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        cov = np.eye(2)
        maha_dist = mahalanobis_distance(a, b, cov=cov)
        euc_dist = euclidean_distance(a, b)
        np.testing.assert_almost_equal(maha_dist, euc_dist, decimal=2)


class TestWassersteinDistance:
    """Tests for wasserstein_distance function."""

    def test_sorted_arrays(self):
        from src.metrics import wasserstein_distance

        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        b = np.array([1, 2, 4, 5, 6], dtype=np.float32)
        dist = wasserstein_distance(a, b)
        assert isinstance(dist, float)

    def test_permutation_invariance(self):
        from src.metrics import wasserstein_distance

        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        b = np.array([1, 2, 4, 5, 6], dtype=np.float32)
        b_perm = np.array([6, 4, 2, 1, 5])
        dist_original = wasserstein_distance(a, b)
        dist_perm = wasserstein_distance(a, b_perm)
        np.testing.assert_almost_equal(dist_original, dist_perm, decimal=5)

    def test_same_distribution(self):
        from src.metrics import wasserstein_distance

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([1, 2, 3], dtype=np.float32)
        dist = wasserstein_distance(a, b)
        assert dist == 0.0

    def test_shifted_distributions(self):
        from src.metrics import wasserstein_distance

        a = np.array([0, 0, 0, 1], dtype=np.float32)
        b = np.array([10, 10, 10, 11], dtype=np.float32)
        dist = wasserstein_distance(a, b)
        assert dist > 0

    def test_multidimensional(self, simple_a, simple_b):
        from src.metrics import wasserstein_distance

        a_flat = simple_a.flatten()
        b_flat = simple_b.flatten()
        dist = wasserstein_distance(a_flat, b_flat)
        assert isinstance(dist, float)


class TestSinkhornDistance:
    """Tests for sinkhorn_distance function."""

    def test_basic_ot(self, simple_a, simple_b):
        from src.metrics import sinkhorn_distance

        dist = sinkhorn_distance(simple_a, simple_b)
        assert isinstance(dist, float)

    def test_symmetry(self, simple_a, simple_b):
        from src.metrics import sinkhorn_distance

        dist_ab = sinkhorn_distance(simple_a, simple_b)
        dist_ba = sinkhorn_distance(simple_b, simple_a)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=2)

    def test_epsilon_sensitivity(self, simple_a, simple_b):
        from src.metrics import sinkhorn_distance

        dist_low_eps = sinkhorn_distance(simple_a, simple_b, epsilon=0.01)
        dist_high_eps = sinkhorn_distance(simple_a, simple_b, epsilon=1.0)
        assert dist_high_eps >= dist_low_eps

    def test_convergence(self, simple_a, simple_b):
        from src.metrics import sinkhorn_distance

        dist = sinkhorn_distance(simple_a, simple_b, max_iter=10)
        assert isinstance(dist, float)

    def test_identical(self, simple_a):
        from src.metrics import sinkhorn_distance

        dist = sinkhorn_distance(simple_a, simple_a)
        np.testing.assert_almost_equal(dist, 0.0, decimal=2)


class TestProcrustesDistance:
    """Tests for procrustes_distance function."""

    def test_orthogonal_alignment(self, orthogonal_a, orthogonal_b):
        from src.metrics import procrustes_distance

        dist = procrustes_distance(orthogonal_a, orthogonal_b)
        assert isinstance(dist, float)

    def test_scaled_versions(self, simple_a):
        from src.metrics import procrustes_distance

        scaled = simple_a * 2.0
        dist = procrustes_distance(simple_a, scaled)
        assert dist >= 0

    def test_symmetry(self, simple_a, simple_b):
        from src.metrics import procrustes_distance

        dist_ab = procrustes_distance(simple_a, simple_b)
        dist_ba = procrustes_distance(simple_b, simple_a)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=3)

    def test_identical(self, simple_a):
        from src.metrics import procrustes_distance

        dist = procrustes_distance(simple_a, simple_a)
        np.testing.assert_almost_equal(dist, 0.0, decimal=3)

    def test_different_n_points(self):
        from src.metrics import procrustes_distance

        rng = np.random.default_rng(42)
        a = rng.standard_normal((10, 2)).astype(np.float32)
        b = rng.standard_normal((5, 2)).astype(np.float32)
        dist = procrustes_distance(a, b)
        assert isinstance(dist, float)


class TestChamferDistance:
    """Tests for chamfer_distance function."""

    def test_symmetry(self, simple_a, simple_b):
        from src.metrics import chamfer_distance

        dist_ab = chamfer_distance(simple_a, simple_b)
        dist_ba = chamfer_distance(simple_b, simple_a)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=5)

    def test_forward_only(self, simple_a, simple_b):
        from src.metrics import chamfer_distance

        dist = chamfer_distance(simple_a, simple_b)
        assert dist >= 0

    def test_identical(self, simple_a):
        from src.metrics import chamfer_distance

        dist = chamfer_distance(simple_a, simple_a)
        assert dist == 0.0

    def test_larger_clouds(self, larger_a, larger_b):
        from src.metrics import chamfer_distance

        dist = chamfer_distance(larger_a, larger_b)
        assert dist > 0


class TestHausdorffDistance:
    """Tests for hausdorff_distance function."""

    def test_symmetry(self, simple_a, simple_b):
        from src.metrics import hausdorff_distance

        dist_ab = hausdorff_distance(simple_a, simple_b)
        dist_ba = hausdorff_distance(simple_b, simple_a)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=5)

    def test_identical(self, simple_a):
        from src.metrics import hausdorff_distance

        dist = hausdorff_distance(simple_a, simple_a)
        assert dist == 0.0

    def test_one_way_matters(self):
        from src.metrics import hausdorff_distance

        a = np.array([[0, 0], [10, 0]], dtype=np.float32)
        b = np.array([[0, 0], [0, 10]], dtype=np.float32)
        dist = hausdorff_distance(a, b)
        assert dist > 0

    def test_disjoint_sets(self):
        from src.metrics import hausdorff_distance

        a = np.array([[0, 0], [1, 1]], dtype=np.float32)
        b = np.array([[100, 100], [101, 101]], dtype=np.float32)
        dist = hausdorff_distance(a, b)
        assert dist > 90


class TestComputeMetric:
    """Tests for compute_metric unified API."""

    def test_all_metric_names(self, simple_a, simple_b):
        from src.metrics import METRIC_NAMES, compute_metric

        for metric_name in METRIC_NAMES:
            dist = compute_metric(simple_a, simple_b, metric_name)
            assert isinstance(dist, float)

    def test_unknown_metric(self, simple_a, simple_b):
        from src.metrics import compute_metric

        with pytest.raises(ValueError, match='Unknown metric'):
            compute_metric(simple_a, simple_b, 'invalid_metric')

    def test_extra_kwargs_cos(self):
        from src.metrics import compute_metric

        a = np.array([[0, 0], [1, 0]], dtype=np.float32)
        b = np.array([[0, 0], [0, 1]], dtype=np.float32)
        dist = compute_metric(a, b, 'cosine')
        np.testing.assert_almost_equal(dist, 1.0, decimal=5)

    def test_extra_kwargs_sinkhorn_epsilon(self, simple_a, simple_b):
        from src.metrics import compute_metric

        dist = compute_metric(simple_a, simple_b, 'sinkhorn', epsilon=0.1)
        assert isinstance(dist, float)


class TestMetricProperties:
    """Parametrized tests for metric properties across all metrics."""

    @pytest.mark.parametrize('metric', ['euclidean', 'cosine', 'chamfer'])
    def test_non_negative(self, metric, simple_a, simple_b):
        from src.metrics import compute_metric

        dist = compute_metric(simple_a, simple_b, metric)
        assert dist >= 0

    @pytest.mark.parametrize('metric', ['euclidean', 'cosine', 'chamfer', 'hausdorff'])
    def test_symmetry(self, metric, simple_a, simple_b):
        from src.metrics import compute_metric

        dist_ab = compute_metric(simple_a, simple_b, metric)
        dist_ba = compute_metric(simple_b, simple_a, metric)
        np.testing.assert_almost_equal(dist_ab, dist_ba, decimal=4)

    @pytest.mark.parametrize('metric', ['euclidean', 'cosine', 'chamfer'])
    def test_identity(self, metric):
        from src.metrics import compute_metric

        a = np.array([[1.0, 0.0]], dtype=np.float32)
        dist = compute_metric(a, a, metric)
        np.testing.assert_almost_equal(dist, 0.0, decimal=2)


class TestMetricEdgeCases:
    """Edge cases and error handling."""

    def test_large_point_sets(self):
        from src.metrics import euclidean_distance

        rng = np.random.default_rng(42)
        a = rng.standard_normal((100, 2)).astype(np.float32)
        b = rng.standard_normal((100, 2)).astype(np.float32)
        dist = euclidean_distance(a, b)
        assert dist > 0

    def test_single_cluster(self):
        from src.metrics import euclidean_distance

        a = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.float32)
        b = np.array([[2, 2], [2, 2]], dtype=np.float32)
        dist = euclidean_distance(a, b)
        assert dist > 0

    def test_high_dimension(self):
        from src.metrics import euclidean_distance

        rng = np.random.default_rng(42)
        a = rng.standard_normal((10, 50)).astype(np.float32)
        b = rng.standard_normal((10, 50)).astype(np.float32)
        dist = euclidean_distance(a, b)
        assert dist > 0


class TestJaccardPrototypeSimilarity:
    """Tests for Jaccard prototype similarity functions."""

    def test_identical_clusters(self):
        from src.metrics import jaccard_prototype_similarity

        cluster_indices_a = {
            0: np.array([0, 1, 2]),
            1: np.array([3, 4, 5]),
            2: np.array([6, 7, 8]),
        }
        cluster_indices_b = {
            0: np.array([0, 1, 2]),
            1: np.array([3, 4, 5]),
            2: np.array([6, 7, 8]),
        }
        sim_matrix = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)
        np.testing.assert_almost_equal(sim_matrix.diagonal(), 1.0, decimal=5)

    def test_no_overlap(self):
        from src.metrics import jaccard_prototype_similarity

        cluster_indices_a = {
            0: np.array([0, 1, 2]),
            1: np.array([3, 4, 5]),
        }
        cluster_indices_b = {
            0: np.array([10, 11, 12]),
            1: np.array([13, 14, 15]),
        }
        sim_matrix = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)
        np.testing.assert_almost_equal(sim_matrix, 0.0, decimal=5)

    def test_partial_overlap(self):
        from src.metrics import jaccard_prototype_similarity

        cluster_indices_a = {
            0: np.array([0, 1, 2, 3]),
        }
        cluster_indices_b = {
            0: np.array([1, 2, 3, 4]),
        }
        sim_matrix = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)
        intersection = 3
        union = 5
        expected = intersection / union
        np.testing.assert_almost_equal(sim_matrix[0, 0], expected, decimal=5)

    def test_different_n_prototypes(self):
        from src.metrics import jaccard_prototype_similarity

        cluster_indices_a = {
            0: np.array([0, 1]),
            1: np.array([2, 3]),
        }
        cluster_indices_b = {
            0: np.array([0, 1]),
            1: np.array([2, 3]),
            2: np.array([4, 5]),
        }
        sim_matrix = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)
        assert sim_matrix.shape == (2, 3)

    def test_compute_jaccard_metrics(self):
        from src.metrics import compute_jaccard_metrics

        cluster_indices_a = {
            0: np.array([0, 1, 2]),
            1: np.array([3, 4, 5]),
            2: np.array([6, 7, 8]),
        }
        cluster_indices_b = {
            0: np.array([0, 1, 2]),
            1: np.array([3, 4, 5]),
            2: np.array([9, 10]),
        }
        metrics = compute_jaccard_metrics(
            cluster_indices_a, cluster_indices_b, threshold=0.5
        )
        assert 'jaccard_mean' in metrics
        assert 'jaccard_good_match_ratio' in metrics
        assert 0.0 <= metrics['jaccard_mean'] <= 1.0
        assert 0.0 <= metrics['jaccard_good_match_ratio'] <= 1.0

    def test_compute_jaccard_metrics_threshold(self):
        from src.metrics import compute_jaccard_metrics

        cluster_indices_a = {
            0: np.array([0, 1]),
            1: np.array([2, 3]),
        }
        cluster_indices_b = {
            0: np.array([0, 1]),
            1: np.array([5, 6]),
        }
        metrics = compute_jaccard_metrics(
            cluster_indices_a, cluster_indices_b, threshold=0.9
        )
        assert metrics['jaccard_mean'] > 0.0
        assert metrics['jaccard_good_match_ratio'] == 0.5
