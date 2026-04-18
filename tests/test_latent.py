"""Tests for LatentSpace class."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs


@pytest.fixture
def simple_cloud() -> np.ndarray:
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
def larger_cloud() -> np.ndarray:
    """Larger point cloud with 100 points in 5D."""
    X, _ = make_blobs(
        n_samples=100,
        n_features=5,
        centers=10,
        random_state=42,
    )
    return X.astype(np.float32)


@pytest.fixture
def cloud_with_extras() -> tuple[np.ndarray, dict]:
    """20 points with extras dict containing labels."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    labels = np.arange(20)
    extras = {'labels': labels}
    return X, extras


class TestInitialization:
    """Tests for LatentSpace initialization."""

    def test_init_numpy(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud)
        np.testing.assert_array_almost_equal(latent.latent, simple_cloud)

    def test_init_torch(self, simple_cloud):
        from src.objects.latent import LatentSpace

        tensor = torch.from_numpy(simple_cloud)
        latent = LatentSpace(tensor)
        np.testing.assert_array_almost_equal(latent.latent, simple_cloud)

    def test_init_with_extras(self, cloud_with_extras):
        from src.objects.latent import LatentSpace

        X, extras = cloud_with_extras
        latent = LatentSpace(X, extras=extras)
        assert 'labels' in latent.extras
        np.testing.assert_array_equal(latent.extras['labels'], extras['labels'])

    def test_init_default_attrs(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud)
        assert latent.prototypes is None
        assert latent.analysis_operator is None
        assert latent.synthesis_operator is None


class TestProperties:
    """Tests for LatentSpace properties."""

    def test_properties(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud, seed=123)
        assert latent.n_points == 10
        assert latent.n_features == 2
        assert latent.seed == 123
        assert latent.latent.shape == (10, 2)
        np.testing.assert_array_equal(latent.original_indices, np.arange(10))

    def test_extras_empty_by_default(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud)
        assert latent.extras == {}


class TestSubsample:
    """Tests for subsample method."""

    def test_subsample_random(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud, seed=42)
        subsampled = latent.subsample(5, seed=42)

        assert subsampled.n_points == 5
        assert subsampled.n_features == 2
        assert subsampled.prototypes is None

    def test_subsample_random_preserves_indices(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud, seed=42)
        subsampled = latent.subsample(5, seed=42)

        original_subset = simple_cloud[subsampled.original_indices]
        np.testing.assert_array_almost_equal(subsampled.latent, original_subset)

    def test_subsample_random_with_extras(self, cloud_with_extras):
        from src.objects.latent import LatentSpace

        X, extras = cloud_with_extras
        latent = LatentSpace(X, extras=extras)
        subsampled = latent.subsample(10, seed=42)

        assert subsampled.extras is not None
        assert subsampled.original_indices.shape == (10,)

    def test_subsample_prototypes(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        subsampled = latent.subsample(10, compute_prototypes=True, seed=42)

        assert subsampled.n_points == 10
        assert subsampled.n_features == 5
        assert subsampled.prototypes is not None
        assert subsampled.prototypes.shape == (10, 5)

    def test_subsample_prototypes_sets_operators(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        subsampled = latent.subsample(
            10,
            compute_prototypes=True,
            apply_parseval=True,
            seed=42,
        )

        assert subsampled.analysis_operator is not None
        assert subsampled.synthesis_operator is not None
        assert subsampled.analysis_operator.shape == (10, 5)

    def test_subsample_prototypes_empty_indices(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        subsampled = latent.subsample(10, compute_prototypes=True, seed=42)

        assert subsampled.original_indices.shape == (0,)

    def test_subsample_prototypes_no_parseval(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        subsampled = latent.subsample(
            10,
            compute_prototypes=True,
            apply_parseval=False,
            seed=42,
        )

        assert subsampled.analysis_operator is None
        assert subsampled.synthesis_operator is None

    def test_subsample_prototypes_custom_clusterer(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        subsampled = latent.subsample(
            10,
            compute_prototypes=True,
            clusterer_cls=KMeans,
            clusterer_kwargs={'n_init': 10},
            seed=42,
        )

        assert subsampled.n_points == 10


class TestNormalize:
    """Tests for normalize method."""

    def test_normalize_standard(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud)
        normalized = latent.normalize('standard')

        assert normalized.shape == simple_cloud.shape
        np.testing.assert_array_almost_equal(normalized.mean(axis=0), 0, decimal=5)
        np.testing.assert_array_almost_equal(normalized.std(axis=0), 1, decimal=5)

    def test_normalize_minmax(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud)
        normalized = latent.normalize('minmax')

        assert normalized.shape == simple_cloud.shape
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

    def test_normalize_l2(self, simple_cloud):
        from src.objects.latent import LatentSpace

        non_zero_cloud = simple_cloud[1:]
        latent = LatentSpace(non_zero_cloud)
        normalized = latent.normalize('l2')

        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=5)


class TestReduceDimensions:
    """Tests for reduce_dimensions method."""

    def test_reduce_dimensions_pca(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud)
        reduced = latent.reduce_dimensions('pca', n_components=2)

        assert reduced.shape == (100, 2)

    def test_reduce_dimensions_prototype_analysis(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        reduced = latent.reduce_dimensions(
            'prototype_analysis',
            n_components=5,
            prototype_n_samples=3,
            seed=42,
        )

        assert reduced.shape == (100, 5)
        assert latent.prototypes is not None
        assert latent.analysis_operator is not None

    def test_reduce_dimensions_prototype_analysis_sets_attrs(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        latent.reduce_dimensions('prototype_analysis', n_components=5)

        assert latent.prototypes is not None
        assert latent.analysis_operator is not None
        assert latent.synthesis_operator is not None


class TestComputePrototypes:
    """Tests for compute_prototypes method."""

    def test_compute_prototypes_clusterer_cls(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        prototypes = latent.compute_prototypes(
            n_samples=5,
            clusterer_cls=KMeans,
            n_clusters=10,
        )

        assert prototypes.shape == (10, 5)
        assert latent.prototypes is prototypes

    def test_compute_prototypes_external_clusterer(self, larger_cloud):
        from src.objects.latent import LatentSpace

        clusterer = KMeans(n_clusters=10, random_state=42)
        latent = LatentSpace(larger_cloud, seed=42)
        prototypes = latent.compute_prototypes(
            n_samples=5,
            clusterer=clusterer,
        )

        assert prototypes.shape == (10, 5)

    def test_compute_prototypes_precomputed_clusters(self, larger_cloud):
        from src.objects.latent import LatentSpace

        kmeans = KMeans(n_clusters=10, random_state=42)
        clusters = kmeans.fit_predict(larger_cloud)

        latent = LatentSpace(larger_cloud, seed=42)
        prototypes = latent.compute_prototypes(
            n_samples=5,
            clusters=clusters,
        )

        assert prototypes.shape == (10, 5)

    def test_compute_prototypes_dbscan(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        prototypes = latent.compute_prototypes(
            n_samples=2,
            clusterer_cls=DBSCAN,
            clusterer_kwargs={'eps': 1.5, 'min_samples': 2},
        )

        assert prototypes is not None
        assert prototypes.shape[1] == 5

    def test_compute_prototypes_no_parseval(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        latent.compute_prototypes(
            n_samples=5,
            clusterer_cls=KMeans,
            n_clusters=10,
            apply_parseval=False,
        )

        assert latent.analysis_operator is None
        assert latent.synthesis_operator is None

    def test_compute_prototypes_with_parseval(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        latent.compute_prototypes(
            n_samples=5,
            clusterer_cls=KMeans,
            n_clusters=10,
            apply_parseval=True,
        )

        assert latent.analysis_operator is not None
        assert latent.synthesis_operator is not None
        F = latent.analysis_operator
        G = latent.synthesis_operator
        np.testing.assert_array_almost_equal(G, F.T)

    def test_compute_prototypes_error_small_cluster(self):
        from src.objects.latent import LatentSpace

        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        clusters = np.array([0, 0, 1])

        latent = LatentSpace(X, seed=42)
        with pytest.raises(ValueError, match='Cluster 0 has 2 samples'):
            latent.compute_prototypes(n_samples=5, clusters=clusters)

    def test_compute_prototypes_error_no_mode(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud, seed=42)
        with pytest.raises(
            ValueError, match='Provide exactly one of .clusters., .clusterer.'
        ):
            latent.compute_prototypes(n_samples=5)

    def test_compute_prototypes_error_multiple_modes(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud, seed=42)
        clusters = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
        with pytest.raises(
            ValueError, match='Provide exactly one of .clusters., .clusterer.'
        ):
            latent.compute_prototypes(
                n_samples=5,
                clusters=clusters,
                clusterer_cls=KMeans,
            )

    def test_compute_prototypes_n_samples_none(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud, seed=42)
        prototypes = latent.compute_prototypes(
            n_samples=None,
            clusterer_cls=KMeans,
            n_clusters=3,
        )

        assert prototypes.shape == (3, 2)
        assert latent.prototypes is prototypes

    def test_compute_prototypes_n_samples_none_with_indices(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        prototypes, cluster_indices = latent.compute_prototypes(
            n_samples=None,
            clusterer_cls=KMeans,
            n_clusters=10,
            return_cluster_indices=True,
        )

        assert prototypes.shape == (10, 5)
        assert isinstance(cluster_indices, dict)
        assert len(cluster_indices) == 10
        for proto_idx, obs_indices in cluster_indices.items():
            assert isinstance(proto_idx, int)
            assert obs_indices.ndim == 1

    def test_compute_prototypes_default_n_samples(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        prototypes = latent.compute_prototypes(
            clusterer_cls=KMeans,
            n_clusters=10,
        )

        assert prototypes.shape == (10, 5)


class TestApplyOperators:
    """Tests for apply_analysis_operator and apply_synthesis_operator."""

    def test_apply_analysis_operator(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        latent.compute_prototypes(
            n_samples=5,
            clusterer_cls=KMeans,
            n_clusters=10,
            apply_parseval=True,
        )

        transformed = latent.apply_analysis_operator()

        assert transformed.shape == (100, 10)

    def test_apply_analysis_operator_error(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud)
        with pytest.raises(
            ValueError, match='Run compute_prototypes with apply_parseval=True'
        ):
            latent.apply_analysis_operator()

    def test_apply_synthesis_operator_numpy(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        latent.compute_prototypes(
            n_samples=5,
            clusterer_cls=KMeans,
            n_clusters=10,
            apply_parseval=True,
        )

        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 10)).astype(np.float32)
        result = latent.apply_synthesis_operator(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5)

    def test_apply_synthesis_operator_torch(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        latent.compute_prototypes(
            n_samples=5,
            clusterer_cls=KMeans,
            n_clusters=10,
            apply_parseval=True,
        )

        X = torch.randn(5, 10)
        result = latent.apply_synthesis_operator(X)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 5)

    def test_apply_synthesis_operator_error(self, simple_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(simple_cloud)
        with pytest.raises(
            ValueError, match='Run compute_prototypes with apply_parseval=True'
        ):
            latent.apply_synthesis_operator(np.array([[1, 2]]))

    def test_roundtrip_transform(self, larger_cloud):
        from src.objects.latent import LatentSpace

        latent = LatentSpace(larger_cloud, seed=42)
        latent.compute_prototypes(
            n_samples=5,
            clusterer_cls=KMeans,
            n_clusters=10,
            apply_parseval=True,
        )

        F_transformed = latent.apply_analysis_operator()

        assert F_transformed.shape == (100, 10)
