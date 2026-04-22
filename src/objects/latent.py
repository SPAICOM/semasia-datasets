"""Latent space representation with preprocessing and prototype computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from scipy.linalg import solve_triangular
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

if TYPE_CHECKING:
    from sklearn.base import ClusterMixin

NormalizeMethod = Literal['standard', 'minmax', 'l2']


def _instantiate_hdbscan(**kwargs):
    """Instantiate HDBSCAN clusterer (lazy import)."""
    try:
        import hdbscan

        return hdbscan.HDBSCAN(**kwargs)
    except ImportError:
        raise ImportError('hdbscan package not installed. Run: pip install hdbscan')


DimReductionMethod = Literal[
    'pca', 'umap', 'tsne', 'lle', 'isomap', 'prototype_analysis'
]


def _normalize_point_cloud(X: np.ndarray, method: NormalizeMethod) -> np.ndarray:
    """Normalize a point cloud."""
    match method:
        case 'standard':
            mu = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            return ((X - mu) / std).astype(np.float32)
        case 'minmax':
            lo = X.min(axis=0)
            hi = X.max(axis=0)
            rng = hi - lo
            rng[rng == 0] = 1.0
            return ((X - lo) / rng).astype(np.float32)
        case 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (X / norms).astype(np.float32)
        case _:
            raise ValueError(f'Unknown normalize method {method!r}')


def _parseval_frame(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Parseval frame via SVD.

    Returns F (analysis) and G (synthesis) operators where G = F^H.
    """
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    U, _, Vh = torch.linalg.svd(X_tensor, full_matrices=False)
    F = (U @ Vh).numpy()
    G = F.conj().T
    return F, G


class LatentSpace:
    """Latent space representation with preprocessing methods.

    Parameters
    ----------
    latent : np.ndarray | torch.Tensor, shape (n_points, n_features)
        The latent space embeddings.
    labels : np.ndarray | torch.Tensor, shape (n_points,), optional
        Labels for each point (e.g., class labels).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        latent: np.ndarray | torch.Tensor,
        extras: dict[str, np.ndarray | torch.Tensor] | None = None,
        seed: int = 42,
    ):
        if isinstance(latent, torch.Tensor):
            self._latent = latent.detach().cpu().float().numpy()
        else:
            self._latent = np.asarray(latent, dtype=np.float32)

        self._extras: dict[str, np.ndarray] = {}
        if extras is not None:
            for name, arr in extras.items():
                if isinstance(arr, torch.Tensor):
                    self._extras[name] = arr.detach().cpu().numpy()
                else:
                    self._extras[name] = np.asarray(arr)

        self._original_indices = np.arange(self._latent.shape[0])
        self._seed = seed
        torch.manual_seed(seed)

        self._prototypes: np.ndarray | None = None
        self._F: np.ndarray | None = None
        self._G: np.ndarray | None = None
        self._prototypes_to_indices: dict[int, np.ndarray] = {}

        self._whiten_F: np.ndarray | None = None
        self._whiten_G: np.ndarray | None = None
        self._whitening_mean: np.ndarray | None = None
        self._whitening_L: np.ndarray | None = None
        self._whitened_latent: np.ndarray | None = None

    @property
    def latent(self) -> np.ndarray:
        """Returns the latent data array."""
        return self._latent

    @property
    def extras(self) -> dict[str, np.ndarray]:
        """Returns the extras dict (e.g., labels, attributes)."""
        return self._extras

    @property
    def original_indices(self) -> np.ndarray:
        """Returns mapping to original indices after subsampling."""
        return self._original_indices

    @property
    def seed(self) -> int:
        """Returns the random seed."""
        return self._seed

    @property
    def n_points(self) -> int:
        """Returns number of points."""
        return self._latent.shape[0]

    @property
    def n_features(self) -> int:
        """Returns dimensionality of latent space."""
        return self._latent.shape[1]

    @property
    def prototypes(self) -> np.ndarray | None:
        """Returns cluster prototypes if computed."""
        return self._prototypes

    @property
    def analysis_operator(self) -> np.ndarray | None:
        """Returns Parseval frame analysis operator if computed."""
        return self._F

    @property
    def synthesis_operator(self) -> np.ndarray | None:
        """Returns Parseval frame synthesis operator if computed."""
        return self._G

    def subsample(
        self,
        n_points: int,
        seed: int | None = None,
        compute_prototypes: bool = False,
        n_samples: int = 10,
        clusterer_cls: type[ClusterMixin] | None = None,
        clusterer_kwargs: dict | None = None,
        apply_parseval: bool = True,
    ) -> LatentSpace:
        """Subsample the point cloud via random sampling or prototype computation.

        Parameters
        ----------
        n_points : int
            Number of points to sample.
        seed : int, optional
            Random seed. Uses instance seed if None.
        compute_prototypes : bool, default=False
            If True, compute prototypes as the subsample (clustering-based
            subsampling). Ignores random sampling. Each prototype becomes
            a point in the new representation, with n_clusters = n_points.
        n_samples : int, default=10
            Number of samples per cluster to compute prototype (used when
            compute_prototypes=True).
        clusterer_cls : type[ClusterMixin], optional
            Clustering class (e.g., KMeans). Required if compute_prototypes=True.
        clusterer_kwargs : dict, optional
            Additional kwargs passed to the clusterer.
        apply_parseval : bool, default=True
            Whether to apply Parseval frame to prototypes.

        Returns
        -------
        LatentSpace
            New LatentSpace with subsampled data. If compute_prototypes=True,
            the new instance will have prototypes, analysis_operator, and
            synthesis_operator attributes set.

        Raises
        ------
        ValueError
            If compute_prototypes=True but no clusterer_cls is provided.
        """
        if seed is None:
            seed = self._seed
        torch.manual_seed(seed)

        if compute_prototypes:
            # default clustering class
            if clusterer_cls is None:
                clusterer_cls = KMeans

            prototypes = self.compute_prototypes(
                n_samples=n_samples,
                clusterer_cls=clusterer_cls,
                n_clusters=n_points,
                clusterer_kwargs=clusterer_kwargs,
                apply_parseval=apply_parseval,
            )

            new_inst = LatentSpace(
                latent=prototypes,
                seed=seed,
            )
            new_inst._original_indices = np.array([], dtype=int)
            new_inst._prototypes = prototypes
            new_inst._F = self._F
            new_inst._G = self._G
            return new_inst

        rng = np.random.default_rng(seed)
        idx = rng.choice(self._latent.shape[0], size=n_points, replace=False)

        new_extras = None
        if self._extras:
            new_extras = {name: arr[idx] for name, arr in self._extras.items()}

        new_inst = LatentSpace(
            latent=self._latent[idx],
            extras=new_extras,
            seed=seed,
        )
        new_inst._original_indices = self._original_indices[idx]
        return new_inst

    def normalize(
        self, method: NormalizeMethod, inplace: bool = False
    ) -> np.ndarray | LatentSpace:
        """Normalize the point cloud.

        Parameters
        ----------
        method : NormalizeMethod
            Normalization method: 'standard', 'minmax', or 'l2'.
        inplace : bool, default=False
            If True, modify self._latent in place and return self.
            If False, return a copy.

        Returns
        -------
        np.ndarray | LatentSpace
            Normalized latent array, or self if inplace=True.
        """
        result = _normalize_point_cloud(self._latent, method)
        if inplace:
            self._latent = result
            return self
        return result

    def prewhiten(self, inplace: bool = False) -> np.ndarray | LatentSpace:
        """Apply Cholesky whitening to the latent space.

        Whitening decorrelates features and normalizes variance to unity.
        Uses Cholesky decomposition for numerically stable whitening.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modify self._latent in place and return self.
            If False, return a copy.

        Returns
        -------
        np.ndarray | LatentSpace
            Whitened latent array, or self if inplace=True.

        Raises
        ------
        ValueError
            If whitening has already been computed.
        """
        if self._whitening_L is not None:
            raise ValueError(
                'Whitening operators already computed. '
                'Create a new LatentSpace instance to re-whiten.'
            )

        X = self._latent

        mean = X.mean(axis=0, keepdims=True)
        X_centered = X - mean

        C = np.cov(X, rowvar=False)

        eps = 1e-6
        C = C + eps * np.eye(C.shape[0])

        L = np.linalg.cholesky(C)

        whitened = solve_triangular(L, X_centered.T, lower=True).T

        self._whitening_mean = mean.squeeze()
        self._whitening_L = L

        if inplace:
            self._latent = whitened
            self._whitened_latent = whitened
            return self

        return whitened

    def dewhiten(self, inplace: bool = False) -> np.ndarray | LatentSpace:
        """Apply the inverse of Cholesky whitening (dewhitening).

        Parameters
        ----------
        inplace : bool, default=False
            If True, modify self._latent in place and return self.

        Returns
        -------
        np.ndarray | LatentSpace
            Dewhitened latent array, or self if inplace=True.

        Raises
        ------
        ValueError
            If whitening operators have not been computed.
        """
        if self._whitening_L is None:
            raise ValueError('Whitening operators not computed. Run prewhiten() first.')

        if inplace and self._whitened_latent is None:
            raise ValueError(
                'No whitened data to dewhiten in place. '
                'Call prewhiten(inplace=True) first.'
            )

        if inplace:
            restored = (
                self._whitened_latent @ self._whitening_L.T
            ) + self._whitening_mean
            self._latent = restored
            self._whitened_latent = None
            return self

        whitened = self._whitened_latent
        if whitened is None:
            whitened = self._latent
        return (whitened @ self._whitening_L.T) + self._whitening_mean

    def apply_whitening_operator(
        self, X: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Apply whitening operator to input data.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor, shape (n, n_features)
            Input data to transform.

        Returns
        -------
        np.ndarray | torch.Tensor
            Whitened data. Output type matches input type.

        Raises
        ------
        ValueError
            If whitening has not been computed.
        """
        if self._whitening_L is None:
            raise ValueError('Whitening operators not computed. Run prewhiten() first.')

        if isinstance(X, torch.Tensor):
            X_arr = X.detach().float()
            L = torch.from_numpy(self._whitening_L)
            X_centered = X_arr - torch.from_numpy(self._whitening_mean)
            return torch.linalg.solve_triangular(L, X_centered.T, upper=False).T
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            X_centered = X_arr - self._whitening_mean
            return solve_triangular(
                self._whitening_L, X_centered.T, lower=True
            ).T.astype(np.float32)

    def apply_dewhitening_operator(
        self, X: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Apply dewhitening operator to input data.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor, shape (n, n_features)
            Input data to transform.

        Returns
        -------
        np.ndarray | torch.Tensor
            Dewhitened data. Output type matches input type.

        Raises
        ------
        ValueError
            If whitening has not been computed.
        """
        if self._whitening_L is None:
            raise ValueError('Whitening operators not computed. Run prewhiten() first.')

        if isinstance(X, torch.Tensor):
            X_arr = X.detach().float()
            L = torch.from_numpy(self._whitening_L).float()
            mean = torch.from_numpy(self._whitening_mean).float()
            return (L @ X_arr.T + mean[:, None]).T
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            return (self._whitening_L @ X_arr.T).T + self._whitening_mean

    def reduce_dimensions(
        self,
        method: DimReductionMethod,
        n_components: int,
        seed: int | None = None,
        prototype_n_samples: int = 10,
        prototype_clusterer_cls: type[ClusterMixin] | None = None,
        prototype_clusterer_kwargs: dict | None = None,
    ) -> np.ndarray:
        """Apply dimensionality reduction.

        Parameters
        ----------
        method : DimReductionMethod
            Reduction method. Choices: pca, umap, tsne, lle, isomap,
            prototype_analysis.
        n_components : int
            Target dimensionality. For prototype_analysis, this becomes
            the number of clusters.
        seed : int, optional
            Random seed.
        prototype_n_samples : int, default=10
            Number of samples per cluster for prototype_analysis.
        prototype_clusterer_cls : type[ClusterMixin], optional
            Clustering class for prototype_analysis. Defaults to KMeans.
        prototype_clusterer_kwargs : dict, optional
            Additional kwargs for the clusterer in prototype_analysis.
        """
        if seed is None:
            seed = self._seed
        rng = np.random.default_rng(seed)
        n_components = min(n_components, self._latent.shape[1], self._latent.shape[0])

        match method:
            case 'pca':
                return (
                    PCA(n_components=n_components, random_state=rng.integers(2**31))
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )
            case 'umap':
                import umap

                return (
                    umap.UMAP(
                        n_components=n_components, random_state=rng.integers(2**31)
                    )
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )
            case 'tsne':
                from sklearn.manifold import TSNE

                return (
                    TSNE(n_components=n_components, random_state=rng.integers(2**31))
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )
            case 'lle':
                return (
                    LocallyLinearEmbedding(
                        n_components=n_components,
                        random_state=rng.integers(2**31),
                    )
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )
            case 'isomap':
                return (
                    Isomap(n_components=n_components)
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )
            case 'prototype_analysis':
                if prototype_clusterer_cls is None:
                    from sklearn.cluster import KMeans

                    prototype_clusterer_cls = KMeans

                self.compute_prototypes(
                    n_samples=prototype_n_samples,
                    clusterer_cls=prototype_clusterer_cls,
                    n_clusters=n_components,
                    clusterer_kwargs=prototype_clusterer_kwargs,
                    apply_parseval=True,
                )
                return self.apply_analysis_operator()
            case _:
                raise ValueError(f'Unknown dim_reduction method {method!r}')

    def compute_prototypes(
        self,
        n_samples: int | None = 10,
        clusters: np.ndarray | None = None,
        clusterer: object | None = None,
        clusterer_cls: type[ClusterMixin] | None = None,
        n_clusters: int | None = None,
        clusterer_kwargs: dict | None = None,
        apply_parseval: bool = True,
        return_cluster_indices: bool = False,
        prewhiten: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[int, np.ndarray]]:
        """Compute cluster prototypes with optional Parseval frame.

        Results are saved to ``prototypes``, ``F``, and ``G`` attributes.

        This method supports three mutually exclusive clustering modes:

        1. **Precomputed clusters**: provide ``clusters``
        2. **External clusterer**: provide ``clusterer``
        3. **Internal clustering**: provide ``clusterer_cls``

        Internal clustering supports both:
        - clusterers that require ``n_clusters`` (e.g. KMeans)
        - clusterers that do not (e.g. DBSCAN)

        Parameters
        ----------
        n_samples : int | None
            Number of samples per cluster to compute prototype.
            If None, use all observations in each cluster to compute centroid.
        clusters : np.ndarray, optional
            Precomputed cluster assignments.
        clusterer : object, optional
            External clustering object implementing ``fit_predict(X)``.
        clusterer_cls : type[ClusterMixin], optional
            Clustering class (e.g., KMeans, DBSCAN).
        n_clusters : int, optional
            Number of clusters (required if clusterer_cls needs it).
        clusterer_kwargs : dict, optional
            Additional kwargs for clusterer.
        apply_parseval : bool
            Whether to apply Parseval frame to prototypes.
        return_cluster_indices : bool
            If True, also return a dict mapping prototype indices to
            arrays of observation indices that belong to each prototype.
        prewhiten : bool, default=False
            If True, apply PCA whitening before clustering and compute
            prototypes in the whitened space. This does not modify
            self._latent in place.

        Returns
        -------
        prototypes : np.ndarray, shape (k, n_features)
            Prototype vectors for each cluster.
        cluster_indices : dict[int, np.ndarray], optional
            Only returned if ``return_cluster_indices=True``.
            Maps prototype index to array of observation indices.
        """
        clusterer_kwargs = clusterer_kwargs or {}

        if clusters is None and clusterer is None and clusterer_cls is None:
            clusterer_cls = KMeans

        latent_for_clustering = self._latent
        if prewhiten and self._whiten_F is None:
            whitened = self.prewhiten()
            latent_for_clustering = whitened

        modes = (
            clusters is not None,
            clusterer is not None,
            clusterer_cls is not None,
        )
        if sum(modes) != 1:
            raise ValueError(
                'Provide exactly one of `clusters`, `clusterer`, or `clusterer_cls`.'
            )

        if clusters is not None:
            clusters = np.asarray(clusters)

        elif clusterer is not None:
            clusters = clusterer.fit_predict(latent_for_clustering)

        else:
            if n_clusters is not None:
                clusterer_kwargs.setdefault('n_clusters', n_clusters)
                if self._seed is not None:
                    clusterer_kwargs.setdefault('random_state', self._seed)

            clusterer = clusterer_cls(**clusterer_kwargs)
            clusters = clusterer.fit_predict(latent_for_clustering)
        unique_clusters = np.unique(clusters)

        prototypes = np.empty(
            (unique_clusters.shape[0], self._latent.shape[1]),
            dtype=np.float32,
        )

        for i, c in enumerate(unique_clusters):
            in_cluster = self._latent[clusters == c]

            if n_samples is not None and in_cluster.shape[0] < n_samples:
                raise ValueError(
                    f'Cluster {c} has {in_cluster.shape[0]} samples, '
                    f'but n_samples={n_samples}.'
                )

            if n_samples is None:
                prototypes[i] = in_cluster.mean(axis=0)
            else:
                rng = np.random.default_rng(self._seed + i)
                idx = rng.choice(in_cluster.shape[0], size=n_samples, replace=False)
                prototypes[i] = in_cluster[idx].mean(axis=0)

        self._prototypes = prototypes

        if apply_parseval:
            self._F, self._G = _parseval_frame(prototypes)
        else:
            self._F = prototypes
            self._G = prototypes.T

        self._prototypes_to_indices: dict[int, np.ndarray] = {}
        if return_cluster_indices:
            for i, c in enumerate(unique_clusters):
                obs_indices = np.where(clusters == c)[0]
                self._prototypes_to_indices[i] = obs_indices
            return prototypes, self._prototypes_to_indices

        return prototypes

    def set_operators(self, use_parseval: bool = True) -> None:
        """Set analysis and synthesis operators from existing prototypes.

        Parameters
        ----------
        use_parseval : bool
            If True, compute Parseval frame operators via SVD.
            If False, set raw operators: F = prototypes, G = F.T.
        """
        if self._prototypes is None:
            raise ValueError(
                "No prototypes computed. Run compute_prototypes() ' first."
            )

        if use_parseval:
            self._F, self._G = _parseval_frame(self._prototypes)
        else:
            self._F = self._prototypes
            self._G = self._prototypes.T

    def compute_artificial_prototypes(
        self,
        cluster_indices: dict[int, np.ndarray],
        n_samples: int | None = None,
        apply_parseval: bool = True,
    ) -> np.ndarray:
        """Compute prototypes using cluster assignments from another model.

        Given cluster assignments produced by a different ``LatentSpace``
        (same observations, different model), computes prototypes from
        *this* model's latent space using those external assignments.
        The Parseval frame operators are updated accordingly.

        Parameters
        ----------
        cluster_indices : dict[int, np.ndarray]
            Cluster assignments from another ``LatentSpace``, mapping
            prototype index → array of observation indices into the shared
            dataset.  All indices must be valid for this instance's latent.
        n_samples : int | None
            Number of observations per cluster to average when computing
            each prototype centroid.  ``None`` uses all observations.
        apply_parseval : bool
            Whether to recompute the Parseval frame operators ``F`` and
            ``G`` from the new prototypes.

        Returns
        -------
        prototypes : np.ndarray, shape (k, n_features)
            Prototype vectors computed from this model's latent space
            using the external cluster structure.
        """
        n_proto = len(cluster_indices)
        prototypes = np.empty((n_proto, self._latent.shape[1]), dtype=np.float32)

        for i, obs_indices in cluster_indices.items():
            in_cluster = self._latent[obs_indices]

            if n_samples is not None:
                rng = np.random.default_rng(self._seed + i)
                size = min(n_samples, len(in_cluster))
                in_cluster = in_cluster[
                    rng.choice(len(in_cluster), size=size, replace=False)
                ]

            prototypes[i] = in_cluster.mean(axis=0)

        self._prototypes = prototypes
        self._prototypes_to_indices = {
            i: np.asarray(obs) for i, obs in cluster_indices.items()
        }

        if apply_parseval:
            self._F, self._G = _parseval_frame(prototypes)
        else:
            self._F = prototypes
            self._G = prototypes.T

        return prototypes

    def apply_analysis_operator(
        self, X: np.ndarray | None = None, use_whitening: bool = False
    ) -> np.ndarray:
        """Apply the Parseval frame analysis operator.

        Parameters
        ----------
        X : np.ndarray, optional
            Input data to transform. If None, transforms self._latent.
        use_whitening : bool, default=False
            If True, use the whitening operator instead of the prototype
            analysis operator.

        Returns
        -------
        np.ndarray, shape (n_points, k)
            Transformed latent data using the analysis operator F.

        Raises
        ------
        ValueError
            If X is provided but prototypes have not been computed
            (when use_whitening=False).
        ValueError
            If X is provided but whitening has not been computed
            (when use_whitening=True).
        ValueError
            If prototypes have not been computed
            (when use_whitening=False and X is None).
        ValueError
            If whitening has not been computed
            (when use_whitening=True and X is None).
        """
        if X is not None:
            X = np.asarray(X, dtype=np.float32)
            if use_whitening:
                if self._whitening_L is None:
                    raise ValueError(
                        'Whitening operators not computed. Run prewhiten() first.'
                    )
                X_centered = X - self._whitening_mean
                return solve_triangular(
                    self._whitening_L, X_centered.T, lower=True
                ).T.astype(np.float32)

            if self._F is None:
                raise ValueError('Run compute_prototypes() first.')
            return (X @ self._F.T).astype(np.float32)

        if use_whitening:
            if self._whitening_L is None:
                raise ValueError(
                    'Whitening operators not computed. Run prewhiten() first.'
                )
            X_centered = self._latent - self._whitening_mean
            return solve_triangular(
                self._whitening_L, X_centered.T, lower=True
            ).T.astype(np.float32)

        if self._F is None:
            raise ValueError('Run compute_prototypes() first.')
        return (self._latent @ self._F.T).astype(np.float32)

    def apply_synthesis_operator(
        self,
        X: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Apply the Parseval frame synthesis operator G.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor, shape (n, k)
            Input data in prototype space to transform to original feature space.

        Returns
        -------
        np.ndarray | torch.Tensor, shape (n, n_features)
            Transformed data using the synthesis operator G.
            Output type matches input type.

        Raises
        ------
        ValueError
            If prototypes have not been computed yet.
        """
        if self._G is None:
            raise ValueError('Run compute_prototypes with apply_parseval=True first.')

        if isinstance(X, torch.Tensor):
            G_tensor = torch.from_numpy(self._G.T)
            return X.detach().float() @ G_tensor
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            return (X_arr @ self._G.T).astype(np.float32)


if __name__ == '__main__':
    pass
