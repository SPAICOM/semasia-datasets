"""Anchor selection strategies for relative representations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from sklearn.base import ClusterMixin

AnchorStrategy = Literal['prototype']


class Anchor:
    """Anchor points derived from a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray or torch.Tensor, shape (n_points, n_features)
        The source point cloud from which anchors are selected.
    strategy : AnchorStrategy, default='prototype'
        Anchor selection strategy.  Currently only ``'prototype'`` is
        supported: cluster the point cloud and use cluster centroids.
    seed : int, default=42
        Random seed for reproducible clustering and centroid sampling.
    """

    def __init__(
        self,
        point_cloud: np.ndarray | torch.Tensor,
        strategy: AnchorStrategy = 'prototype',
        seed: int = 42,
    ) -> None:
        if isinstance(point_cloud, torch.Tensor):
            self._point_cloud = point_cloud.detach().cpu().float().numpy()
        else:
            self._point_cloud = np.asarray(point_cloud, dtype=np.float32)

        self._strategy = strategy
        self._seed = seed
        self._anchors: np.ndarray | None = None
        self._cluster_indices: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        n_anchors: int | None = None,
        n_samples: int | None = 10,
        clusters: np.ndarray | None = None,
        clusterer: object | None = None,
        clusterer_cls: type[ClusterMixin] | None = None,
        clusterer_kwargs: dict | None = None,
        centroid_data: np.ndarray | None = None,
    ) -> Anchor:
        """Compute anchor points.

        Parameters
        ----------
        n_anchors : int, optional
            Number of anchors.  Used as ``n_clusters`` when ``clusterer_cls``
            requires it.  Ignored when ``clusters`` or ``clusterer`` are
            provided.
        n_samples : int | None, default=10
            Observations per cluster used to estimate the centroid.
            ``None`` uses every point in the cluster.
        clusters : np.ndarray, optional
            Precomputed cluster label array, shape ``(n_points,)``.
        clusterer : object, optional
            Fitted or unfitted clustering object implementing
            ``fit_predict(X)``.
        clusterer_cls : type[ClusterMixin], optional
            Clustering class (e.g. ``KMeans``, ``DBSCAN``).  Defaults to
            ``KMeans`` when all three clustering arguments are ``None``.
        clusterer_kwargs : dict, optional
            Extra kwargs passed to ``clusterer_cls``.
        centroid_data : np.ndarray, optional
            Point cloud used for centroid averaging.  Defaults to the
            constructor's *point_cloud*.  Pass ``LatentSpace._latent`` here
            when clustering on a whitened copy so centroids stay in the
            original space.

        Returns
        -------
        self
        """
        match self._strategy:
            case 'prototype':
                self._fit_prototype(
                    n_anchors=n_anchors,
                    n_samples=n_samples,
                    clusters=clusters,
                    clusterer=clusterer,
                    clusterer_cls=clusterer_cls,
                    clusterer_kwargs=clusterer_kwargs,
                    centroid_data=centroid_data,
                )
            case _:
                raise ValueError(
                    f'Unknown anchor strategy {self._strategy!r}. '
                    "Currently supported: 'prototype'."
                )
        return self

    def get_anchors(self) -> np.ndarray:
        """Return anchor points, shape ``(k, n_features)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._anchors is None:
            raise RuntimeError('Anchor.fit() must be called before get_anchors().')
        return self._anchors

    @property
    def cluster_indices(self) -> dict[int, np.ndarray]:
        """Mapping from anchor index to the observation indices in its cluster."""
        return self._cluster_indices

    @property
    def n_anchors(self) -> int:
        """Number of anchors (0 before :meth:`fit` is called)."""
        return 0 if self._anchors is None else self._anchors.shape[0]

    @property
    def is_fitted(self) -> bool:
        """True after :meth:`fit` has been called successfully."""
        return self._anchors is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_prototype(
        self,
        n_anchors: int | None,
        n_samples: int | None,
        clusters: np.ndarray | None,
        clusterer: object | None,
        clusterer_cls: type[ClusterMixin] | None,
        clusterer_kwargs: dict | None,
        centroid_data: np.ndarray | None,
    ) -> None:
        clusterer_kwargs = dict(clusterer_kwargs) if clusterer_kwargs else {}

        # Default to KMeans when no clustering mode is specified
        if clusters is None and clusterer is None and clusterer_cls is None:
            clusterer_cls = KMeans

        modes = (
            clusters is not None,
            clusterer is not None,
            clusterer_cls is not None,
        )
        if sum(modes) != 1:
            raise ValueError(
                'Provide exactly one of `clusters`, `clusterer`, or `clusterer_cls`.'
            )

        # Resolve cluster labels
        if clusters is not None:
            cluster_labels = np.asarray(clusters)

        elif clusterer is not None:
            cluster_labels = clusterer.fit_predict(self._point_cloud)

        else:
            if n_anchors is not None:
                clusterer_kwargs.setdefault('n_clusters', n_anchors)
            clusterer_kwargs.setdefault('random_state', self._seed)
            cluster_labels = clusterer_cls(**clusterer_kwargs).fit_predict(
                self._point_cloud
            )

        # Data used for centroid averaging (may differ from clustering data)
        data_for_centroids = (
            centroid_data if centroid_data is not None else self._point_cloud
        )

        unique_clusters = np.unique(cluster_labels)
        k = unique_clusters.shape[0]
        d = data_for_centroids.shape[1]
        anchors = np.empty((k, d), dtype=np.float32)

        for i, c in enumerate(unique_clusters):
            mask = cluster_labels == c
            in_cluster = data_for_centroids[mask]

            if n_samples is not None and in_cluster.shape[0] < n_samples:
                raise ValueError(
                    f'Cluster {c} has {in_cluster.shape[0]} samples, '
                    f'but n_samples={n_samples}.'
                )

            if n_samples is None:
                anchors[i] = in_cluster.mean(axis=0)
            else:
                rng = np.random.default_rng(self._seed + i)
                idx = rng.choice(in_cluster.shape[0], size=n_samples, replace=False)
                anchors[i] = in_cluster[idx].mean(axis=0)

            self._cluster_indices[i] = np.where(mask)[0]

        self._anchors = anchors


__all__ = ['Anchor', 'AnchorStrategy']
