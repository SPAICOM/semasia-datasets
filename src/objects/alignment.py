"""Alignment methods for pairs of LatentSpace objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial.distance import cdist

from src.objects.latent import LatentSpace
from src.objects.anchor import AnchorStrategy

if TYPE_CHECKING:
    from sklearn.base import ClusterMixin
    from src.tsp import LaplacianType

AlignmentMethod = Literal['relative', 'cca', 'graph', 'ot']


class AlignmentProblem:
    """Align two latent spaces via one of several strategies.

    Parameters
    ----------
    space_a, space_b : LatentSpace
        The two spaces to align.  They may have different ``n_features``
        depending on the alignment method chosen (see :meth:`align`).

    Attributes
    ----------
    aligned_a, aligned_b : LatentSpace or None
        Set after a successful call to :meth:`align`.
    """

    def __init__(self, space_a: LatentSpace, space_b: LatentSpace) -> None:
        self.space_a = space_a
        self.space_b = space_b
        self.aligned_a: LatentSpace | None = None
        self.aligned_b: LatentSpace | None = None

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def align(
        self,
        method: AlignmentMethod,
        **kwargs,
    ) -> tuple[LatentSpace, LatentSpace]:
        """Align the two spaces using the requested strategy.

        Parameters
        ----------
        method : {'relative', 'cca', 'graph', 'ot'}
            Alignment strategy:

            - ``'relative'``: project each space into its own anchor-relative
              coordinate frame (same anchor parameters applied to both).
              Accepts all :meth:`LatentSpace.get_relative` keyword arguments.
            - ``'cca'``: Canonical Correlation Analysis — find linear
              projections that maximise cross-space correlations.
              Keyword: ``n_components`` (int, default 2).
              Requires ``space_a.n_points == space_b.n_points``.
            - ``'graph'``: spectral / functional-map alignment — build KNN
              graphs, compute Laplacian eigenvectors, derive a functional
              map between the two spectral spaces.
              Keywords: ``k_neighbors`` (int, 10), ``n_eigvecs`` (int, 10),
              ``normalization`` (LaplacianType, 'symmetric'),
              ``metric`` (str, 'euclidean'), ``weighted`` (bool, False),
              ``mutual`` (bool, False).
              Requires ``space_a.n_points == space_b.n_points``.
            - ``'ot'``: optimal-transport alignment via Sinkhorn — maps each
              point to a barycentric combination of the other space's points.
              Keywords: ``epsilon`` (float, 0.1), ``max_iter`` (int, 1000),
              ``cost_metric`` (str, 'euclidean').
              Requires ``space_a.n_features == space_b.n_features``.
        **kwargs
            Method-specific keyword arguments (see above).

        Returns
        -------
        aligned_a, aligned_b : tuple[LatentSpace, LatentSpace]
            Aligned representations stored as new ``LatentSpace`` objects.
            Also saved to :attr:`aligned_a` and :attr:`aligned_b`.
        """
        match method:
            case 'relative':
                result = self._align_relative(**kwargs)
            case 'cca':
                result = self._align_cca(**kwargs)
            case 'graph':
                result = self._align_graph(**kwargs)
            case 'ot':
                result = self._align_ot(**kwargs)
            case _:
                raise ValueError(
                    f'Unknown alignment method {method!r}. '
                    "Choices: 'relative', 'cca', 'graph', 'ot'."
                )

        self.aligned_a, self.aligned_b = result
        return result

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _align_relative(
        self,
        strategy: AnchorStrategy = 'prototype',
        n_anchors: int | None = None,
        clusterer_cls: type[ClusterMixin] | None = None,
        clusterer_kwargs: dict | None = None,
        n_samples: int | None = 10,
        apply_parseval: bool = True,
        force_recompute: bool = False,
    ) -> tuple[LatentSpace, LatentSpace]:
        """Project each space into a shared anchor-relative frame.

        Anchors are selected from ``space_a`` only.  The resulting cluster
        assignments are then injected into ``space_b`` so that both spaces use
        the same set of semantic anchor points (expressed in each space's own
        feature dimensions).  This yields comparable ``(n_points, n_anchors)``
        representations.

        Requires ``space_a.n_points == space_b.n_points`` (point correspondence).
        Spaces may have different ``n_features``.
        """
        if self.space_a.n_points != self.space_b.n_points:
            raise ValueError(
                f'Relative alignment requires matching n_points; '
                f'space_a has {self.space_a.n_points}, '
                f'space_b has {self.space_b.n_points}.'
            )

        # Compute anchors from space_a
        self.space_a.compute_prototypes(
            n_clusters=n_anchors,
            n_samples=n_samples,
            clusterer_cls=clusterer_cls,
            clusterer_kwargs=clusterer_kwargs,
            apply_parseval=apply_parseval,
        )

        # Reconstruct per-point cluster label array from space_a's assignments
        cluster_labels = np.empty(self.space_a.n_points, dtype=int)
        for anchor_idx, point_indices in self.space_a.anchor.cluster_indices.items():
            cluster_labels[point_indices] = anchor_idx

        # Inject the same cluster assignments into space_b so its anchors are
        # the corresponding points expressed in space_b's feature space
        self.space_b.compute_prototypes(
            clusters=cluster_labels,
            n_samples=n_samples,
            apply_parseval=apply_parseval,
        )

        rel_a = LatentSpace(
            self.space_a.apply_analysis_operator(),
            extras=self.space_a.extras,
            seed=self.space_a.seed,
        )
        rel_b = LatentSpace(
            self.space_b.apply_analysis_operator(),
            extras=self.space_b.extras,
            seed=self.space_b.seed,
        )
        return rel_a, rel_b

    def _align_cca(
        self,
        n_components: int = 2,
    ) -> tuple[LatentSpace, LatentSpace]:
        """Canonical Correlation Analysis alignment.

        Finds linear projections W_a and W_b such that the projections
        ``latent_a @ W_a`` and ``latent_b @ W_b`` have maximum Pearson
        correlation.  Both output spaces have shape ``(n_points, n_components)``.

        Requires ``space_a.n_points == space_b.n_points``.
        """
        from sklearn.cross_decomposition import CCA

        if self.space_a.n_points != self.space_b.n_points:
            raise ValueError(
                f'CCA requires matching n_points; '
                f'space_a has {self.space_a.n_points}, '
                f'space_b has {self.space_b.n_points}.'
            )

        n_components = min(
            n_components,
            self.space_a.n_features,
            self.space_b.n_features,
            self.space_a.n_points - 1,
        )

        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(self.space_a.latent, self.space_b.latent)

        aligned_a = LatentSpace(
            X_c.astype(np.float32),
            extras=self.space_a.extras,
            seed=self.space_a.seed,
        )
        aligned_b = LatentSpace(
            Y_c.astype(np.float32),
            extras=self.space_b.extras,
            seed=self.space_b.seed,
        )
        return aligned_a, aligned_b

    def _align_graph(
        self,
        k_neighbors: int = 10,
        n_eigvecs: int = 10,
        normalization: LaplacianType = 'symmetric',
        metric: str = 'euclidean',
        weighted: bool = False,
        mutual: bool = False,
    ) -> tuple[LatentSpace, LatentSpace]:
        """Spectral / functional-map graph alignment.

        Builds a KNN graph for each space, computes the ``n_eigvecs`` smallest
        Laplacian eigenvectors, and derives a functional map
        ``C = Phi_b.T @ Phi_a`` (shape ``(n_eigvecs, n_eigvecs)``).  The
        aligned representations are the spectral embeddings themselves
        ``(n_points, n_eigvecs)``; the functional map is stored as
        :attr:`functional_map` for downstream use.

        Requires ``space_a.n_points == space_b.n_points``.
        """
        from src.objects.graph import Graph

        if self.space_a.n_points != self.space_b.n_points:
            raise ValueError(
                f'Graph alignment requires matching n_points; '
                f'space_a has {self.space_a.n_points}, '
                f'space_b has {self.space_b.n_points}.'
            )

        knn_kwargs = dict(k=k_neighbors, metric=metric, weighted=weighted, mutual=mutual)

        graph_a = Graph.from_point_cloud(self.space_a.latent, **knn_kwargs)
        graph_b = Graph.from_point_cloud(self.space_b.latent, **knn_kwargs)

        _, Phi_a = graph_a.compute_eigenvectors(k=n_eigvecs, normalization=normalization)
        _, Phi_b = graph_b.compute_eigenvectors(k=n_eigvecs, normalization=normalization)

        # Resolve sign ambiguity: each eigenvector's largest-magnitude element
        # is made positive so that C is comparable across runs.
        for j in range(Phi_a.shape[1]):
            if Phi_a[np.argmax(np.abs(Phi_a[:, j])), j] < 0:
                Phi_a[:, j] *= -1.0
        for j in range(Phi_b.shape[1]):
            if Phi_b[np.argmax(np.abs(Phi_b[:, j])), j] < 0:
                Phi_b[:, j] *= -1.0

        # Functional map C: converts spectral coefficients from A to B.
        self.functional_map: np.ndarray = Phi_b.T @ Phi_a  # (k, k)

        aligned_a = LatentSpace(
            Phi_a.astype(np.float32),
            extras=self.space_a.extras,
            seed=self.space_a.seed,
        )
        aligned_b = LatentSpace(
            Phi_b.astype(np.float32),
            extras=self.space_b.extras,
            seed=self.space_b.seed,
        )
        return aligned_a, aligned_b

    def _align_ot(
        self,
        epsilon: float = 0.1,
        max_iter: int = 1000,
        cost_metric: str = 'euclidean',
    ) -> tuple[LatentSpace, LatentSpace]:
        """Optimal-transport alignment via Sinkhorn.

        Solves the entropic-regularised optimal-transport problem between the
        two point clouds (uniform marginals) to obtain a transport plan
        ``T`` of shape ``(n_a, n_b)``.  Each point is then mapped to the
        barycentric combination of the other space's points:

        - ``aligned_A[i] = sum_j T[i,j] * latent_b[j]``  (row-normalised)
        - ``aligned_B[j] = sum_i T[i,j] * latent_a[i]``  (column-normalised)

        Requires ``space_a.n_features == space_b.n_features`` (needed to form
        the ground-cost matrix).  Use ``'relative'`` or ``'cca'`` first if
        the spaces have different dimensionalities.
        """
        if self.space_a.n_features != self.space_b.n_features:
            raise ValueError(
                f'OT alignment requires matching n_features; '
                f'space_a has {self.space_a.n_features}, '
                f'space_b has {self.space_b.n_features}. '
                "Apply 'relative' or 'cca' alignment first to bring both "
                'spaces to the same dimensionality.'
            )

        A = self.space_a.latent.astype(np.float64)
        B = self.space_b.latent.astype(np.float64)
        n_a, n_b = A.shape[0], B.shape[0]

        M = cdist(A, B, metric=cost_metric)  # (n_a, n_b) ground-cost matrix

        mu = np.ones(n_a) / n_a
        nu = np.ones(n_b) / n_b
        K = np.exp(-M / epsilon)

        u = np.ones(n_a)
        for _ in range(max_iter):
            v = nu / (K.T @ u + 1e-10)
            u_new = mu / (K @ v + 1e-10)
            if np.max(np.abs(u_new - u)) < 1e-7:
                u = u_new
                break
            u = u_new

        v = nu / (K.T @ u + 1e-10)
        T = u[:, None] * K * v[None, :]  # (n_a, n_b) transport plan

        # Barycentric projections
        T_row = T / (T.sum(axis=1, keepdims=True) + 1e-10)
        aligned_A = (T_row @ B).astype(np.float32)

        T_col = T / (T.sum(axis=0, keepdims=True) + 1e-10)
        aligned_B = (T_col.T @ A).astype(np.float32)

        aligned_a = LatentSpace(
            aligned_A,
            extras=self.space_a.extras,
            seed=self.space_a.seed,
        )
        aligned_b = LatentSpace(
            aligned_B,
            extras=self.space_b.extras,
            seed=self.space_b.seed,
        )
        return aligned_a, aligned_b


__all__ = ['AlignmentProblem', 'AlignmentMethod']
