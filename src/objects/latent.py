"""Latent space representation with preprocessing and prototype computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg import solve_triangular
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.cluster import KMeans
import umap

from src.tsp import LaplacianType
from src.objects.anchor import Anchor, AnchorStrategy
from src.objects.graph import Graph

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
    'pca', 'umap', 'tsne', 'lle', 'isomap', 'prototype_analysis', 'eigen_laplacian'
]

GraphLayout = Literal['kamada_kawai', 'spectral']


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

        self._pc_method: str | None = None
        self._pc_embedding: np.ndarray | None = None
        self._pc_axes: np.ndarray | None = None

        self._knn_graph: Graph | None = None
        self._anchor: Anchor | None = None

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

    @property
    def pc_embedding(self) -> np.ndarray | None:
        """Returns cached principal-component scores ``(n_points, n_components)``."""
        return self._pc_embedding

    @property
    def pc_axes(self) -> np.ndarray | None:
        """Returns principal-component loading axes ``(n_components, n_features)``.

        Only available for linear methods (``'pca'``, ``'prototype_analysis'``);
        ``None`` for non-linear reductions.
        """
        return self._pc_axes

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

                return (
                    umap.UMAP(
                        n_components=n_components, random_state=rng.integers(2**31)
                    )
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )
            case 'tsne':

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

    @property
    def knn_graph(self) -> Graph | None:
        """Returns the cached KNN graph if built."""
        return self._knn_graph

    @property
    def anchor(self) -> Anchor | None:
        """Returns the cached Anchor object if computed."""
        return self._anchor

    def build_knn_graph(
        self,
        k: int,
        metric: str = 'euclidean',
        weighted: bool = False,
        mutual: bool = False,
    ) -> Graph:
        """Build and cache a KNN graph from the latent space.

        Parameters
        ----------
        k : int
            Number of nearest neighbours per point.
        metric : str
            Distance metric.
        weighted : bool
            If True, use Gaussian-kernel edge weights.
        mutual : bool
            If True, keep only mutual nearest-neighbour edges.

        Returns
        -------
        Graph
            Graph object wrapping the KNN adjacency matrix.
        """
        self._knn_graph = Graph.from_point_cloud(
            self._latent, k=k, metric=metric, weighted=weighted, mutual=mutual
        )
        return self._knn_graph

    def plot_knn_graph(
        self,
        layout: GraphLayout | np.ndarray = 'kamada_kawai',
        laplacian_normalization: LaplacianType = 'symmetric',
        node_color: str | np.ndarray | None = None,
        node_size: float = 20.0,
        edge_alpha: float = 0.4,
        edge_width: float = 0.5,
        cmap: str = 'tab10',
        figsize: tuple[float, float] = (7.0, 7.0),
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Draw the cached KNN graph.

        Parameters
        ----------
        layout : {'kamada_kawai', 'spectral'} or np.ndarray
            Node placement strategy:

            - ``'kamada_kawai'``: energy-minimisation layout via NetworkX.
            - ``'spectral'``: 2-D coordinates from the 2nd and 3rd smallest
              Laplacian eigenvectors (Fiedler vector and the next one).
            - ``np.ndarray``, shape ``(n_points, 2)``: fixed external
              coordinates passed directly.
        laplacian_normalization : LaplacianType, default='symmetric'
            Laplacian variant used only when ``layout='spectral'``.
        node_color : str, np.ndarray, or None
            Colour specification for nodes:

            - ``str``: key into ``self.extras`` — the corresponding array is
              used as per-node scalar/label values.
            - ``np.ndarray``, shape ``(n_points,)``: per-node scalar or
              integer values mapped through *cmap*.
            - ``None``: uniform colour (matplotlib default).
        node_size : float, default=20.0
            Marker size for nodes.
        edge_alpha : float, default=0.4
            Transparency of edges.
        edge_width : float, default=0.5
            Line width of edges.
        cmap : str, default='tab10'
            Matplotlib colormap applied when *node_color* is an array.
        figsize : (float, float), default=(7.0, 7.0)
            Figure size in inches. Ignored when *ax* is provided.
        ax : plt.Axes, optional
            Existing axes to draw on. A new figure is created when ``None``.
        title : str, optional
            Axes title. Auto-generated from the layout name when ``None``.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes

        Raises
        ------
        ValueError
            If no KNN graph has been built yet (call ``build_knn_graph`` first).
        ValueError
            If *layout* is an array whose shape does not match ``(n_points, 2)``.
        """
        if self._knn_graph is None:
            raise ValueError('No KNN graph found. Call build_knn_graph() first.')

        # Resolve node colour: extras key → array
        color_values: np.ndarray | None = None
        if isinstance(node_color, str):
            if node_color not in self._extras:
                raise ValueError(
                    f'Key {node_color!r} not found in extras. '
                    f'Available: {list(self._extras.keys())}.'
                )
            color_values = self._extras[node_color]
        elif isinstance(node_color, np.ndarray):
            color_values = node_color

        return self._knn_graph.plot(
            layout=layout,
            laplacian_normalization=laplacian_normalization,
            node_color=color_values,
            node_size=node_size,
            edge_alpha=edge_alpha,
            edge_width=edge_width,
            cmap=cmap,
            figsize=figsize,
            ax=ax,
            title=title,
        )

    def compute_principal_components(
        self,
        method: DimReductionMethod,
        n_components: int,
        k: int | None = None,
        seed: int | None = None,
        prototype_n_samples: int = 10,
        prototype_clusterer_cls: type[ClusterMixin] | None = None,
        prototype_clusterer_kwargs: dict | None = None,
        knn_k: int = 10,
        laplacian_normalization: LaplacianType = 'symmetric',
        knn_metric: str = 'euclidean',
        knn_weighted: bool = False,
        knn_mutual: bool = False,
    ) -> np.ndarray:
        """Compute, cache, and return the leading principal-component axes.

        Fits the chosen dimensionality-reduction model on ``self._latent`` and
        stores two attributes:

        * :attr:`pc_embedding` — projected scores, shape ``(n_points, n_components)``.
        * :attr:`pc_axes` — component directions in the original feature space,
          shape ``(n_components, n_features)``.  Available only for linear methods
          (``'pca'``, ``'prototype_analysis'``); ``None`` for non-linear ones.

        Parameters
        ----------
        method : DimReductionMethod
            Reduction algorithm.  Choices: ``'pca'``, ``'umap'``, ``'tsne'``,
            ``'lle'``, ``'isomap'``, ``'prototype_analysis'``,
            ``'eigen_laplacian'``.
        n_components : int
            Total number of components to compute.
        k : int, optional
            Number of leading axes to return.  Defaults to *n_components*.
        seed : int, optional
            Random seed.  Falls back to the instance seed when omitted.
        prototype_n_samples : int, default=10
            Samples per cluster for ``'prototype_analysis'``.
        prototype_clusterer_cls : type[ClusterMixin], optional
            Clustering class for ``'prototype_analysis'``.  Defaults to KMeans.
        prototype_clusterer_kwargs : dict, optional
            Extra kwargs for the clusterer.
        knn_k : int, default=10
            Number of nearest neighbours for ``'eigen_laplacian'``.
        laplacian_normalization : LaplacianType, default='symmetric'
            Laplacian variant for ``'eigen_laplacian'``.
        knn_metric : str, default='euclidean'
            Distance metric for KNN graph construction.
        knn_weighted : bool, default=False
            Use Gaussian-kernel edge weights in the KNN graph.
        knn_mutual : bool, default=False
            Keep only mutual nearest-neighbour edges.

        Returns
        -------
        np.ndarray, shape (k, n_features)
            The first *k* principal-component axes (loading vectors) for linear
            methods.  For non-linear methods and ``'eigen_laplacian'``, where a
            feature-space direction is undefined, returns the embedding score
            vectors transposed, shape ``(k, n_points)``.
        """
        if seed is None:
            seed = self._seed
        rng = np.random.default_rng(seed)
        n_components = min(n_components, self._latent.shape[1], self._latent.shape[0])
        k_eff = min(k if k is not None else n_components, n_components)

        axes: np.ndarray | None = None

        match method:
            case 'pca':
                pca = PCA(
                    n_components=n_components,
                    random_state=rng.integers(2**31),
                )
                embedding = pca.fit_transform(self._latent).astype(np.float32)
                axes = pca.components_.astype(np.float32)

            case 'umap':
                embedding = (
                    umap.UMAP(
                        n_components=n_components,
                        random_state=rng.integers(2**31),
                    )
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )

            case 'tsne':
                embedding = (
                    TSNE(
                        n_components=n_components,
                        random_state=rng.integers(2**31),
                    )
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )

            case 'lle':
                embedding = (
                    LocallyLinearEmbedding(
                        n_components=n_components,
                        random_state=rng.integers(2**31),
                    )
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )

            case 'isomap':
                embedding = (
                    Isomap(n_components=n_components)
                    .fit_transform(self._latent)
                    .astype(np.float32)
                )

            case 'prototype_analysis':
                if prototype_clusterer_cls is None:
                    prototype_clusterer_cls = KMeans

                self.compute_prototypes(
                    n_samples=prototype_n_samples,
                    clusterer_cls=prototype_clusterer_cls,
                    n_clusters=n_components,
                    clusterer_kwargs=prototype_clusterer_kwargs,
                    apply_parseval=True,
                )
                embedding = self.apply_analysis_operator()
                axes = self._F.astype(np.float32) if self._F is not None else None

            case 'eigen_laplacian':
                self.build_knn_graph(
                    k=knn_k,
                    metric=knn_metric,
                    weighted=knn_weighted,
                    mutual=knn_mutual,
                )
                _, eigvecs = self._knn_graph.compute_eigenvectors(
                    k=n_components, normalization=laplacian_normalization
                )
                embedding = eigvecs.astype(np.float32)

            case _:
                raise ValueError(f'Unknown dim_reduction method {method!r}')

        self._pc_method = method
        self._pc_embedding = embedding
        self._pc_axes = axes

        if axes is not None:
            return axes[:k_eff]
        # non-linear methods: return embedding score vectors as rows
        return embedding[:, :k_eff].T

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
        latent_for_clustering = self._latent
        if prewhiten and self._whiten_F is None:
            whitened = self.prewhiten()
            latent_for_clustering = whitened

        anchor = Anchor(latent_for_clustering, strategy='prototype', seed=self._seed)
        anchor.fit(
            n_anchors=n_clusters,
            n_samples=n_samples,
            clusters=clusters,
            clusterer=clusterer,
            clusterer_cls=clusterer_cls,
            clusterer_kwargs=clusterer_kwargs,
            # centroids always in original (non-whitened) space
            centroid_data=self._latent,
        )

        prototypes = anchor.get_anchors()
        self._anchor = anchor
        self._prototypes = prototypes
        self._prototypes_to_indices = anchor.cluster_indices

        if apply_parseval:
            self._F, self._G = _parseval_frame(prototypes)
        else:
            self._F = prototypes
            self._G = prototypes.T

        if return_cluster_indices:
            return prototypes, self._prototypes_to_indices

        return prototypes

    def get_relative(
        self,
        strategy: AnchorStrategy = 'prototype',
        n_anchors: int | None = None,
        clusterer_cls: type[ClusterMixin] | None = None,
        clusterer_kwargs: dict | None = None,
        n_samples: int | None = 10,
        apply_parseval: bool = True,
        force_recompute: bool = False,
    ) -> LatentSpace:
        """Project the latent space into anchor-relative coordinates.

        Each point is expressed as its projection onto the anchor directions
        (analysis operator), yielding an ``(n_points, k)`` representation
        where ``k`` is the number of anchors.  This is the *relative
        representation* of the latent space with respect to the chosen anchors.

        If the analysis operator ``F`` has already been computed and
        ``force_recompute=False``, the existing anchors are reused and anchor
        parameters are ignored (a warning is emitted).

        Parameters
        ----------
        strategy : AnchorStrategy, default='prototype'
            Anchor selection strategy.
        n_anchors : int, optional
            Number of anchors (clusters).  Ignored when operators are cached.
        clusterer_cls : type[ClusterMixin], optional
            Clustering class.  Defaults to ``KMeans``.
        clusterer_kwargs : dict, optional
            Extra kwargs for the clusterer.
        n_samples : int | None, default=10
            Observations per cluster used for centroid estimation.
        apply_parseval : bool, default=True
            Whether to orthonormalise the anchor frame via SVD.
        force_recompute : bool, default=False
            If True, recompute anchors even when ``F`` is already set.

        Returns
        -------
        LatentSpace
            New instance whose latent data are the relative coordinates,
            shape ``(n_points, k)``.
        """
        import warnings

        if self._F is not None and not force_recompute:
            warnings.warn(
                'Reusing cached analysis operator F. '
                'Pass force_recompute=True or call compute_prototypes() first '
                'to use different anchor parameters.',
                stacklevel=2,
            )
        else:
            self.compute_prototypes(
                n_samples=n_samples,
                clusterer_cls=clusterer_cls,
                n_clusters=n_anchors,
                clusterer_kwargs=clusterer_kwargs,
                apply_parseval=apply_parseval,
            )

        relative = self.apply_analysis_operator()
        return LatentSpace(relative, extras=self._extras, seed=self._seed)

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
