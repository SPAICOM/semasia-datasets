"""Graph object with cached Laplacian, spectral decomposition, and structural metrics."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

from src.tsp import (
    LaplacianType,
    build_knn_graph,
    compute_eigenvectors,
    compute_laplacian,
)
from src.metrics.graph import compute_graph_metrics

GraphLayout = Literal['kamada_kawai', 'spectral']


class Graph:
    """Undirected weighted graph with cached Laplacian, eigenvectors, and metrics.

    The primary internal representation is a sparse adjacency matrix.
    NetworkX graph, Laplacian, Laplacian eigenpairs, and structural metrics
    are computed on demand and cached for efficient reuse.

    Parameters
    ----------
    adjacency : sp.spmatrix or np.ndarray, shape (n, n)
        Symmetric adjacency matrix.  Converted to CSR float32 internally.

    Examples
    --------
    >>> g = Graph.from_point_cloud(X, k=10)
    >>> g.compute_laplacian('symmetric')
    >>> vals, vecs = g.compute_eigenvectors(k=8)
    >>> metrics = g.compute_metrics()
    >>> fig, ax = g.plot(layout='spectral', node_color=labels)
    """

    def __init__(self, adjacency: sp.spmatrix | np.ndarray) -> None:
        if not sp.issparse(adjacency):
            adjacency = sp.csr_matrix(np.asarray(adjacency, dtype=np.float32))
        self._adjacency: sp.csr_matrix = adjacency.tocsr().astype(np.float32)
        self._laplacian: sp.csr_matrix | None = None
        self._laplacian_type: LaplacianType | None = None
        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None
        self._nx_graph: nx.Graph | None = None
        self._metrics: dict | None = None

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_point_cloud(
        cls,
        point_cloud: np.ndarray,
        k: int,
        metric: str = 'euclidean',
        weighted: bool = False,
        mutual: bool = False,
    ) -> Graph:
        """Build a KNN graph from a point cloud.

        Parameters
        ----------
        point_cloud : np.ndarray or torch.Tensor, shape (n_points, n_features)
            Input embeddings.
        k : int
            Number of nearest neighbours per point (self excluded).
        metric : str
            Distance metric accepted by
            :class:`sklearn.neighbors.NearestNeighbors`.
        weighted : bool
            If True, edge weights are Gaussian-kernel similarities.
        mutual : bool
            If True, keep only mutual nearest-neighbour edges (intersection).

        Returns
        -------
        Graph
        """
        adjacency = build_knn_graph(
            point_cloud, k=k, metric=metric, weighted=weighted, mutual=mutual
        )
        return cls(adjacency)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def adjacency(self) -> sp.csr_matrix:
        """Symmetric sparse adjacency matrix, shape (n_nodes, n_nodes)."""
        return self._adjacency

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self._adjacency.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of undirected edges (each counted once)."""
        return int(self._adjacency.nnz // 2)

    @property
    def laplacian(self) -> sp.csr_matrix | None:
        """Cached Laplacian matrix. ``None`` until :meth:`compute_laplacian` is called."""
        return self._laplacian

    @property
    def laplacian_type(self) -> LaplacianType | None:
        """Normalization of the currently cached Laplacian."""
        return self._laplacian_type

    @property
    def eigenvalues(self) -> np.ndarray | None:
        """Cached eigenvalues, shape (k,). ``None`` until :meth:`compute_eigenvectors` is called."""
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray | None:
        """Cached eigenvectors, shape (n_nodes, k). ``None`` until :meth:`compute_eigenvectors` is called."""
        return self._eigenvectors

    @property
    def nx_graph(self) -> nx.Graph:
        """NetworkX graph built lazily from the adjacency matrix on first access."""
        if self._nx_graph is None:
            self._nx_graph = nx.from_scipy_sparse_array(self._adjacency)
        return self._nx_graph

    @property
    def metrics(self) -> dict | None:
        """Cached structural metrics. ``None`` until :meth:`compute_metrics` is called."""
        return self._metrics

    # ------------------------------------------------------------------
    # Computation methods
    # ------------------------------------------------------------------

    def compute_laplacian(
        self,
        normalization: LaplacianType = 'symmetric',
    ) -> sp.csr_matrix:
        """Compute and cache the graph Laplacian.

        Changing *normalization* invalidates the cached eigenvectors.

        Parameters
        ----------
        normalization : {'unnormalized', 'symmetric', 'random_walk'}
            Laplacian variant to compute.

        Returns
        -------
        sp.csr_matrix, shape (n_nodes, n_nodes)
        """
        if normalization != self._laplacian_type:
            self._eigenvalues = None
            self._eigenvectors = None
        self._laplacian = compute_laplacian(
            self._adjacency, normalization=normalization
        )
        self._laplacian_type = normalization
        return self._laplacian

    def compute_eigenvectors(
        self,
        k: int,
        normalization: LaplacianType | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and cache the *k* smallest Laplacian eigenpairs.

        If *normalization* is given and differs from the cached Laplacian type,
        the Laplacian is recomputed first.  When no Laplacian is cached yet,
        ``'symmetric'`` is used as the default.

        Parameters
        ----------
        k : int
            Number of eigenpairs to extract.
        normalization : LaplacianType, optional
            Laplacian normalization.  Uses the currently cached type when ``None``.

        Returns
        -------
        eigenvalues : np.ndarray, shape (k,)
        eigenvectors : np.ndarray, shape (n_nodes, k)
        """
        if normalization is not None and normalization != self._laplacian_type:
            self.compute_laplacian(normalization)
        elif self._laplacian is None:
            self.compute_laplacian('symmetric')

        vals, vecs = compute_eigenvectors(self._laplacian, k=k)
        self._eigenvalues = vals
        self._eigenvectors = vecs
        return vals, vecs

    def compute_metrics(self, eigengap_k: int = 1) -> dict:
        """Compute all structural graph metrics and cache them.

        Uses the cached Laplacian when available; computes the default
        ``'symmetric'`` Laplacian on first call if none has been cached yet.

        Parameters
        ----------
        eigengap_k : int
            Position at which the eigengap is measured (see
            :func:`src.metrics.graph.eigengap`).

        Returns
        -------
        dict
            Metric name → value.  Also stored in :attr:`metrics`.

            Keys: ``'cycle_length'``, ``'number_of_cycles'``,
            ``'mean_square_clustering'``, ``'schultz_index'``,
            ``'disorder_number'``, ``'wiener_index'``, ``'girth'``,
            ``'eigengap'``, ``'n_connected_components'``,
            ``'graph_diameter'``, ``'density'``, ``'gutman_index'``,
            ``'degree_distribution'``.
        """
        if self._laplacian is None:
            self.compute_laplacian('symmetric')

        self._metrics = compute_graph_metrics(
            self._adjacency, laplacian=self._laplacian, eigengap_k=eigengap_k
        )
        return self._metrics

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        layout: GraphLayout | np.ndarray = 'kamada_kawai',
        laplacian_normalization: LaplacianType = 'symmetric',
        node_color: np.ndarray | None = None,
        node_size: float = 20.0,
        edge_alpha: float = 0.4,
        edge_width: float = 0.5,
        cmap: str = 'tab10',
        figsize: tuple[float, float] = (7.0, 7.0),
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Draw the graph.

        Parameters
        ----------
        layout : {'kamada_kawai', 'spectral'} or np.ndarray
            Node placement strategy:

            - ``'kamada_kawai'``: energy-minimisation layout via NetworkX.
            - ``'spectral'``: 2-D coordinates from the 2nd and 3rd smallest
              Laplacian eigenvectors (Fiedler vector and the next one).
              Reuses cached eigenvectors when their normalization matches
              *laplacian_normalization* and at least 3 were computed.
            - ``np.ndarray``, shape ``(n_nodes, 2)``: fixed external coordinates.
        laplacian_normalization : LaplacianType, default='symmetric'
            Laplacian variant used for the spectral layout.
        node_color : np.ndarray, shape (n_nodes,), optional
            Per-node scalar or integer values mapped through *cmap*.
            ``None`` gives uniform default colour.
        node_size : float, default=20.0
            Marker size for nodes.
        edge_alpha : float, default=0.4
            Edge transparency.
        edge_width : float, default=0.5
            Edge line width.
        cmap : str, default='tab10'
            Matplotlib colormap for *node_color*.
        figsize : (float, float), default=(7.0, 7.0)
            Figure size in inches.  Ignored when *ax* is provided.
        ax : plt.Axes, optional
            Existing axes to draw into.
        title : str, optional
            Axes title.  Auto-generated when ``None``.

        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes

        Raises
        ------
        ValueError
            If *layout* is an array with wrong shape, or an unknown string.
        """
        G = self.nx_graph

        if isinstance(layout, np.ndarray):
            coords = np.asarray(layout, dtype=float)
            if coords.shape != (self.n_nodes, 2):
                raise ValueError(
                    f'External layout must have shape ({self.n_nodes}, 2), '
                    f'got {coords.shape}.'
                )
            pos = {i: coords[i] for i in range(self.n_nodes)}
            layout_name = 'external'

        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
            layout_name = 'Kamada–Kawai'

        elif layout == 'spectral':
            need_recompute = (
                self._eigenvectors is None
                or self._eigenvectors.shape[1] < 3
                or self._laplacian_type != laplacian_normalization
            )
            if need_recompute:
                self.compute_eigenvectors(k=3, normalization=laplacian_normalization)
            coords = self._eigenvectors[:, 1:3].astype(float)
            pos = {i: coords[i] for i in range(self.n_nodes)}
            layout_name = f'spectral ({laplacian_normalization})'

        else:
            raise ValueError(
                f'Unknown layout {layout!r}. '
                "Choices: 'kamada_kawai', 'spectral', or an (n_nodes, 2) array."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            alpha=edge_alpha,
            width=edge_width,
            edge_color='gray',
        )

        if node_color is not None:
            nc = nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_size=node_size,
                node_color=node_color,
                cmap=cmap,
            )
            fig.colorbar(nc, ax=ax, fraction=0.03, pad=0.02)
        else:
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)

        ax.set_axis_off()
        ax.set_title(
            title if title is not None else f'Graph — {layout_name} layout',
            fontsize=10,
        )
        fig.tight_layout()
        return fig, ax

    def __repr__(self) -> str:
        lap = self._laplacian_type or 'none'
        eig = 'none' if self._eigenvectors is None else str(self._eigenvectors.shape[1])
        return (
            f'Graph(n_nodes={self.n_nodes}, n_edges={self.n_edges}, '
            f'laplacian={lap}, eigenvectors={eig})'
        )


__all__ = ['Graph', 'GraphLayout']
