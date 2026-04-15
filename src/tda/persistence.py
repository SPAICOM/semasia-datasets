"""Persistence diagram computation via simplicial filtrations (scikit-tda / ripser)."""

from typing import Literal

import numpy as np
import torch
import umap
from ripser import ripser
from sklearn.decomposition import PCA

SimplicialFilter = Literal['VietorisRips']
NormalizeMethod = Literal['standard', 'minmax', 'l2']
DimReductionMethod = Literal['pca', 'umap', 'tsne']

# Maps filter name -> ripser kwargs
_FILTER_KWARGS: dict[str, dict] = {
    'VietorisRips': {'distance_matrix': False},
}


def _normalize_point_cloud(X: np.ndarray, method: str) -> np.ndarray:
    """Normalize a point cloud in-place.

    Parameters
    ----------
    X:
        Array of shape ``(n_points, n_features)``.
    method:
        ``'standard'`` – zero mean, unit std per feature.
        ``'minmax'``   – scale each feature to [0, 1].
        ``'l2'``       – L2-normalise each sample to unit norm.
    """
    if method == 'standard':
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero for constant features
        return ((X - mu) / std).astype(np.float32)
    elif method == 'minmax':
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        rng = hi - lo
        rng[rng == 0] = 1.0
        return ((X - lo) / rng).astype(np.float32)
    elif method == 'l2':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (X / norms).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown normalize method {method!r}. Choices: 'standard', 'minmax', 'l2'."
        )


def _reduce_dimensions(
    X: np.ndarray,
    method: str,
    n_components: int,
    seed: int,
) -> np.ndarray:
    """Apply dimensionality reduction to a point cloud.

    Parameters
    ----------
    X:
        Array of shape ``(n_points, n_features)``.
    method:
        ``'pca'``, ``'umap'``, or ``'tsne'``.
    n_components:
        Target number of dimensions.
    seed:
        Random seed for reproducibility.
    """
    n_components = min(n_components, X.shape[1], X.shape[0])

    if method == 'pca':
        return (
            PCA(n_components=n_components, random_state=seed)
            .fit_transform(X)
            .astype(np.float32)
        )

    elif method == 'umap':
        return (
            umap.UMAP(n_components=n_components, random_state=seed)
            .fit_transform(X)
            .astype(np.float32)
        )

    elif method == 'tsne':
        from sklearn.manifold import TSNE

        return (
            TSNE(n_components=n_components, random_state=seed)
            .fit_transform(X)
            .astype(np.float32)
        )

    else:
        raise ValueError(
            f"Unknown dim_reduction method {method!r}. Choices: 'pca', 'umap', 'tsne'."
        )


def compute_persistence_diagram(
    point_cloud: torch.Tensor | np.ndarray,
    max_dim: int = 4,
    simplicial_filter: SimplicialFilter = 'VietorisRips',
    max_points: int = 1000,
    seed: int = 42,
    normalize: NormalizeMethod | None = None,
    dim_reduction: DimReductionMethod | None = None,
    dim_reduction_components: int = 50,
    **filtration_kwargs,  # forwarded to ripser() — e.g. metric='cosine', thresh=2.0
) -> list[np.ndarray]:
    """Compute persistence diagrams for a point cloud via a simplicial filtration.

    Parameters
    ----------
    point_cloud:
        Array of shape ``(n_points, n_features)`` – e.g. a batch of latent
        space embeddings.
    max_dim:
        Maximum homological dimension (inclusive). Dimensions 0, 1, …,
        ``max_dim`` are all computed.
    simplicial_filter:
        Filtration method.  Currently supported: ``'VietorisRips'`` (via
        ripser).  Extra filtrations (e.g. Čech, Alpha) can be added once
        the corresponding backend is installed.
    max_points:
        Maximum number of points passed to the filtration.  When the point
        cloud is larger, a random landmark subset of this size is drawn.
        Vietoris-Rips simplex indices overflow for large clouds (ripser's
        hard limit is roughly 36 billion simplices), so keeping this at or
        below ~1000 is strongly recommended.  Set to ``-1`` to disable
        subsampling (use only if you know the cloud is small enough).
    seed:
        Random seed used for the landmark subsample and stochastic dim
        reduction methods.
    normalize:
        Optional point-cloud normalisation applied *after* subsampling and
        *before* dim reduction / filtration.

        * ``'standard'`` – zero mean, unit std per feature.
        * ``'minmax'``   – scale each feature to [0, 1].
        * ``'l2'``       – L2-normalise each sample to unit norm.
        * ``None``       – no normalisation (default).
    dim_reduction:
        Optional dimensionality reduction applied after normalisation and
        before the filtration.  Useful when the embedding dimension is large
        (e.g. 512-d or 768-d) because Vietoris-Rips is expensive in high
        dimensions.

        * ``'pca'``  – scikit-learn PCA (fast, linear).
        * ``'umap'`` – UMAP (requires ``umap-learn``).
        * ``'tsne'`` – t-SNE (slow; mainly useful for 2-3 components).
        * ``None``   – no reduction (default).
    dim_reduction_components:
        Target number of dimensions for the reduction step.  Ignored when
        ``dim_reduction`` is ``None``.
    **filtration_kwargs:
        Extra keyword arguments forwarded to :func:`ripser.ripser`
        (e.g. ``metric='cosine'``, ``thresh=2.0``, ``coeff=3``).

    Returns
    -------
    dgms : list of np.ndarray
        ``dgms[d]`` has shape ``(n_pts_d, 2)`` with columns
        ``[birth, death]``.  Points that never die have ``death == np.inf``.
        Length of the list is ``max_dim + 1``.
    """
    if simplicial_filter not in _FILTER_KWARGS:
        raise ValueError(
            f'Unknown simplicial_filter {simplicial_filter!r}. '
            f'Available: {list(_FILTER_KWARGS)}'
        )

    if isinstance(point_cloud, torch.Tensor):
        X = point_cloud.detach().cpu().float().numpy()
    else:
        X = np.asarray(point_cloud, dtype=np.float32)

    # subsampling
    if max_points > 0 and X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]

    # normalizing
    if normalize is not None:
        X = _normalize_point_cloud(X, normalize)

    # dimensionality reduction
    if dim_reduction is not None:
        X = _reduce_dimensions(X, dim_reduction, dim_reduction_components, seed)

    kwargs = {**_FILTER_KWARGS[simplicial_filter], **filtration_kwargs}
    result = ripser(X, maxdim=max_dim, **kwargs)

    return result['dgms']  # list of (n_pts_d, 2) arrays, length max_dim+1
