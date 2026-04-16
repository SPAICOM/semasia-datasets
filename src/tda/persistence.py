"""Persistence diagram computation via simplicial filtrations (ripser)."""

from typing import Literal

import numpy as np
import torch
from ripser import ripser

SimplicialFilter = Literal['VietorisRips']

_FILTER_KWARGS: dict[str, dict] = {
    'VietorisRips': {'distance_matrix': False},
}


def compute_persistence_diagram(
    point_cloud: torch.Tensor | np.ndarray,
    max_dim: int = 4,
    simplicial_filter: SimplicialFilter = 'VietorisRips',
    **filtration_kwargs,
) -> list[np.ndarray]:
    """Compute persistence diagrams from a pre-processed point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray | torch.Tensor, shape (n_points, n_features)
        Pre-processed (subsampled, normalized, dim-reduced) point cloud.
    max_dim : int
        Maximum homological dimension (inclusive).
    simplicial_filter : str
        Filtration method (default: 'VietorisRips').
    **filtration_kwargs
        Extra kwargs for ripser.ripser() (e.g., metric='euclidean', thresh=2.0).

    Returns
    -------
    list of np.ndarray
        ``dgms[d]`` has shape ``(n_pts_d, 2)`` with columns ``[birth, death]``.
        Points that never die have ``death == np.inf``.
    """
    match simplicial_filter:
        case 'VietorisRips':
            kwargs = {**_FILTER_KWARGS['VietorisRips'], **filtration_kwargs}
        case _:
            raise ValueError(
                f'Unknown simplicial_filter {simplicial_filter!r}. '
                f'Available: {list(_FILTER_KWARGS)}'
            )

    if isinstance(point_cloud, torch.Tensor):
        X = point_cloud.detach().cpu().float().numpy()
    else:
        X = np.asarray(point_cloud, dtype=np.float32)

    result = ripser(X, maxdim=max_dim, **kwargs)
    return result['dgms']


__all__ = [
    'compute_persistence_diagram',
    'SimplicialFilter',
]
