"""Topological Data Analysis utilities for latent space characterisation.

Public API
----------
compute_tda_features
    Orchestrates all TDA computations and returns a flat dict ready to be
    merged into a parquet row.
TDA_KEYS
    Ordered list of keys produced by :func:`compute_tda_features`.
    Use it to pre-initialise a dataframe row with ``None`` values.
SimplicialFilter
    Type alias listing the accepted filtration names.
"""

from typing import Literal

from .distances import (
    DistanceType,
    betti_curve_distance,
    bottleneck_distance,
    compute_distance,
    hausdorff_distance,
    wasserstein_distance,
)
from .features import (
    compute_betti_curve,
    compute_diagram_entropy,
    compute_persistence_image,
)
from .persistence import (
    SimplicialFilter,
    compute_persistence_diagram,
)
from .signature import compute_tda_features

__all__ = [
    'compute_tda_features',
    'compute_persistence_diagram',
    'TDA_KEYS',
    'SimplicialFilter',
    'DistanceType',
    'compute_diagram_entropy',
    'compute_persistence_image',
    'compute_betti_curve',
    'bottleneck_distance',
    'wasserstein_distance',
    'hausdorff_distance',
    'betti_curve_distance',
    'compute_distance',
]

# Type aliases (canonical source is LatentSpace in src/objects)
NormalizeMethod = Literal['standard', 'minmax', 'l2']
DimReductionMethod = Literal['pca', 'umap', 'tsne', 'lle', 'isomap']
DimReductionMethodExtended = DimReductionMethod  # alias for backwards compat

# Canonical column names produced by compute_tda_features
TDA_KEYS: list[str] = [
    'persistence_diagram',  # list[list[float]]  shape (n_pts, 3)
    'diagram_entropy',  # list[float]         shape (max_dim+1,)
    'persistence_image',  # list[list[list[float]]]  shape (max_dim+1, n_bins, n_bins)
    'betti_curve',  # list[list[float]]   shape (max_dim+1, n_bins)
]
