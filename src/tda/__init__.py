"""Topological Data Analysis utilities for latent space characterisation.

Public API
----------
compute_tda_signature
    Orchestrates all TDA computations and returns a flat dict ready to be
    merged into a parquet row.
TDA_KEYS
    Ordered list of keys produced by :func:`compute_tda_signature`.
    Use it to pre-initialise a dataframe row with ``None`` values.
SimplicialFilter
    Type alias listing the accepted filtration names.
"""

from .features import (
    compute_betti_curve,
    compute_diagram_entropy,
    compute_persistence_image,
)
from .persistence import (
    DimReductionMethod,
    NormalizeMethod,
    SimplicialFilter,
    compute_persistence_diagram,
)
from .distances import (
    DistanceType,
    bottleneck_distance,
    betti_curve_distance,
    compute_distance,
    hausdorff_distance,
    wasserstein_distance,
)
from .signature import compute_tda_signature

__all__ = [
    'compute_tda_signature',
    'TDA_KEYS',
    'SimplicialFilter',
    'NormalizeMethod',
    'DimReductionMethod',
    'DistanceType',
    'compute_persistence_diagram',
    'compute_diagram_entropy',
    'compute_persistence_image',
    'compute_betti_curve',
    'bottleneck_distance',
    'wasserstein_distance',
    'hausdorff_distance',
    'betti_curve_distance',
    'compute_distance',
]

# Canonical column names produced by compute_tda_signature
TDA_KEYS: list[str] = [
    'persistence_diagram',  # list[list[float]]  shape (n_pts, 3)
    'diagram_entropy',  # list[float]         shape (max_dim+1,)
    'persistence_image',  # list[list[list[float]]]  shape (max_dim+1, n_bins, n_bins)
    'betti_curve',  # list[list[float]]   shape (max_dim+1, n_bins)
]
