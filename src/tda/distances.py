"""Distances between persistence diagrams and topological signatures."""

from typing import Literal

import numpy as np
from persim import bottleneck, wasserstein

DistanceType = Literal['bottleneck', 'wasserstein', 'hausdorff', 'betti_curve']


def _parse_diagram(flat: list[list[float]]) -> list[np.ndarray]:
    """Convert the flat ``[birth, death, dim]`` list stored in parquet to a
    list of per-dimension ``(n_pts, 2)`` arrays (ripser format).

    Infinite-death points are kept as-is; the distance functions handle them.
    """
    arr = np.array(flat, dtype=np.float64)  # (n_pts, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return []
    dims = sorted({int(d) for d in arr[:, 2]})
    max_dim = dims[-1] if dims else 0
    dgms: list[np.ndarray] = []
    for d in range(max_dim + 1):
        mask = arr[:, 2] == d
        dgms.append(arr[mask, :2])  # (n_pts_d, 2)
    return dgms


def _align_dims(
    dgms_a: list[np.ndarray],
    dgms_b: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (dgm_a_d, dgm_b_d) pairs, one per dimension.

    Shorter diagram list is padded with empty arrays so both have the same
    number of dimensions.
    """
    n = max(len(dgms_a), len(dgms_b))
    empty = np.empty((0, 2), dtype=np.float64)
    pairs = []
    for d in range(n):
        a = dgms_a[d] if d < len(dgms_a) else empty
        b = dgms_b[d] if d < len(dgms_b) else empty
        pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Bottleneck distance
# ---------------------------------------------------------------------------

def bottleneck_distance(
    diagram_a: list[list[float]],
    diagram_b: list[list[float]],
) -> float:
    """Bottleneck distance between two persistence diagrams.

    The bottleneck distance is the infimum over all matchings of the maximum
    cost assigned to any matched pair (including diagonal projections):

        d_B(A, B) = inf_{γ} sup_{p ∈ A} ‖p − γ(p)‖_∞

    Computed per homological dimension and summed across dimensions.

    Parameters
    ----------
    diagram_a, diagram_b:
        Flat ``[birth, death, dim]`` lists as stored in parquet (``persistence_diagram``
        column).

    Returns
    -------
    float
        Sum of per-dimension bottleneck distances.
    """
    dgms_a = _parse_diagram(diagram_a)
    dgms_b = _parse_diagram(diagram_b)
    total = 0.0
    for a, b in _align_dims(dgms_a, dgms_b):
        # persim.bottleneck expects finite-only diagrams
        a_fin = a[np.isfinite(a[:, 1])] if len(a) else a
        b_fin = b[np.isfinite(b[:, 1])] if len(b) else b
        if len(a_fin) == 0 and len(b_fin) == 0:
            continue
        total += bottleneck(a_fin, b_fin)
    return float(total)


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------

def wasserstein_distance(
    diagram_a: list[list[float]],
    diagram_b: list[list[float]],
    p: int = 2,
) -> float:
    """p-Wasserstein distance between two persistence diagrams.

    The p-Wasserstein distance is the L^p-cost of the optimal matching
    between diagram points (including diagonal projections):

        W_p(A, B) = ( inf_{γ} Σ_{x ∈ A} ‖x − γ(x)‖_∞^p )^{1/p}

    Computed per homological dimension and summed across dimensions.

    Parameters
    ----------
    diagram_a, diagram_b:
        Flat ``[birth, death, dim]`` lists as stored in parquet.
    p:
        Wasserstein exponent. Default is 2 (2-Wasserstein).

    Returns
    -------
    float
        Sum of per-dimension Wasserstein distances.
    """
    dgms_a = _parse_diagram(diagram_a)
    dgms_b = _parse_diagram(diagram_b)
    total = 0.0
    for a, b in _align_dims(dgms_a, dgms_b):
        a_fin = a[np.isfinite(a[:, 1])] if len(a) else a
        b_fin = b[np.isfinite(b[:, 1])] if len(b) else b
        if len(a_fin) == 0 and len(b_fin) == 0:
            continue
        total += wasserstein(a_fin, b_fin, matching=False)
    return float(total)


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------

def hausdorff_distance(
    diagram_a: list[list[float]],
    diagram_b: list[list[float]],
) -> float:
    """Symmetric Hausdorff distance between two persistence diagrams.

    Both diagrams are treated as point sets in ℝ² (birth, death).  The
    distance is defined as:

        d_H(A, B) = max( max_{a ∈ A} min_{b ∈ B} ‖a − b‖₂,
                         max_{b ∈ B} min_{a ∈ A} ‖b − a‖₂ )

    Computed per homological dimension and summed across dimensions.

    Parameters
    ----------
    diagram_a, diagram_b:
        Flat ``[birth, death, dim]`` lists as stored in parquet.

    Returns
    -------
    float
        Sum of per-dimension Hausdorff distances.
    """
    dgms_a = _parse_diagram(diagram_a)
    dgms_b = _parse_diagram(diagram_b)
    total = 0.0
    for a, b in _align_dims(dgms_a, dgms_b):
        a_fin = a[np.isfinite(a[:, 1])] if len(a) else a
        b_fin = b[np.isfinite(b[:, 1])] if len(b) else b
        if len(a_fin) == 0 and len(b_fin) == 0:
            continue
        if len(a_fin) == 0 or len(b_fin) == 0:
            # One diagram is empty: Hausdorff distance is the largest persistence
            nonempty = a_fin if len(a_fin) else b_fin
            lifetimes = nonempty[:, 1] - nonempty[:, 0]
            total += float(lifetimes.max())
            continue
        # min_{b ∈ B} ‖a_i − b‖₂  for each a_i
        diff_ab = a_fin[:, np.newaxis, :] - b_fin[np.newaxis, :, :]  # (|A|, |B|, 2)
        dist_ab = np.linalg.norm(diff_ab, axis=2)                    # (|A|, |B|)
        forward = dist_ab.min(axis=1).max()                          # max_a min_b

        diff_ba = b_fin[:, np.newaxis, :] - a_fin[np.newaxis, :, :]  # (|B|, |A|, 2)
        dist_ba = np.linalg.norm(diff_ba, axis=2)                    # (|B|, |A|)
        backward = dist_ba.min(axis=1).max()                         # max_b min_a

        total += float(max(forward, backward))
    return float(total)


# ---------------------------------------------------------------------------
# Betti curve distance
# ---------------------------------------------------------------------------

def betti_curve_distance(
    betti_a: list[list[float]],
    betti_b: list[list[float]],
    norm: Literal['l1', 'l2', 'linf'] = 'l2',
) -> float:
    """L^p distance between two Betti number curves.

    Each Betti curve is a 2-D array of shape ``(max_dim+1, n_bins)``.  The
    distance is computed dimension-wise and summed:

        d(A, B) = Σ_d  ‖β_d^A − β_d^B‖_p / n_bins

    The curves are normalised by ``n_bins`` so that the result is independent
    of the grid resolution.

    Parameters
    ----------
    betti_a, betti_b:
        Betti curves as stored in the parquet ``betti_curve`` column –
        nested lists of shape ``(max_dim+1, n_bins)``.
    norm:
        Which norm to use: ``'l1'``, ``'l2'``, or ``'linf'`` (sup-norm).

    Returns
    -------
    float
        Distance between the two Betti curves.
    """
    ca = np.array(betti_a, dtype=np.float64)  # (n_dims_a, n_bins_a)
    cb = np.array(betti_b, dtype=np.float64)  # (n_dims_b, n_bins_b)

    n_dims = max(ca.shape[0], cb.shape[0])
    n_bins = max(ca.shape[1], cb.shape[1])

    # Pad dimensions and bins with zeros so both arrays have the same shape
    def _pad(x: np.ndarray, target_dims: int, target_bins: int) -> np.ndarray:
        padded = np.zeros((target_dims, target_bins), dtype=np.float64)
        padded[: x.shape[0], : x.shape[1]] = x
        return padded

    ca = _pad(ca, n_dims, n_bins)
    cb = _pad(cb, n_dims, n_bins)

    diff = ca - cb  # (n_dims, n_bins)

    if norm == 'l2':
        return float(np.sqrt((diff**2).sum(axis=1)).sum() / n_bins)
    elif norm == 'l1':
        return float(np.abs(diff).sum(axis=1).sum() / n_bins)
    elif norm == 'linf':
        return float(np.abs(diff).max(axis=1).sum())
    else:
        raise ValueError(f"Unknown norm {norm!r}. Choices: 'l1', 'l2', 'linf'.")


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def compute_distance(
    sig_a: dict,
    sig_b: dict,
    distance_type: DistanceType,
    **kwargs,
) -> float:
    """Compute a distance between two TDA signatures.

    Parameters
    ----------
    sig_a, sig_b:
        Dictionaries with keys ``'persistence_diagram'`` and ``'betti_curve'``
        as returned by reading a parquet row (values are already extracted from
        their single-element list wrappers).
    distance_type:
        One of ``'bottleneck'``, ``'wasserstein'``, ``'hausdorff'``,
        ``'betti_curve'``.
    **kwargs:
        Forwarded to the underlying distance function (e.g. ``p=1`` for
        Wasserstein, ``norm='l1'`` for Betti curve).

    Returns
    -------
    float
    """
    if distance_type == 'bottleneck':
        return bottleneck_distance(sig_a['persistence_diagram'], sig_b['persistence_diagram'])
    elif distance_type == 'wasserstein':
        return wasserstein_distance(
            sig_a['persistence_diagram'], sig_b['persistence_diagram'], **kwargs
        )
    elif distance_type == 'hausdorff':
        return hausdorff_distance(sig_a['persistence_diagram'], sig_b['persistence_diagram'])
    elif distance_type == 'betti_curve':
        return betti_curve_distance(sig_a['betti_curve'], sig_b['betti_curve'], **kwargs)
    else:
        raise ValueError(
            f"Unknown distance_type {distance_type!r}. "
            f"Choices: 'bottleneck', 'wasserstein', 'hausdorff', 'betti_curve'."
        )
