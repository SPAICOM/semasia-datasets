"""
Point Cloud Metrics — single latent space
==========================================
Geometry-aware metrics to characterise the spread, structure,
and dimensionality of a single point cloud Z (n_samples, n_dims).

Usage:
    python pointcloud_metrics.py

Or import and call compute_all_metrics(Z).
"""

import numpy as np

# ── 1. Scale / spread ─────────────────────────────────────────────────────────


def total_spread(Z: np.ndarray) -> float:
    """Sum of per-dim variances (trace of covariance). O(n·d)."""
    return float(np.var(Z, axis=0).sum())


def mean_distance_to_centroid(Z: np.ndarray) -> float:
    """Mean L2 distance of each point from the centroid. O(n·d)."""
    return float(np.linalg.norm(Z - Z.mean(axis=0), axis=1).mean())


def std_distance_to_centroid(Z: np.ndarray) -> float:
    """Std of L2 distances to centroid — how uniform the spread is. O(n·d)."""
    return float(np.linalg.norm(Z - Z.mean(axis=0), axis=1).std())


def per_dim_variance(Z: np.ndarray) -> dict:
    """Per-dimension variance and std. O(n·d)."""
    v = np.var(Z, axis=0)
    return {
        'per_dim_variance': v.tolist(),
        'mean_variance': float(v.mean()),
        'max_variance': float(v.max()),
        'min_variance': float(v.min()),
        'mean_std': float(np.sqrt(v).mean()),
    }


# ── 2. Dimensionality / shape ─────────────────────────────────────────────────


def effective_rank_fast(Z: np.ndarray) -> float:
    """
    Effective rank via per-dim variances (no SVD). O(n·d).
    Captures how many dimensions actively carry variance.
    """
    v = np.var(Z, axis=0)
    v = v[v > 0]
    p = v / v.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio_fast(Z: np.ndarray) -> float:
    """Participation ratio via per-dim variances. O(n·d)."""
    v = np.var(Z, axis=0)
    return float(v.sum() ** 2 / (v**2).sum())


def pca_summary(Z: np.ndarray, n_components: int = 10) -> dict:
    """
    Fast PCA via covariance matrix. O(n·d + d³).
    Returns top eigenvalues and explained variance ratios.
    """
    Z_c = Z - Z.mean(axis=0)
    cov = np.cov(Z_c, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]  # descending
    eigvals = eigvals[eigvals > 0]
    ratios = eigvals / eigvals.sum()
    cumsum = np.cumsum(ratios)
    k90 = int(np.searchsorted(cumsum, 0.90)) + 1

    return {
        'top_eigenvalues': eigvals[:n_components].tolist(),
        'explained_variance_ratio': ratios[:n_components].tolist(),
        'n_components_90pct': k90,
    }


# ── 3. Isotropy ───────────────────────────────────────────────────────────────


def isotropy(Z: np.ndarray) -> float:
    """
    Ratio of min to max per-dim variance.
    1.0 = perfectly isotropic, ~0 = strongly anisotropic.
    """
    v = np.var(Z, axis=0)
    return float(v.min() / v.max()) if v.max() > 0 else 0.0


def anisotropy_ratio(Z: np.ndarray) -> float:
    """Ratio of largest to smallest per-dim variance."""
    v = np.var(Z, axis=0)
    return float(v.max() / v.min()) if v.min() > 0 else float('inf')


# ── 4. Density ────────────────────────────────────────────────────────────────


def density_estimate(Z: np.ndarray) -> float:
    """
    Approximate density: n_points / total_spread.
    Higher = more points packed into less variance space.
    """
    spread = total_spread(Z)
    return float(len(Z) / spread) if spread > 0 else float('inf')


# ── 5. PCA-derived metrics ───────────────────────────────────────────────


def n_components_90pct(Z: np.ndarray) -> float:
    """Number of PCA components needed to explain 90% of variance."""
    return float(pca_summary(Z)['n_components_90pct'])


def explained_var_ratio_top1(Z: np.ndarray) -> float:
    """Explained variance ratio by top-1 principal component."""
    return float(pca_summary(Z)['explained_variance_ratio'][0])


def explained_var_ratio_top3(Z: np.ndarray) -> float:
    """Explained variance ratio by top-3 principal components."""
    ratios = pca_summary(Z)['explained_variance_ratio']
    return float(sum(ratios[:3]))


def top_eigenvalue_ratio(Z: np.ndarray) -> float:
    """Ratio of top eigenvalue to sum of all eigenvalues."""
    eigs = pca_summary(Z)['top_eigenvalues']
    return float(eigs[0] / sum(eigs))


# ── Aggregate runner ──────────────────────────────────────────────────────────


def compute_all_metrics(Z: np.ndarray, verbose: bool = True) -> dict:
    """
    Compute and (optionally) print all metrics for a single point cloud.

    Parameters
    ----------
    Z       : ndarray, shape (n_samples, n_dims)
    verbose : bool — print a human-readable summary

    Returns
    -------
    dict with all computed metrics
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f'Z must be 2-D (n_samples, n_dims), got {Z.shape}')

    results = {
        'n_samples': len(Z),
        'n_dims': Z.shape[1],
        'total_spread': total_spread(Z),
        'mean_dist_to_centroid': mean_distance_to_centroid(Z),
        'std_dist_to_centroid': std_distance_to_centroid(Z),
        'per_dim': per_dim_variance(Z),
        'effective_rank': effective_rank_fast(Z),
        'participation_ratio': participation_ratio_fast(Z),
        'pca': pca_summary(Z),
        'isotropy': isotropy(Z),
        'anisotropy_ratio': anisotropy_ratio(Z),
        'density': density_estimate(Z),
    }

    if verbose:
        _pretty_print(results)

    return results


def _pretty_print(r: dict):
    sep = '─' * 55
    print(f'\n{sep}')
    print('  POINT CLOUD METRICS')
    print(sep)
    print(f'  Samples: {r["n_samples"]}   Dims: {r["n_dims"]}')

    print('\n[1] SCALE / SPREAD')
    print(f'  Total spread (var trace):    {r["total_spread"]:.4f}')
    print(f'  Mean dist to centroid:       {r["mean_dist_to_centroid"]:.4f}')
    print(f'  Std  dist to centroid:       {r["std_dist_to_centroid"]:.4f}')
    print(f'  Mean std per dim:            {r["per_dim"]["mean_std"]:.4f}')
    print(f'  Max  variance dim:           {r["per_dim"]["max_variance"]:.4f}')
    print(f'  Min  variance dim:           {r["per_dim"]["min_variance"]:.4f}')

    print('\n[2] DIMENSIONALITY / SHAPE')
    print(f'  Effective rank:              {r["effective_rank"]:.4f}')
    print(f'  Participation ratio:         {r["participation_ratio"]:.4f}')
    print(f'  Dims for 90% variance:       {r["pca"]["n_components_90pct"]}')
    top3 = [f'{v:.2f}' for v in r['pca']['top_eigenvalues'][:3]]
    print(f'  Top-3 eigenvalues:           {top3}')

    print('\n[3] ISOTROPY')
    print(f'  Isotropy  (1=sphere):        {r["isotropy"]:.4f}')
    print(f'  Anisotropy ratio (max/min):  {r["anisotropy_ratio"]:.4f}')

    print('\n[4] DENSITY')
    print(f'  Density (n / spread):        {r["density"]:.6f}')
    print(f'\n{sep}\n')


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((300, 20)) * np.linspace(1, 5, 20)  # anisotropic cloud
    compute_all_metrics(Z, verbose=True)
