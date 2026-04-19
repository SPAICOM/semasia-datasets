"""Entropy-based metrics for latent spaces."""

from __future__ import annotations

import numpy as np


def spectral_entropy(Z: np.ndarray) -> float:
    """Compute spectral entropy from SVD singular values.

    Parameters
    ----------
    Z : np.ndarray, shape (n, d)
        The latent space embedding matrix.

    Returns
    -------
    float
        Spectral entropy H = -sum(p_i * log(p_i)) where p_i are normalized
        singular values.
    """
    Z_c = Z - Z.mean(axis=0)
    sv = np.linalg.svd(Z_c, compute_uv=False)
    sv = sv[sv > 0]
    p = sv / sv.sum()
    return float(-np.sum(p * np.log(p)))


def effective_rank(Z: np.ndarray) -> float:
    """Compute effective rank (entropy rank) of the latent space.

    Parameters
    ----------
    Z : np.ndarray, shape (n, d)
        The latent space embedding matrix.

    Returns
    -------
    float
        Effective rank = exp(spectral_entropy). Represents the effective
        number of dimensions.
    """
    return float(np.exp(spectral_entropy(Z)))


def participation_ratio(Z: np.ndarray) -> float:
    """Compute participation ratio of the latent space.

    Parameters
    ----------
    Z : np.ndarray, shape (n, d)
        The latent space embedding matrix.

    Returns
    -------
    float
        Participation ratio = (sum(s_i^2))^2 / sum(s_i^4). Represents the
        effective number of components with significant variance.
    """
    Z_c = Z - Z.mean(axis=0)
    sv = np.linalg.svd(Z_c, compute_uv=False)
    sv2 = sv**2
    return float(sv2.sum() ** 2 / (sv2**2).sum())


def compute_entropy_metrics(Z: np.ndarray) -> dict[str, float]:
    """Compute all entropy-based metrics.

    Parameters
    ----------
    Z : np.ndarray, shape (n, d)
        The latent space embedding matrix.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'spectral_entropy': Spectral entropy
        - 'effective_rank': Effective rank
        - 'participation_ratio': Participation ratio
    """
    return {
        'spectral_entropy': spectral_entropy(Z),
        'effective_rank': effective_rank(Z),
        'participation_ratio': participation_ratio(Z),
    }
