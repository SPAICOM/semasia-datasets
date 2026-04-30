"""Latent-space visualisation: principal-component correlation heatmaps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.objects.latent import DimReductionMethod, LatentSpace


def _pearson_cross_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation between every column pair of A and B.

    Parameters
    ----------
    A : np.ndarray, shape (n, ka)
    B : np.ndarray, shape (n, kb)

    Returns
    -------
    np.ndarray, shape (ka, kb)
        ``C[i, j] = pearson_r(A[:, i], B[:, j])``.
    """

    # centre and normalise each column to zero-mean unit-variance
    def _standardise(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return (X - mu) / std

    n = A.shape[0]
    As = _standardise(A.astype(np.float64))
    Bs = _standardise(B.astype(np.float64))
    return (As.T @ Bs) / n  # (ka, kb)


def plot_pc_correlation_heatmap(
    latent_a: LatentSpace,
    latent_b: LatentSpace,
    method: DimReductionMethod,
    n_components: int,
    k: int | None = None,
    label_a: str = 'Space A',
    label_b: str = 'Space B',
    use_absolute: bool = False,
    cmap: str = 'coolwarm',
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a Pearson-correlation heatmap between principal components of two
    latent spaces.

    Both spaces must encode the same set of samples (identical ``n_points``).
    Principal components are derived via ``LatentSpace.compute_principal_components``
    using the requested dimensionality-reduction *method*.  Entry ``(i, j)`` of
    the heatmap is the Pearson correlation between the *i*-th component of
    *latent_a* and the *j*-th component of *latent_b*.

    Parameters
    ----------
    latent_a, latent_b : LatentSpace
        The two latent spaces to compare.  Must have the same ``n_points``.
    method : DimReductionMethod
        Dimensionality-reduction algorithm passed to
        ``compute_principal_components``.  Choices: ``'pca'``, ``'umap'``,
        ``'tsne'``, ``'lle'``, ``'isomap'``, ``'prototype_analysis'``.
    n_components : int
        Total number of components to compute before selecting *k*.
    k : int, optional
        Number of leading components to include in the heatmap.
        Defaults to *n_components*.
    label_a, label_b : str
        Axis labels identifying the two spaces (e.g. model names).
    use_absolute : bool
        If True, plot ``|correlation|`` instead of the signed value.
    cmap : str
        Matplotlib colormap.  ``'coolwarm'`` (signed) or ``'Blues'``
        (absolute) work well.
    figsize : (float, float), optional
        Figure size in inches.  Auto-sized from *k* when not provided.
    seed : int
        Random seed forwarded to ``compute_principal_components``.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    """
    if latent_a.n_points != latent_b.n_points:
        raise ValueError(
            f'Both spaces must have the same number of points '
            f'({latent_a.n_points} vs {latent_b.n_points}).'
        )

    # populate pc_embedding on both spaces; ignore the returned axes
    latent_a.compute_principal_components(
        method=method, n_components=n_components, k=k, seed=seed
    )
    latent_b.compute_principal_components(
        method=method, n_components=n_components, k=k, seed=seed
    )

    k_eff_a = latent_a.pc_embedding.shape[1]
    k_eff_b = latent_b.pc_embedding.shape[1]
    k_use = min(k if k is not None else k_eff_a, k_eff_a, k_eff_b)

    # correlate point-wise scores: each PC is a length-n_points vector
    scores_a = latent_a.pc_embedding[:, :k_use]
    scores_b = latent_b.pc_embedding[:, :k_use]
    C = _pearson_cross_correlation(scores_a, scores_b)
    if use_absolute:
        C = np.abs(C)

    k_eff = C.shape[0]
    size = figsize or (max(4.0, k_eff * 0.6), max(3.5, k_eff * 0.55))

    fig, ax = plt.subplots(figsize=size)

    vmax = 1.0
    vmin = -1.0 if not use_absolute else 0.0
    im = ax.imshow(C, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('|Pearson r|' if use_absolute else 'Pearson r', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ticks = np.arange(k_eff)
    tick_labels = [f'PC{i + 1}' for i in range(k_eff)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel(label_b, fontsize=10)
    ax.set_ylabel(label_a, fontsize=10)
    ax.set_title(
        f'PC correlation  [{method}]  {label_a} vs {label_b}',
        fontsize=10,
        pad=8,
    )

    # annotate cells with correlation values
    for i in range(k_eff):
        for j in range(k_eff):
            val = C[i, j]
            text_color = 'white' if abs(val) > 0.65 else 'black'
            ax.text(
                j,
                i,
                f'{val:.2f}',
                ha='center',
                va='center',
                fontsize=max(5, 9 - k_eff // 4),
                color=text_color,
            )

    fig.tight_layout()
    return fig, ax


__all__ = ['plot_pc_correlation_heatmap']
