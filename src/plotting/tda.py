"""TDA visualisation: persistence diagrams, persistence images, distance graphs."""

from typing import TYPE_CHECKING, Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    import polars as pl

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

DIM_COLORS: list[str] = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
]

FAMILY_PALETTE: list[str] = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#aec7e8',
    '#ffbb78',
    '#98df8a',
    '#ff9896',
    '#c5b0d5',
    '#c49c94',
    '#f7b6d2',
    '#c7c7c7',
    '#dbdb8d',
    '#9edae5',
]

UNKNOWN_COLOR: str = '#cccccc'
DEFAULT_NODE_COLOR: str = '#4a90d9'


# ---------------------------------------------------------------------------
# Persistence diagram
# ---------------------------------------------------------------------------


def plot_persistence_diagram(
    ax: plt.Axes,
    pts: np.ndarray,
    title: str = '',
) -> None:
    """Scatter-plot a persistence diagram on *ax*.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    pts : np.ndarray, shape (n_pts, 3)
        Array with columns ``[birth, death, dim]`` as produced by
        ``compute_tda_features``.  Infinite death values are silently ignored
        when drawing the diagonal reference line.
    title : str
        Axes title.
    """
    if pts.size == 0:
        ax.text(
            0.5, 0.5, 'empty diagram', ha='center', va='center', transform=ax.transAxes
        )
        return

    dims = sorted({int(d) for d in pts[:, 2]})
    all_finite = pts[np.isfinite(pts[:, 1])]

    for dim in dims:
        mask = (pts[:, 2] == dim) & np.isfinite(pts[:, 1])
        if mask.any():
            ax.scatter(
                pts[mask, 0],
                pts[mask, 1],
                s=8,
                alpha=0.6,
                color=DIM_COLORS[dim % len(DIM_COLORS)],
                label=f'H{dim}',
            )

    if len(all_finite):
        lo = min(all_finite[:, 0].min(), all_finite[:, 1].min())
        hi = max(all_finite[:, 0].max(), all_finite[:, 1].max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.legend(loc='lower right', markerscale=2, fontsize=7)
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Persistence images
# ---------------------------------------------------------------------------


def plot_persistence_images(
    axes: list[plt.Axes],
    images_nested: list,
    max_dim: int,
) -> None:
    """Plot one persistence image per homological dimension.

    Parameters
    ----------
    axes : list[plt.Axes]
        One Axes per homological dimension (length ``max_dim + 1``).
    images_nested : list
        Nested list of shape ``(max_dim+1, n_bins, n_bins)`` as stored in the
        ``persistence_image`` parquet column.
    max_dim : int
        Highest homological dimension.
    """
    for dim in range(max_dim + 1):
        ax = axes[dim]
        img = np.array(images_nested[dim])
        im = ax.imshow(img.T, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        ax.set_xlabel('Birth axis')
        ax.set_ylabel('Persistence axis')
        ax.set_title(f'H{dim}', fontsize=9)


# ---------------------------------------------------------------------------
# Graph layout algorithms
# ---------------------------------------------------------------------------


def compute_mds_layout(
    models: list[str],
    df: 'pl.DataFrame',
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Compute MDS 2-D layout from a pairwise TDA distance DataFrame.

    Parameters
    ----------
    models : list[str]
        Ordered list of model names defining the node set.
    df : pl.DataFrame
        Pairwise distances with columns ``model_a``, ``model_b``, ``distance``.
    seed : int
        Random seed for MDS initialisation.

    Returns
    -------
    dict mapping model name → (x, y) coordinates.
    """
    from sklearn.manifold import MDS

    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    dist_max = float(df['distance'].max())
    D = np.full((n, n), dist_max, dtype=float)
    np.fill_diagonal(D, 0.0)

    for row in df.iter_rows(named=True):
        i = idx.get(row['model_a'])
        j = idx.get(row['model_b'])
        if i is not None and j is not None:
            D[i, j] = row['distance']
            D[j, i] = row['distance']

    pos_array = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=seed,
        normalized_stress='auto',
        n_init=4,
    ).fit_transform(D)

    return {
        m: (float(pos_array[i, 0]), float(pos_array[i, 1]))
        for i, m in enumerate(models)
    }


def compute_circular_layout(
    models: list[str],
) -> dict[str, tuple[float, float]]:
    """Place nodes uniformly on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False)
    return {m: (float(np.cos(a)), float(np.sin(a))) for m, a in zip(models, angles)}


def compute_random_layout(
    models: list[str],
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Place nodes at random positions in [−1, 1]²."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1, 1, size=(len(models), 2))
    return {m: (float(coords[i, 0]), float(coords[i, 1])) for i, m in enumerate(models)}


def compute_layout(
    name: Literal['mds', 'circular', 'random'],
    models: list[str],
    df: 'pl.DataFrame',
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Dispatch to the requested layout algorithm.

    Parameters
    ----------
    name : {'mds', 'circular', 'random'}
        Layout algorithm.  Unknown names fall back to ``'mds'`` with a warning.
    models : list[str]
        Node names.
    df : pl.DataFrame
        Full pairwise distance table (needed by MDS).
    seed : int
        Random seed.
    """
    match name:
        case 'mds':
            return compute_mds_layout(models, df, seed)
        case 'circular':
            return compute_circular_layout(models)
        case 'random':
            return compute_random_layout(models, seed)
        case _:
            print(f'[WARN] Unknown layout {name!r}, falling back to mds')
            return compute_mds_layout(models, df, seed)


# ---------------------------------------------------------------------------
# TDA distance graph
# ---------------------------------------------------------------------------


def plot_tda_distance_graph(
    fig: plt.Figure,
    ax: plt.Axes,
    df_edges: 'pl.DataFrame',
    pos: dict[str, tuple[float, float]],
    models: list[str],
    node_colors: list[str] | None = None,
    edge_width: float = 0.8,
    node_size: float = 10,
    cmap: str = 'coolwarm',
    node_label: Literal['none', 'short', 'full'] = 'short',
    family_handles: list[mpatches.Patch] | None = None,
    distance_label: str = 'distance',
    title: str = '',
) -> None:
    """Draw a TDA model-distance graph onto existing Axes.

    Edges are coloured blue → red by distance via *cmap*.  Node positions are
    given by *pos* (from one of the ``compute_*_layout`` helpers).

    Parameters
    ----------
    fig : plt.Figure
        Parent figure (needed for the colorbar).
    ax : plt.Axes
        Target axes (``ax.set_aspect('equal')`` and ``ax.axis('off')`` are
        applied internally).
    df_edges : pl.DataFrame
        Filtered edge table with columns ``model_a``, ``model_b``, ``distance``.
    pos : dict
        Mapping model name → ``(x, y)`` coordinates.
    models : list[str]
        Ordered list of nodes to draw (must be a subset of *pos* keys).
    node_colors : list[str], optional
        Per-node fill colour.  Defaults to :data:`DEFAULT_NODE_COLOR` for all.
    edge_width : float
        LineCollection linewidth.
    node_size : float
        Scatter marker size (in points, not points²).
    cmap : str
        Matplotlib colormap name for edge colouring.
    node_label : {'none', 'short', 'full'}
        Label style: ``'none'`` skips labels; ``'short'`` uses the first
        dot-separated token; ``'full'`` uses the full model name.
    family_handles : list[mpatches.Patch], optional
        Pre-built legend patches for the family colour legend.
    distance_label : str
        Colorbar label suffix (e.g. the metric name).
    title : str
        Axes title.
    """
    ax.set_aspect('equal')
    ax.axis('off')

    if node_colors is None:
        node_colors = [DEFAULT_NODE_COLOR] * len(models)

    distances = df_edges['distance'].to_numpy()
    d_min, d_max = float(distances.min()), float(distances.max())
    norm = Normalize(vmin=d_min, vmax=d_max)
    colormap = plt.get_cmap(cmap)

    segments, edge_values = [], []
    for row in df_edges.iter_rows(named=True):
        x0, y0 = pos[row['model_a']]
        x1, y1 = pos[row['model_b']]
        segments.append([(x0, y0), (x1, y1)])
        edge_values.append(row['distance'])

    lc = LineCollection(
        segments, cmap=colormap, norm=norm, linewidths=edge_width, alpha=0.7, zorder=1
    )
    lc.set_array(np.array(edge_values))
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(distance_label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    xs = [pos[m][0] for m in models]
    ys = [pos[m][1] for m in models]
    ax.scatter(
        xs,
        ys,
        c=node_colors,
        s=node_size**2,
        zorder=2,
        edgecolors='white',
        linewidths=0.8,
    )

    if node_label != 'none':
        for m, x, y in zip(models, xs, ys):
            label = m.split('.')[0] if node_label == 'short' else m
            ax.text(x, y, label, fontsize=5, ha='center', va='bottom', clip_on=True)

    if family_handles:
        ax.legend(
            handles=family_handles,
            fontsize=7,
            title_fontsize=8,
            loc='upper left',
            framealpha=0.85,
            markerscale=1.2,
        )

    if title:
        ax.set_title(title, fontsize=11, pad=10)

    ax.autoscale()


__all__ = [
    'DIM_COLORS',
    'FAMILY_PALETTE',
    'UNKNOWN_COLOR',
    'DEFAULT_NODE_COLOR',
    'plot_persistence_diagram',
    'plot_persistence_images',
    'compute_mds_layout',
    'compute_circular_layout',
    'compute_random_layout',
    'compute_layout',
    'plot_tda_distance_graph',
]
