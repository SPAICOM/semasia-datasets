"""Visualize a TDA model-distance graph and save it as a PNG.

Each node is a model from the TDA distances file. Each edge is colored
blue → red based on the TDA distance between the two models. The graph
is filtered to the k edges with the lowest distance. Nodes can optionally
be colored by architectural family from the model registry.

Layout uses Multidimensional Scaling on the full pairwise distance matrix
so that topologically similar models cluster together spatially.

Usage:
    python scripts/tda_plot.py
    python scripts/tda_plot.py k_edges=30
    python scripts/tda_plot.py k_edges=30 color_by_family=true
    python scripts/tda_plot.py dataset=cifar10 split=train distance_metric=bottleneck k_edges=50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from omegaconf import DictConfig
from sklearn.manifold import MDS

REGISTRY_URL = 'hf://datasets/spaicom-lab/model-registry/**/*.parquet'

FAMILY_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]
UNKNOWN_COLOR = '#cccccc'
DEFAULT_NODE_COLOR = '#4a90d9'


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _mds_layout(
    models: list[str],
    df: pl.DataFrame,
    seed: int,
) -> dict[str, tuple[float, float]]:
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

    return {m: (float(pos_array[i, 0]), float(pos_array[i, 1])) for i, m in enumerate(models)}


def _circular_layout(models: list[str]) -> dict[str, tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False)
    return {m: (float(np.cos(a)), float(np.sin(a))) for m, a in zip(models, angles)}


def _random_layout(models: list[str], seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1, 1, size=(len(models), 2))
    return {m: (float(coords[i, 0]), float(coords[i, 1])) for i, m in enumerate(models)}


def compute_layout(
    name: str,
    models: list[str],
    df: pl.DataFrame,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if name == 'mds':
        return _mds_layout(models, df, seed)
    elif name == 'circular':
        return _circular_layout(models)
    elif name == 'random':
        return _random_layout(models, seed)
    else:
        print(f'[WARN] Unknown layout {name!r}, falling back to mds')
        return _mds_layout(models, df, seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path='../configs/hydra/',
    config_name='tda_plot',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current = Path('.')

    # ------------------------------------------------------------------
    # Load distances
    # ------------------------------------------------------------------
    dist_path = (
        current
        / 'results/tda_distances'
        / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{cfg.split}__{cfg.distance_metric}.parquet'
    )

    if not dist_path.exists():
        raise FileNotFoundError(
            f'TDA distances file not found: {dist_path}\n'
            'Check repo_id, prefix, dataset, split, and distance_metric.'
        )

    print(f'[INFO] Loading TDA distances from {dist_path}')
    df_all = pl.read_parquet(dist_path)
    all_models = sorted(
        set(df_all['model_a'].to_list()) | set(df_all['model_b'].to_list())
    )
    print(f'  {len(df_all)} pairs · {len(all_models)} unique models')

    # ------------------------------------------------------------------
    # Filter to k edges with lowest distance
    # ------------------------------------------------------------------
    k = cfg.get('k_edges')
    df_edges = df_all.sort('distance')
    if k is not None and int(k) < len(df_edges):
        df_edges = df_edges.head(int(k))
        print(f'[INFO] Retaining {len(df_edges)} edges with lowest distance (k_edges={k})')
    else:
        print(f'[INFO] Retaining all {len(df_edges)} edges')

    edge_models = sorted(
        set(df_edges['model_a'].to_list()) | set(df_edges['model_b'].to_list())
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    d_min_check = float(df_edges['distance'].min())
    d_max_check = float(df_edges['distance'].max())
    layout_name = cfg.layout
    if d_max_check == d_min_check:
        print(
            f'[WARN] All distances are identical ({d_max_check}). '
            'MDS layout is degenerate — falling back to circular.'
        )
        layout_name = 'circular'

    print(f'[INFO] Computing {layout_name} layout for {len(edge_models)} nodes...')
    df_layout = df_all.filter(
        pl.col('model_a').is_in(edge_models) & pl.col('model_b').is_in(edge_models)
    )
    pos = compute_layout(layout_name, edge_models, df_layout, cfg.layout_seed)

    # ------------------------------------------------------------------
    # Optional family coloring
    # ------------------------------------------------------------------
    node_colors = [DEFAULT_NODE_COLOR] * len(edge_models)
    family_handles: list[mpatches.Patch] = []

    if cfg.color_by_family:
        family_col = cfg.get('family_column', 'family')
        print(f'[INFO] Loading model registry for node coloring by {family_col!r}...')
        registry = pl.read_parquet(REGISTRY_URL).select(['model_name', family_col])
        model_to_family: dict[str, str | None] = dict(
            registry.filter(pl.col('model_name').is_in(edge_models)).iter_rows()
        )
        families = sorted({v for v in model_to_family.values() if v is not None})
        family_to_color = {
            fam: FAMILY_PALETTE[i % len(FAMILY_PALETTE)]
            for i, fam in enumerate(families)
        }
        print(f'  Found {len(families)} distinct families')

        node_colors = [
            family_to_color.get(model_to_family.get(m), UNKNOWN_COLOR)
            for m in edge_models
        ]
        family_handles = [
            mpatches.Patch(color=c, label=f) for f, c in family_to_color.items()
        ]

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    dpi = 150
    fig_w = cfg.fig_height / dpi * (16 / 9)  # widescreen aspect
    fig_h = cfg.fig_height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')

    distances = df_edges['distance'].to_numpy()
    d_min, d_max = float(distances.min()), float(distances.max())
    norm = Normalize(vmin=d_min, vmax=d_max)
    cmap = plt.get_cmap('coolwarm')  # blue (low) → red (high)

    # Build LineCollection — much faster than drawing one segment at a time
    segments = []
    edge_values = []
    for row in df_edges.iter_rows(named=True):
        x0, y0 = pos[row['model_a']]
        x1, y1 = pos[row['model_b']]
        segments.append([(x0, y0), (x1, y1)])
        edge_values.append(row['distance'])

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=cfg.edge_width,
        alpha=0.7,
        zorder=1,
    )
    lc.set_array(np.array(edge_values))
    ax.add_collection(lc)

    # Colorbar for edges
    cbar = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f'{cfg.distance_metric} distance', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Nodes
    xs = [pos[m][0] for m in edge_models]
    ys = [pos[m][1] for m in edge_models]
    ax.scatter(
        xs, ys,
        c=node_colors,
        s=cfg.node_size ** 2,
        zorder=2,
        edgecolors='white',
        linewidths=0.8,
    )

    # Node labels
    label_mode = cfg.get('node_label', 'short')
    if label_mode != 'none':
        for m, x, y in zip(edge_models, xs, ys):
            label = m.split('.')[0] if label_mode == 'short' else m
            ax.text(
                x, y,
                label,
                fontsize=5,
                ha='center',
                va='bottom',
                clip_on=True,
            )

    # Family legend
    if family_handles:
        ax.legend(
            handles=family_handles,
            title=cfg.get('family_column', 'family').replace('_', ' ').title(),
            fontsize=7,
            title_fontsize=8,
            loc='upper left',
            framealpha=0.85,
            markerscale=1.2,
        )

    ax.set_title(
        f'TDA Model Distance Graph  ·  {cfg.dataset} / {cfg.split}  ·  '
        f'{cfg.distance_metric}  ·  top {len(df_edges)} edges',
        fontsize=11,
        pad=10,
    )
    ax.autoscale()
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_dir = current / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{cfg.dataset}_{cfg.split}_{cfg.distance_metric}_k{len(df_edges)}'

    out_png = output_dir / f'{stem}.png'
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    print(f'[OUT] Saved → {out_png}')

    if cfg.get('show', False):
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    main()
