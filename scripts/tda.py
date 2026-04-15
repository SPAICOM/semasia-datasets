import logging
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))
logging.getLogger('httpx').setLevel(logging.WARNING)

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import DictConfig

RESULTS_DIR = Path('results/tda_signatures')
OUTPUT_DIR = Path('results/tda_plots')

_DIM_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def _plot_diagram(ax: plt.Axes, pts: np.ndarray, title: str = '') -> None:
    """Scatter-plot a persistence diagram on *ax*."""
    if pts.size == 0:
        ax.text(0.5, 0.5, 'empty diagram', ha='center', va='center', transform=ax.transAxes)
        return

    dims = sorted({int(d) for d in pts[:, 2]})
    all_finite = pts[np.isfinite(pts[:, 1])]

    for dim in dims:
        finite_mask = (pts[:, 2] == dim) & np.isfinite(pts[:, 1])
        if finite_mask.any():
            ax.scatter(
                pts[finite_mask, 0],
                pts[finite_mask, 1],
                s=8,
                alpha=0.6,
                color=_DIM_COLORS[dim % len(_DIM_COLORS)],
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


def _plot_images(axes: list, images_nested: list, max_dim: int) -> None:
    """Plot one persistence image per homological dimension."""
    for dim in range(max_dim + 1):
        ax = axes[dim]
        img = np.array(images_nested[dim])  # (n_bins, n_bins)
        im = ax.imshow(img.T, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        ax.set_xlabel('Birth axis')
        ax.set_ylabel('Persistence axis')
        ax.set_title(f'H{dim}', fontsize=9)


@hydra.main(
    config_path='../configs/hydra/',
    config_name='tda_extraction',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    repo_dataset = f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}'
    pattern = f'{repo_dataset}__*'

    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        print(f'No files found matching: {RESULTS_DIR / pattern}')
        return

    df = pl.read_parquet(files)

    model_pattern: str | None = cfg.get('model') or None
    if model_pattern:
        df = df.filter(pl.col('model').str.contains(model_pattern))

    if df.is_empty():
        print(f'No rows left after filtering model ~ {model_pattern!r}')
        return

    print(f'Plotting {len(df)} row(s) | model filter: {model_pattern or "none"}')
    max_dim: int = cfg.tda.max_dim
    n_dims = max_dim + 1

    for row in df.iter_rows(named=True):
        model_name: str = row['model']
        split: str = row['split']

        out_dir = OUTPUT_DIR / cfg.dataset / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        pts = np.array(row['persistence_diagram'])   # (n_pts, 3): [birth, death, dim]
        images_nested = row['persistence_image']      # list[list[list[float]]] (n_dims, n_bins, n_bins)

        # --- persistence diagram ---
        fig_diag, ax_diag = plt.subplots(figsize=(6, 6))
        _plot_diagram(
            ax_diag,
            pts,
            title=f'{model_name} | {cfg.dataset} | {split}',
        )
        fig_diag.tight_layout()
        diag_path = out_dir / f'{split}__persistence_diagram.png'
        fig_diag.savefig(diag_path, dpi=150)
        plt.close(fig_diag)
        print(f'  Saved: {diag_path}')

        # --- persistence images ---
        fig_img, axes_img = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5))
        axes_list = [axes_img] if n_dims == 1 else list(axes_img)
        fig_img.suptitle(
            f'Persistence Images – {model_name} | {cfg.dataset} | {split}',
            fontsize=10,
        )
        _plot_images(axes_list, images_nested, max_dim)
        fig_img.tight_layout()
        img_path = out_dir / f'{split}__persistence_image.png'
        fig_img.savefig(img_path, dpi=150)
        plt.close(fig_img)
        print(f'  Saved: {img_path}')


if __name__ == '__main__':
    main()
