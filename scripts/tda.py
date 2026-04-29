import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.plotting.tda import plot_persistence_diagram, plot_persistence_images

logging.getLogger('httpx').setLevel(logging.WARNING)

RESULTS_DIR = Path('results/tda_signatures')
OUTPUT_DIR = Path('results/tda_plots')


@hydra.main(
    config_path='../configs/hydra/',
    config_name='tda_extraction',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    repo_dataset = f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}'
    pattern = f'{repo_dataset}__*.parquet'

    df = pl.read_parquet(RESULTS_DIR / pattern)
    if df.is_empty():
        print(f'No files found matching: {RESULTS_DIR / pattern}')
        return

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

        pts = np.array(row['persistence_diagram'])  # (n_pts, 3): [birth, death, dim]
        images_nested = row[
            'persistence_image'
        ]  # list[list[list[float]]] (n_dims, n_bins, n_bins)

        # --- persistence diagram ---
        fig_diag, ax_diag = plt.subplots(figsize=(6, 6))
        plot_persistence_diagram(
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
        plot_persistence_images(axes_list, images_nested, max_dim)
        fig_img.tight_layout()
        img_path = out_dir / f'{split}__persistence_image.png'
        fig_img.savefig(img_path, dpi=150)
        plt.close(fig_img)
        print(f'  Saved: {img_path}')


if __name__ == '__main__':
    main()
