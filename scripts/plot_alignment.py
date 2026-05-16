"""Plot alignment metrics vs compression (k) for each dataset.

One figure per metric per dataset. All methods are overlaid in the same axes;
lines show mean ± standard error across model pairs.

Usage:
    uv run python scripts/plot_alignment.py
    uv run python scripts/plot_alignment.py metrics=[accuracy] log_x=true
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from omegaconf import DictConfig

_METHOD_PALETTE = {
    'proto': '#636EFA',
    'cca': '#EF553B',
    'linear': '#00CC96',
}

_NM_COLOR = '#636363'  # grey for the No Mismatch baseline


@hydra.main(
    config_path='../configs/hydra/',
    config_name='plot_alignment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    results_dir = Path(cfg.results_dir)
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(results_dir.glob('*.parquet'))
    if not parquet_files:
        print(f'[WARN] No parquet files found in {results_dir}')
        return

    frames: list[pl.DataFrame] = []
    for path in parquet_files:
        parts = path.stem.split('__')
        dataset = parts[2] if len(parts) >= 3 else path.stem
        df = pl.read_parquet(path).with_columns(
            [
                pl.lit(dataset).alias('dataset'),
                (pl.col('model_a') + ' → ' + pl.col('model_b')).alias('pair'),
            ]
        )
        frames.append(df)

    data = pl.concat(frames, how='diagonal').to_pandas()
    data['k'] = data['k'].astype(int)

    methods: list[str] = list(cfg.get('methods', ['proto', 'cca', 'linear']))
    nm_data = data[data['method'] == 'no_mismatch'].copy()
    data = data[data['method'].isin(methods)]
    if data.empty:
        print(f'[WARN] No data left after filtering for methods {methods}.')
        return

    metrics: list[str] = list(cfg.metrics)
    available = [m for m in metrics if m in data.columns]
    if not available:
        print(f'[WARN] None of the requested metrics {metrics} found in data.')
        return

    present_methods = [m for m in methods if m in data['method'].values]
    palette = {m: _METHOD_PALETTE[m] for m in present_methods if m in _METHOD_PALETTE}
    log_x: bool = bool(cfg.get('log_x', False))

    datasets = data['dataset'].unique()
    print(
        f'[INFO] Datasets: {list(datasets)}  |  Methods: {present_methods}  |  Metrics: {available}'
    )

    sns.set_theme(style='whitegrid', font='serif', font_scale=1.1)

    for dataset in datasets:
        subset = data[data['dataset'] == dataset]
        k_vals = sorted(subset['k'].unique())

        for metric in available:
            fig, ax = plt.subplots(figsize=(6, 4))

            sns.lineplot(
                data=subset,
                x='k',
                y=metric,
                hue='method',
                hue_order=present_methods,
                errorbar='se',
                palette=palette,
                markers=True,
                dashes=False,
                linewidth=1.5,
                ax=ax,
            )

            # No Mismatch baseline: horizontal line (mean ± SE across pairs)
            has_nm = False
            nm_subset = nm_data[
                (nm_data['dataset'] == dataset) & nm_data[metric].notna()
            ]
            if metric != 'mse' and not nm_subset.empty:
                nm_mean = nm_subset[metric].mean()
                ax.axhline(
                    nm_mean,
                    color=_NM_COLOR,
                    linestyle='--',
                    linewidth=1.5,
                    label='No Mismatch',
                )
                has_nm = True

            ax.set_xlabel('k', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(dataset, fontsize=13)

            if log_x:
                ax.set_xscale('log', base=2)

            ax.set_xticks(k_vals)
            ax.set_xticklabels([str(k) for k in k_vals])
            ax.set_xlim(k_vals[0], k_vals[-1])
            if metric == 'accuracy':
                ax.set_ylim(None, 1)

            _LABEL_MAP = {
                'proto': 'Proto',
                'cca': 'CCA',
                'linear': 'Linear',
                'No Mismatch': 'No Mismatch',
            }
            handles, labels = ax.get_legend_handles_labels()
            labels = [_LABEL_MAP.get(l, l) for l in labels]
            ncol = len(present_methods) + (1 if has_nm else 0)
            ax.legend(
                handles,
                labels,
                title=None,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.28),
                ncol=ncol,
                frameon=True,
            )

            fig.tight_layout()
            stem = f'{dataset}__{metric}'
            fig.savefig(plots_dir / f'{stem}.pdf', bbox_inches='tight')
            plt.close(fig)
            print(f'  [SAVED] {stem}.pdf')

    print(f'\n[DONE] Plots saved to {plots_dir}/')


if __name__ == '__main__':
    main()
