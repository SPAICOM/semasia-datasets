"""Plot alignment metrics vs compression (K): accuracy (top) and MSE (bottom).

For each (dataset, method, K) the thick line is the mean over the model pairs of the
seed-averaged value; the band is the 95% t interval of that mean across pairs; the
thin lines are the individual (seed-averaged) pairs. No Mismatch is a dashed line in
the accuracy panels only.

Usage:
    uv run python scripts/plot_alignment.py
    uv run python scripts/plot_alignment.py mse_col=nmse
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from src.utils.alignment_aggregate import aggregate_pairs, load_results, per_pair

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _pick_font(candidates: list[str]) -> str:
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    return 'serif'


@hydra.main(
    config_path='../configs/hydra/',
    config_name='plot_alignment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    metric_cols = {'accuracy': cfg.accuracy_col, 'mse': cfg.mse_col}
    cols = list(metric_cols.values())

    data = load_results(cfg.results_dir)
    pairs = per_pair(data, cols)
    agg = aggregate_pairs(pairs, cols, confidence=cfg.confidence)

    out_data = Path(cfg.figure_data)
    out_data.parent.mkdir(parents=True, exist_ok=True)
    agg.rename({'k': 'K'}).write_csv(out_data)

    font = _pick_font(list(cfg.font_candidates))
    print(f'[INFO] Font used: {font}')
    sns.set_theme(style='whitegrid', font=font, font_scale=1.25)
    mpl.rcParams['pdf.fonttype'] = 42  # embed TrueType fonts, text stays vector

    methods = list(cfg.methods)
    datasets = list(cfg.titles.keys())
    datasets = [d for d in datasets if d in set(agg['dataset'].unique())]
    ks = sorted(k for k in pairs['k'].unique().to_list() if k > 0)
    colors, labels = dict(cfg.colors), dict(cfg.labels)

    fig, axes = plt.subplots(
        2, len(datasets), figsize=tuple(cfg.figsize), squeeze=False
    )
    for j, dataset in enumerate(datasets):
        for i, (kind, col) in enumerate(metric_cols.items()):
            ax = axes[i, j]
            for method in methods:
                color = colors[method]
                sel = agg.filter(
                    (pl.col('dataset') == dataset)
                    & (pl.col('metric') == col)
                    & (pl.col('method') == method)
                ).sort('k')
                for _, g in pairs.filter(
                    (pl.col('dataset') == dataset) & (pl.col('method') == method)
                ).group_by('model_a'):
                    g = g.sort('k')
                    ax.plot(
                        g['k'],
                        g[col],
                        color=color,
                        alpha=cfg.pair_alpha,
                        linewidth=cfg.pair_linewidth,
                        zorder=2,
                    )
                ax.fill_between(
                    sel['k'],
                    sel['ci_low'],
                    sel['ci_high'],
                    color=color,
                    alpha=cfg.band_alpha,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    sel['k'],
                    sel['mean'],
                    color=color,
                    linewidth=cfg.mean_linewidth,
                    zorder=3,
                )
            if kind == 'accuracy':
                nm = agg.filter(
                    (pl.col('dataset') == dataset)
                    & (pl.col('metric') == col)
                    & (pl.col('method') == 'no_mismatch')
                )
                ax.axhline(
                    float(nm['mean'][0]),
                    color=colors['no_mismatch'],
                    linestyle='--',
                    linewidth=1.5,
                    zorder=3,
                )
                ax.set_ylim(top=1.0)
            ax.set_xscale('log', base=2)
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(ks[0], ks[-1])
            ax.set_ylabel(
                {'accuracy': 'Accuracy', 'mse': 'MSE'}[kind] if j == 0 else ''
            )
            ax.set_xlabel(cfg.xlabel if i == 1 else '')
            if i == 0:
                ax.set_title(cfg.titles[dataset])

    handles = [
        Line2D([0], [0], color=colors[m], linewidth=cfg.mean_linewidth) for m in methods
    ]
    handles.append(
        Line2D([0], [0], color=colors['no_mismatch'], linestyle='--', linewidth=1.5)
    )
    names = [labels[m] for m in methods] + [labels['no_mismatch']]
    fig.legend(
        handles,
        names,
        loc='upper center',
        ncol=len(names),
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_pdf = Path(cfg.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'alignment.pdf')
    plt.close(fig)
    print(f'[SAVED] {out_pdf}  and  {plots_dir / "alignment.pdf"}')
    print(f'[SAVED] {out_data}  (n_pairs={sorted(set(agg["n_pairs"].to_list()))})')


if __name__ == '__main__':
    main()
