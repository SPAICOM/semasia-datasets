"""Plot alignment metrics vs compression (K) as grids: receivers x datasets.

One PDF per entry of ``figures`` (accuracy for the main text, MSE for the
supplementary material). Rows are receivers, because the No Mismatch bound and the
raw MSE are specific to the receiver; columns are datasets.

For each (receiver, dataset, method, K) the thick line is the mean over the model
pairs of that receiver of the seed-averaged value; the band is the 95% t interval of
that mean across pairs; the thin lines are the individual (seed-averaged) pairs.
No Mismatch is a dashed line in the accuracy panels only.

Usage:
    uv run python scripts/plot_alignment.py
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


def _draw_figure(
    fig_cfg: DictConfig,
    cfg: DictConfig,
    pairs: pl.DataFrame,
    agg: pl.DataFrame,
    receivers: list[dict],
    datasets: list[str],
    ks: list[int],
) -> None:
    col = fig_cfg.metric_col
    colors, labels = dict(cfg.colors), dict(cfg.labels)
    methods = list(cfg.methods)
    n_rows, n_cols = len(receivers), len(datasets)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cfg.figsize[0], cfg.figsize[1] * n_rows / 2),
        squeeze=False,
    )
    for i, rx in enumerate(receivers):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            where = (pl.col('dataset') == dataset) & (pl.col('model_b') == rx['model'])
            for method in methods:
                color = colors[method]
                sel = agg.filter(
                    where & (pl.col('metric') == col) & (pl.col('method') == method)
                ).sort('k')
                if cfg.pair_lines:
                    for _, g in pairs.filter(
                        where & (pl.col('method') == method)
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
                if cfg.band:
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
            if fig_cfg.no_mismatch:
                nm = agg.filter(
                    where
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
            ax.set_ylabel(fig_cfg.ylabel if j == 0 else '')
            ax.set_xlabel(cfg.xlabel if i == n_rows - 1 else '')
            if i == 0:
                ax.set_title(cfg.titles[dataset])

    handles = [
        Line2D([0], [0], color=colors[m], linewidth=cfg.mean_linewidth) for m in methods
    ]
    names = [labels[m] for m in methods]
    if fig_cfg.no_mismatch:
        handles.append(
            Line2D([0], [0], color=colors['no_mismatch'], linestyle='--', linewidth=1.5)
        )
        names.append(labels['no_mismatch'])
    fig.legend(
        handles,
        names,
        loc='upper center',
        ncol=len(names),
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(cfg.row_title_margin, 0, 1, 0.94 if n_rows > 1 else 0.92))
    # row titles: the receiver name, to the left of the y labels
    for i, rx in enumerate(receivers):
        box = axes[i, 0].get_position()
        fig.text(
            cfg.row_title_x,
            (box.y0 + box.y1) / 2,
            rx['label'],
            rotation=90,
            ha='center',
            va='center',
            fontweight='bold',
        )

    out_pdf = Path(fig_cfg.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'[SAVED] {out_pdf}')


@hydra.main(
    config_path='../configs/hydra/',
    config_name='plot_alignment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    dirs = [d for d in cfg.results_dirs if list(Path(d).glob('*.parquet'))]
    print(f'[INFO] Result directories with data: {dirs}')
    cols = sorted({f.metric_col for f in cfg.figures})
    data = load_results(dirs)
    pairs = per_pair(data, cols)
    agg = aggregate_pairs(pairs, cols, confidence=cfg.confidence)

    out_data = Path(cfg.figure_data)
    out_data.parent.mkdir(parents=True, exist_ok=True)
    agg.rename({'k': 'K'}).write_csv(out_data)

    font = _pick_font(list(cfg.font_candidates))
    print(f'[INFO] Font used: {font}')
    sns.set_theme(style='whitegrid', font=font, font_scale=1.25)
    mpl.rcParams['pdf.fonttype'] = 42  # embed TrueType fonts, text stays vector

    present = set(agg['model_b'].unique().to_list())
    receivers = [dict(r) for r in cfg.receivers if r['model'] in present]
    datasets = [d for d in cfg.titles if d in set(agg['dataset'].unique())]
    ks = sorted(k for k in pairs['k'].unique().to_list() if k > 0)
    for rx in receivers:
        n = pairs.filter(pl.col('model_b') == rx['model'])['model_a'].n_unique()
        print(f'[INFO] receiver {rx["model"]}: {n} transmitters')

    for fig_cfg in cfg.figures:
        _draw_figure(fig_cfg, cfg, pairs, agg, receivers, datasets, ks)
    print(f'[SAVED] {out_data}')


if __name__ == '__main__':
    main()
