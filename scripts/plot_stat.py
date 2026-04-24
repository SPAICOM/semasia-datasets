"""Generate publication-ready figures from statistical regression results.

Usage:
    uv run scripts/plot_stat.py
    uv run scripts/plot_stat.py metric=isotropy stat_case=raw analysis_type=augmentation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import polars as pl
from omegaconf import DictConfig

from src.visualizations.statistical import (
    plot_forest,
    plot_group_means,
    plot_partial_regression,
    plot_residual_diagnostics,
)


@hydra.main(
    config_path='../configs/hydra/',
    config_name='plot_stat',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    results_dir = Path(cfg.results_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stat_cases = cfg.get('stat_cases', ['raw', 'proto_no_prewhiten', 'proto_prewhiten'])
    metrics = cfg.get('metrics', None)

    # --- Figure 1: Forest plots ---
    print('\n[Figure 1] Forest plots...')
    plot_forest(results_dir, output_dir, stat_cases=stat_cases, metrics=metrics)

    # --- Load raw data for Figures 2–4 ---
    raw_path = (
        results_dir / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{cfg.split}.parquet'
    )
    if not raw_path.exists():
        print(f'[ERROR] Raw data not found: {raw_path}')
        return

    df = pl.read_parquet(raw_path)
    metric = cfg.metric
    stat_case = cfg.stat_case
    analysis_type = cfg.analysis_type

    print(
        f'\n[Figures 2-4] metric={metric},'
        f' stat_case={stat_case}, analysis_type={analysis_type}'
    )

    # --- Figure 2: Group means ---
    print('\n[Figure 2] Group means...')
    plot_group_means(df, metric, stat_case, analysis_type, output_dir)

    # --- Figure 3: Partial regression ---
    print('\n[Figure 3] Partial regression...')
    plot_partial_regression(df, metric, stat_case, analysis_type, output_dir)

    # --- Figure 4: Residual diagnostics ---
    print('\n[Figure 4] Residual diagnostics...')
    plot_residual_diagnostics(df, metric, stat_case, analysis_type, output_dir)

    print(f'\n[DONE] All figures saved to {output_dir}/')


if __name__ == '__main__':
    main()
