"""Generate publication-ready forest plots from statistical regression results.

Usage:
    uv run scripts/plot_stat.py
    uv run scripts/plot_stat.py regression_type=pooled
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig

from src.visualizations.statistical import plot_forest


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

    print('\n[Forest plots]')
    print('  Case A: pooled...')
    plot_forest(
        results_dir, output_dir, stat_cases=stat_cases, regression_type='pooled'
    )

    print('  Case B: within-dataset...')
    plot_forest(
        results_dir, output_dir, stat_cases=stat_cases, regression_type='within'
    )

    print('  Case C: interaction...')
    plot_forest(
        results_dir, output_dir, stat_cases=stat_cases, regression_type='interaction'
    )

    print(f'\n[DONE] All figures saved to {output_dir}/')


if __name__ == '__main__':
    main()
