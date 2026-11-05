"""Aggregation of the semantic-alignment results used by the figure and the report.

Pipeline: seed-average within each (dataset, pair, method, K), then aggregate across
pairs with a t-based confidence interval of the mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

GROUP = ['dataset', 'model_a', 'method', 'k']


def load_results(results_dir: str | Path) -> pl.DataFrame:
    """Concatenate all per-pair parquet files of a results directory."""
    files = sorted(Path(results_dir).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'no parquet files in {results_dir}')
    return pl.concat([pl.read_parquet(f) for f in files], how='diagonal')


def per_pair(df: pl.DataFrame, metrics: list[str]) -> pl.DataFrame:
    """Average over seeds within each (dataset, pair, method, K)."""
    return (
        df.group_by(GROUP, maintain_order=True)
        .agg([pl.col(m).mean() for m in metrics] + [pl.len().alias('n_seeds')])
        .sort(GROUP)
    )


def aggregate_pairs(
    pairs: pl.DataFrame, metrics: list[str], confidence: float = 0.95
) -> pl.DataFrame:
    """Mean, SD (ddof=1), SE and t-interval of the mean across pairs.

    Long format: dataset, method, k, metric, mean, sd, se, ci_low, ci_high, n_pairs.
    The interval is mean +/- t(1-(1-confidence)/2, n-1) * SD / sqrt(n).
    """
    rows = []
    for (dataset, method, k), g in pairs.group_by(
        ['dataset', 'method', 'k'], maintain_order=True
    ):
        for metric in metrics:
            vals = g[metric].drop_nulls().to_numpy().astype(np.float64)
            n = len(vals)
            if n == 0:
                continue
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if n > 1 else float('nan')
            se = sd / np.sqrt(n) if n > 1 else float('nan')
            half = (
                float(stats.t.ppf(1 - (1 - confidence) / 2, n - 1)) * se
                if n > 1
                else float('nan')
            )
            rows.append(
                {
                    'dataset': dataset,
                    'method': method,
                    'k': int(k),
                    'metric': metric,
                    'mean': mean,
                    'sd': sd,
                    'se': se,
                    'ci_low': mean - half,
                    'ci_high': mean + half,
                    'n_pairs': n,
                }
            )
    return pl.DataFrame(rows).sort(['dataset', 'metric', 'method', 'k'])
