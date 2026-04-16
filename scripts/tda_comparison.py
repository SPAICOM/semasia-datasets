"""Pairwise TDA distance comparison across models for a given dataset.

For every pair of models that share the same dataset × split, the script
computes a chosen persistence-diagram distance and saves the full pairwise
distance matrix to a parquet file.

Usage
-----
Run via Hydra (from the repo root):

    python scripts/tda_comparison.py dataset=cifar10 comparison.distance=bottleneck
    python scripts/tda_comparison.py dataset=cifar10 comparison.distance=wasserstein
    python scripts/tda_comparison.py dataset=cifar10 comparison.distance=hausdorff
    python scripts/tda_comparison.py dataset=cifar10 comparison.distance=betti_curve

The output is written to::

    results/tda_distances/<repo_id>__<prefix><dataset>__<split>__<distance>.parquet

Each output parquet has columns: ``model_a``, ``model_b``, ``distance``.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.tda.distances import DistanceType, compute_distance

logging.getLogger('httpx').setLevel(logging.WARNING)

RESULTS_DIR = Path('results/tda_signatures')
OUTPUT_DIR = Path('results/tda_distances')


def _load_signatures(dataset_prefix: str) -> pl.DataFrame:
    """Load all parquet files matching *dataset_prefix* into a single DataFrame."""
    pattern = f'{dataset_prefix}__*.parquet'
    df = pl.read_parquet(RESULTS_DIR / pattern)
    if df.is_empty():
        raise FileNotFoundError(
            f'No signature files found matching: {RESULTS_DIR / pattern}\n'
            'Run scripts/tda_extraction.py first.'
        )
    # Unwrap single-element list columns produced by compute_tda_features
    for col in (
        'persistence_diagram',
        'betti_curve',
        'persistence_image',
        'diagram_entropy',
    ):
        if col in df.columns:
            df = df.with_columns(pl.col(col).list.first().alias(col))
    return df


def _build_distance_matrix(
    df: pl.DataFrame,
    split: str,
    distance_type: DistanceType,
    **distance_kwargs,
) -> pl.DataFrame:
    """Compute pairwise distances for a single *split* and return a DataFrame.

    Returns
    -------
    pl.DataFrame with columns ``model_a``, ``model_b``, ``distance``.
    """
    split_df = df.filter(pl.col('split') == split)
    models = split_df['model'].unique().sort().to_list()
    n = len(models)

    if n < 2:
        print(f'  [{split}] Only {n} model(s) — nothing to compare.')
        return pl.DataFrame({'model_a': [], 'model_b': [], 'distance': []})

    # Index rows by model name for fast lookup
    sig_by_model: dict[str, dict] = {}
    for row in split_df.iter_rows(named=True):
        sig_by_model[row['model']] = {
            'persistence_diagram': row['persistence_diagram'],
            'betti_curve': row['betti_curve'],
        }

    pairs = list(combinations(models, 2))
    model_a_col: list[str] = []
    model_b_col: list[str] = []
    dist_col: list[float] = []

    for ma, mb in tqdm(
        pairs, desc=f'  [{split}] computing {distance_type}', leave=False
    ):
        d = compute_distance(
            sig_by_model[ma],
            sig_by_model[mb],
            distance_type,
            **distance_kwargs,
        )
        model_a_col.append(ma)
        model_b_col.append(mb)
        dist_col.append(d)

    return pl.DataFrame(
        {
            'model_a': model_a_col,
            'model_b': model_b_col,
            'distance': dist_col,
        }
    )


@hydra.main(
    config_path='../configs/hydra/',
    config_name='tda_extraction',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """Entry point – driven by the shared tda_extraction config plus overrides."""
    # Pull comparison-specific overrides (injected via CLI or defaults)
    comp_cfg = OmegaConf.to_container(cfg.get('comparison', {}), resolve=True)
    distance_type: DistanceType = comp_cfg.get('distance', 'bottleneck')
    # Extra kwargs forwarded to the distance function (e.g. p=1 for Wasserstein)
    distance_kwargs: dict = {k: v for k, v in comp_cfg.items() if k != 'distance'}

    valid = ('bottleneck', 'wasserstein', 'hausdorff', 'betti_curve')
    if distance_type not in valid:
        raise ValueError(
            f'comparison.distance must be one of {valid}, got {distance_type!r}.'
        )

    dataset_prefix = f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}'
    print(f'Dataset  : {dataset_prefix}')
    print(f'Distance : {distance_type}  kwargs={distance_kwargs}')

    df = _load_signatures(dataset_prefix)
    splits = df['split'].unique().sort().to_list()
    print(f'Splits   : {splits}')
    print(f'Models   : {df["model"].n_unique()}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        dist_df = _build_distance_matrix(df, split, distance_type, **distance_kwargs)
        if dist_df.is_empty():
            continue

        out_path = OUTPUT_DIR / f'{dataset_prefix}__{split}__{distance_type}.parquet'
        dist_df.write_parquet(out_path)
        print(f'  Saved {len(dist_df)} pairs → {out_path}')


if __name__ == '__main__':
    main()
