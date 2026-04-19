"""Statistical analysis on latent embeddings.

Usage:
    python scripts/stat_analysis.py dataset=cifar10
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.metrics.entropy import (
    effective_rank,
    participation_ratio,
    spectral_entropy,
)
from src.metrics.pointcloud import (
    anisotropy_ratio,
    density_estimate,
    effective_rank_fast,
    explained_var_ratio_top1,
    explained_var_ratio_top3,
    isotropy,
    mean_distance_to_centroid,
    n_components_90pct,
    participation_ratio_fast,
    std_distance_to_centroid,
    top_eigenvalue_ratio,
    total_spread,
)
from src.objects import LatentSpace

METRIC_DISPATCH = {
    'total_spread': total_spread,
    'mean_distance_to_centroid': mean_distance_to_centroid,
    'std_distance_to_centroid': std_distance_to_centroid,
    'effective_rank_fast': effective_rank_fast,
    'participation_ratio_fast': participation_ratio_fast,
    'isotropy': isotropy,
    'anisotropy_ratio': anisotropy_ratio,
    'density': density_estimate,
    'n_components_90pct': n_components_90pct,
    'explained_var_ratio_top1': explained_var_ratio_top1,
    'explained_var_ratio_top3': explained_var_ratio_top3,
    'top_eigenvalue_ratio': top_eigenvalue_ratio,
    'spectral_entropy': spectral_entropy,
    'effective_rank': effective_rank,
    'participation_ratio': participation_ratio,
}

logging.getLogger('httpx').setLevel(logging.WARNING)

DATASET_SPLITS = {
    'cifar10': {'train', 'test'},
    'cifar100': {'train', 'test'},
    'mnist': {'train', 'test'},
    'fashion_mnist': {'train', 'test'},
    'imagenet-1k': {'validation', 'test'},
    'tiny-imagenet': {'train'},
    'celeba': {'train', 'test'},
    'svhn': {'train', 'test'},
}


@hydra.main(
    config_path='../configs/hydra/',
    config_name='stat_analysis',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current: Path = Path('.')
    results_dir: Path = current / 'results/stat'

    results_dir.mkdir(parents=True, exist_ok=True)

    dataset: str = f'{cfg.repo_id}/{cfg.prefix}{cfg.dataset}'

    split = cfg.split
    valid_splits = DATASET_SPLITS.get(cfg.dataset, {'train'})
    if split not in valid_splits:
        raise ValueError(f'Invalid split {split!r} for dataset {cfg.dataset}.')

    print('\n[INFO] Loading model registry...')
    model_df = pl.read_parquet('hf://datasets/spaicom-lab/model-registry/**/*.parquet')
    model_df = model_df.with_columns(
        pl.col('model_name').str.split('.').list.first().alias('arch_key')
    )

    pairs_df = build_pairs(model_df)
    print(f'  Total pairs: {len(pairs_df)}')
    for at in pairs_df['analysis_type'].unique().to_list():
        count = len(pairs_df.filter(pl.col('analysis_type') == at))
        print(f'    {at}: {count}')

    regression_only = cfg.get('regression_only', False)

    if regression_only:
        print('\n[MODE] Running regression on existing metrics...')
        metrics_path = (
            results_dir
            / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{cfg.split}.parquet'
        )
        if not metrics_path.exists():
            print(f'[ERROR] Metrics file not found: {metrics_path}')
            print('  Run compute phase first.')
            return

        df = pl.read_parquet(metrics_path)
        outcomes = cfg.get(
            'metrics', ['effective_rank_fast', 'participation_ratio_fast']
        )
        run_stat_regression(df, results_dir, outcomes)
        return

    clusterer_cls = hydra.utils.instantiate(cfg.clustering)

    all_models = set(
        pairs_df['control_model'].to_list() + pairs_df['treatment_model'].to_list()
    )
    all_models.discard(None)

    print(f'  Total unique models to process: {len(all_models)}')

    download_only = cfg.get('download_only', False)

    if download_only:
        print('\n[PHASE 1] Downloading all model latents (no computation)...')
    else:
        print('\n[PHASE 2] Computing metrics from cached latents...')

    if download_only:
        for model_name in tqdm(sorted(all_models), desc='Downloading latents'):
            try:
                load_latent(dataset, model_name, split)
            except Exception as e:
                print(f'Error downloading {model_name}: {e}')
                continue

        print('\n[COMPLETE] All latents downloaded.')
        print('  Run again with download_only: false for computation.')
        return

    print('\n[INFO] Computing metrics for unique models...')
    model_metrics = {}
    for model_name in tqdm(sorted(all_models), desc='Computing metrics'):
        try:
            model_metrics[model_name] = process_model(
                dataset=dataset,
                model_name=model_name,
                split=split,
                cfg=cfg,
                clusterer_cls=clusterer_cls,
            )
        except Exception as e:
            print(f'Error with {model_name}: {e}')
            continue

    print(f'  Computed metrics for {len(model_metrics)} models')

    print('\n[INFO] Building results...')
    all_results = []

    for row in pairs_df.iter_rows(named=True):
        analysis_type = row['analysis_type']
        control_model = row['control_model']
        treatment_model = row['treatment_model']
        arch_key = row['arch_key']

        control_metrics = model_metrics.get(control_model)
        if control_metrics is not None:
            all_results.append(
                {
                    'analysis_type': analysis_type,
                    'arch_key': arch_key,
                    'model_name': control_model,
                    'is_treatment': 0,
                    **control_metrics,
                }
            )

        treatment_metrics = model_metrics.get(treatment_model)
        if treatment_metrics is not None:
            all_results.append(
                {
                    'analysis_type': analysis_type,
                    'arch_key': arch_key,
                    'model_name': treatment_model,
                    'is_treatment': 1,
                    **treatment_metrics,
                }
            )

    if all_results:
        df = pl.DataFrame(all_results)
        if cfg.get('save_results', True):
            output_path = (
                results_dir
                / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{split}.parquet'
            )
            df.write_parquet(output_path)
            print(f'\n[Saved] {len(df)} results to {output_path}')
        print('\n[RESULTS] DataFrame preview:')
        print(df.head(10))

        outcomes = cfg.get(
            'metrics', ['effective_rank_fast', 'participation_ratio_fast']
        )
        run_stat_regression(df, results_dir, outcomes)
    else:
        print('\n[ERROR] No results to save.')


def build_pairs(model_df: pl.DataFrame) -> pl.DataFrame:
    """Build all analysis pairs with correct constraints.

    1. dataset_change: same (arch_key, aug), both no ft, different ds
    2. finetuning_large: same (arch_key, pretrain_dataset, aug), with/without ft
    3. finetuning_final: in1k no ft vs any large->ft to 1k, same (arch_key, aug)
    4. augmentation: same (arch_key, pretrain_dataset, ft), with/without aug
    """
    rows = []

    no_ft = model_df.filter(pl.col('pretrain_ft').is_null())
    ft_to_1k = model_df.filter(pl.col('pretrain_ft') == 'ImageNet-1K')

    # ============================================================
    # 1. DATASET CHANGE: same (arch_key, aug), both no ft, different ds
    # ============================================================
    for aug_val in [None] + model_df.filter(pl.col('pretrain_aug').is_not_null())[
        'pretrain_aug'
    ].unique().to_list():
        if aug_val is None:
            aug_no_ft = no_ft.filter(pl.col('pretrain_aug').is_null())
        else:
            aug_no_ft = no_ft.filter(pl.col('pretrain_aug') == aug_val)

        arch_datasets = aug_no_ft.group_by('arch_key').agg(
            pl.col('model_name').alias('models'),
            pl.col('pretrain_dataset').alias('datasets'),
        )

        for row in arch_datasets.iter_rows(named=True):
            arch_key = row['arch_key']
            models = row['models']
            datasets = row['datasets']

            for i, (m1, d1) in enumerate(zip(models, datasets)):
                for m2, d2 in zip(models[i + 1 :], datasets[i + 1 :]):
                    rows.append(
                        {
                            'analysis_type': 'dataset_change',
                            'control_model': m1,
                            'treatment_model': m2,
                            'arch_key': arch_key,
                        }
                    )

    # ============================================================
    # 2. FINETUNING_LARGE: same (arch_key, pretrain_dataset, aug), with/without ft
    # ============================================================
    for aug_val in [None] + model_df.filter(pl.col('pretrain_aug').is_not_null())[
        'pretrain_aug'
    ].unique().to_list():
        if aug_val is None:
            no_ft_filtered = no_ft.filter(pl.col('pretrain_aug').is_null())
            ft_filtered = ft_to_1k.filter(pl.col('pretrain_aug').is_null())
        else:
            no_ft_filtered = no_ft.filter(pl.col('pretrain_aug') == aug_val)
            ft_filtered = ft_to_1k.filter(pl.col('pretrain_aug') == aug_val)

        pairs_all = ft_filtered.select(
            ['arch_key', 'model_name', 'pretrain_dataset']
        ).join(
            no_ft_filtered.select(['arch_key', 'model_name', 'pretrain_dataset']),
            on=['arch_key', 'pretrain_dataset'],
            suffix='_ctrl',
        )

        pairs = pairs_all.filter(pl.col('pretrain_dataset').is_not_null())

        rows.extend(
            [
                {
                    'analysis_type': 'finetuning_large',
                    'control_model': row['model_name_ctrl'],
                    'treatment_model': row['model_name'],
                    'arch_key': row['arch_key'],
                }
                for row in pairs.iter_rows(named=True)
            ]
        )

    # ============================================================
    # 3. FINETUNING_FINAL: control=ImageNet-1K no ft, treatment=any→ft→1K
    # Same (arch_key, aug), control is in1k no ft, treatment is ft to 1k
    # ============================================================
    in1k_no_ft = model_df.filter(
        (pl.col('pretrain_dataset') == 'ImageNet-1K') & pl.col('pretrain_ft').is_null()
    )

    for aug_val in [None] + model_df.filter(pl.col('pretrain_aug').is_not_null())[
        'pretrain_aug'
    ].unique().to_list():
        if aug_val is None:
            in1k_no_ft_aug = in1k_no_ft.filter(pl.col('pretrain_aug').is_null())
            ft_to_1k_aug = ft_to_1k.filter(pl.col('pretrain_aug').is_null())
        else:
            in1k_no_ft_aug = in1k_no_ft.filter(pl.col('pretrain_aug') == aug_val)
            ft_to_1k_aug = ft_to_1k.filter(pl.col('pretrain_aug') == aug_val)

        pairs = ft_to_1k_aug.select(['arch_key', 'model_name']).join(
            in1k_no_ft_aug.select(['arch_key', 'model_name']),
            on='arch_key',
            suffix='_ctrl',
        )

        rows.extend(
            [
                {
                    'analysis_type': 'finetuning_final',
                    'control_model': row['model_name_ctrl'],
                    'treatment_model': row['model_name'],
                    'arch_key': row['arch_key'],
                }
                for row in pairs.iter_rows(named=True)
            ]
        )

    # ============================================================
    # 4. AUGMENTATION: same (arch_key, pretrain_dataset, ft), with/without aug
    # ============================================================
    all_with_aug = model_df.filter(pl.col('pretrain_aug').is_not_null())
    aug_combos = all_with_aug.group_by(
        ['arch_key', 'pretrain_dataset', 'pretrain_ft']
    ).agg(
        pl.col('model_name').alias('aug_models'),
    )
    no_aug_models = model_df.filter(pl.col('pretrain_aug').is_null())

    for row in aug_combos.iter_rows(named=True):
        arch_key = row['arch_key']
        pretrain_ds = row['pretrain_dataset']
        pretrain_ft_val = row['pretrain_ft']

        if pretrain_ds is None:
            if pretrain_ft_val is None:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('pretrain_dataset').is_null())
                    & (pl.col('pretrain_ft').is_null())
                )['model_name'].to_list()
            else:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('pretrain_dataset').is_null())
                    & (pl.col('pretrain_ft') == pretrain_ft_val)
                )['model_name'].to_list()
        else:
            if pretrain_ft_val is None:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('pretrain_dataset') == pretrain_ds)
                    & (pl.col('pretrain_ft').is_null())
                )['model_name'].to_list()
            else:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('pretrain_dataset') == pretrain_ds)
                    & (pl.col('pretrain_ft') == pretrain_ft_val)
                )['model_name'].to_list()

        for aug_model, no_aug_model in [
            (a, n) for a in row['aug_models'] for n in no_aug_matches
        ]:
            rows.append(
                {
                    'analysis_type': 'augmentation',
                    'control_model': no_aug_model,
                    'treatment_model': aug_model,
                    'arch_key': arch_key,
                }
            )

    return pl.DataFrame(rows)


def load_latent(dataset: str, model: str, split: str) -> np.ndarray:
    """Load latent embeddings from HuggingFace dataset."""
    data = load_dataset(dataset, model, split=split).with_format('torch')
    latent: torch.Tensor = torch.vstack(list(data['embedding']))
    return latent.detach().cpu().float().numpy()


def process_model(
    dataset: str,
    model_name: str,
    split: str,
    cfg: DictConfig,
    clusterer_cls,
) -> dict | None:
    """Process a single model: load latent, optionally compute prototypes,
    compute metrics based on config."""
    latent = load_latent(dataset, model_name, split)

    if cfg.get('use_prototypes', False):
        ls = LatentSpace(latent, seed=cfg.seed)
        ls.compute_prototypes(
            n_samples=cfg.get('n_samples', 10),
            clusterer_cls=clusterer_cls,
            n_clusters=cfg.n_prototypes,
            apply_parseval=True,
            return_cluster_indices=True,
        )
        data = ls.apply_analysis_operator()
    else:
        data = latent

    metric_names = cfg.get(
        'metrics',
        [
            'effective_rank_fast',
            'participation_ratio_fast',
        ],
    )

    results = {}
    for name in metric_names:
        if name in METRIC_DISPATCH:
            results[name] = METRIC_DISPATCH[name](data)
        else:
            print(f'[WARN] Unknown metric: {name}')

    return results


def run_stat_regression(
    metrics_df: pl.DataFrame, results_dir: Path, outcomes: list
) -> None:
    """Run regression on computed metrics."""
    for outcome in outcomes:
        _run_stat_for_outcome(metrics_df, outcome, results_dir)

    print(f'\n[COMPLETE] Regression results saved to {results_dir}')


def _run_stat_for_outcome(
    metrics_df: pl.DataFrame,
    outcome: str,
    results_dir: Path,
) -> None:
    """Run regression for a specific outcome variable."""
    df = metrics_df.to_pandas()
    results = []

    analysis_types = df['analysis_type'].unique()

    for analysis_type in analysis_types:
        subset = df[df['analysis_type'] == analysis_type].copy()
        if len(subset) < 2:
            continue

        try:
            m = smf.ols(
                f'{outcome} ~ is_treatment + C(arch_key)',
                data=subset,
            ).fit(cov_type='HC3')

            beta = m.params.get('is_treatment', np.nan)
            se = m.bse.get('is_treatment', np.nan)
            t_stat = m.tvalues.get('is_treatment', np.nan)
            p_value = m.pvalues.get('is_treatment', np.nan)
            conf_int = m.conf_int()
            ci_lower = (
                conf_int.loc['is_treatment', 0]
                if 'is_treatment' in conf_int.index
                else np.nan
            )
            ci_upper = (
                conf_int.loc['is_treatment', 1]
                if 'is_treatment' in conf_int.index
                else np.nan
            )

            results.append(
                {
                    'analysis_type': analysis_type,
                    'outcome': outcome,
                    'beta': beta,
                    'se': se,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'r_squared': m.rsquared,
                    'n_obs': int(m.nobs),
                    'df_resid': m.df_resid,
                }
            )
        except Exception as e:
            print(f'[WARN] stat failed for {analysis_type}/{outcome}: {e}')

    if results:
        df_out = pl.DataFrame(results)
        output_path = results_dir / f'stat__{outcome}.parquet'
        df_out.write_parquet(output_path)
        print(f'  Saved: {output_path.name}')


if __name__ == '__main__':
    main()
