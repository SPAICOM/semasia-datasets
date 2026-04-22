"""Statistical analysis on latent embeddings using precomputed metrics.

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
from omegaconf import DictConfig

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

IMAGENET_VARIANTS = [
    'ImageNet-1K',
    'ImageNet-12K',
    'ImageNet-21K',
    'ImageNet-Winter-21K',
    'ImageNet-22K',
]

IMAGENET_DIFFICULTY = {ds: i for i, ds in enumerate(IMAGENET_VARIANTS)}


@hydra.main(
    config_path='../configs/hydra/',
    config_name='stat_analysis',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current: Path = Path('.')
    results_dir: Path = current / 'results/stat'

    results_dir.mkdir(parents=True, exist_ok=True)

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

    print('\n[INFO] Loading precomputed metrics...')
    metrics_path = (
        Path('results/compute_metrics')
        / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{split}_stat.parquet'
    )
    if not metrics_path.exists():
        print(f'[ERROR] Metrics file not found: {metrics_path}')
        print('  Run compute_metrics.py first to generate metrics.')
        return

    metrics_df = pl.read_parquet(metrics_path)
    print(f'  Loaded {len(metrics_df)} metric rows')

    available_models = set(metrics_df['model'].unique().to_list())
    print(f'  Available models in metrics: {len(available_models)}')

    print('\n[INFO] Building results from pairs...')
    all_results = []

    missing_models = set()

    for row in pairs_df.iter_rows(named=True):
        analysis_type = row['analysis_type']
        control_model = row['control_model']
        treatment_model = row['treatment_model']
        arch_key = row['arch_key']

        if control_model not in available_models:
            missing_models.add(control_model)
            continue
        if treatment_model not in available_models:
            missing_models.add(treatment_model)
            continue

        control_metrics = metrics_df.filter(pl.col('model') == control_model)
        treatment_metrics = metrics_df.filter(pl.col('model') == treatment_model)

        stat_cases = ['raw', 'proto_no_prewhiten', 'proto_prewhiten']
        for stat_case in stat_cases:
            control_case = control_metrics.filter(pl.col('stat_case') == stat_case)
            treatment_case = treatment_metrics.filter(pl.col('stat_case') == stat_case)

            if len(control_case) == 0 or len(treatment_case) == 0:
                continue

            metric_cols = [
                c
                for c in control_case.columns
                if c
                not in [
                    'arch_key',
                    'stat_case',
                    'n_prototypes',
                    'prewhiten',
                    'model',
                    'split',
                    'dataset',
                ]
            ]
            control_row = control_case.to_dicts()[0]
            treatment_row = treatment_case.to_dicts()[0]

            all_results.append(
                {
                    'analysis_type': analysis_type,
                    'arch_key': arch_key,
                    'model_name': control_model,
                    'is_treatment': 0,
                    'stat_case': stat_case,
                    **{k: control_row[k] for k in metric_cols},
                }
            )

            all_results.append(
                {
                    'analysis_type': analysis_type,
                    'arch_key': arch_key,
                    'model_name': treatment_model,
                    'is_treatment': 1,
                    'stat_case': stat_case,
                    **{k: treatment_row[k] for k in metric_cols},
                }
            )

    if missing_models:
        print(f'\n[WARN] Missing models ({len(missing_models)}):')
        for m in sorted(missing_models)[:10]:
            print(f'  - {m}')
        if len(missing_models) > 10:
            print(f'  ... and {len(missing_models) - 10} more')

    if not all_results:
        print('\n[ERROR] No results to save.')
        return

    df = pl.DataFrame(all_results)
    output_path = (
        results_dir / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{split}.parquet'
    )
    df.write_parquet(output_path)
    print(f'\n[Saved] {len(df)} results to {output_path}')

    print('\n[RESULTS] DataFrame preview:')
    print(df.head(10))

    outcomes = cfg.get('metrics', ['effective_rank_fast', 'participation_ratio_fast'])
    run_stat_regression(df, results_dir, outcomes)


def build_pairs(model_df: pl.DataFrame) -> pl.DataFrame:
    """Build all analysis pairs with correct constraints.

    1. dataset_change: same (arch_key, aug), both no ft, different ds
       control = harder dataset, treatment = easier dataset
    2. large_vs_finetuned: same (arch_key, pretrain_dataset, aug), with/without ft
       control = finetuned to IN-1K, treatment = no finetuning
    3. small_vs_finetuned: in1k no ft vs any large->ft to 1k, same (arch_key, aug)
       control = large->ft to IN-1K, treatment = IN-1K no ft
    4. augmentation: same (arch_key, pretrain_dataset, ft), with/without aug
       control = with augmentation, treatment = no augmentation

    Only considers ImageNet variants as pretrain datasets.
    """
    rows = []

    imagenet_df = model_df.filter(pl.col('pretrain_dataset').is_in(IMAGENET_VARIANTS))

    no_ft = imagenet_df.filter(pl.col('pretrain_ft').is_null())
    ft_to_1k = imagenet_df.filter(pl.col('pretrain_ft') == 'ImageNet-1K')

    # 1. DATASET CHANGE: same (arch_key, aug), both no ft, different ds
    #    treatment = smaller/easier dataset, control = larger/harder dataset
    for aug_val in [None] + imagenet_df.filter(pl.col('pretrain_aug').is_not_null())[
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

            sorted_pairs = []
            for i, (m1, d1) in enumerate(zip(models, datasets)):
                for m2, d2 in zip(models[i + 1 :], datasets[i + 1 :]):
                    if d1 == d2:
                        continue
                    diff1 = IMAGENET_DIFFICULTY.get(d1, 999)
                    diff2 = IMAGENET_DIFFICULTY.get(d2, 999)
                    if diff1 < diff2:
                        sorted_pairs.append((m1, d1, m2, d2))
                    else:
                        sorted_pairs.append((m2, d2, m1, d1))

            for easier_model, easier_ds, harder_model, harder_ds in sorted_pairs:
                rows.append(
                    {
                        'analysis_type': 'dataset_change',
                        'control_model': easier_model,
                        'treatment_model': harder_model,
                        'arch_key': arch_key,
                    }
                )

    # 2. FINETUNING_LARGE: same (arch_key, pretrain_dataset, aug), with/without ft
    for aug_val in [None] + imagenet_df.filter(pl.col('pretrain_aug').is_not_null())[
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
                    'analysis_type': 'large_vs_finetuned',
                    'control_model': row['model_name_ctrl'],
                    'treatment_model': row['model_name'],
                    'arch_key': row['arch_key'],
                }
                for row in pairs.iter_rows(named=True)
            ]
        )

    # 3. SMALL_VS_FINETUNED: control=large->ft->1K, treatment=in1k no ft
    in1k_no_ft = imagenet_df.filter(
        (pl.col('pretrain_dataset') == 'ImageNet-1K') & pl.col('pretrain_ft').is_null()
    )

    for aug_val in [None] + imagenet_df.filter(pl.col('pretrain_aug').is_not_null())[
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
                    'analysis_type': 'small_vs_finetuned',
                    'control_model': row['model_name_ctrl'],
                    'treatment_model': row['model_name'],
                    'arch_key': row['arch_key'],
                }
                for row in pairs.iter_rows(named=True)
            ]
        )

    # 4. AUGMENTATION: control=no aug, treatment=with aug
    all_with_aug = imagenet_df.filter(pl.col('pretrain_aug').is_not_null())
    aug_combos = all_with_aug.group_by(
        ['arch_key', 'pretrain_dataset', 'pretrain_ft']
    ).agg(
        pl.col('model_name').alias('aug_models'),
    )
    no_aug_models = imagenet_df.filter(pl.col('pretrain_aug').is_null())

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


def run_stat_regression(
    metrics_df: pl.DataFrame, results_dir: Path, outcomes: list
) -> None:
    """Run regression on computed metrics for each analysis case."""
    stat_cases = metrics_df['stat_case'].unique().to_list()

    for case in stat_cases:
        print(f'\n[REGRESSION] Stat case: {case}')

        df_case = metrics_df.filter(pl.col('stat_case') == case)

        for outcome in outcomes:
            _run_stat_for_outcome(df_case, outcome, results_dir, case)

    print(f'\n[COMPLETE] Regression results saved to {results_dir}')


def _run_stat_for_outcome(
    metrics_df: pl.DataFrame,
    outcome: str,
    results_dir: Path,
    stat_case: str,
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
                    'stat_case': stat_case,
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
            print(f'[WARN] stat failed for {analysis_type}/{stat_case}/{outcome}: {e}')

    if results:
        df_out = pl.DataFrame(results)
        output_path = results_dir / f'stat__{outcome}__{stat_case}.parquet'
        df_out.write_parquet(output_path)
        print(f'  Saved: {output_path.name}')


if __name__ == '__main__':
    main()
