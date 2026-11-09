"""Statistical analysis on latent embeddings using precomputed metrics.

This script regresses geometric metrics (e.g., effective_rank, isotropy) on treatment
indicators to quantify the effect of architectural choices on embedding structure.

The regression uses a binary treatment indicator where:
    - 0 = control (baseline architecture)
    - 1 = treatment (modified architecture)

Config keys beyond the yaml (pass with +key=value): registry_path (local parquet instead of the Hub),
metrics_dir (default results/compute_metrics), results_dir (default results/stat),
same_pretrain_tasks (default true), density_log10 (default true). pretrain_dataset is compared as a
set of corpora and pretrain_ft as an ordered sequence (see build_pairs).

Three regression specifications are used:

1. POOLED: metric ~ is_treatment + C(dataset) + C(arch_key)
   - Tests if treatment affects target after controlling for dataset identity.
   - If beta != 0 (p < 0.05), the effect is independent of which dataset.
   - Equivalent to: "Does the treatment matter, regardless of evaluation dataset?"

2. WITHIN: metric ~ is_treatment + C(arch_key)   [run per dataset]
   - Same regression, estimated separately within each dataset.
   - Tests if treatment effect holds within a given dataset.
   - Equivalent to: "Does the treatment matter for CIFAR-10? For MNIST?"

3. INTERACTION: metric ~ is_treatment * C(dataset) + C(dataset) + C(arch_key)
   - Tests if treatment effect differs by dataset (interaction term).
   - If beta_interaction != 0 (p < 0.05), the effect is dataset-dependent.
   - Equivalent to: "Does the treatment effect depend on which dataset?"

Inference (pooled and within): the same model appears in many pairs, so standard errors
are clustered on the union of model ids (dyadic-robust), p-values and 95% CIs come from a
wild cluster bootstrap-t (B = 9,999), and the (condition, metric) tests of each regression
type are corrected with Benjamini-Hochberg. Each result is classified as ``robust``,
``sample-sensitive`` or ``null`` by re-estimating on random subsets of pairs in which every
model appears at most once (see ``src/stats/inference.py``). The interaction specification
still reports HC3 standard errors (``inference`` column).

Usage:
    uv run scripts/stat_analysis.py

Output:
    - combined.parquet: raw per-model data with dataset column
    - stat__{metric}__{stat_case}___pooled.parquet  (beta, se, p_value, p_bh, ci, status)
    - stat__{metric}__{stat_case}___{dataset}.parquet    (within-dataset)
    - stat__{metric}__{stat_case}___interaction.parquet
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import polars as pl
import statsmodels.formula.api as smf
from omegaconf import DictConfig

from src.stats.inference import (
    bh,
    classify,
    condition_rng,
    infer_condition,
    pair_frame,
)

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

VIT_SIZE_ORDER = {
    'tiny': 0,
    'small': 1,
    'base': 2,
    'medium': 3,
    'large': 4,
    'huge': 5,
    'gigantic': 6,
    'enormous': 7,
}

SIZE_ORDER = {
    'xxsmall': 0,
    'xsmall': 1,
    'tiny': 2,
    'small': 3,
    'base': 4,
    'medium': 5,
    'large': 6,
    'xlarge': 7,
    'huge': 8,
    'gigantic': 9,
    'enormous': 10,
}

FAMILY_SIZE_ORDERS = {
    'ViT': {
        'tiny': 0,
        'small': 1,
        'base': 2,
        'medium': 3,
        'large': 4,
        'huge': 5,
        'gigantic': 6,
        'enormous': 7,
    },
    'ConvNeXt': {
        'tiny': 4,
        'small': 5,
        'base': 6,
        'large': 7,
        'xlarge': 8,
        'huge': 9,
    },
    'Swin': {'tiny': 0, 'small': 1, 'base': 2, 'large': 3},
    'DeiT': {'tiny': 0, 'small': 1, 'base': 2, 'medium': 3, 'large': 4, 'huge': 5},
    'MaxViT': {'tiny': 0, 'small': 1, 'base': 2, 'large': 3, 'xlarge': 4},
    'MobileNet': {
        'small': 3,
        'medium': 4,
        'large': 5,
    },
    'MobileViT': {
        'xxsmall': 0,
        'xsmall': 1,
        'small': 2,
        'medium': 3,
        'large': 4,
        'xlarge': 5,
    },
}


@hydra.main(
    config_path='../configs/hydra/',
    config_name='stat_analysis',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current: Path = Path('.')
    results_dir: Path = current / cfg.get('results_dir', 'results/stat')
    metrics_dir: Path = current / cfg.get('metrics_dir', 'results/compute_metrics')
    results_dir.mkdir(parents=True, exist_ok=True)

    split = cfg.split

    # Normalise datasets to list (backward compat for string / single-item configs)
    raw_datasets = cfg.get('datasets', [])
    datasets = [raw_datasets] if isinstance(raw_datasets, str) else list(raw_datasets)

    for ds in datasets:
        valid_splits = DATASET_SPLITS.get(ds, {'train'})
        if split not in valid_splits:
            raise ValueError(f'Invalid split {split!r} for dataset {ds!r}.')

    print('\n[INFO] Loading model registry...')
    model_df = pl.read_parquet(
        cfg.get(
            'registry_path',
            'hf://datasets/spaicom-lab/semasia-model-registry/**/*.parquet',
        )
    )
    model_df = model_df.with_columns(
        pl.col('model_name').str.split('.').list.first().alias('arch_key')
    )

    pairs_df = build_pairs(model_df)
    # Matching condition: control and treatment share the same set of pretrain_tasks
    # (config key same_pretrain_tasks, default true; false gives the pairs without this condition).
    if cfg.get('same_pretrain_tasks', True):
        tasks = model_df.select('model_name', 'pretrain_tasks')
        pairs_df = (
            pairs_df.join(
                tasks.rename({'model_name': 'control_model', 'pretrain_tasks': '_tc'}),
                on='control_model',
            )
            .join(
                tasks.rename({'model_name': 'treatment_model', 'pretrain_tasks': '_tt'}),
                on='treatment_model',
            )
            .filter(pl.col('_tc').list.sort().list.join('|') == pl.col('_tt').list.sort().list.join('|'))
            .drop('_tc', '_tt')
        )
    # Attach architecture family from the registry using the control model.
    pairs_df = pairs_df.join(
        model_df.select(['model_name', 'family']).rename(
            {'model_name': 'control_model'}
        ),
        on='control_model',
        how='left',
    ).sort(['analysis_type', 'control_model', 'treatment_model'])  # reproducible order
    print(f'  Total pairs: {len(pairs_df)}')
    for at in pairs_df['analysis_type'].unique().to_list():
        count = len(pairs_df.filter(pl.col('analysis_type') == at))
        print(f'    {at}: {count}')

    print('\n[INFO] Loading precomputed metrics...')
    all_metrics = []
    for ds in datasets:
        metrics_path = (
            metrics_dir
            / f'{cfg.repo_id}__{cfg.prefix}{ds}__{split}_stat.parquet'
        )
        if not metrics_path.exists():
            print(f'  [WARN] Metrics not found for {ds}: {metrics_path}')
            continue
        m_df = pl.read_parquet(metrics_path).with_columns(pl.lit(ds).alias('dataset'))
        print(f'  Loaded {len(m_df)} rows from {ds}')

        if cfg.get('compute_probing', False):
            probing_path = (
                metrics_dir
                / f'{cfg.repo_id}__{cfg.prefix}{ds}__{split}_probing.parquet'
            )
            if probing_path.exists():
                probing_df = pl.read_parquet(probing_path)
                probing_metrics = probing_df.select(
                    [
                        'model',
                        'probing_accuracy',
                        'probing_recall',
                        'probing_precision',
                        'probing_f1',
                    ]
                )
                m_df = m_df.join(probing_metrics, on='model', how='left')

        all_metrics.append(m_df)

    if not all_metrics:
        print('[ERROR] No metrics loaded.')
        return

    metrics_df = pl.concat(all_metrics)
    print(f'  Total metric rows: {len(metrics_df)}')
    print(f'  Available models: {metrics_df["model"].n_unique()}')

    print('\n[INFO] Building results from pairs...')
    all_results = []
    missing_models = set()

    for ds in datasets:
        ds_metrics = metrics_df.filter(pl.col('dataset') == ds)
        available_ds_models = set(ds_metrics['model'].unique().to_list())

        for row in pairs_df.iter_rows(named=True):
            analysis_type = row['analysis_type']
            control_model = row['control_model']
            treatment_model = row['treatment_model']
            arch_key = row['arch_key']
            arch_family = row.get('family') or arch_key

            if control_model not in available_ds_models:
                missing_models.add(control_model)
                continue
            if treatment_model not in available_ds_models:
                missing_models.add(treatment_model)
                continue

            control_metrics = ds_metrics.filter(pl.col('model') == control_model)
            treatment_metrics = ds_metrics.filter(pl.col('model') == treatment_model)

            for stat_case in ['raw']:
                control_case = control_metrics.filter(pl.col('stat_case') == stat_case)
                treatment_case = treatment_metrics.filter(
                    pl.col('stat_case') == stat_case
                )

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
                        'arch_family': arch_family,
                        'dataset': ds,
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
                        'arch_family': arch_family,
                        'dataset': ds,
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
    # Density is heavy-tailed: analysed on the log10 scale (config key density_log10, default true; false keeps the raw scale).
    if cfg.get('density_log10', True):
        df = df.with_columns(pl.col('density').log10())

    null_counts = {c: df[c].null_count() for c in df.columns if df[c].null_count() > 0}
    if null_counts:
        print('\n[WARN] Null values found in combined DataFrame:')
        for col, count in null_counts.items():
            print(f'  {col}: {count} nulls ({100 * count / len(df):.1f}%)')
    else:
        print('\n[OK] No null values in combined DataFrame.')

    output_path = results_dir / 'combined.parquet'
    df.write_parquet(output_path)
    print(f'\n[Saved] {len(df)} results to {output_path}')

    print('\n[RESULTS] DataFrame preview:')
    print(df.head(10))

    outcomes = cfg.get('metrics', ['effective_rank_fast', 'participation_ratio_fast'])
    if cfg.get('compute_probing', False):
        probing_outcomes = cfg.get(
            'probing_outcomes',
            [
                'probing_accuracy',
                'probing_recall',
                'probing_precision',
                'probing_f1',
            ],
        )
        outcomes.extend(probing_outcomes)
    regression_types = cfg.get('regression_types', ['pooled'])
    inference = dict(cfg.get('inference', {}))
    run_stat_regression(
        df, results_dir, outcomes, ['raw'], regression_types, inference=inference
    )
    write_latex_table(results_dir, outcomes)
    write_obs_table(df, results_dir)


def build_pairs(model_df: pl.DataFrame) -> pl.DataFrame:
    """Build all analysis pairs with correct constraints.

    1. dataset_change: same (arch_key, aug), both no ft, different ds
       control = easier dataset, treatment = harder dataset
    2. large_vs_finetuned (Specialization): same (arch_key, pretrain_dataset,
       aug), pretrained on a larger ImageNet variant than IN-1K
       control = no finetuning, treatment = same checkpoint finetuned to IN-1K
    3. small_vs_finetuned (Transfer Learning): same (arch_key, aug)
       control = trained directly on IN-1K (no ft),
       treatment = pretrained on a larger variant, then finetuned to IN-1K
    4. augmentation: same (arch_key, pretrain_dataset, full ft chain), with/without aug
       control = with augmentation, treatment = no augmentation
    5. model_scale: same (pretrain_dataset, aug, ft), vit family only
       control = smaller model, treatment = larger model

    Only considers ImageNet variants as pretrain datasets.
    pretrain_dataset (a list) is compared as a SET of corpora via '_ds_key'; pretrain_ft (a list) as a SEQUENCE via '_ft_key'.
    """
    rows = []

    # pretrain_ft is an ORDERED LIST of fine-tuning datasets (null / empty = no fine-tuning).
    # '_ft_key' = the whole chain as a string, used where the same fine-tuning is required.
    model_df = model_df.with_columns(
        pl.when(pl.col('pretrain_ft').list.len() > 0)
        .then(pl.col('pretrain_ft').list.join(' > '))
        .otherwise(None)
        .alias('_ft_key')
    )
    # pretrain_dataset is a LIST of pretraining corpora, treated as a SET: '_ds_key' = the sorted elements joined
    # (the order of the pretraining corpora has no meaning; pretrain_ft, in contrast, is a SEQUENCE: '_ft_key' keeps its order).
    model_df = model_df.with_columns(pl.col('pretrain_dataset').list.sort().list.join(' + ').alias('_ds_key'))

    # ImageNet variants only: every corpus of the pretraining is an ImageNet variant
    imagenet_df = model_df.filter(
        pl.col('pretrain_dataset').list.eval(pl.element().is_in(IMAGENET_VARIANTS)).list.all()
    )

    no_ft = imagenet_df.filter(pl.col('_ft_key').is_null())
    # Fine-tuned treatments of Specialization / Transfer Learning must come from a pretraining on a
    # LARGER ImageNet variant than IN-1K (IN-12K / 21K / 22K ...) and be fine-tuned to IN-1K
    # (IN-1K in the fine-tuning chain); IN-1K -> IN-1K models are excluded.
    ft_to_1k = imagenet_df.filter(
        pl.col('pretrain_ft').list.contains('ImageNet-1K')
        & (pl.col('_ds_key') != 'ImageNet-1K')
    )


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
            pl.col('_ds_key').alias('datasets'),
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
            ['arch_key', 'model_name', '_ds_key']
        ).join(
            no_ft_filtered.select(['arch_key', 'model_name', '_ds_key']),
            on=['arch_key', '_ds_key'],
            suffix='_ctrl',
        )

        pairs = pairs_all.filter(pl.col('_ds_key').is_not_null())

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

    # 3. TRANSFER LEARNING (small_vs_finetuned):
    #    control=large->ft->1K, treatment=in1k no ft
    in1k_no_ft = imagenet_df.filter(
        (pl.col('_ds_key') == 'ImageNet-1K') & pl.col('_ft_key').is_null()
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
        ['arch_key', '_ds_key', '_ft_key']
    ).agg(
        pl.col('model_name').alias('aug_models'),
    )
    no_aug_models = imagenet_df.filter(pl.col('pretrain_aug').is_null())

    for row in aug_combos.iter_rows(named=True):
        arch_key = row['arch_key']
        pretrain_ds = row['_ds_key']
        pretrain_ft_val = row['_ft_key']

        if pretrain_ds is None:
            if pretrain_ft_val is None:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('_ds_key').is_null())
                    & (pl.col('_ft_key').is_null())
                )['model_name'].to_list()
            else:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('_ds_key').is_null())
                    & (pl.col('_ft_key') == pretrain_ft_val)
                )['model_name'].to_list()
        else:
            if pretrain_ft_val is None:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('_ds_key') == pretrain_ds)
                    & (pl.col('_ft_key').is_null())
                )['model_name'].to_list()
            else:
                no_aug_matches = no_aug_models.filter(
                    (pl.col('arch_key') == arch_key)
                    & (pl.col('_ds_key') == pretrain_ds)
                    & (pl.col('_ft_key') == pretrain_ft_val)
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

    # 5. MODEL SCALE: same (pretrain_dataset, aug, ft), vit family only
    #    control = smaller model, treatment = larger model
    rows.extend(_build_model_scale_pairs(imagenet_df))

    return pl.DataFrame(rows)


def _build_model_scale_pairs(imagenet_df: pl.DataFrame) -> list[dict]:
    """Build model scale pairs for multiple architecture families.

    For each family in FAMILY_SIZE_ORDERS:
        1. Filter to that family
        2. Group by (pretrain_dataset, pretrain_aug, pretrain_ft)
        3. Order by size (primary), num_parameters (secondary)
        4. Create pairs: smaller = control, larger = treatment

    arch_key is set to lowercase family name for regression control.
    """
    rows = []

    for family, family_order in FAMILY_SIZE_ORDERS.items():
        # Filter to one family first (ensures consistent group definitions)
        family_df = imagenet_df.filter(pl.col('family') == family)

        if len(family_df) == 0:
            continue

        group_cols = ['_ds_key', 'pretrain_aug', '_ft_key']  # _ds_key: set of pretraining corpora (see build_pairs)
        grouped = family_df.group_by(group_cols).agg(
            pl.col('model_name').alias('models'),
            pl.col('size').alias('sizes'),
            pl.col('num_parameters').alias('params'),
        )

        for row in grouped.iter_rows(named=True):
            models = row['models']
            sizes = row['sizes']
            params = row['params']

            valid_indices = []
            for i in range(len(models)):
                size_val = sizes[i].lower() if sizes[i] is not None else None

                if size_val not in family_order:
                    continue

                size_idx = family_order[size_val]
                param_val = params[i] if params[i] is not None else 0
                valid_indices.append((i, size_idx, param_val))

            valid_indices.sort(key=lambda x: (x[1], x[2]))

            for i in range(len(valid_indices)):
                idx1 = valid_indices[i][0]
                for j in range(i + 1, len(valid_indices)):
                    idx2 = valid_indices[j][0]
                    smaller_model = models[idx1]
                    larger_model = models[idx2]
                    rows.append(
                        {
                            'analysis_type': 'model_scale',
                            'control_model': smaller_model,
                            'treatment_model': larger_model,
                            'arch_key': family.lower(),
                        }
                    )

    return rows


def run_stat_regression(
    metrics_df: pl.DataFrame,
    results_dir: Path,
    outcomes: list,
    stat_cases: list | None = None,
    regression_types: list | None = None,
    inference: dict | None = None,
) -> None:
    """Run regression on computed metrics for each analysis case.

    Parameters
    ----------
    metrics_df : raw per-model DataFrame with all stat_cases and a `dataset` column,
        rows ordered as (control, treatment) pairs
    results_dir : directory for output parquets
    outcomes : list of metric column names to regress
    stat_cases : list of stat_cases to process (e.g., ['raw', 'proto_no_prewhiten']).
        If None, uses all stat_cases present in metrics_df.
    regression_types : list of regression modes to run.
        'pooled'   -> metric ~ is_treatment + C(dataset) + C(arch_family)
        'within'   -> same regression per dataset slice
        'interaction' -> metric ~ is_treatment * C(dataset) + C(dataset) + C(arch_family)
        If None, defaults to ['pooled'].
    inference : options of the clustered inference (pooled / within):
        n_boot (9999), n_subsamples (200), seed (0), alpha (0.05).

    Regressions are run on raw (unstandardized) metric values.  A pooled
    control-group standard deviation is computed per metric and saved alongside
    the coefficients so the plot can rescale β for cross-metric comparability
    without altering p-values.
    """
    if stat_cases is None:
        stat_cases = metrics_df['stat_case'].unique().to_list()
    if regression_types is None:
        regression_types = ['pooled']

    FORMULA_SUFFIXES = {
        'pooled': ' + C(dataset) + C(arch_family)',
        'interaction': ' * C(dataset) + C(arch_family)',
    }

    for case in stat_cases:
        print(f'\n[REGRESSION] Stat case: {case}')

        df_case = metrics_df.filter(pl.col('stat_case') == case)
        if len(df_case) == 0:
            print(f'  [WARN] No data for stat_case {case}, skipping')
            continue

        present = [o for o in outcomes if o in df_case.columns]

        # Pooled control std across all analysis_types and datasets — single
        # scale factor per metric used by the plot for cross-metric comparability.
        control_std = {
            o: float(df_case.filter(pl.col('is_treatment') == 0)[o].drop_nulls().std())
            for o in present
        }

        clustered = [r for r in regression_types if r in ('pooled', 'within')]
        if clustered:
            _run_clustered_inference(
                df_case, present, results_dir, case, clustered, control_std, inference
            )

        if 'interaction' in regression_types:
            for outcome in present:
                _run_stat_for_outcome(
                    df_case,
                    outcome,
                    results_dir,
                    case,
                    'interaction',
                    FORMULA_SUFFIXES['interaction'],
                    pooled_control_std=control_std[outcome],
                )

    print(f'\n[COMPLETE] Regression results saved to {results_dir}')


def _run_clustered_inference(
    df_case: pl.DataFrame,
    outcomes: list[str],
    results_dir: Path,
    stat_case: str,
    reg_types: list[str],
    control_std: dict[str, float],
    inference: dict | None = None,
) -> None:
    """Model-union clustered inference for the pooled and within regressions.

    One call estimates every (condition, metric) of a regression slice so that the
    Benjamini-Hochberg correction runs over all of its tests, then writes one parquet
    per metric with the file naming used by the plots.
    """
    opts = {'n_boot': 9999, 'n_subsamples': 200, 'seed': 0, 'alpha': 0.05}
    opts.update(inference or {})

    if df_case.select(outcomes).null_count().sum_horizontal().item() > 0:
        raise ValueError('null metric values: clustered inference needs complete pairs')

    slices: list[tuple[str, str | None, pl.DataFrame]] = []
    if 'pooled' in reg_types:
        slices.append(('pooled', None, df_case))
    if 'within' in reg_types:
        for ds in sorted(df_case['dataset'].unique().to_list()):
            slices.append(('within', ds, df_case.filter(pl.col('dataset') == ds)))

    for reg_type, dataset, frame in slices:
        label = reg_type if dataset is None else f'{reg_type}/{dataset}'
        print(f'  [{label}] model-union clustering, wild bootstrap B={opts["n_boot"]}')
        pairs = pair_frame(frame, outcomes)
        rows = []
        for analysis_type in frame['analysis_type'].unique(maintain_order=True):
            sub = pairs.filter(pl.col('analysis_type') == analysis_type)
            if len(sub) < 2:
                continue
            rng = condition_rng(opts['seed'], reg_type, dataset or 'all', analysis_type)
            for r in infer_condition(
                sub, outcomes, rng, opts['n_boot'], opts['n_subsamples']
            ):
                rows.append(
                    {
                        'analysis_type': analysis_type,
                        'stat_case': stat_case,
                        'regression_type': reg_type,
                        **r,
                        'control_std': control_std[r['outcome']],
                        'term': 'is_treatment',
                        'inference': 'model_union_wcb',
                        **({'dataset': dataset} if dataset else {}),
                    }
                )
        if not rows:
            continue
        res = pl.DataFrame(rows)
        p_bh = bh(res['p_value'].to_numpy())
        status = classify(
            p_bh,
            res['beta'].to_numpy(),
            res['sub_beta'].to_numpy(),
            res['sub_p_median'].to_numpy(),
            res['sub_share_sig'].to_numpy(),
            opts['alpha'],
        )
        res = res.with_columns(pl.Series('p_bh', p_bh), pl.Series('status', status))

        tag = f'__{dataset}' if dataset else f'___{reg_type}'
        for outcome in outcomes:
            part = res.filter(pl.col('outcome') == outcome)
            if len(part):
                path = results_dir / f'stat__{outcome}__{stat_case}{tag}.parquet'
                part.write_parquet(path)
                print(f'  Saved: {path.name}')


def _run_stat_for_outcome(
    metrics_df: pl.DataFrame,
    outcome: str,
    results_dir: Path,
    stat_case: str,
    reg_type: str,
    formula_suffix: str,
    *,
    pooled_control_std: float,
    dataset: str | None = None,
) -> None:
    """Run regression for a specific outcome variable on raw (unstandardized) values.

    Parameters
    ----------
    metrics_df : DataFrame slice to regress (all data or single-dataset slice)
    outcome : metric column name
    results_dir : output directory
    stat_case : stat_case label for output filename
    reg_type : one of 'pooled', 'within', 'interaction'
    formula_suffix : RHS terms appended after 'is_treatment'
    pooled_control_std : pooled control-group SD of this metric (across all
        analysis_types and datasets). Saved in results so the plot can rescale
        β for cross-metric comparability without changing p-values.
    dataset : for 'within' mode, the dataset name (used in filename)
    """
    results = []
    analysis_types = metrics_df['analysis_type'].unique().to_list()

    for analysis_type in analysis_types:
        subset_pl = metrics_df.filter(pl.col('analysis_type') == analysis_type)
        if len(subset_pl) < 2:
            continue

        subset = subset_pl.drop_nulls(subset=[outcome]).to_pandas()
        if len(subset) < 2:
            continue

        try:
            formula = f'{outcome} ~ is_treatment{formula_suffix}'
            m = smf.ols(formula, data=subset).fit(cov_type='HC3')

            for term in [
                'is_treatment',
                'is_treatment:C(dataset)',
                'is_treatment:C(dataset)[T.',
            ]:
                if term in m.params.index:
                    beta = m.params[term]
                    se = m.bse[term]
                    t_stat = m.tvalues[term]
                    p_value = m.pvalues[term]
                    conf_int = m.conf_int()
                    ci_lower = float(conf_int.loc[term, 0])
                    ci_upper = float(conf_int.loc[term, 1])
                    interaction_label = term
                    break
            else:
                continue

            row = {
                'analysis_type': analysis_type,
                'stat_case': stat_case,
                'outcome': outcome,
                'regression_type': reg_type,
                'beta': beta,
                'se': se,
                't_stat': t_stat,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'r_squared': m.rsquared,
                'n_obs': int(m.nobs),
                'df_resid': m.df_resid,
                'control_std': pooled_control_std,
                'term': interaction_label,
                'inference': 'HC3',
            }

            if reg_type == 'within' and dataset is not None:
                row['dataset'] = dataset

            results.append(row)
        except Exception as e:
            print(f'[WARN] stat failed for {analysis_type}/{stat_case}/{outcome}: {e}')

    if results:
        df_out = pl.DataFrame(results)
        if dataset:
            tag = f'__{dataset}'
        elif reg_type == 'within':
            tag = ''
        else:
            tag = f'___{reg_type}'
        output_path = results_dir / f'stat__{outcome}__{stat_case}{tag}.parquet'
        df_out.write_parquet(output_path)
        print(f'  Saved: {output_path.name}')


def _stars(p: float) -> str:
    if p < 0.01:
        return r'^{***}'
    if p < 0.05:
        return r'^{**}'
    if p < 0.10:
        return r'^{*}'
    return ''


def write_latex_table(
    results_dir: Path,
    outcomes: list[str],
    regression_type: str = 'pooled',
) -> None:
    """Write a single LaTeX table of unstandardized regression coefficients.

    Reads the pooled results written by :func:`run_stat_regression`, so that β is in
    the original metric units and significance follows the primary inference (model-union
    clustering, wild bootstrap, Benjamini-Hochberg).  One row per metric; each cell shows
    β with significance stars and the BH-adjusted p-value in parentheses in a smaller
    font on the same line; a dagger marks sample-sensitive results.  The table is wrapped
    in \\resizebox so it always fits the text width.
    """
    from src.visualizations.statistical import (
        ANALYSIS_LABELS,
        ANALYSIS_ORDER,
        METRIC_GROUPS,
        METRIC_LABELS,
        load_regression_results,
    )

    reg = load_regression_results(results_dir)
    if reg.is_empty():
        print('[WARN] No regression results found; skipping LaTeX table.')
        return
    reg = reg.filter(
        (pl.col('regression_type') == regression_type)
        & (pl.col('stat_case_parsed') == 'raw')
        & pl.col('metric').is_in(outcomes)
    )
    if reg.is_empty() or 'p_bh' not in reg.columns:
        print('[WARN] No clustered regression results for LaTeX table.')
        return

    # (analysis_type, metric) -> (beta, p_bh, status)
    lookup = {
        (r['analysis_type'], r['metric']): (
            float(r['beta']),
            float(r['p_bh']),
            r['status'],
        )
        for r in reg.iter_rows(named=True)
    }
    present_analyses = [a for a in ANALYSIS_ORDER if any(k[0] == a for k in lookup)]
    present_metrics = {k[1] for k in lookup}

    n_cols = len(present_analyses)
    n_total = n_cols + 1
    col_spec = 'l' + 'r' * n_cols
    col_headers = ' & '.join(ANALYSIS_LABELS.get(a, a) for a in present_analyses)

    lines: list[str] = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Pooled OLS regression coefficients $\beta$ for embedding geometry '
        r'and linear probing metrics across five pretraining conditions. '
        r'Each $\beta$ is the estimated treatment effect in original '
        r'metric units, from \textit{metric} $\sim$ '
        r'\textit{is\_treatment} $+$ \texttt{C(dataset)} $+$ '
        r'\texttt{C(arch\_family)}. '
        r'Standard errors are clustered on the union of control and treatment model ids; '
        r'$p$-values come from a wild cluster bootstrap-$t$ ($B=9{,}999$) and are '
        r'Benjamini--Hochberg adjusted over all tests (shown in parentheses; stars '
        r'refer to the adjusted values). $\dagger$: significant, but not in subsamples '
        r'in which each model appears at most once. '
        r'Metrics are grouped by category; '
        r'conditions are defined in Table~\ref{tab:conditions}.}',
        r'\label{tab:regression_results}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        f' & {col_headers} \\\\',
        r'\midrule',
    ]

    for group_name, group_metrics in METRIC_GROUPS.items():
        group_outcomes = [
            m for m in group_metrics if m in outcomes and m in present_metrics
        ]
        if not group_outcomes:
            continue

        lines.append(
            r'\multicolumn{' + str(n_total) + r'}{l}{\textit{' + group_name + r'}} \\'
        )

        for metric in group_outcomes:
            metric_label = METRIC_LABELS.get(metric, metric)
            cells: list[str] = []
            for analysis in present_analyses:
                entry = lookup.get((analysis, metric))
                if entry is not None:
                    beta, p, status = entry
                    dagger = r'^{\dagger}' if status == 'sample-sensitive' else ''
                    cells.append(
                        f'${{\\textstyle {beta:.3f}{_stars(p)}{dagger}}}$ '
                        f'{{\\scriptsize $({p:.3f})$}}'
                    )
                else:
                    cells.append('---')

            lines.append(f'{metric_label} & {" & ".join(cells)} \\\\')

        lines.append(r'\addlinespace[2pt]')

    lines += [
        r'\bottomrule',
        r'\multicolumn{'
        + str(n_total)
        + r'}{l}{\footnotesize $^{***}p_{BH}<0.01$; $^{**}p_{BH}<0.05$; '
        r'$^{*}p_{BH}<0.10$} \\',
        r'\end{tabular}',
        r'}',  # close \resizebox
        r'\end{table}',
    ]

    output_path = results_dir / f'latex_table___{regression_type}.tex'
    output_path.write_text('\n'.join(lines))
    print(f'\n[Saved] LaTeX table to {output_path}')


def write_obs_table(
    df: pl.DataFrame,
    results_dir: Path,
) -> None:
    """Write a LaTeX table with the number of
    observations per analysis type and dataset."""
    from src.visualizations.statistical import (
        ANALYSIS_LABELS,
        ANALYSIS_ORDER,
        DATASET_LABELS,
    )

    datasets = sorted(df['dataset'].unique().to_list())
    present_analyses = [
        a for a in ANALYSIS_ORDER if a in df['analysis_type'].unique().to_list()
    ]

    # Count pairs (divide by 2 since each pair has a control and treatment row)
    counts = (
        df.group_by(['analysis_type', 'dataset'])
        .agg((pl.len() // 2).alias('n_pairs'))
        .sort(['analysis_type', 'dataset'])
    )

    # Build lookup: (analysis_type, dataset) -> n_pairs
    lookup = {
        (row['analysis_type'], row['dataset']): row['n_pairs']
        for row in counts.iter_rows(named=True)
    }

    ds_headers = ' & '.join(DATASET_LABELS.get(d, d) for d in datasets)
    col_spec = 'l' + 'r' * (len(datasets) + 1)

    lines: list[str] = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Number of matched pairs per pretraining '
        r'condition and evaluation dataset.}',
        r'\label{tab:obs_counts}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        f'Condition & {ds_headers} & Total \\\\',
        r'\midrule',
    ]

    for analysis in present_analyses:
        label = ANALYSIS_LABELS.get(analysis, analysis)
        row_counts = [lookup.get((analysis, d), 0) for d in datasets]
        total = sum(row_counts)
        cells = ' & '.join(str(c) for c in row_counts)
        lines.append(f'{label} & {cells} & {total} \\\\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    output_path = results_dir / 'obs_table.tex'
    output_path.write_text('\n'.join(lines))
    print(f'\n[Saved] Observations table to {output_path}')


if __name__ == '__main__':
    main()
