"""Compute metrics for latent embeddings: stat metrics + TDA persistence diagrams.

Usage:
    python scripts/compute_metrics.py dataset=cifar10
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from joblib import Parallel, delayed
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from scripts.stat_analysis import (
    build_pairs,
)
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
from src.tda import compute_tda_features


def get_label_column(base_dataset: str) -> str:
    """Get the label column name from dataset config."""
    from pathlib import Path

    import yaml

    config_path = (
        Path(__file__).parent.parent
        / 'configs'
        / 'hydra'
        / 'dataset'
        / f'{base_dataset}.yaml'
    )
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        extras = config.get('extras', ['label'])
        return extras[0] if extras else 'label'
    return 'label'


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
    config_name='compute_metrics',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current: Path = Path('.')
    results_dir: Path = current / 'results/compute_metrics'

    results_dir.mkdir(parents=True, exist_ok=True)

    dataset: str = f'{cfg.repo_id}/{cfg.prefix}{cfg.dataset}'

    split = cfg.split
    valid_splits = DATASET_SPLITS.get(cfg.dataset, {'train'})
    if split not in valid_splits:
        raise ValueError(f'Invalid split {split!r} for dataset {cfg.dataset}.')

    print('\n[INFO] Loading model registry...')
    model_df = pl.read_parquet('hf://datasets/spaicom-lab/semasia-model-registry/**/*.parquet')
    model_df = model_df.with_columns(
        pl.col('model_name').str.split('.').list.first().alias('arch_key')
    )

    pairs_df = build_pairs(model_df)
    all_models = set(
        pairs_df['control_model'].to_list() + pairs_df['treatment_model'].to_list()
    )
    all_models.discard(None)
    print(f'  Models from stat_analysis pairs: {len(all_models)}')

    model_filter = cfg.get('model')
    if model_filter is not None:
        model_df = model_df.filter(pl.col('model_name').str.contains(model_filter))
        all_models = all_models & set(model_df['model_name'].unique().to_list())

    clusterer_cls = hydra.utils.instantiate(cfg.clustering)

    limit_models = cfg.get('limit_models')
    if limit_models is not None:
        all_models = set(sorted(all_models)[:limit_models])
        print(f'  Limited to {limit_models} models for testing')

    output_path = (
        results_dir / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{split}.parquet'
    )
    output_path_stat = output_path.with_name(output_path.stem + '_stat.parquet')
    output_path_tda = output_path.with_name(output_path.stem + '_tda.parquet')
    output_path_probing = output_path.with_name(output_path.stem + '_probing.parquet')

    force_recompute = cfg.get('force_recompute', False)
    compute_stat = cfg.get('compute_stat', True)
    compute_tda = cfg.get('compute_tda', False)
    compute_probing = cfg.get('compute_probing', False)

    if compute_stat and output_path_stat.exists() and not force_recompute:
        print('\n[INFO] Checking existing stat results for incremental computation...')
        try:
            existing_df = pl.read_parquet(output_path_stat)
            existing_models_stat = set(existing_df['model'].unique().to_list())
            print(f'  Found {len(existing_models_stat)} existing stat models')
        except Exception as e:
            print(f'  Could not read existing stat file: {e}')
            existing_models_stat = set()
    else:
        existing_models_stat = set()

    if compute_tda and output_path_tda.exists() and not force_recompute:
        print('\n[INFO] Checking existing TDA results for incremental computation...')
        try:
            existing_df = pl.read_parquet(output_path_tda)
            existing_models_tda = set(existing_df['model'].unique().to_list())
            print(f'  Found {len(existing_models_tda)} existing TDA models')
        except Exception as e:
            print(f'  Could not read existing TDA file: {e}')
            existing_models_tda = set()
    else:
        existing_models_tda = set()

    # Probing check
    if compute_probing and output_path_probing.exists() and not force_recompute:
        print('\n[INFO] Checking existing probing results...')
        try:
            existing_df = pl.read_parquet(output_path_probing)
            existing_models_probing = set(existing_df['model'].unique().to_list())
            print(f'  Found {len(existing_models_probing)} existing probing models')
        except Exception as e:
            print(f'  Could not read probing: {e}')
            existing_models_probing = set()
    else:
        existing_models_probing = set()

    # Fix skip logic - if compute_probing is enabled, only check probing file
    if compute_probing:
        existing_models = (
            existing_models_probing if output_path_probing.exists() else set()
        )
    else:
        existing_models = existing_models_stat | existing_models_tda

    if existing_models and not force_recompute:
        settings_cols = [
            'n_prototypes',
            'prewhiten',
            'tda_max_points',
            'tda_dim_reduction',
            'tda_dim_reduction_components',
            'tda_normalize',
        ]

        settings_changed = False
        if compute_stat and output_path_stat.exists():
            try:
                existing_settings = (
                    pl.read_parquet(output_path_stat)
                    .select(settings_cols)
                    .head(1)
                    .to_dicts()[0]
                    if len(pl.read_parquet(output_path_stat)) > 0
                    else {}
                )
                current_settings = {
                    'n_prototypes': cfg.n_prototypes,
                    'prewhiten': None,
                    'tda_max_points': cfg.tda.max_points,
                    'tda_dim_reduction': cfg.tda.dim_reduction,
                    'tda_dim_reduction_components': cfg.tda.dim_reduction_components,
                    'tda_normalize': cfg.tda.normalize,
                }
                settings_changed = existing_settings != current_settings
            except Exception:
                pass
        elif compute_tda and output_path_tda.exists():
            try:
                existing_settings = (
                    pl.read_parquet(output_path_tda)
                    .select(settings_cols)
                    .head(1)
                    .to_dicts()[0]
                    if len(pl.read_parquet(output_path_tda)) > 0
                    else {}
                )
                current_settings = {
                    'n_prototypes': cfg.n_prototypes,
                    'prewhiten': None,
                    'tda_max_points': cfg.tda.max_points,
                    'tda_dim_reduction': cfg.tda.dim_reduction,
                    'tda_dim_reduction_components': cfg.tda.dim_reduction_components,
                    'tda_normalize': cfg.tda.normalize,
                }
                settings_changed = existing_settings != current_settings
            except Exception:
                pass

        if settings_changed:
            print('  Settings changed: recomputing all models')
            all_models_to_process = all_models
        else:
            models_to_skip = existing_models & all_models
            all_models_to_process = all_models - models_to_skip
            print(f'  Skipping {len(models_to_skip)} already-computed models')
            print(f'  Computing {len(all_models_to_process)} new models')

        if not all_models_to_process:
            print('\n[COMPLETE] All models already computed.')
            print(f'  Stat results: {output_path_stat}')
            print(f'  TDA results: {output_path_tda}')
            return
    else:
        if force_recompute:
            print('\n[INFO] Force recompute enabled: recomputing all models')
        all_models_to_process = all_models

    all_models = all_models_to_process
    print(f'  Total unique models to process: {len(all_models)}')

    download_only = cfg.get('download_only', False)

    if download_only:
        print('\n[PHASE 1] Downloading all model latents (no computation)...')
        for model_name in tqdm(sorted(all_models), desc='Downloading latents'):
            try:
                load_latent(dataset, model_name, split)
            except Exception as e:
                print(f'Error downloading {model_name}: {e}')
                continue

        print('\n[COMPLETE] All latents downloaded.')
        print('  Run again with download_only: false for computation.')
        return

    print('\n[PHASE 2] Computing metrics from cached latents...')

    n_jobs = cfg.get('n_jobs')
    compute_stat = cfg.get('compute_stat', True)
    compute_tda = cfg.get('compute_tda', False)
    compute_probing = cfg.get('compute_probing', False)

    def save_model_results(
        model_name: str,
        metrics: list,
        model_df: pl.DataFrame,
        output_path_stat: Path,
        output_path_tda: Path,
        output_path_probing: Path,
    ):
        """Save results for a single model incrementally to separate parquet files."""
        if not metrics:
            return

        model_arch_key = (
            model_df.filter(pl.col('model_name') == model_name)
            .select('arch_key')
            .head(1)
            .to_dicts()
        )
        arch_key = model_arch_key[0]['arch_key'] if model_arch_key else None

        stat_results = []
        tda_results = []
        probing_results = []

        for case_result in metrics:
            result_row = {
                'arch_key': arch_key,
                **case_result,
            }
            if 'stat_case' in case_result:
                stat_results.append(result_row)
            elif 'tda_case' in case_result:
                tda_results.append(result_row)
            elif 'probing_case' in case_result:
                probing_results.append(result_row)

        def _save_to_parquet(results: list, output_path: Path, dedup_cols: list):
            if not results:
                return
            new_df = pl.DataFrame(results)
            try:
                if output_path.exists():
                    existing_df = pl.read_parquet(output_path)
                    combined_df = pl.concat([existing_df, new_df])
                    combined_df = combined_df.unique(
                        subset=dedup_cols,
                        keep='first',
                    )
                    combined_df.write_parquet(output_path)
                else:
                    new_df.write_parquet(output_path)
            except Exception as e:
                print(f'  Error saving to {output_path}: {e}')

        if compute_stat and stat_results:
            _save_to_parquet(stat_results, output_path_stat, ['model', 'stat_case'])
        if compute_tda and tda_results:
            _save_to_parquet(tda_results, output_path_tda, ['model', 'tda_case'])
        if compute_probing and probing_results:
            _save_to_parquet(
                probing_results, output_path_probing, ['model', 'probing_case']
            )

    def process_model_wrapper(model_name):
        try:
            return model_name, process_model(
                dataset=dataset,
                model_name=model_name,
                split=split,
                cfg=cfg,
                clusterer_cls=clusterer_cls,
            )
        except Exception as e:
            print(f'Error with {model_name}: {e}')
            return model_name, None

    model_list = sorted(all_models)

    probe_method = cfg.get('probing_method', 'autodiff')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Probing using device: {device}, method: {probe_method}')
    if n_jobs is None or n_jobs == 1:
        for model_name in tqdm(model_list, desc='Computing metrics'):
            name, metrics = process_model_wrapper(model_name)
            if metrics is not None:
                save_model_results(
                    name,
                    metrics,
                    model_df,
                    output_path_stat,
                    output_path_tda,
                    output_path_probing,
                )
    else:

        def process_and_save(model_name):
            name, metrics = process_model_wrapper(model_name)
            if metrics is not None:
                save_model_results(
                    name,
                    metrics,
                    model_df,
                    output_path_stat,
                    output_path_tda,
                    output_path_probing,
                )
            return name, metrics is not None

        with tqdm_joblib(desc='Computing metrics', total=len(model_list)):
            Parallel(n_jobs=n_jobs)(
                delayed(process_and_save)(name) for name in model_list
            )

    if compute_stat:
        print(f'\n[COMPLETE] Stat results saved to {output_path_stat}')
    if compute_tda:
        print(f'[COMPLETE] TDA results saved to {output_path_tda}')
    if compute_probing:
        print(f'[COMPLETE] Probing results saved to {output_path_probing}')


def load_latent(dataset: str, model: str, split: str) -> np.ndarray:
    """Load latent embeddings from HuggingFace dataset."""
    data = load_dataset(dataset, model, split=split).with_format('torch')
    latent: torch.Tensor = torch.vstack(list(data['embedding']))
    return latent.detach().cpu().float().numpy()


def load_latent_with_labels(
    dataset: str,
    model: str,
    split: str,
    label_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load latent embeddings and labels from HuggingFace dataset."""
    data = load_dataset(dataset, model, split=split).with_format('torch')
    latent: torch.Tensor = torch.vstack(list(data['embedding']))
    labels: np.ndarray = np.array(data[label_col])
    return latent.detach().cpu().float().numpy(), labels


def _compute_probing_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    batch_size: int = 256,
    epochs: int = 100,
    lr: float = 0.001,
    method: str = 'autodiff',
) -> dict:
    """Compute linear probing metrics using PyTorch linear layer."""
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if method == 'lstsq':
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(device)
        Y_onehot = torch.nn.functional.one_hot(y_t, n_classes).float()

        W = torch.linalg.lstsq(X_t, Y_onehot).solution

        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = (X_test_t @ W).argmax(dim=1).cpu().numpy()
    else:
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = nn.Linear(n_features, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred = model(X_test_t).argmax(dim=1).cpu().numpy()

    return {
        'probing_accuracy': accuracy_score(y_test, y_pred),
        'probing_recall': recall_score(
            y_test, y_pred, average='macro', zero_division=0
        ),
        'probing_precision': precision_score(
            y_test, y_pred, average='macro', zero_division=0
        ),
        'probing_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }


def _compute_stat_metrics(data: np.ndarray, cfg: DictConfig) -> dict:
    """Compute all stat metrics for given data."""
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


def _compute_tda_features(data: np.ndarray, cfg: DictConfig) -> dict:
    """Compute TDA features from data."""
    tda_cfg = cfg.tda

    max_points = tda_cfg.get('max_points', 1000)
    dim_reduction = tda_cfg.get('dim_reduction')
    dim_reduction_components = tda_cfg.get('dim_reduction_components', 20)
    normalize = tda_cfg.get('normalize')

    ls = LatentSpace(data, seed=cfg.seed)

    if max_points > 0 and ls.n_points > max_points:
        ls = ls.subsample(max_points, seed=cfg.seed)

    if normalize is not None and normalize != 'null':
        latent_processed = ls.normalize(normalize)
    else:
        latent_processed = ls.latent

    if dim_reduction is not None and dim_reduction != 'null':
        latent_processed = ls.reduce_dimensions(
            dim_reduction,
            dim_reduction_components,
            seed=cfg.seed,
        )

    max_dim = tda_cfg.get('max_dim', 2)
    simplicial_filter = tda_cfg.get('simplicial_filter', 'VietorisRips')
    n_bins = tda_cfg.get('n_bins', 100)
    sigma = tda_cfg.get('sigma', 0.1)
    metric = tda_cfg.get('metric', 'euclidean')

    result = compute_tda_features(
        latent_processed,
        max_dim=max_dim,
        simplicial_filter=simplicial_filter,
        n_bins=n_bins,
        sigma=sigma,
        metric=metric,
    )

    result['tda_simplicial_filter'] = simplicial_filter
    result['tda_max_dim'] = max_dim
    result['tda_sigma'] = sigma
    result['tda_metric'] = metric
    result['tda_n_bins'] = n_bins

    return result


def process_model(
    dataset: str,
    model_name: str,
    split: str,
    cfg: DictConfig,
    clusterer_cls,
) -> list[dict]:
    """Process a single model: compute stat and/or TDA metrics based on config."""
    compute_stat = cfg.get('compute_stat', True)
    compute_tda = cfg.get('compute_tda', False)
    stat_cases = cfg.get('stat_cases', ['raw', 'proto_no_prewhiten', 'proto_prewhiten'])
    tda_cases = cfg.get('tda_cases', ['absolute', 'proto_no_prewhiten'])

    latent = load_latent(dataset, model_name, split)

    n_clusters = cfg.n_prototypes
    n_samples = cfg.get('n_samples', 10)
    seed = cfg.seed

    tda_cfg = cfg.tda
    max_points = tda_cfg.get('max_points', 1000)
    dim_reduction = tda_cfg.get('dim_reduction')
    dim_reduction_components = tda_cfg.get('dim_reduction_components', 20)
    normalize = tda_cfg.get('normalize')

    common_cols = {
        'model': model_name,
        'split': split,
        'dataset': dataset,
    }

    results_all = []

    if compute_stat:
        for stat_case in stat_cases:
            if stat_case == 'raw':
                results_case = _compute_stat_metrics(latent, cfg)
                results_case.update(
                    {
                        'stat_case': 'raw',
                        'n_prototypes': 0,
                        'prewhiten': False,
                    }
                )
            elif stat_case == 'proto_no_prewhiten':
                ls = LatentSpace(latent, seed=seed)
                ls.compute_prototypes(
                    n_samples=n_samples,
                    clusterer_cls=clusterer_cls,
                    n_clusters=n_clusters,
                    apply_parseval=True,
                    return_cluster_indices=False,
                    prewhiten=False,
                )
                data_proto = ls.apply_analysis_operator()
                results_case = _compute_stat_metrics(data_proto, cfg)
                results_case.update(
                    {
                        'stat_case': 'proto_no_prewhiten',
                        'n_prototypes': n_clusters,
                        'prewhiten': False,
                    }
                )
            elif stat_case == 'proto_prewhiten':
                ls = LatentSpace(latent, seed=seed)
                ls.compute_prototypes(
                    n_samples=n_samples,
                    clusterer_cls=clusterer_cls,
                    n_clusters=n_clusters,
                    apply_parseval=True,
                    return_cluster_indices=False,
                    prewhiten=True,
                )
                data_proto = ls.apply_analysis_operator()
                results_case = _compute_stat_metrics(data_proto, cfg)
                results_case.update(
                    {
                        'stat_case': 'proto_prewhiten',
                        'n_prototypes': n_clusters,
                        'prewhiten': True,
                    }
                )
            else:
                continue

            results_case.update(common_cols)
            results_all.append(results_case)

    if compute_tda:
        tda_idx = 0
        for tda_case in tda_cases:
            if tda_case == 'absolute':
                ls_tda = LatentSpace(latent, seed=seed)
                tda_result = _compute_tda_features(ls_tda.latent, cfg)
                results_tda = {
                    'tda_case': 'absolute',
                    'tda_max_points': max_points,
                    'tda_dim_reduction': dim_reduction,
                    'tda_dim_reduction_components': dim_reduction_components,
                    'tda_normalize': normalize,
                }
            elif tda_case == 'proto_no_prewhiten':
                ls_tda = LatentSpace(latent, seed=seed)
                ls_tda.compute_prototypes(
                    n_samples=n_samples,
                    clusterer_cls=clusterer_cls,
                    n_clusters=n_clusters,
                    apply_parseval=True,
                    return_cluster_indices=False,
                    prewhiten=False,
                )
                data_tda = ls_tda.apply_analysis_operator()
                tda_result = _compute_tda_features(data_tda, cfg)
                results_tda = {
                    'tda_case': 'proto_no_prewhiten',
                    'tda_max_points': max_points,
                    'tda_dim_reduction': dim_reduction,
                    'tda_dim_reduction_components': dim_reduction_components,
                    'tda_normalize': normalize,
                }
            else:
                continue

            results_tda.update(tda_result)
            results_tda.update(common_cols)

            if compute_stat and tda_idx < len(results_all):
                results_all[tda_idx].update(results_tda)
            else:
                results_all.append(results_tda)

            tda_idx += 1

    compute_probing = cfg.get('compute_probing', False)
    if compute_probing:
        probe_test_split = cfg.get('probe_test_split', 'test')
        probe_batch_size = cfg.get('probing_batch_size', 256)
        probe_epochs = cfg.get('probing_epochs', 100)
        probe_lr = cfg.get('probing_lr', 0.001)
        probe_method = cfg.get('probing_method', 'autodiff')
        label_col = get_label_column(cfg.dataset)
        valid_splits = DATASET_SPLITS.get(cfg.dataset, {'train'})
        if probe_test_split not in valid_splits:
            if 'validation' in valid_splits:
                probe_test_split = 'validation'
            else:
                probe_test_split = list(valid_splits)[0]
        X_train, y_train = load_latent_with_labels(
            dataset, model_name, 'train', label_col
        )
        X_test, y_test = load_latent_with_labels(
            dataset, model_name, probe_test_split, label_col
        )
        probe_result = _compute_probing_metrics(
            X_train,
            y_train,
            X_test,
            y_test,
            seed,
            batch_size=probe_batch_size,
            epochs=probe_epochs,
            lr=probe_lr,
            method=probe_method,
        )
        probe_result.update(
            {
                'probing_case': 'raw',
                'probing_method': probe_method,
                'label_column': label_col,
                'train_split': 'train',
                'test_split': probe_test_split,
            }
        )
        probe_result.update(common_cols)
        results_all.append(probe_result)

    return results_all


if __name__ == '__main__':
    main()
