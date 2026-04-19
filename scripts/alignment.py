"""Align latent spaces across model pairs and compute distance metrics.

Usage:
    python scripts/alignment.py clustering.name=kmeans n_prototypes=[50,100]
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from src import remove_matching
from src.metrics.alignment import (
    compute_jaccard_metrics,
    compute_metric,
    jaccard_prototype_similarity,
)
from src.objects import LatentSpace

matplotlib.use('Agg')

logging.getLogger('httpx').setLevel(logging.WARNING)

DATASET_SPLITS = {
    'cifar10': {'train', 'test'},
    'cifar100': {'train', 'test'},
    'mnist': {'train', 'test'},
    'fashion_mnist': {'train', 'test'},
    'imagenet-1k': {'validation', 'test'},
    'tiny-imagenet': {'train'},
    'celeba': {'train', 'test'},
    'shvn': {'train', 'test'},
}


@hydra.main(
    config_path='../configs/hydra/',
    config_name='alignment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    CURRENT: Path = Path('.')
    RESULTS: Path = CURRENT / 'results/alignments/'
    HEATMAPS: Path = RESULTS / 'heatmaps/'

    RESULTS.mkdir(parents=True, exist_ok=True)
    if cfg.get('save_heatmaps', True):
        HEATMAPS.mkdir(parents=True, exist_ok=True)

    dataset: str = f'{cfg.repo_id}/{cfg.prefix}{cfg.dataset}'
    hub_pattern: str = f'datasets--{cfg.repo_id}--{cfg.prefix}{cfg.dataset}*'

    split = cfg.split
    valid_splits = DATASET_SPLITS.get(cfg.dataset, {'train'})
    if split not in valid_splits:
        raise ValueError(f'Invalid split {split!r} for dataset {cfg.dataset}.')

    model_df = pl.read_parquet('hf://datasets/spaicom-lab/model-registry/**/*.parquet')
    models: list[str] = (
        model_df.select('model_name')
        .unique()
        .sort('model_name')['model_name']
        .to_list()
    )

    clusterer_cls = hydra.utils.instantiate(cfg.clustering)

    jaccard_threshold = cfg.get('jaccard_threshold', 0.5)
    save_heatmaps = cfg.get('save_heatmaps', True)

    all_results = []

    for n_proto in cfg.n_prototypes:
        print(f'\n=== n_prototypes={n_proto} ===\n')

        for model_a in tqdm(models, desc='Reference models'):
            try:
                latent_a = load_latent(dataset, model_a, split)
                ls_a = LatentSpace(latent_a, seed=cfg.seed)
                ls_a.compute_prototypes(
                    n_samples=10,
                    clusterer_cls=clusterer_cls,
                    n_clusters=n_proto,
                    apply_parseval=True,
                    return_cluster_indices=True,
                )
                la = ls_a.latent
                F_a = ls_a.analysis_operator
                G_a = ls_a.synthesis_operator
                cluster_indices_a = ls_a._prototypes_to_indices
            except Exception as e:
                print(f'Error with {model_a}: {e}')
                continue

            for model_b in models:
                if model_a == model_b:
                    continue

                try:
                    latent_b = load_latent(dataset, model_b, split)
                    ls_b = LatentSpace(latent_b, seed=cfg.seed)
                    ls_b.compute_prototypes(
                        n_samples=10,
                        clusterer_cls=clusterer_cls,
                        n_clusters=n_proto,
                        apply_parseval=True,
                        return_cluster_indices=True,
                    )
                    lb = ls_b.latent
                    F_b = ls_b.analysis_operator
                    G_b = ls_b.synthesis_operator
                    cluster_indices_b = ls_b._prototypes_to_indices

                    results = compute_alignment_metrics(
                        la,
                        lb,
                        F_a,
                        G_a,
                        F_b,
                        G_b,
                        n_proto,
                        split,
                        model_a,
                        model_b,
                        cfg.dataset,
                        cfg.metrics,
                    )

                    jaccard_result = compute_jaccard_metrics(
                        cluster_indices_a, cluster_indices_b, jaccard_threshold
                    )

                    heatmap_path = None
                    if save_heatmaps:
                        heatmap_path = save_heatmap(
                            cluster_indices_a,
                            cluster_indices_b,
                            model_a,
                            model_b,
                            split,
                            n_proto,
                            HEATMAPS,
                        )

                    for result in results:
                        result.update(jaccard_result)
                        result['heatmap_path'] = heatmap_path

                    all_results.extend(results)

                    cache_pattern_b = (
                        f'{dataset.replace("/", "__")}___{model_b.replace("/", "__")}*'
                    )
                    remove_matching('~/.cache/huggingface/datasets/', cache_pattern_b)
                    remove_matching('~/.cache/huggingface/hub/', hub_pattern)

                except Exception as e:
                    print(f'Error with ({model_a}, {model_b}): {e}')
                    continue

            if latent_a is not None:
                cache_pattern_a = (
                    f'{dataset.replace("/", "__")}___{model_a.replace("/", "__")}*'
                )
                remove_matching('~/.cache/huggingface/datasets/', cache_pattern_a)
                remove_matching('~/.cache/huggingface/hub/', hub_pattern)

    if all_results:
        df = pl.DataFrame(all_results)
        output_path = (
            RESULTS / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{split}.parquet'
        )
        df.write_parquet(output_path)
        print(f'\nSaved {len(df)} results to {output_path}')
    else:
        print('\nNo results to save.')


def load_latent(dataset: str, model: str, split: str) -> np.ndarray:
    data = load_dataset(dataset, model, split=split).with_format('torch')
    latent: torch.Tensor = torch.vstack(list(data['embedding']))
    return latent.detach().cpu().float().numpy()


def compute_alignment_metrics(
    la: np.ndarray,
    lb: np.ndarray,
    F_a: np.ndarray,
    G_a: np.ndarray,
    F_b: np.ndarray,
    G_b: np.ndarray,
    n_prototypes: int,
    split: str,
    model_a: str,
    model_b: str,
    dataset_name: str,
    metrics: list,
) -> list[dict]:
    try:
        a_in_protospace = la @ F_a.T
        b_in_protospace = lb @ F_b.T

        a_recon_to_b = la @ F_a.T @ G_b.T
        b_recon_to_a = lb @ F_b.T @ G_a.T

        metrics_a_to_b = {}
        for metric_name in metrics:
            metrics_a_to_b[metric_name] = compute_metric(
                a_in_protospace, b_in_protospace, metric_name
            )
            metrics_a_to_b[f'{metric_name}_recon'] = compute_metric(
                a_recon_to_b, lb, metric_name
            )

        metrics_b_to_a = {}
        for metric_name in metrics:
            metrics_b_to_a[metric_name] = compute_metric(
                b_in_protospace, a_in_protospace, metric_name
            )
            metrics_b_to_a[f'{metric_name}_recon'] = compute_metric(
                b_recon_to_a, la, metric_name
            )

        result_a_to_b = {
            'n_prototypes': n_prototypes,
            'split': split,
            'source': model_a,
            'target': model_b,
            'dataset': dataset_name,
            **metrics_a_to_b,
        }

        result_b_to_a = {
            'n_prototypes': n_prototypes,
            'split': split,
            'source': model_b,
            'target': model_a,
            'dataset': dataset_name,
            **metrics_b_to_a,
        }

        return [result_a_to_b, result_b_to_a]

    except Exception:
        import traceback

        traceback.print_exc()
        return [
            {
                'n_prototypes': n_prototypes,
                'split': split,
                'source': model_a,
                'target': model_b,
                'dataset': dataset_name,
            },
            {
                'n_prototypes': n_prototypes,
                'split': split,
                'source': model_b,
                'target': model_a,
                'dataset': dataset_name,
            },
        ]


def save_heatmap(
    cluster_indices_a: dict,
    cluster_indices_b: dict,
    model_a: str,
    model_b: str,
    split: str,
    n_prototypes: int,
    heatmap_dir: Path,
) -> str:
    sim_matrix = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)

    n_a, n_b = sim_matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_b * 0.5), max(5, n_a * 0.5)))

    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Jaccard Similarity', fontsize=10)

    ax.set_xlabel(f'{model_b} Prototypes', fontsize=10)
    ax.set_ylabel(f'{model_a} Prototypes', fontsize=10)
    ax.set_title(
        (
            'Jaccard Prototype Similarity\n'
            f'{model_a} -> {model_b} | {split} | n={n_prototypes}'
        ),
        fontsize=11,
    )

    best_matches_a = sim_matrix.argmax(axis=1)
    best_matches_b = sim_matrix.argmax(axis=0)

    for i in range(n_a):
        j = best_matches_a[i]
        circle = plt.Circle(
            (j + 0.5, i + 0.5),
            radius=max(0.15, min(0.3, 0.3 * (n_a / 50))),
            color='none',
            ec='black',
            linewidth=2,
        )
        ax.add_patch(circle)

    for j in range(n_b):
        i = best_matches_b[j]
        circle = plt.Circle(
            (j + 0.5, i + 0.5),
            radius=max(0.15, min(0.3, 0.3 * (n_a / 50))),
            color='none',
            ec='blue',
            linewidth=2,
            linestyle='--',
        )
        ax.add_patch(circle)

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    for i, j in zip(row_ind, col_ind):
        ax.plot([j + 0.5, j + 0.5], [i - 0.3, i + 0.3], 'k-', alpha=0.3, linewidth=0.5)

    ax.set_xticks(np.arange(n_b) + 0.5)
    ax.set_yticks(np.arange(n_a) + 0.5)
    ax.set_xlim(0, n_b)
    ax.set_ylim(n_a, 0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    filename = f'{model_a}__{model_b}__{split}__{n_prototypes}.png'
    filepath = heatmap_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


if __name__ == '__main__':
    main()
