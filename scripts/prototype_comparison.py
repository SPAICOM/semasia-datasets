"""Compute prototype similarity metrics between two models."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use('Agg')

from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from src.metrics.prototypes import (
    MatchGroup,
    compute_all_metrics,
    compute_matching,
    foscttm_score,
    jaccard_matrix,
)
from src.objects import LatentSpace
from src.visualizations.prototypes import (
    plot_bipartite_merge,
    plot_cluster_pair_images,
    plot_force_graph,
    plot_prototype_heatmap,
    plot_similarity_profile,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logging.getLogger('httpx').setLevel(logging.WARNING)

MODEL_A_DEFAULT = 'vit_base_patch16_224.augreg_in1k'
MODEL_B_DEFAULT = 'vit_base_patch16_224.augreg_in21k'

# Maps semantic dataset name → (HuggingFace dataset id, image column name)
_BASE_IMAGE_DATASETS: dict[str, tuple[str, str]] = {
    'cifar10': ('cifar10', 'img'),
    'mnist': ('mnist', 'image'),
    'fashion_mnist': ('fashion_mnist', 'image'),
    'oxford-flowers': ('nelorth/oxford-flowers', 'image'),
}


@dataclass
class PrototypeResult:
    """Container for prototype comparison results."""

    dataset: str
    split: str
    model_a: str
    model_b: str
    n_prototypes: int
    f1: float
    hungarian: float
    entropy: float
    foscttm_raw_a2b: float
    foscttm_raw_b2a: float
    foscttm_raw_mean: float
    foscttm_proto_a2b: float
    foscttm_proto_b2a: float
    foscttm_proto_mean: float


def _cluster_centroids(data: np.ndarray, indices: dict[int, np.ndarray]) -> np.ndarray:
    """Compute cluster centroids in the original (non-whitened) data space."""
    k = len(indices)
    centroids = np.zeros((k, data.shape[1]))
    for i, idx in indices.items():
        centroids[i] = data[idx].mean(axis=0)
    return centroids


def _cluster_indices_to_sets(
    cluster_indices: dict[int, np.ndarray],
) -> list[set]:
    """Convert cluster indices dict to list of sets."""
    return [set(indices) for indices in cluster_indices.values()]


def _save_plotly(fig, path: Path) -> Path:
    try:
        fig.write_image(path)
        return path
    except Exception:
        path = path.with_suffix('.html')
        fig.write_html(path)
        return path


@hydra.main(
    config_path='../configs/hydra/',
    config_name='prototype_comparison',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current: Path = Path('.')
    results_dir: Path = current / 'results' / 'prototype_comparison'
    plots_dir: Path = results_dir / 'plots'

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    datasets_list = list(cfg.datasets)
    dataset_label = '+'.join(datasets_list)
    split = cfg.split
    model_a = cfg.model_a
    model_b = cfg.model_b
    seed = cfg.seed
    n_prototypes_list = cfg.n_prototypes
    prewhiten = cfg.get('prewhiten', True)
    threshold = cfg.get('jaccard_threshold', 0.5)
    n_cluster_samples = cfg.get('n_cluster_samples', 8)
    matching_method = cfg.get('matching_method', 'hungarian')
    n_foscttm = cfg.get('n_foscttm', 5000)

    if model_a is None or model_b is None:
        raise ValueError('model_a and model_b must be specified')

    latents_a, latents_b = [], []
    img_datasets: list[tuple] = []
    dataset_boundaries: list[int] = [0]

    for ds_name in datasets_list:
        dataset_path = f'{cfg.repo_id}/{cfg.prefix}{ds_name}'
        print(f'\n[INFO] Loading {model_a} from {dataset_path} ({split})...')
        data_a = load_dataset(dataset_path, model_a, split=split).with_format('torch')
        la: np.ndarray = torch.vstack(list(data_a['embedding'])).float().numpy()

        print(f'[INFO] Loading {model_b} from {dataset_path} ({split})...')
        data_b = load_dataset(dataset_path, model_b, split=split).with_format('torch')
        lb: np.ndarray = torch.vstack(list(data_b['embedding'])).float().numpy()

        # Load base dataset for original images (embedding configs have no image column)
        base_hf_id, img_col = _BASE_IMAGE_DATASETS.get(ds_name, (ds_name, 'image'))
        print(f'[INFO] Loading images from {base_hf_id} ({split})...')
        try:
            ds_img = load_dataset(base_hf_id, split=split)
            img_datasets.append((ds_img, img_col))
        except Exception as exc:
            print(
                f'  [WARN] Could not load {base_hf_id}: {exc} — images will be skipped'
            )
            img_datasets.append((None, None))

        dataset_boundaries.append(dataset_boundaries[-1] + len(la))
        latents_a.append(la)
        latents_b.append(lb)

    latent_a = np.vstack(latents_a)
    latent_b = np.vstack(latents_b)
    print(f'\n  Combined latent shapes: A={latent_a.shape}, B={latent_b.shape}')

    if latent_a.shape[0] != latent_b.shape[0]:
        min_len = min(latent_a.shape[0], latent_b.shape[0])
        latent_a = latent_a[:min_len]
        latent_b = latent_b[:min_len]
        print(f'  Trimmed to {min_len} samples')

    # Raw FOSCTTM on subsampled embeddings (requires same embedding dimension)
    if latent_a.shape[1] == latent_b.shape[1]:
        rng_sub = np.random.default_rng(seed)
        n_sub = min(n_foscttm, latent_a.shape[0])
        sub_idx = rng_sub.choice(latent_a.shape[0], size=n_sub, replace=False)
        foscttm_raw = foscttm_score(latent_a[sub_idx], latent_b[sub_idx])
        print(
            f'\n[FOSCTTM raw] n={n_sub}'
            f'  a→b={foscttm_raw["a2b"]:.4f}'
            f'  b→a={foscttm_raw["b2a"]:.4f}'
            f'  mean={foscttm_raw["mean"]:.4f}'
        )
    else:
        print(
            f'[WARN] Embedding dims differ ({latent_a.shape[1]} vs'
            f' {latent_b.shape[1]}), skipping raw FOSCTTM'
        )
        foscttm_raw = {'a2b': float('nan'), 'b2a': float('nan'), 'mean': float('nan')}

    heatmap_ks_cfg = cfg.get('heatmap_ks', None)
    heatmap_ks: set[int] = (
        set(heatmap_ks_cfg)
        if heatmap_ks_cfg is not None
        else {list(n_prototypes_list)[-1]}
    )
    selected_metrics: list[str] = list(
        cfg.get('metrics', ['f1', 'hungarian', 'entropy'])
    )

    results: list[PrototypeResult] = []
    J_by_k: dict[int, np.ndarray] = {}
    indices_a_by_k: dict[int, dict] = {}
    indices_b_by_k: dict[int, dict] = {}

    print(f'\n[INFO] Computing prototypes for k={n_prototypes_list}...')
    for k in n_prototypes_list:
        print(f'  k={k}...')

        ls_a = LatentSpace(latent_a, seed=seed)
        if prewhiten:
            ls_a.prewhiten(inplace=True)

        _, indices_a = ls_a.compute_prototypes(
            n_samples=None,
            clusterer_cls=KMeans,
            n_clusters=k,
            return_cluster_indices=True,
        )

        ls_b = LatentSpace(latent_b, seed=seed)
        if prewhiten:
            ls_b.prewhiten(inplace=True)

        _, indices_b = ls_b.compute_prototypes(
            n_samples=None,
            clusterer_cls=KMeans,
            n_clusters=k,
            return_cluster_indices=True,
        )

        clusters_a = _cluster_indices_to_sets(indices_a)
        clusters_b = _cluster_indices_to_sets(indices_b)

        J = jaccard_matrix(clusters_a, clusters_b)

        metrics = compute_all_metrics(J, threshold)

        # Proto FOSCTTM: centroids in original space, ordered by Hungarian matching
        if latent_a.shape[1] == latent_b.shape[1]:
            proto_a = _cluster_centroids(latent_a, indices_a)
            proto_b = _cluster_centroids(latent_b, indices_b)
            row_ind, col_ind = linear_sum_assignment(-J)
            foscttm_proto = foscttm_score(proto_a[row_ind], proto_b[col_ind])
        else:
            foscttm_proto = {
                'a2b': float('nan'),
                'b2a': float('nan'),
                'mean': float('nan'),
            }

        result = PrototypeResult(
            dataset=dataset_label,
            split=split,
            model_a=model_a,
            model_b=model_b,
            n_prototypes=k,
            f1=metrics['f1'],
            hungarian=metrics['hungarian'],
            entropy=metrics['entropy'],
            foscttm_raw_a2b=foscttm_raw['a2b'],
            foscttm_raw_b2a=foscttm_raw['b2a'],
            foscttm_raw_mean=foscttm_raw['mean'],
            foscttm_proto_a2b=foscttm_proto['a2b'],
            foscttm_proto_b2a=foscttm_proto['b2a'],
            foscttm_proto_mean=foscttm_proto['mean'],
        )
        results.append(result)

        print(
            f'    f1={metrics["f1"]:.3f}'
            f'  hung={metrics["hungarian"]:.3f}'
            f'  ent={metrics["entropy"]:.3f}'
            f'  foscttm proto a→b={foscttm_proto["a2b"]:.4f}'
            f'  b→a={foscttm_proto["b2a"]:.4f}'
            f'  mean={foscttm_proto["mean"]:.4f}'
        )

        if k in heatmap_ks:
            J_by_k[k] = J
            indices_a_by_k[k] = dict(indices_a)
            indices_b_by_k[k] = dict(indices_b)

    results_df = pl.DataFrame(
        [
            {
                'dataset': r.dataset,
                'split': r.split,
                'model_a': r.model_a,
                'model_b': r.model_b,
                'n_prototypes': r.n_prototypes,
                'f1': r.f1,
                'hungarian': r.hungarian,
                'entropy': r.entropy,
                'foscttm_raw_a2b': r.foscttm_raw_a2b,
                'foscttm_raw_b2a': r.foscttm_raw_b2a,
                'foscttm_raw_mean': r.foscttm_raw_mean,
                'foscttm_proto_a2b': r.foscttm_proto_a2b,
                'foscttm_proto_b2a': r.foscttm_proto_b2a,
                'foscttm_proto_mean': r.foscttm_proto_mean,
            }
            for r in results
        ]
    )

    if cfg.get('save_results', True):
        output_path = (
            results_dir / f'{model_a}__{model_b}__{dataset_label}__{split}.parquet'
        )
        results_df.write_parquet(output_path)
        print(f'\n[COMPLETE] Results saved to {output_path}')

    if cfg.get('save_heatmaps', True):
        for k, J_k in sorted(J_by_k.items()):
            k_dir = plots_dir / f'k{k}'
            k_dir.mkdir(exist_ok=True)

            groups: list[MatchGroup] = compute_matching(
                J_k, method=matching_method, threshold=threshold
            )
            matched_pairs: set[tuple[int, int]] = {
                (i, j) for g in groups for i in g.a_clusters for j in g.b_clusters
            }

            heatmap_path = _save_plotly(
                plot_prototype_heatmap(J_k, model_a, model_b),
                k_dir / 'heatmap.pdf',
            )
            print(f'[COMPLETE] Heatmap saved to {heatmap_path}')

            bipartite_path = _save_plotly(
                plot_bipartite_merge(
                    J_k,
                    model_a,
                    model_b,
                    threshold=threshold,
                    matched_pairs=matched_pairs,
                ),
                k_dir / 'bipartite.pdf',
            )
            print(f'[COMPLETE] Bipartite graph saved to {bipartite_path}')

            force_path = _save_plotly(
                plot_force_graph(
                    J_k,
                    model_a,
                    model_b,
                    threshold=threshold,
                    matched_pairs=matched_pairs,
                    groups=groups,
                ),
                k_dir / 'force.pdf',
            )
            print(f'[COMPLETE] Force graph saved to {force_path}')

            # Cluster group images
            idx_a = indices_a_by_k[k]
            idx_b = indices_b_by_k[k]
            print(f'  Saving cluster group images for k={k} ({matching_method})...')
            for rank, group in enumerate(groups):
                a_label = '+'.join(str(i) for i in group.a_clusters)
                b_label = '+'.join(str(j) for j in group.b_clusters)
                a_indices = np.concatenate([idx_a[i] for i in group.a_clusters])
                b_indices = np.concatenate([idx_b[j] for j in group.b_clusters])
                pair_fig = plot_cluster_pair_images(
                    a_indices,
                    b_indices,
                    img_datasets,
                    dataset_boundaries,
                    n_samples=n_cluster_samples,
                    model_a=model_a,
                    model_b=model_b,
                    jaccard=group.score,
                    cluster_a=a_label,
                    cluster_b=b_label,
                    seed=seed,
                )
                fname = f'group_{rank:02d}_a{a_label}_b{b_label}_s{group.score:.3f}.pdf'
                pair_path = k_dir / fname
                pair_fig.savefig(pair_path, bbox_inches='tight')
                plt.close(pair_fig)
                print(f'    G{rank:02d}: A[{a_label}]↔B[{b_label}] s={group.score:.3f}')

        all_profile_data = {
            'f1': [r.f1 for r in results],
            'hungarian': [r.hungarian for r in results],
            'entropy': [r.entropy for r in results],
        }
        profile_data = {
            m: all_profile_data[m] for m in selected_metrics if m in all_profile_data
        }
        profile_fig = plot_similarity_profile(
            n_prototypes_list,
            profile_data,
        )
        profile_path = _save_plotly(
            profile_fig,
            plots_dir / f'{model_a}__{model_b}_profile.pdf',
        )
        print(f'[COMPLETE] Profile saved to {profile_path}')

    print('\n[INFO] Summary:')
    print(results_df)


if __name__ == '__main__':
    main()
