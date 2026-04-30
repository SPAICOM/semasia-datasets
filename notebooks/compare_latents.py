import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path().resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans

from src.objects import LatentSpace
from src.objects.latent import DimReductionMethod
from src.plotting.latent import _pearson_cross_correlation


def _check_models_in_registry(models: list[str]) -> None:
    model_df = pl.read_parquet('hf://datasets/spaicom-lab/model-registry/**/*.parquet')
    available_models = set(model_df['model_name'].unique().to_list())
    missing = [m for m in models if m not in available_models]
    if missing:
        raise ValueError(
            f'Model(s) not found in model registry: {missing}\n'
            f'Available models: {sorted(available_models)[:20]}... (showing first 20)'
        )


def _check_models_in_dataset(models: list[str], dataset: str) -> None:
    from datasets import get_dataset_config_names

    try:
        available = set(get_dataset_config_names(dataset))
        missing = [m for m in models if m not in available]
        if missing:
            raise ValueError(
                f'Model(s) not found in dataset {dataset}: {missing}\n'
                f'Use scripts/encode_dataset_all_timm.py to encode missing models.'
            )
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f'Error checking dataset {dataset}: {e}') from e


def _load_raw(model: str, dataset: str) -> np.ndarray:
    data = load_dataset(dataset, model, split='test').with_format('torch')
    return torch.vstack(list(data['embedding'])).float().numpy()


def _whiten(X: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_triangular

    mean = X.mean(axis=0)
    X_c = X - mean
    C = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    L = np.linalg.cholesky(C)
    return solve_triangular(L, X_c.T, lower=True).T.astype(np.float32)


def _cosine_sim(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    return (X_n @ A_n.T).astype(np.float32)


def _correlation_matrix(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    method: DimReductionMethod,
    n_components: int,
    k: int | None,
    seed: int = 42,
) -> np.ndarray:
    la = LatentSpace(arr_a, seed=seed)
    lb = LatentSpace(arr_b, seed=seed)
    la.compute_principal_components(
        method=method, n_components=n_components, k=k, seed=seed
    )
    lb.compute_principal_components(
        method=method, n_components=n_components, k=k, seed=seed
    )

    k_eff_a = la.pc_embedding.shape[1]
    k_eff_b = lb.pc_embedding.shape[1]
    k_use = min(k if k is not None else k_eff_a, k_eff_a, k_eff_b)

    return _pearson_cross_correlation(
        la.pc_embedding[:, :k_use],
        lb.pc_embedding[:, :k_use],
    )


def compare_latents(
    pairs: list[tuple[str, str]],
    dataset: str,
    methods: list[DimReductionMethod],
    n_components: int = 50,
    k: int | None = None,
    n_anchors: int | None = None,
    output_path: str | None = None,
    repo_id: str = 'spaicom-lab',
    prefix: str = 'semantic-',
    figsize: tuple[float, float] = (20, 8),
    cmap: str = 'coolwarm',
) -> tuple[plt.Figure, np.ndarray]:
    """Plot an (n_methods × n_pairs) grid of PC-correlation heatmaps.

    Each row corresponds to one dimensionality-reduction method; each column
    to one model pair.  Pair representations (whitening, KMeans, cosine
    projection) are computed once and reused across all method rows.
    """
    if not pairs:
        raise ValueError('Need at least one pair.')
    if not methods:
        raise ValueError('Need at least one method.')

    # unique models, insertion-order preserved
    models: list[str] = list(dict.fromkeys(m for a, b in pairs for m in (a, b)))

    _check_models_in_registry(models)
    full_dataset = f'{repo_id}/{prefix}{dataset}'
    _check_models_in_dataset(models, full_dataset)

    print(f'Loading {len(models)} unique latent spaces...')
    raw: dict[str, np.ndarray] = {m: _load_raw(m, full_dataset) for m in models}

    # --- pre-compute pair representations once (shared across all method rows) ---
    if n_anchors is not None:
        print('Whitening...')
        whitened: dict[str, np.ndarray] = {m: _whiten(raw[m]) for m in models}

        print('Clustering (KMeans)...')
        km: dict[str, KMeans] = {
            m: KMeans(n_clusters=n_anchors, random_state=42, n_init='auto').fit(
                whitened[m]
            )
            for m in models
        }

        rel: dict[str, np.ndarray] = {
            m: _cosine_sim(whitened[m], km[m].cluster_centers_.astype(np.float32))
            for m in models
        }

    # (arr_a, arr_b) per pair — input arrays for _correlation_matrix
    pair_arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for model_a, model_b in pairs:
        if n_anchors is not None:
            labels_a = km[model_a].labels_
            wb = whitened[model_b]
            anchors_b = np.array(
                [wb[labels_a == c].mean(axis=0) for c in range(n_anchors)],
                dtype=np.float32,
            )
            pair_arrays[(model_a, model_b)] = (rel[model_a], _cosine_sim(wb, anchors_b))
        else:
            pair_arrays[(model_a, model_b)] = (raw[model_a], raw[model_b])

    # --- build grid: rows = methods, cols = pairs ---
    n_rows, n_cols = len(methods), len(pairs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    title_suffix = f'  [cosine relative, {n_anchors} anchors]' if n_anchors else ''
    fig.suptitle(f'PC correlation{title_suffix}', fontsize=13, y=1.01)

    vmin, vmax = -1.0, 1.0
    im = None

    for row, method in enumerate(methods):
        print(f'Computing correlations [{method}]...')
        for col, (model_a, model_b) in enumerate(pairs):
            arr_a, arr_b = pair_arrays[(model_a, model_b)]
            C = _correlation_matrix(arr_a, arr_b, method, n_components, k)

            ax = axes[row, col]
            im = ax.imshow(C, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(model_b, fontsize=7, labelpad=3)

            # method label on the left-most column; model_a label elsewhere
            if col == 0:
                ax.set_ylabel(f'[{method}]\n{model_a}', fontsize=7, labelpad=3)
            else:
                ax.set_ylabel(model_a, fontsize=7, labelpad=3)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label('Pearson r', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {output_path}')
    else:
        plt.show()

    return fig, axes
