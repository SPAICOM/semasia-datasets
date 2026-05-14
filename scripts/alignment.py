"""Semantic alignment evaluation over pairs of models.

Transmits model A's test embeddings into model B's raw embedding space via
two methods and evaluates the quality of the transmission on the test set.

Methods
-------
proto  : prewhitening + KMeans + Parseval frame (injected strategy).
cca    : Canonical Correlation Analysis on raw embeddings.
linear : prewhitening + rank-k truncated linear map  M_k = U S_k Vt.

Run with:
    uv run python scripts/alignment.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn.functional as torch_F
from datasets import load_dataset
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from src.objects import LatentSpace

if TYPE_CHECKING:
    from omegaconf import DictConfig

logging.getLogger('httpx').setLevel(logging.WARNING)


@dataclass
class AlignmentResult:
    model_a: str
    model_b: str
    method: str  # 'proto' | 'cca' | 'linear'
    k: int
    metrics: dict[str, float] = field(default_factory=dict)


def _load_split(
    dataset_path: str, model: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    ds = load_dataset(dataset_path, model, split=split)
    embeddings = (
        torch.vstack(list(ds.with_format('torch')['embedding'])).float().numpy()
    )
    labels = np.array(ds['label'])
    return embeddings, labels


def _lstsq_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
    n_classes = int(y_tr.max().item()) + 1
    Y_oh = torch_F.one_hot(y_tr, n_classes).float()
    W = torch.linalg.lstsq(X_tr, Y_oh).solution
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    return (X_te @ W).argmax(dim=1).cpu().numpy()


def _compute_metrics(
    selected: dict[str, bool],
    b_hat: np.ndarray,
    b_test: np.ndarray,
    b_train: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if selected.get('accuracy', False):
        preds = _lstsq_predict(b_train, labels_train, b_hat, device)
        out['accuracy'] = float(np.mean(preds == labels_test))
    if selected.get('mse', False):
        out['mse'] = float(np.mean((b_hat - b_test) ** 2))
    return out


def _proto_transmit(
    la_train: np.ndarray,
    lb_train: np.ndarray,
    la_test: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Transmit A_test to B space via prewhitening + Parseval (injected)."""
    ls_a = LatentSpace(la_train, seed=seed)
    ls_a.prewhiten(inplace=True)
    _, indices_a = ls_a.compute_prototypes(
        n_samples=None,
        clusterer_cls=KMeans,
        n_clusters=k,
        apply_parseval=True,
        return_cluster_indices=True,
    )

    la_test_w = ls_a.apply_whitening_operator(la_test)
    a_test_proto = ls_a.apply_analysis_operator(la_test_w)

    ls_b = LatentSpace(lb_train, seed=seed)
    ls_b.prewhiten(inplace=True)
    ls_b.compute_injected_prototypes(indices_a, apply_parseval=True)

    b_hat = ls_b.apply_dewhitening_operator(
        ls_b.apply_synthesis_operator(a_test_proto)
    )
    return b_hat


def _fit_linear(
    la_train: np.ndarray,
    lb_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LatentSpace]:
    """Prewhiten both spaces, solve M = argmin ||B_w - A_w M||²_F, return SVD of M.

    Returns (U, s, Vt, la_test_w_fn, ls_b) where applying the rank-k map is:
        M_k = U[:, :k] @ diag(s[:k]) @ Vt[:k, :]
        b_hat = ls_b.apply_dewhitening_operator(la_test_w @ M_k)
    """
    ls_a = LatentSpace(la_train, seed=seed)
    ls_a.prewhiten(inplace=True)

    ls_b = LatentSpace(lb_train, seed=seed)
    ls_b.prewhiten(inplace=True)

    M, _, _, _ = np.linalg.lstsq(ls_a.latent, ls_b.latent, rcond=None)
    U, s, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    return U.astype(np.float32), s.astype(np.float32), Vt.astype(np.float32), ls_a, ls_b


def _linear_transmit(
    la_test: np.ndarray,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    ls_a: LatentSpace,
    ls_b: LatentSpace,
    k: int,
) -> np.ndarray:
    """Apply rank-k truncated linear map: M_k = U S_k Vt, then dewhiten."""
    k = min(k, len(s))
    s_k = np.zeros_like(s)
    s_k[:k] = s[:k]
    M_k = (U * s_k) @ Vt                               # (d_a, d_b)
    la_test_w = ls_a.apply_whitening_operator(la_test)
    return ls_b.apply_dewhitening_operator((la_test_w @ M_k).astype(np.float32))


def _fit_cca(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit CCA from scratch via SVD of the whitened cross-covariance.

    Returns (W_a, W_b, x_mean, y_mean) where W_a (d_a, k) and W_b (d_b, k)
    are the canonical directions for X and Y respectively.
    """
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Xc = X - x_mean
    Yc = Y - y_mean
    n = Xc.shape[0]

    Sxx = Xc.T @ Xc / (n - 1) + eps * np.eye(Xc.shape[1])
    Syy = Yc.T @ Yc / (n - 1) + eps * np.eye(Yc.shape[1])
    Sxy = Xc.T @ Yc / (n - 1)

    def _inv_sqrt(S: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(S)
        vals = np.maximum(vals, 1e-10)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    Sxx_isqrt = _inv_sqrt(Sxx)
    Syy_isqrt = _inv_sqrt(Syy)

    T = Sxx_isqrt @ Sxy @ Syy_isqrt
    U, _, Vt = np.linalg.svd(T, full_matrices=False)

    k = min(k, U.shape[1], Vt.shape[0])
    W_a = (Sxx_isqrt @ U[:, :k]).astype(np.float32)
    W_b = (Syy_isqrt @ Vt[:k].T).astype(np.float32)
    return W_a, W_b, x_mean.astype(np.float32), y_mean.astype(np.float32)


def _cca_transmit(
    la_train: np.ndarray,
    lb_train: np.ndarray,
    la_test: np.ndarray,
    k: int,
) -> np.ndarray:
    """Transmit A_test to B space via CCA on raw embeddings.

    Projects A_test into the k-dimensional canonical space shared with B,
    then reconstructs in B's original space via the pseudo-inverse of W_b.
    """
    W_a, W_b, x_mean, y_mean = _fit_cca(la_train, lb_train, k)
    a_test_c = (la_test - x_mean) @ W_a          # (n_test, k)
    W_b_pinv = np.linalg.pinv(W_b)               # (k, d_b)
    return (a_test_c @ W_b_pinv + y_mean).astype(np.float32)


@hydra.main(
    config_path='../configs/hydra/',
    config_name='alignment_methods',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current = Path('.')
    results_dir = current / 'results' / 'alignment'
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = cfg.seed
    k_values: list[int] = list(cfg.k_values)
    selected_metrics: dict[str, bool] = dict(cfg.metrics)
    selected_methods: list[str] = list(cfg.get('methods', ['proto', 'cca']))
    model_b: str = cfg.model_b
    models_a: list[str] = list(cfg.models_a)
    datasets: list[str] = list(cfg.datasets)

    tqdm.write(f'[INFO] Datasets: {datasets}')
    tqdm.write(f'[INFO] Model B: {model_b}')
    tqdm.write(f'[INFO] Models A: {models_a}')
    tqdm.write(f'[INFO] Methods: {selected_methods}  |  k values: {k_values}')
    tqdm.write(f'[INFO] Metrics: {[m for m, v in selected_metrics.items() if v]}')

    for dataset in tqdm(datasets, desc='Datasets', unit='dataset'):
        dataset_path = f'{cfg.repo_id}/{cfg.prefix}{dataset}'
        tqdm.write(f'\n[INFO] Dataset: {dataset_path}')

        # Load model_b once per dataset (shared across all model_a iterations)
        tqdm.write(f'  Loading {model_b} train...')
        lb_train, labels_train = _load_split(dataset_path, model_b, cfg.train_split)
        tqdm.write(f'  Loading {model_b} test...')
        lb_test, labels_test = _load_split(dataset_path, model_b, cfg.test_split)

        # No Mismatch baseline is the same for all model_a in this dataset
        nm_metrics: dict[str, float] = {}
        if selected_metrics.get('accuracy', False):
            preds_nm = _lstsq_predict(lb_train, labels_train, lb_test, device)
            nm_metrics['accuracy'] = float(np.mean(preds_nm == labels_test))
        if selected_metrics.get('mse', False):
            nm_metrics['mse'] = 0.0
        tqdm.write(f'  no_mismatch: {nm_metrics}')

        for model_a in tqdm(models_a, desc='Models A', unit='model'):
            out_path = results_dir / f'{model_a}__{model_b}__{dataset}.parquet'

            # Skip if already complete
            if out_path.exists():
                _existing = pl.read_parquet(out_path)
                if _existing['method'].is_in(['no_mismatch']).any():
                    tqdm.write(f'[SKIP] {model_a} → {model_b} ({out_path.name})')
                    continue

            tqdm.write(f'\n{"="*60}')
            tqdm.write(f'[PAIR] {model_a}  →  {model_b}')

            if cfg.get('save_results', True):
                nm_row = pl.DataFrame([{
                    'model_a': model_a, 'model_b': model_b,
                    'method': 'no_mismatch', 'k': 0, **nm_metrics,
                }])
                if out_path.exists():
                    df = pl.concat([pl.read_parquet(out_path), nm_row], how='diagonal')
                    df.write_parquet(out_path)
                    tqdm.write(f'  [PATCHED] {out_path.name}')
                    continue

            # ── Full alignment (only for brand-new parquets) ───────────────────
            tqdm.write(f'  Loading {model_a} train...')
            la_train, _ = _load_split(dataset_path, model_a, cfg.train_split)
            tqdm.write(f'  Loading {model_a} test...')
            la_test, _ = _load_split(dataset_path, model_a, cfg.test_split)

            tqdm.write(
                f'  Train A={la_train.shape}  B={lb_train.shape}'
                f'  |  Test A={la_test.shape}  B={lb_test.shape}'
            )

            results: list[AlignmentResult] = []
            results.append(AlignmentResult(model_a, model_b, 'no_mismatch', 0, nm_metrics))

            linear_svd = None
            if 'linear' in selected_methods:
                tqdm.write('  linear: fitting SVD of linear map...')
                linear_svd = _fit_linear(la_train, lb_train, seed)

            for k in tqdm(k_values, desc='  k values', unit='k', leave=False):
                tqdm.write(f'\n  [k={k}]')

                if 'proto' in selected_methods:
                    tqdm.write('    proto: transmitting...')
                    b_hat_proto = _proto_transmit(la_train, lb_train, la_test, k, seed)
                    m_proto = _compute_metrics(
                        selected_metrics, b_hat_proto, lb_test,
                        lb_train, labels_train, labels_test, device,
                    )
                    tqdm.write(f'    proto: {m_proto}')
                    results.append(AlignmentResult(model_a, model_b, 'proto', k, m_proto))

                if 'cca' in selected_methods:
                    tqdm.write('    cca:   transmitting...')
                    b_hat_cca = _cca_transmit(la_train, lb_train, la_test, k)
                    m_cca = _compute_metrics(
                        selected_metrics, b_hat_cca, lb_test,
                        lb_train, labels_train, labels_test, device,
                    )
                    tqdm.write(f'    cca:   {m_cca}')
                    results.append(AlignmentResult(model_a, model_b, 'cca', k, m_cca))

                if linear_svd is not None:
                    U, s, Vt, ls_a_lin, ls_b_lin = linear_svd
                    b_hat_lin = _linear_transmit(la_test, U, s, Vt, ls_a_lin, ls_b_lin, k)
                    m_lin = _compute_metrics(
                        selected_metrics, b_hat_lin, lb_test,
                        lb_train, labels_train, labels_test, device,
                    )
                    tqdm.write(f'    linear: {m_lin}')
                    results.append(AlignmentResult(model_a, model_b, 'linear', k, m_lin))

            # ── Save parquet ───────────────────────────────────────────────────
            if cfg.get('save_results', True):
                rows = []
                for r in results:
                    row = {'model_a': r.model_a, 'model_b': r.model_b, 'method': r.method, 'k': r.k}
                    row.update(r.metrics)
                    rows.append(row)
                df = pl.DataFrame(rows)
                df.write_parquet(out_path)
                tqdm.write(f'\n  [SAVED] {out_path}')
                tqdm.write(str(df))


if __name__ == '__main__':
    main()
