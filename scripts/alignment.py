"""Semantic alignment evaluation over pairs of models.

Transmits model A's test embeddings into model B's raw embedding space via
several methods and evaluates the quality of the transmission on the test set.

Methods
-------
proto  : prewhitening + KMeans + Parseval frame (injected strategy).
cca    : Canonical Correlation Analysis on raw embeddings.
linear : prewhitening + rank-k truncated linear map  M_k = U S_k Vt.

Accuracy after communication is measured with a linear probe fit on the
receiver's raw TRAIN embeddings (``decoder: ridge`` in float64 on CPU, or the
legacy ``decoder: lstsq``). Only proto (KMeans) is stochastic and is run for every
seed in ``seeds``; cca and linear are deterministic and run once.

Run with:
    uv run python scripts/alignment.py
    uv run python scripts/alignment.py decoder=lstsq results_dir=<dir>
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn.functional as torch_F
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from src.objects import LatentSpace

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig

logging.getLogger('httpx').setLevel(logging.WARNING)


def _load_split(
    dataset_path: str, model: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    ds = load_dataset(dataset_path, model, split=split)
    embeddings = (
        torch.vstack(list(ds.with_format('torch')['embedding'])).float().numpy()
    )
    labels = np.array(ds['label'])
    return embeddings, labels


def _one_hot(y: np.ndarray, n_classes: int, dtype: type) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=dtype)
    out[np.arange(len(y)), y] = 1
    return out


def _lstsq_probe(
    X_train: np.ndarray, y_train: np.ndarray, device: torch.device
) -> Callable[[np.ndarray], np.ndarray]:
    """Legacy decoder: bare ``torch.linalg.lstsq`` in float32 (device dependent)."""
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
    n_classes = int(y_tr.max().item()) + 1
    Y_oh = torch_F.one_hot(y_tr, n_classes).float()
    W = torch.linalg.lstsq(X_tr, Y_oh).solution

    def predict(X: np.ndarray) -> np.ndarray:
        X_te = torch.tensor(X, dtype=torch.float32, device=device)
        return (X_te @ W).argmax(dim=1).cpu().numpy()

    return predict


def _ridge_probes(
    X_train: np.ndarray, y_train: np.ndarray, lam_rels: list[float]
) -> dict[float, Callable[[np.ndarray], np.ndarray]]:
    """Ridge probes W = (X^T X + lam I)^-1 X^T Y, float64 on CPU, no intercept.

    lam = lam_rel * trace(X^T X) / d, one probe per entry of ``lam_rels``.
    """
    X = X_train.astype(np.float64)
    n_classes = int(y_train.max()) + 1
    Y = _one_hot(y_train, n_classes, np.float64)
    gram = X.T @ X
    xty = X.T @ Y
    d = gram.shape[0]
    scale = np.trace(gram) / d
    eye = np.eye(d)

    def make(lam_rel: float) -> Callable[[np.ndarray], np.ndarray]:
        W = np.linalg.solve(gram + lam_rel * scale * eye, xty)

        def predict(X_new: np.ndarray) -> np.ndarray:
            return (X_new.astype(np.float64) @ W).argmax(axis=1)

        return predict

    return {lam: make(lam) for lam in lam_rels}


def _make_probes(
    decoder: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lam_rels: list[float],
    lstsq_device: torch.device,
) -> dict[float, Callable[[np.ndarray], np.ndarray]]:
    """Decoders keyed by lam_rel (the lstsq decoder ignores lam and has one entry)."""
    if decoder == 'ridge':
        return _ridge_probes(X_train, y_train, lam_rels)
    if decoder == 'lstsq':
        return {lam_rels[0]: _lstsq_probe(X_train, y_train, lstsq_device)}
    raise ValueError(f'unknown decoder {decoder!r} (expected ridge | lstsq)')


def _error_metrics(b_hat: np.ndarray, b_test: np.ndarray) -> dict[str, float]:
    """Reconstruction error in the raw receiver space.

    mse            : mean over all entries (= per-sample squared norm / d_b)
    mse_per_sample : mean over samples of the squared L2 norm of the error
    nmse           : total squared error / total squared norm of the targets
    """
    b_hat64 = b_hat.astype(np.float64)
    b_test64 = b_test.astype(np.float64)
    sq = (b_hat64 - b_test64) ** 2
    return {
        'mse': float(sq.mean()),
        'mse_per_sample': float(sq.sum(axis=1).mean()),
        'nmse': float(sq.sum() / (b_test64**2).sum()),
        'd_b': int(b_test.shape[1]),
    }


def _evaluate(
    selected: dict[str, bool],
    b_hat: np.ndarray,
    b_test: np.ndarray,
    probes: dict[float, Callable[[np.ndarray], np.ndarray]],
    labels_test: np.ndarray,
) -> tuple[dict[str, float], dict[float, float]]:
    """Return (metrics, accuracy per lam_rel). The first probe is the main one."""
    out: dict[str, float] = {}
    acc_by_lam: dict[float, float] = {}
    if selected.get('accuracy', False):
        for lam, predict in probes.items():
            acc_by_lam[lam] = float(np.mean(predict(b_hat) == labels_test))
        out['accuracy'] = acc_by_lam[next(iter(probes))]
    if selected.get('mse', False):
        out.update(_error_metrics(b_hat, b_test))
    return out, acc_by_lam


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

    b_hat = ls_b.apply_dewhitening_operator(ls_b.apply_synthesis_operator(a_test_proto))
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
    M_k = (U * s_k) @ Vt  # (d_a, d_b)
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
    a_test_c = (la_test - x_mean) @ W_a  # (n_test, k)
    W_b_pinv = np.linalg.pinv(W_b)  # (k, d_b)
    return (a_test_c @ W_b_pinv + y_mean).astype(np.float32)


def _git_info() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ['git', *args], capture_output=True, text=True, check=False
        ).stdout.rstrip('\n')

    dirty = [
        line[3:]
        for line in run('status', '--porcelain', '--untracked-files=no').splitlines()
    ]
    return {
        'commit': run('rev-parse', 'HEAD').strip(),
        'branch': run('rev-parse', '--abbrev-ref', 'HEAD').strip(),
        'dirty_tracked_files': dirty,
    }


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'n/a'


def _write_manifest(
    path: Path, cfg: DictConfig, info: dict[str, dict], lstsq_device: torch.device
) -> None:
    manifest = {
        'date_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'git': _git_info(),
        'device': {
            'cuda_available': torch.cuda.is_available(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'lstsq_device': str(lstsq_device),
            'ridge_decoder_device': 'cpu (float64)',
            'transmission_methods_device': 'cpu (numpy / scikit-learn)',
        },
        'versions': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'scikit-learn': _pkg_version('scikit-learn'),
            'seaborn': _pkg_version('seaborn'),
            'polars': pl.__version__,
        },
        'decoder': cfg.decoder,
        'lam_rel': float(cfg.lam_rel),
        'lam_sweep': list(cfg.get('lam_sweep', [])),
        'seeds': list(cfg.seeds),
        'config': OmegaConf.to_container(cfg, resolve=True),
        **info,
    }
    path.write_text(json.dumps(manifest, indent=2, default=str))


@hydra.main(
    config_path='../configs/hydra/',
    config_name='alignment_methods',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    decoder: str = cfg.decoder
    lam_rel = float(cfg.lam_rel)
    lam_sweep = [float(x) for x in cfg.get('lam_sweep', [])]
    lam_all = [lam_rel] + [lam for lam in lam_sweep if lam != lam_rel]
    lambda_dir = Path(cfg.lambda_dir)
    if decoder == 'ridge' and lam_sweep:
        lambda_dir.mkdir(parents=True, exist_ok=True)

    # Only the legacy lstsq decoder uses a device; ridge is always CPU float64.
    lstsq_device = torch.device(cfg.lstsq_device)
    if lstsq_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('lstsq_device=cuda requested but CUDA is not available')

    seeds: list[int] = list(cfg.seeds)
    k_values: list[int] = list(cfg.k_values)
    selected_metrics: dict[str, bool] = dict(cfg.metrics)
    selected_methods: list[str] = list(cfg.get('methods', ['proto', 'cca']))
    model_b: str = cfg.model_b
    models_a: list[str] = list(cfg.models_a)
    datasets: list[str] = list(cfg.datasets)
    overwrite: bool = bool(cfg.get('overwrite', False))

    tqdm.write(f'[INFO] Datasets: {datasets}')
    tqdm.write(f'[INFO] Model B: {model_b}')
    tqdm.write(f'[INFO] Models A: {models_a}')
    tqdm.write(f'[INFO] Methods: {selected_methods}  |  k values: {k_values}')
    tqdm.write(f'[INFO] Decoder: {decoder}  lam_rel={lam_rel}  sweep={lam_sweep}')
    tqdm.write(f'[INFO] Seeds (proto): {seeds}  |  results_dir: {results_dir}')

    manifest_path = results_dir / 'manifest.json'
    info: dict[str, dict] = {'datasets': {}, 'models': {}, 'pairs': {}}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        info = {key: old.get(key, {}) for key in info}

    for dataset in tqdm(datasets, desc='Datasets', unit='dataset'):
        dataset_path = f'{cfg.repo_id}/{cfg.prefix}{dataset}'
        tqdm.write(f'\n[INFO] Dataset: {dataset_path}')

        tqdm.write(f'  Loading {model_b} train...')
        lb_train, labels_train = _load_split(dataset_path, model_b, cfg.train_split)
        tqdm.write(f'  Loading {model_b} test...')
        lb_test, labels_test = _load_split(dataset_path, model_b, cfg.test_split)
        info['datasets'][dataset] = {
            'n_train': int(lb_train.shape[0]),
            'n_test': int(lb_test.shape[0]),
            'n_classes': int(labels_train.max()) + 1,
            'train_split': cfg.train_split,
            'test_split': cfg.test_split,
        }
        info['models'][model_b] = {
            'latent_dim': int(lb_train.shape[1]),
            'role': 'receiver',
        }

        probes = _make_probes(decoder, lb_train, labels_train, lam_all, lstsq_device)

        # No Mismatch baseline: probe applied to the receiver's own test embeddings
        nm_metrics: dict[str, float] = {}
        nm_acc_by_lam: dict[float, float] = {}
        if selected_metrics.get('accuracy', False):
            nm_metrics, nm_acc_by_lam = _evaluate(
                {'accuracy': True}, lb_test, lb_test, probes, labels_test
            )
        if selected_metrics.get('mse', False):
            nm_metrics.update(
                {
                    'mse': 0.0,
                    'mse_per_sample': 0.0,
                    'nmse': 0.0,
                    'd_b': int(lb_test.shape[1]),
                }
            )
        tqdm.write(f'  no_mismatch: {nm_metrics}')

        for model_a in tqdm(models_a, desc='Models A', unit='model'):
            out_path = results_dir / f'{model_a}__{model_b}__{dataset}.parquet'
            if out_path.exists() and not overwrite:
                tqdm.write(f'[SKIP] {model_a} -> {model_b} ({out_path.name})')
                continue

            tqdm.write(f'\n{"=" * 60}')
            tqdm.write(f'[PAIR] {model_a}  ->  {model_b}')

            tqdm.write(f'  Loading {model_a} train...')
            la_train, _ = _load_split(dataset_path, model_a, cfg.train_split)
            tqdm.write(f'  Loading {model_a} test...')
            la_test, _ = _load_split(dataset_path, model_a, cfg.test_split)
            info['models'][model_a] = {
                'latent_dim': int(la_train.shape[1]),
                'role': 'transmitter',
            }
            info['pairs'][f'{model_a}__{dataset}'] = {
                'd_a': int(la_train.shape[1]),
                'd_b': int(lb_train.shape[1]),
                'n_train': int(la_train.shape[0]),
                'n_test': int(la_test.shape[0]),
            }
            tqdm.write(
                f'  Train A={la_train.shape}  B={lb_train.shape}'
                f'  |  Test A={la_test.shape}  B={lb_test.shape}'
            )

            base = {'model_a': model_a, 'model_b': model_b, 'dataset': dataset}
            rows: list[dict] = []
            sweep_rows: list[dict] = []

            def add(
                method: str,
                k: int,
                seed: int | None,
                metrics: dict[str, float],
                acc_by_lam: dict[float, float],
                base: dict = base,
                rows: list[dict] = rows,
                sweep_rows: list[dict] = sweep_rows,
            ) -> None:
                rows.append(
                    {
                        **base,
                        'method': method,
                        'k': k,
                        'seed': seed,
                        'decoder': decoder,
                        'lam_rel': lam_rel if decoder == 'ridge' else None,
                        **metrics,
                    }
                )
                if decoder == 'ridge' and lam_sweep:
                    sweep_rows.extend(
                        {
                            **base,
                            'method': method,
                            'k': k,
                            'seed': seed,
                            'lam_rel': lam,
                            'accuracy': acc,
                        }
                        for lam, acc in acc_by_lam.items()
                    )

            add('no_mismatch', 0, None, nm_metrics, nm_acc_by_lam)

            linear_svd = None
            if 'linear' in selected_methods:
                tqdm.write('  linear: fitting SVD of linear map...')
                linear_svd = _fit_linear(la_train, lb_train, seeds[0])

            for k in tqdm(k_values, desc='  k values', unit='k', leave=False):
                tqdm.write(f'\n  [k={k}]')

                if 'proto' in selected_methods:
                    for seed in seeds:
                        b_hat = _proto_transmit(la_train, lb_train, la_test, k, seed)
                        m, acc = _evaluate(
                            selected_metrics, b_hat, lb_test, probes, labels_test
                        )
                        tqdm.write(f'    proto (seed {seed}): {m}')
                        add('proto', k, seed, m, acc)

                if 'cca' in selected_methods:
                    b_hat = _cca_transmit(la_train, lb_train, la_test, k)
                    m, acc = _evaluate(
                        selected_metrics, b_hat, lb_test, probes, labels_test
                    )
                    tqdm.write(f'    cca:    {m}')
                    add('cca', k, None, m, acc)

                if linear_svd is not None:
                    U, s, Vt, ls_a_lin, ls_b_lin = linear_svd
                    b_hat = _linear_transmit(la_test, U, s, Vt, ls_a_lin, ls_b_lin, k)
                    m, acc = _evaluate(
                        selected_metrics, b_hat, lb_test, probes, labels_test
                    )
                    tqdm.write(f'    linear: {m}')
                    add('linear', k, None, m, acc)

            if cfg.get('save_results', True):
                df = pl.DataFrame(rows, infer_schema_length=None)
                df.write_parquet(out_path)
                if sweep_rows:
                    pl.DataFrame(sweep_rows, infer_schema_length=None).write_parquet(
                        lambda_dir / out_path.name
                    )
                tqdm.write(f'\n  [SAVED] {out_path}')
                _write_manifest(manifest_path, cfg, info, lstsq_device)

    _write_manifest(manifest_path, cfg, info, lstsq_device)
    tqdm.write(f'[DONE] manifest: {manifest_path}')


if __name__ == '__main__':
    main()
