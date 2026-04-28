"""Transmit model A's test embeddings to model B's raw embedding space via
injected or Hungarian-aligned prototypes, and evaluate classification quality
via lstsq linear probing trained on model B's raw train embeddings."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use('Agg')

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn.functional as torch_F
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.metrics.alignment import align_prototypes
from src.objects import LatentSpace

if TYPE_CHECKING:
    from omegaconf import DictConfig

logging.getLogger('httpx').setLevel(logging.WARNING)

_PLOTLY_STYLE: dict = {
    'font': {'family': 'Times New Roman', 'size': 12},
    'paper_bgcolor': 'white',
    'plot_bgcolor': 'white',
}

_METRIC_COLORS = {
    'accuracy': '#636EFA',
    'precision': '#00CC96',
    'recall': '#AB63FA',
    'f1': '#FFA15A',
    'injected': '#636EFA',
    'hungarian': '#636EFA',
}


def _metric_color(name: str) -> str:
    for key, color in _METRIC_COLORS.items():
        if key in name.lower():
            return color
    return '#333333'


@dataclass
class AlignmentResult:
    strategy: str
    dataset: str
    train_split: str
    test_split: str
    model_a: str
    model_b: str
    n_prototypes: int
    mse: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def _load_split(
    dataset_path: str, model: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, labels) for one model/split combination."""
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


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(
            precision_score(y_true, y_pred, average='macro', zero_division=0)
        ),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def _save_plotly(fig, path: Path) -> Path:
    fig.write_image(path.with_suffix('.png'))
    try:
        fig.write_image(path.with_suffix('.pdf'))
        return path.with_suffix('.pdf')
    except Exception:
        html_path = path.with_suffix('.html')
        fig.write_html(html_path)
        return html_path


def _plot_line(
    x: list[int],
    series: dict[str, list[float]],
    xaxis_title: str,
    yaxis_title: str,
    yaxis_range: list | None = None,
    hlines: dict[str, float] | None = None,
    hline_colors: dict[str, str] | None = None,
    dashes: dict[str, str] | None = None,
) -> object:
    """Line plot with optional horizontal reference lines.

    Parameters
    ----------
    dashes : dict[series_name -> dash style], e.g. {'Hungarian': 'dot'}
    hlines : dict[label -> y_value] for horizontal reference lines
    hline_colors : dict[label -> color]; falls back to _metric_color(label)
    """
    import plotly.graph_objects as go

    dashes = dashes or {}
    hline_colors = hline_colors or {}
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode='lines+markers',
                name=name,
                line={
                    'color': _metric_color(name),
                    'width': 2,
                    'dash': dashes.get(name, 'solid'),
                },
                marker={'size': 7},
            )
        )
    if hlines:
        for label, value in hlines.items():
            color = hline_colors.get(label, _metric_color(label))
            fig.add_hline(
                y=value,
                line_dash='dash',
                line_color=color,
                line_width=1.5,
                opacity=0.6,
                annotation_text=label,
                annotation_position='right',
                annotation_font={'family': 'Times New Roman', 'size': 10},
            )
    layout = dict(
        **_PLOTLY_STYLE,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend={'orientation': 'h', 'y': 1.05, 'x': 0.5, 'xanchor': 'center'},
        hovermode='x unified',
        height=400,
        margin={'t': 30, 'b': 50, 'l': 60, 'r': 80},
    )
    if yaxis_range is not None:
        layout['yaxis_range'] = yaxis_range
    fig.update_layout(**layout)
    fig.update_xaxes(
        range=[x[0], x[-1]],
        tickmode='array',
        tickvals=x,
        tickangle=0,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.4)',
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.4)',
    )
    return fig


@hydra.main(
    config_path='../configs/hydra/',
    config_name='proto_alignment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    current = Path('.')
    results_dir = current / 'results' / 'proto_alignment'
    plots_dir = results_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    train_split = cfg.train_split
    test_split = cfg.test_split
    model_a = cfg.model_a
    model_b = cfg.model_b
    seed = cfg.seed
    n_prototypes_list: list[int] = list(cfg.n_prototypes)
    selected_metrics: list[str] = list(
        cfg.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
    )

    dataset_path = f'{cfg.repo_id}/{cfg.prefix}{dataset}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load embeddings ───────────────────────────────────────────────────────
    print(f'\n[INFO] Loading embeddings from {dataset_path}')
    print(f'  {model_a}  train/{train_split}...')
    la_train, labels_train = _load_split(dataset_path, model_a, train_split)
    print(f'  {model_b}  train/{train_split}...')
    lb_train, _ = _load_split(dataset_path, model_b, train_split)
    print(f'  {model_a}  test/{test_split}...')
    la_test, labels_test = _load_split(dataset_path, model_a, test_split)
    print(f'  {model_b}  test/{test_split}...')
    lb_test, _ = _load_split(dataset_path, model_b, test_split)

    print(
        f'\n  Train  A={la_train.shape}  B={lb_train.shape}'
        f'\n  Test   A={la_test.shape}   B={lb_test.shape}'
    )

    # ── B baseline: lstsq on raw B train, evaluated on raw B test ────────────
    print('\n[INFO] Computing B baseline...')
    preds_base = _lstsq_predict(lb_train, labels_train, lb_test, device)
    baseline_metrics = _clf_metrics(labels_test, preds_base)
    print(
        f'  acc={baseline_metrics["accuracy"]:.4f}'
        f'  prec={baseline_metrics["precision"]:.4f}'
        f'  rec={baseline_metrics["recall"]:.4f}'
        f'  f1={baseline_metrics["f1"]:.4f}'
    )

    # ── Per-k alignment + evaluation ─────────────────────────────────────────
    results: list[AlignmentResult] = []

    for k in n_prototypes_list:
        print(f'\n[INFO] k={k} prototypes...')

        # Fit A's prototype space (prewhiten → KMeans → Parseval)
        ls_a = LatentSpace(la_train, seed=seed)
        ls_a.prewhiten(inplace=True)
        _, indices_a = ls_a.compute_prototypes(
            n_samples=None,
            clusterer_cls=KMeans,
            n_clusters=k,
            apply_parseval=True,
            return_cluster_indices=True,
        )

        # Whiten A test and project to prototype space
        la_test_w = ls_a.apply_whitening_operator(la_test)
        a_test_proto = ls_a.apply_analysis_operator(la_test_w)

        # ── Injected strategy ─────────────────────────────────────────────────
        ls_b_inj = LatentSpace(lb_train, seed=seed)
        ls_b_inj.prewhiten(inplace=True)
        ls_b_inj.compute_injected_prototypes(indices_a, apply_parseval=True)

        lb_test_w_inj = ls_b_inj.apply_whitening_operator(lb_test)
        b_test_proto_inj = ls_b_inj.apply_analysis_operator(lb_test_w_inj)
        mse_inj = float(np.mean((a_test_proto - b_test_proto_inj) ** 2))

        a_raw_b_inj = ls_b_inj.apply_dewhitening_operator(
            ls_b_inj.apply_synthesis_operator(a_test_proto)
        )
        preds_inj = _lstsq_predict(lb_train, labels_train, a_raw_b_inj, device)
        m_inj = _clf_metrics(labels_test, preds_inj)

        print(
            f'  [injected]  MSE={mse_inj:.4f}  acc={m_inj["accuracy"]:.4f}'
            f'  prec={m_inj["precision"]:.4f}  rec={m_inj["recall"]:.4f}'
            f'  f1={m_inj["f1"]:.4f}'
        )
        results.append(
            AlignmentResult(
                strategy='injected',
                dataset=dataset,
                train_split=train_split,
                test_split=test_split,
                model_a=model_a,
                model_b=model_b,
                n_prototypes=k,
                mse=mse_inj,
                **m_inj,
            )
        )

        # ── Hungarian strategy ────────────────────────────────────────────────
        ls_b_hun = LatentSpace(lb_train, seed=seed)
        ls_b_hun.prewhiten(inplace=True)
        _, indices_b_hun = ls_b_hun.compute_prototypes(
            n_samples=None,
            clusterer_cls=KMeans,
            n_clusters=k,
            apply_parseval=True,
            return_cluster_indices=True,
        )

        perm = align_prototypes(indices_a, indices_b_hun)
        # Remap A's prototype coords to B's ordering: A[i] → B[perm[i]]
        a_proto_hun = np.zeros_like(a_test_proto)
        a_proto_hun[:, perm] = a_test_proto

        lb_test_w_hun = ls_b_hun.apply_whitening_operator(lb_test)
        b_test_proto_hun = ls_b_hun.apply_analysis_operator(lb_test_w_hun)
        mse_hun = float(np.mean((a_proto_hun - b_test_proto_hun) ** 2))

        a_raw_b_hun = ls_b_hun.apply_dewhitening_operator(
            ls_b_hun.apply_synthesis_operator(a_proto_hun)
        )
        preds_hun = _lstsq_predict(lb_train, labels_train, a_raw_b_hun, device)
        m_hun = _clf_metrics(labels_test, preds_hun)

        print(
            f'  [hungarian] MSE={mse_hun:.4f}  acc={m_hun["accuracy"]:.4f}'
            f'  prec={m_hun["precision"]:.4f}  rec={m_hun["recall"]:.4f}'
            f'  f1={m_hun["f1"]:.4f}'
        )
        results.append(
            AlignmentResult(
                strategy='hungarian',
                dataset=dataset,
                train_split=train_split,
                test_split=test_split,
                model_a=model_a,
                model_b=model_b,
                n_prototypes=k,
                mse=mse_hun,
                **m_hun,
            )
        )

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pl.DataFrame(
        [
            {
                'strategy': r.strategy,
                'dataset': r.dataset,
                'train_split': r.train_split,
                'test_split': r.test_split,
                'model_a': r.model_a,
                'model_b': r.model_b,
                'n_prototypes': r.n_prototypes,
                'mse': r.mse,
                'accuracy': r.accuracy,
                'precision': r.precision,
                'recall': r.recall,
                'f1': r.f1,
            }
            for r in results
        ]
    )

    if cfg.get('save_results', True):
        out_path = results_dir / f'{model_a}__{model_b}__{dataset}.parquet'
        results_df.write_parquet(out_path)
        print(f'\n[COMPLETE] Results saved to {out_path}')

    print('\n[INFO] Summary:')
    print(results_df)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if cfg.get('save_plots', True):
        stem = f'{model_a}__{model_b}__{dataset}'

        res_inj = [r for r in results if r.strategy == 'injected']
        res_hun = [r for r in results if r.strategy == 'hungarian']

        # MSE vs k (both strategies)
        mse_series: dict[str, list[float]] = {
            'MSE (injected)': [r.mse for r in res_inj],
            'MSE (hungarian)': [r.mse for r in res_hun],
        }
        mse_dashes = {'MSE (hungarian)': 'dot'}
        mse_fig = _plot_line(
            n_prototypes_list,
            mse_series,
            xaxis_title='Prototypes',
            yaxis_title='MSE',
            dashes=mse_dashes,
        )
        mse_path = _save_plotly(mse_fig, plots_dir / f'{stem}__mse.pdf')
        print(f'[COMPLETE] MSE plot → {mse_path}')

        # Classification metrics vs k (both strategies + baseline hlines)
        cls_series: dict[str, list[float]] = {}
        cls_dashes: dict[str, str] = {}
        for m in selected_metrics:
            cls_series['Injected'] = [getattr(r, m) for r in res_inj]
            cls_series['Hungarian'] = [getattr(r, m) for r in res_hun]
            cls_dashes['Hungarian'] = 'dot'

        cls_hlines = (
            {'No Mismatch': baseline_metrics['accuracy']}
            if 'accuracy' in selected_metrics
            else {}
        )
        cls_hline_colors = {'No Mismatch': _METRIC_COLORS['accuracy']}
        cls_fig = _plot_line(
            n_prototypes_list,
            cls_series,
            xaxis_title='Prototypes',
            yaxis_title='Accuracy',
            yaxis_range=[0, 1],
            hlines=cls_hlines,
            hline_colors=cls_hline_colors,
            dashes=cls_dashes,
        )
        cls_path = _save_plotly(cls_fig, plots_dir / f'{stem}__classification.pdf')
        print(f'[COMPLETE] Classification plot → {cls_path}')


if __name__ == '__main__':
    main()
