import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import torch
from datasets import load_dataset

from src.objects import AlignmentProblem, LatentSpace
from src.objects.latent import DimReductionMethod
from src.plotting.latent import plot_pc_correlation_heatmap


def _check_models_in_registry(model_a: str, model_b: str) -> None:
    """Check if models exist in the model registry."""
    model_df = pl.read_parquet('hf://datasets/spaicom-lab/model-registry/**/*.parquet')
    available_models = set(model_df['model_name'].unique().to_list())

    missing = []
    if model_a not in available_models:
        missing.append(model_a)
    if model_b not in available_models:
        missing.append(model_b)

    if missing:
        raise ValueError(
            f'Model(s) not found in model registry: {missing}\n'
            f'Available models: {sorted(available_models)[:20]}... (showing first 20)'
        )


def _check_models_in_dataset(model_a: str, model_b: str, dataset: str) -> None:
    """Check if models have embeddings in the dataset."""
    from datasets import get_dataset_config_names

    try:
        available = set(get_dataset_config_names(dataset))

        missing = []
        if model_a not in available:
            missing.append(model_a)
        if model_b not in available:
            missing.append(model_b)

        if missing:
            raise ValueError(
                f'Model(s) not found in dataset {dataset}: {missing}\n'
                f'Use scripts/encode_dataset_all_timm.py to encode missing models.'
            )
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f'Error checking dataset {dataset}: {e}') from e


def main(
    model_a: str,
    model_b: str,
    dataset: str,
    method: DimReductionMethod,
    n_components: int = 50,
    k: int | None = None,
    n_anchors: int | None = None,
    output_path: str | None = None,
    repo_id: str = 'spaicom-lab',
    prefix: str = 'semantic-',
) -> None:
    """Plot correlation heatmap between principal components of two models."""

    _check_models_in_registry(model_a, model_b)

    full_dataset = f'{repo_id}/{prefix}{dataset}'
    _check_models_in_dataset(model_a, model_b, full_dataset)

    latent_a = _load_latent(model_a, full_dataset)
    latent_b = _load_latent(model_b, full_dataset)

    if n_anchors is not None:
        print(f'Aligning via relative representation (n_anchors={n_anchors}) ...')
        latent_a, latent_b = AlignmentProblem(latent_a, latent_b).align(
            'relative',
            strategy='prototype',
            n_anchors=n_anchors,
        )

    fig, ax = plot_pc_correlation_heatmap(
        latent_a=latent_a,
        latent_b=latent_b,
        method=method,
        n_components=n_components,
        k=k,
        label_a=model_a,
        label_b=model_b,
    )

    if n_anchors is not None:
        ax.set_title(
            ax.get_title().replace(
                'PC correlation', f'PC correlation (relative, {n_anchors} anchors)'
            ),
            fontsize=10,
            pad=8,
        )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved heatmap to {output_path}')
    else:
        fig.show()


def _load_latent(model: str, dataset: str) -> LatentSpace:
    """Load latent embeddings from HuggingFace dataset."""
    data = load_dataset(dataset, model, split='test').with_format('torch')
    latent = torch.vstack(list(data['embedding']))
    return LatentSpace(latent, seed=42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot correlation heatmap between two models.'
    )
    parser.add_argument('model_a', help='First model (e.g., facebook/dinov2-base)')
    parser.add_argument('model_b', help='Second model (e.g., facebook/dinov2-large)')
    parser.add_argument('dataset', help='Dataset name on HuggingFace')
    parser.add_argument(
        '--repo-id',
        type=str,
        default='spaicom-lab',
        help='HuggingFace repository ID (default: spaicom-lab)',
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='semantic-',
        help='Dataset prefix (default: semantic-)',
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pca',
        choices=['pca', 'umap', 'tsne', 'lle', 'isomap', 'prototype_analysis'],
        help='Dimensionality reduction method (default: pca)',
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=50,
        help='Number of components to compute (default: 50)',
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Number of leading components to plot (default: n_components)',
    )
    parser.add_argument(
        '--n-anchors',
        type=int,
        default=None,
        help=(
            'If set, align both spaces via anchor-relative representation '
            'before computing PCs (uses AlignmentProblem with proto anchors). '
            'Example: --n-anchors 64'
        ),
    )
    parser.add_argument(
        '--output', type=str, default=None, help='Output file path for the plot'
    )

    args = parser.parse_args()
    main(
        model_a=args.model_a,
        model_b=args.model_b,
        dataset=args.dataset,
        method=args.method,
        n_components=args.n_components,
        k=args.k,
        n_anchors=args.n_anchors,
        output_path=args.output,
        repo_id=args.repo_id,
        prefix=args.prefix,
    )
