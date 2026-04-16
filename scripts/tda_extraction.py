import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src import remove_matching
from src.objects import LatentSpace
from src.tda import TDA_KEYS, compute_tda_features

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

logging.getLogger('httpx').setLevel(logging.WARNING)


@hydra.main(
    config_path='../configs/hydra/',
    config_name='tda_extraction',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """Extract TDA signatures from latent spaces using LatentSpace for preprocessing."""

    CURRENT: Path = Path('.')
    RESULTS: Path = CURRENT / 'results/tda_signatures/'

    RESULTS.mkdir(parents=True, exist_ok=True)

    dataset: str = f'{cfg.repo_id}/{cfg.prefix}{cfg.dataset}'
    cache_pattern: str = f'{cfg.repo_id}___{cfg.prefix}{cfg.dataset}'
    hub_pattern: str = f'datasets--{cfg.repo_id}--{cfg.prefix}{cfg.dataset}*'

    models: list[str] = (
        pl.read_parquet('hf://datasets/spaicom-lab/model-registry/**/*.parquet')
        .filter(pl.col('latent_dim') < cfg.tda.max_points)
        .filter((cfg.model is None) | pl.col('model_name').str.contains(cfg.model))
        .select('model_name')
        .unique()
        .sort('model_name')['model_name']
        .to_list()
    )

    for model in tqdm(models):
        for split in DATASET_SPLITS[cfg.dataset]:
            temp: dict[str, any] = {
                'dataset': dataset,
                'split': split,
                'model': model,
                **dict.fromkeys(TDA_KEYS),
            }

            try:
                data = load_dataset(dataset, model, split=split).with_format('torch')
            except Exception as e:
                print(
                    f"Error loading dataset for model '{model}', split '{split}': {e}"
                )
                continue

            latent: torch.Tensor = torch.vstack(list(data['embedding']))

            extras = None
            if hasattr(cfg.dataset, 'extras') and cfg.dataset.extras:
                extra_names = list(cfg.dataset.extras)
                extras = {
                    name: np.array(data[name])
                    for name in extra_names
                    if name in data.columns
                }

            ls = LatentSpace(latent, extras=extras, seed=cfg.tda.seed)

            if cfg.tda.max_points > 0 and ls.n_points > cfg.tda.max_points:
                ls = ls.subsample(cfg.tda.max_points, seed=cfg.tda.seed)

            if cfg.tda.normalize is not None:
                latent_processed = ls.normalize(cfg.tda.normalize)
            else:
                latent_processed = ls.latent

            if cfg.tda.dim_reduction is not None:
                latent_processed = ls.reduce_dimensions(
                    cfg.tda.dim_reduction,
                    cfg.tda.dim_reduction_components,
                    seed=cfg.tda.seed,
                )

            output_tda = compute_tda_features(
                latent_processed,
                max_dim=cfg.tda.max_dim,
                simplicial_filter=cfg.tda.simplicial_filter,
                n_bins=cfg.tda.n_bins,
                sigma=cfg.tda.sigma,
                metric=cfg.tda.metric,
            )

            pl.DataFrame(temp | output_tda).write_parquet(
                RESULTS
                / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{split}__{model}.parquet'
            )

            remove_matching('~/.cache/huggingface/datasets/', f'{cache_pattern}*')
            remove_matching('~/.cache/huggingface/hub/', hub_pattern)


if __name__ == '__main__':
    main()
