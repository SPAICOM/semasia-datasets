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

    dataset_id: str = cfg.dataset.name.split('/')[-1]
    dataset: str = f'{cfg.repo_id}/{cfg.prefix}{dataset_id}'
    cache_pattern: str = f'{cfg.repo_id}___{cfg.prefix}{dataset_id}'
    hub_pattern: str = f'datasets--{cfg.repo_id}--{cfg.prefix}{dataset_id}*'

    models: list[str] = (
        pl.read_parquet('hf://datasets/spaicom-lab/model-registry/**/*.parquet')
        .filter(pl.col('latent_dim') < cfg.preprocess.max_latent)
        .filter((cfg.model is None) | pl.col('model_name').str.contains(cfg.model))
        .select('model_name')
        .unique()
        .sort('model_name')['model_name']
        .to_list()
    )

    for model in tqdm(models):
        for split in DATASET_SPLITS[dataset_id]:

            if split in cfg.splits:
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
                        if name in data.column_names
                    }

                ls = LatentSpace(latent, extras=extras, seed=cfg.seed)

                if (cfg.preprocess.max_points > 0) and (ls.n_points > cfg.preprocess.max_points):
                    ls = ls.subsample(n_points=cfg.preprocess.max_points,
                                      compute_prototypes=cfg.preprocess.prototypes.enable,
                                      n_samples=cfg.preprocess.prototypes.n_samples,
                                      seed=cfg.seed,)

                if cfg.preprocess.normalize is not None:
                    latent_processed = ls.normalize(cfg.preprocess.normalize)
                else:
                    latent_processed = ls.latent

                if cfg.preprocess.dim_reduction is not None:
                    latent_processed = ls.reduce_dimensions(
                        cfg.preprocess.dim_reduction,
                        cfg.preprocess.dim_reduction_components,
                        seed=cfg.seed,
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
                    / f'{cfg.repo_id}__{cfg.prefix}{dataset_id}__{split}__{model}.parquet'
                )

                remove_matching('~/.cache/huggingface/datasets/', f'{cache_pattern}*')
                remove_matching('~/.cache/huggingface/hub/', hub_pattern)


if __name__ == '__main__':
    main()
