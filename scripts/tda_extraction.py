import logging
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

logging.getLogger('httpx').setLevel(logging.WARNING)

import hydra
import polars as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src import remove_matching
from src.tda import TDA_KEYS, compute_tda_signature

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
    config_name='tda_extraction',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """"""

    # Define paths
    CURRENT: Path = Path('.')
    RESULTS: Path = CURRENT / 'results/tda_signatures/'

    # Create the results directory that will be populated by parquets
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Variables
    dataset: str = f'{cfg.repo_id}/{cfg.prefix}{cfg.dataset}'
    cache_pattern: str = f'{cfg.repo_id}___{cfg.prefix}{cfg.dataset}'
    hub_pattern: str = f'datasets--{cfg.repo_id}--{cfg.prefix}{cfg.dataset}*'

    # Get all the models from the model-registry
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
            # Instantiate the temp dictionary (TDA columns pre-filled with None)
            temp: dict[str, any] = {
                'dataset': dataset,
                'split': split,
                'model': model,
                **dict.fromkeys(TDA_KEYS),
            }

            # At the first time it will download all the splits
            # and then load the fist split
            # From the second split on it will only load it
            try:
                data = load_dataset(dataset, model, split=split).with_format('torch')
            except Exception as e:
                print(
                    f"Error loading dataset for model '{model}', split '{split}': {e}"
                )
                continue

            # Get the latent in torch format
            latent: torch.Tensor = torch.vstack(list(data['embedding']))

            # Compute TDA signature for this model × dataset × split triplet
            output_tda: dict[str, list] = compute_tda_signature(
                latent,
                max_dim=cfg.tda.max_dim,
                simplicial_filter=cfg.tda.simplicial_filter,
                n_bins=cfg.tda.n_bins,
                sigma=cfg.tda.sigma,
                metric=cfg.tda.metric,
                max_points=cfg.tda.max_points,
                seed=cfg.tda.seed,
                normalize=cfg.tda.normalize or None,
                dim_reduction=cfg.tda.dim_reduction or None,
                dim_reduction_components=cfg.tda.dim_reduction_components,
            )

            # Dump the parquet file: one row per dataset-split-model triplet
            (
                pl.DataFrame(temp | output_tda).write_parquet(
                    RESULTS
                    / f'{cfg.repo_id}__{cfg.prefix}{cfg.dataset}__{split}__{model}'
                )
            )

        # Clean the cache when a model is done (all splits are done)
        remove_matching('~/.cache/huggingface/datasets/', f'{cache_pattern}*')
        remove_matching('~/.cache/huggingface/hub/', hub_pattern)

    return None


if __name__ == '__main__':
    main()
