import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import polars as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src import remove_matching

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
        .select('model_name')
        .unique()
        .sort('model_name')['model_name']
        .to_list()
    )

    for model in tqdm(models):
        for split in DATASET_SPLITS[cfg.dataset]:
            # Instantiate the temp dictionary
            temp: dict[str, any] = {
                'dataset': dataset,
                'split': split,
                'key1': None,
            }

            # At the first time it will download all the splits
            # and then load the fist split
            # From the second split on it will only load it
            data = load_dataset(dataset, model, split=split).with_format('torch')

            # Get the latent in torch format
            latent: torch.Tensor = torch.vstack(list(data['embedding']))

            # Dummy computation to check if all works
            print(latent.shape)

            # TODO: Calculate TDA signature
            # I need you to create a function or more, inside the src/tda/ subfolder
            # The function should return a dictionary with
            # as keys the attributes (will be the columns) and as values
            # the measurements.
            # Then update the temp dictionary and pass it to the dataframe
            # Pls initialize the temp also with the tda keys (see example in line 60)
            #
            # ...
            output_tda: dict[str, any] = {'key1': 'value1'}

            # Dump the parquet file: one for each dataset-split-model triplet
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
