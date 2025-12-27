# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "hydra-core",
#     "lightning",
#     "polars",
#     "timm",
# ]
# ///
""""""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from functools import partial

import hydra
import polars as pl
import timm
import torch
from datasets import load_dataset
from huggingface_hub import create_repo
from huggingface_hub.utils import get_token
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.huggingface import (
    collect_models_by_split,
    push_folder_to_hub,
)
from src.io import latents_to_parquet_shards
from src.models.latent import LatentExtractor
from src.models.timm import load_model
from src.utils import collect_local_models_by_split


def collate_fn(
    batch,
    transform,
):
    return torch.stack(
        [transform(sample.convert('RGB')) for sample in batch],
        dim=0,
    )


@hydra.main(
    config_path='../configs/hydra/',
    config_name='config',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """"""
    # Define paths
    CURRENT: Path = Path('.')
    EXPORT_ROOT: Path = CURRENT / 'data'  # Where Parquet trees will be written
    EXPORT_ROOT.mkdir(exist_ok=True, parents=True)

    # Variables
    repo_id: str = f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{cfg.dataset.name}'

    # Log the name of the dataset
    print(f'[INFO] Dataset that will be encoded: {cfg.dataset.name}')

    # Seed everything
    seed_everything(cfg.seed)

    # Retrieve the datasets - both train and test
    dataset = load_dataset(cfg.dataset.name)

    # Create subdirectories
    dataset_export_root = EXPORT_ROOT / cfg.dataset.name
    for split in dataset:
        (dataset_export_root / split).mkdir(parents=True, exist_ok=True)

    # Enumerate ALL pretrained timm models
    # (change to list_models() for non-pretrained too)
    all_models = set(timm.list_models(pretrained=True))
    all_models = sorted(all_models)

    if cfg.models_startwith is not None:
        all_models = [
            model_name
            for model_name in all_models
            if model_name.startswith(cfg.models_startwith)
        ]

    # Get already processed models
    already_processed_models = collect_local_models_by_split(
        dataset_export_root=dataset_export_root
    )

    # Get models already loaded on Hugging Face
    already_loaded_models = collect_models_by_split(
        repo_id=repo_id,
    )

    if not cfg.hf.re_push:
        loaded_models = set().union(*already_loaded_models.values())

        all_models = [
                    model_name
                    for model_name in all_models
                    if model_name not in loaded_models
                ]
        
    # Handling extras
    data = {}
    for split in dataset:
        data[split] = {col: list(dataset[split][col]) for col in cfg.dataset.extras}

    # Define the Trainer
    trainer = Trainer(
        accelerator=cfg.device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    # Hugging Face Token
    token = get_token()
    if token is None:
        print('[WARN] No HF token found. Run `huggingface-cli login` or set HF_TOKEN.')
        return

    # Ensure Hugging Face repo exists
    create_repo(
        repo_id=f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{cfg.dataset.name}',
        repo_type='dataset',
        private=cfg.hf.private,
        exist_ok=True,
        token=token,
    )

    # Compute the latent encodings for each model
    for model_name in tqdm(all_models, desc=f'TIMM {len(all_models)} Models'):
        # Encode each split separately
        for split in dataset:
            if cfg.encode:
                # Check if model has already been loaded in HF
                if (
                    model_name in already_loaded_models.get(split, set())
                ) and not cfg.hf.re_push:
                    continue

                # Check if model has already been processed
                if (
                    model_name in already_processed_models.get(split, set())
                ) and not cfg.re_encode:
                    continue

                print(f'\n\n[INFO] Proceed with {model_name} {split=}')

                try:
                    model = load_model(model_name=model_name, device=cfg.device)

                    model_cfg = resolve_data_config({}, model=model)
                    transform = create_transform(**model_cfg)

                    extractor = LatentExtractor(model=model)
                except Exception as e:
                    print(f'[ERROR][{model_name}] creating model: {e}')
                    continue

                # -----------------------------------------------------------------
                #                   Prepare the DataLoader
                # -----------------------------------------------------------------
                # Training Set
                dataloader = DataLoader(
                    dataset[split][cfg.dataset.data],
                    num_workers=cfg.dataloader.num_workers,
                    pin_memory=cfg.dataloader.pin_memory,
                    batch_size=cfg.dataloader.batch_size,
                    collate_fn=partial(
                        collate_fn,
                        transform=transform,
                    ),
                )
                # -----------------------------------------------------------------

                try:
                    # Encode the data, both train and test
                    latents = trainer.predict(
                        extractor,
                        dataloaders=dataloader,
                    )
                except torch.OutOfMemoryError as e:
                    print(
                        f'[ERROR] Skipped model {model_name} {split=}, because of {e}'
                    )
                    continue

                # Concatenate
                latents = torch.cat(latents, dim=0).contiguous()

                # -----------------------------------------------------------------
                #                   Dump the Latents
                # -----------------------------------------------------------------
                temp = data[split].copy()
                temp['model_name'] = model_name
                temp['embedding'] = latents.numpy()

                # Save the encodings in a polars DataFrame
                df = pl.DataFrame(temp).with_row_index('id')

                # Write the parquet in shards
                latents_to_parquet_shards(
                    df=df,
                    export_path=(dataset_export_root / split) / model_name,
                    max_rows_per_shard=cfg.parquet.max_rows_per_shard,
                )

                # Cleaning:
                # - delete temp
                # - remove model from gpu
                del temp
                model = model.cpu()

            # -----------------------------------------------------------------
            #                   Push dataset to Hugging Face
            # -----------------------------------------------------------------
            if cfg.hf.push:
                local_folder = dataset_export_root / f'{split}/{model_name}'

                if not local_folder.is_dir():
                    print(f'\n[INFO] Skipped HF push for model {model_name} {split=}.')
                    continue

                # Check if model has already been loaded on Hugging Face
                if (
                    model_name in already_loaded_models.get(split, set())
                ) and not cfg.hf.re_push:
                    continue

                # Pust to Hugging Face
                push_folder_to_hub(
                    local_folder=local_folder,
                    path_in_repo=f'{split}/{model_name}',
                    repo_id=repo_id,
                    private=cfg.hf.private,
                    commit_message=cfg.hf.commit_message,
                )

    return None


if __name__ == '__main__':
    main()
