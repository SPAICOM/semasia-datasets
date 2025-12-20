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

from src.huggingface import push_folder_to_hub
from src.io import latents_to_parquet_shards
from src.models.latent import LatentExtractor
from src.models.timm import load_model


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

    # Log the name of the dataset
    print(f'[INFO] Dataset that will be encoded: {cfg.dataset.name}')

    # Create subdirectories
    dataset_export_root = EXPORT_ROOT / cfg.dataset.name
    (dataset_export_root / 'train').mkdir(parents=True, exist_ok=True)
    (dataset_export_root / 'test').mkdir(parents=True, exist_ok=True)

    # Seed everything
    seed_everything(cfg.seed)

    # Retrieve the datasets - both train and test
    train_dataset, test_dataset = load_dataset(cfg.dataset.name).values()

    # Enumerate ALL pretrained timm models
    # (change to list_models() for non-pretrained too)
    all_models = set(timm.list_models(pretrained=True))

    # Get already processed models
    already_processed = {
        p.parent.name for p in (dataset_export_root / 'train').rglob('*.parquet')
    } & {p.parent.name for p in (dataset_export_root / 'test').rglob('*.parquet')}

    print(f'[INFO] Found {len(already_processed)} already processed models.')

    # Remove already processed models
    models_to_do = all_models - already_processed
    models_to_do = sorted(models_to_do)

    # Handling labels and extras
    train_data = {col: list(train_dataset[col]) for col in cfg.dataset.extras}
    test_data = {col: list(test_dataset[col]) for col in cfg.dataset.extras}

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
    for model_name in tqdm(models_to_do, desc=f'TIMM {len(all_models)} Models'):
        if cfg.encode:
            try:
                model = load_model(model_name=model_name, device=cfg.device)

                model_cfg = resolve_data_config({}, model=model)
                transform = create_transform(**model_cfg)

                extractor = LatentExtractor(model=model)
            except Exception as e:
                print(f'[ERROR][{model_name}] creating model: {e}')
                continue

            # -----------------------------------------------------------------
            #                   Prepare the DataLoaders
            # -----------------------------------------------------------------
            # Training Set
            train_dataloader = DataLoader(
                train_dataset[cfg.dataset.data],
                num_workers=cfg.dataloader.num_workers,
                pin_memory=cfg.dataloader.pin_memory,
                batch_size=cfg.dataloader.batch_size,
                collate_fn=partial(
                    collate_fn,
                    transform=transform,
                ),
            )

            # Test Set
            test_dataloader = DataLoader(
                test_dataset[cfg.dataset.data],
                num_workers=cfg.dataloader.num_workers,
                pin_memory=cfg.dataloader.pin_memory,
                batch_size=cfg.dataloader.batch_size,
                collate_fn=partial(
                    collate_fn,
                    transform=transform,
                ),
            )
            # -----------------------------------------------------------------

            # Encode the data, both train and test
            train_latents, test_latents = trainer.predict(
                extractor,
                dataloaders=[train_dataloader, test_dataloader],
            )

            # Concatenate
            train_latents = torch.cat(train_latents, dim=0).contiguous()
            test_latents = torch.cat(test_latents, dim=0).contiguous()

            # -----------------------------------------------------------------
            #                   Dump the Training Set
            # -----------------------------------------------------------------
            temp = train_data.copy()
            temp['model_name'] = model_name
            temp['embedding'] = train_latents.numpy()

            # Save the encodings in a polars DataFrame
            df = pl.DataFrame(temp).with_row_index('id')

            # Write the parquet in shards
            latents_to_parquet_shards(
                df=df,
                export_path=(dataset_export_root / 'train') / model_name,
                max_rows_per_shard=cfg.parquet.max_rows_per_shard,
            )

            del temp
            # -----------------------------------------------------------------
            #                   Dump the Test Set
            # -----------------------------------------------------------------
            temp = test_data.copy()
            temp['model_name'] = model_name
            temp['embedding'] = test_latents.numpy()

            # Save the encodings in a polars DataFrame
            df = pl.DataFrame(temp).with_row_index('id')

            # Write the parquet in shards
            latents_to_parquet_shards(
                df=df,
                export_path=(dataset_export_root / 'test') / model_name,
                max_rows_per_shard=cfg.parquet.max_rows_per_shard,
            )

            del temp
            # -----------------------------------------------------------------

            # Remove model from gpu
            model = model.cpu()

        # -----------------------------------------------------------------
        #                   Push dataset to Hugging Face
        # -----------------------------------------------------------------
        if cfg.hf.push:
            # Pust the Train Set
            push_folder_to_hub(
                local_folder=dataset_export_root / f'train/{model_name}',
                path_in_repo=f'train/{model_name}',
                repo_id=f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{cfg.dataset.name}',
                private=cfg.hf.private,
                commit_message=cfg.hf.commit_message,
            )

            # Pust the Test Set
            push_folder_to_hub(
                local_folder=dataset_export_root / f'test/{model_name}',
                path_in_repo=f'test/{model_name}',
                repo_id=f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{cfg.dataset.name}',
                private=cfg.hf.private,
                commit_message=cfg.hf.commit_message,
            )

        break

    return None


if __name__ == '__main__':
    main()
