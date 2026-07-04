"""
Encode dataset images using timm models and upload to Hugging Face.

This script processes images from a dataset using pretrained timm models,
extracts latent embeddings, saves them as Parquet shards, and optionally
uploads them to a Hugging Face repository.

Usage:
    uv run scripts/encode_dataset_all_timm.py dataset=cifar10 hf.push=false
"""

import sys
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import polars as pl
import timm
import torch
from datasets import load_dataset
from huggingface_hub import create_repo
from huggingface_hub.utils import get_token
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import (
    LatentExtractor,
    collect_local_models_by_split,
    collect_models_by_split,
    encoder_params_below_threshold,
    latents_to_parquet_shards,
    load_model,
    push_folder_to_hub,
    remove_matching,
)


def collate_fn(batch: list[Any], transform: Any) -> torch.Tensor:
    """
    Collate a batch of images with the given transform.

    Parameters
    ----------
    batch : list[Any]
        List of image samples.
    transform : Any
        Transform to apply to each image.

    Returns
    -------
    torch.Tensor
        Stacked and transformed images.
    """
    return torch.stack([transform(sample.convert('RGB')) for sample in batch], dim=0)


@hydra.main(
    config_path='../configs/hydra/',
    config_name='encode_timm',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """
    Encode dataset images using timm models.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing dataset, model, trainer, and
        HuggingFace settings.
    """
    current: Path = Path('.')
    export_root: Path = current / 'data'
    export_root.mkdir(exist_ok=True, parents=True)

    dataset_name = cfg.dataset.name.split('/')[-1]
    repo_id: str = f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{dataset_name}'

    print(f'[INFO] Dataset: {cfg.dataset.name}')

    seed_everything(cfg.seed)

    dataset = load_dataset(cfg.dataset.name)

    dataset_export_root = export_root / dataset_name
    for split in cfg.dataset.split:
        (dataset_export_root / split).mkdir(parents=True, exist_ok=True)

    data: dict[str, dict[str, list[Any]]] = {}
    for split in cfg.dataset.split:
        data[split] = {col: list(dataset[split][col]) for col in cfg.dataset.extras}

    token = get_token()
    if token is None:
        print('[WARN] No HF token found. Run `huggingface-cli login` or set HF_TOKEN.')
        return

    create_repo(
        repo_id=repo_id,
        repo_type='dataset',
        private=cfg.hf.private,
        exist_ok=True,
        token=token,
    )

    all_models = set(timm.list_models(pretrained=True))
    all_models = sorted(all_models)

    if cfg.models_startwith is not None:
        all_models = [m for m in all_models if m.startswith(cfg.models_startwith)]

    already_processed_models = collect_local_models_by_split(
        dataset_export_root=dataset_export_root
    )
    already_loaded_models = collect_models_by_split(repo_id=repo_id)

    if not cfg.hf.re_push and already_loaded_models:
        loaded_models = set.intersection(*already_loaded_models.values())
    else:
        loaded_models = set()
    all_models = [m for m in all_models if m not in loaded_models]

    if not cfg.re_encode and not cfg.hf.re_push and already_processed_models:
        processed_models = set.intersection(*already_processed_models.values())
    else:
        processed_models = set()
    all_models = [m for m in all_models if m not in processed_models]

    trainer = Trainer(**cfg.trainer)

    for model_name in tqdm(all_models, desc=f'TIMM {len(all_models)} Models'):
        print(f'\n\n[INFO] Checking {model_name} parameters...', end='\t')
        if not encoder_params_below_threshold(
            model_name=model_name,
            threshold=cfg.parameters_threshold,
        ):
            print('[SKIPPED]')
            continue
        print('[OK]')

        for split in cfg.dataset.split:
            if cfg.encode:
                if (
                    model_name in already_loaded_models.get(split, set())
                ) and not cfg.hf.re_push:
                    continue

                if (
                    model_name in already_processed_models.get(split, set())
                ) and not cfg.re_encode:
                    continue

                print(f'\n\n[INFO] Processing {model_name} {split=}')

                try:
                    model = load_model(model_name=model_name)
                    model_cfg = resolve_data_config({}, model=model)
                    transform = create_transform(**model_cfg)
                    extractor = LatentExtractor(model=model)
                except Exception as e:
                    print(f'[ERROR][{model_name}] creating model: {e}')
                    continue

                dataloader = DataLoader(
                    dataset[split][cfg.dataset.data],
                    num_workers=cfg.dataloader.num_workers,
                    pin_memory=cfg.dataloader.pin_memory,
                    batch_size=cfg.dataloader.batch_size,
                    collate_fn=partial(collate_fn, transform=transform),
                )

                try:
                    latents = trainer.predict(extractor, dataloaders=dataloader)
                except torch.OutOfMemoryError as e:
                    print(f'[ERROR] Skipped {model_name} {split=}: {e}')
                    continue

                latents = torch.cat(latents, dim=0).contiguous()

                temp = data[split].copy()
                temp['model_name'] = model_name
                temp['embedding'] = latents.numpy()

                df = pl.DataFrame(temp).with_row_index('id')

                latents_to_parquet_shards(
                    df=df,
                    export_path=(dataset_export_root / split) / model_name,
                    max_rows_per_shard=cfg.parquet.max_rows_per_shard,
                )

                del temp
                model = model.cpu()

            if cfg.hf.push:
                local_folder = dataset_export_root / f'{split}/{model_name}'

                if not local_folder.is_dir():
                    print(f'\n[INFO] Skipped HF push for {model_name} {split=}.')
                    continue

                if (
                    model_name in already_loaded_models.get(split, set())
                ) and not cfg.hf.re_push:
                    continue

                push_folder_to_hub(
                    local_folder=local_folder,
                    path_in_repo=f'{split}/{model_name}',
                    repo_id=repo_id,
                    private=cfg.hf.private,
                    commit_message=cfg.hf.commit_message,
                )

        remove_matching('~/.cache/huggingface/hub', '*timm*')


if __name__ == '__main__':
    main()
