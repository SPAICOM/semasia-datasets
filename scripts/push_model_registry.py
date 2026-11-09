"""
Push timm model metadata registry to Hugging Face.

This script:
1. Lists all pretrained timm models
2. Extracts metadata for each model
3. Saves incrementally to parquet files (robust to interruptions)
4. Pushes to HuggingFace dataset spaicom-lab/model-registry
5. Generates and pushes README

``latent_dim`` is the length of the saved embedding, computed by ``get_latent_dim``
inside ``get_model_metadata``.

Usage:
    uv run scripts/push_model_registry.py            # build and push
    uv run scripts/push_model_registry.py --no-push  # build locally only
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import timm
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token
from tqdm.auto import tqdm

from src.utils.huggingface import generate_model_registry_readme
from src.utils.timm_metadata import (
    add_derived_columns,
    get_model_metadata,
    refresh_pretrain_dataset,
    refresh_pretrain_ft,
    refresh_pretrain_org,
)

BATCH_SIZE = 100

METADATA_SCHEMA: dict[str, pl.DataType] = {
    'model_name': pl.String,
    'family': pl.String,
    'macro_family': pl.String,
    'model_version': pl.String,
    'size': pl.String,
    'depth_code': pl.String,
    'width_code': pl.String,
    'patch_size': pl.Int64,
    'input_resolution': pl.Int64,
    'window_size': pl.Int64,
    'stride_code': pl.String,
    'head_type': pl.String,
    'num_registers': pl.Int64,
    'positional_encoding': pl.String,
    'activation': pl.String,
    'pe_scope': pl.String,
    'is_distilled': pl.Boolean,
    'is_pruned': pl.Boolean,
    'is_legacy': pl.Boolean,
    'is_gap': pl.Boolean,
    'uses_rmlp': pl.Boolean,
    'uses_rw': pl.Boolean,
    'uses_cr': pl.Boolean,
    'uses_ns': pl.Boolean,
    'uses_abswin': pl.Boolean,
    'uses_quickgelu': pl.Boolean,
    'uses_ts': pl.Boolean,
    'uses_aa': pl.Boolean,
    'pretrain_config': pl.String,
    'pretrain_org': pl.String,
    'pretrain_dataset': pl.String,
    'pretrain_dataset_size': pl.String,
    'pretrain_method': pl.String,
    'pretrain_ft': pl.List(pl.String),  # ordered fine-tuning chain, None = no fine-tuning
    'pretrain_resolution': pl.Int64,
    'pretrain_ft_resolution': pl.Int64,
    'pretrain_epochs': pl.Int64,
    'pretrain_tokens': pl.String,
    'pretrain_aug': pl.String,
    'pretrain_i18n': pl.Boolean,
    'num_parameters': pl.Int64,
    'latent_dim': pl.Int64,
    'pretrain_tasks': pl.List(pl.String),
    'paradigm': pl.String,
}


def normalize_batch(batch: list[dict]) -> pl.DataFrame:
    """Create DataFrame and enforce consistent schema."""
    df = pl.DataFrame(batch, schema_overrides=METADATA_SCHEMA)
    return df


def load_failed_models(path: Path) -> list[str]:
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_failed_models(path: Path, failed: list[str]) -> None:
    path.write_text(json.dumps(failed, indent=2))


def load_existing_models(path: Path) -> set[str]:
    if path.exists():
        df = pl.read_parquet(path)
        return set(df['model_name'].to_list())
    return set()


def main() -> None:
    """Extract timm model metadata and (optionally) push to HuggingFace."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no-push', action='store_true', help='update the local registry only'
    )
    args = parser.parse_args()

    repo_id = 'spaicom-lab/semasia-model-registry'
    local_path = Path('data/semasia-model-registry')
    temp_dir = local_path / 'temp'
    local_path.mkdir(parents=True, exist_ok=True)

    parquet_path = local_path / 'semasia_model_registry.parquet'
    failed_path = local_path / 'failed_models.json'

    if parquet_path.exists() and temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    model_names = timm.list_models('*', pretrained=True)
    model_names = [m for m in model_names if '.' in m]

    processed = load_existing_models(parquet_path)
    failed = load_failed_models(failed_path)

    models_to_process = [
        m for m in model_names if m not in processed and m not in failed
    ]

    print(f'Found {len(model_names)} pretrained timm models with valid names')
    print(f'Already processed: {len(processed)}')
    print(f'Failed (will skip): {len(failed)}')
    print(f'To process: {len(models_to_process)}')

    if not models_to_process:
        print('Nothing new to process. Using existing parquet.')
        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
        else:
            df = pl.DataFrame(schema=METADATA_SCHEMA)
    else:
        start_time = time.time()
        batch: list[dict] = []
        batch_num = 0
        total_new = 0

        with tqdm(models_to_process, desc='Extracting metadata', leave=True) as pbar:
            for model_name in pbar:
                pbar.set_postfix_str(model_name[:40])
                try:
                    meta = get_model_metadata(model_name)
                    batch.append(meta)
                except Exception:
                    failed.append(model_name)
                    save_failed_models(failed_path, failed)

                if len(batch) >= BATCH_SIZE:
                    batch_num += 1
                    batch_df = normalize_batch(batch)
                    batch_path = temp_dir / f'batch_{batch_num:03d}.parquet'
                    batch_df.write_parquet(batch_path)
                    total_new += len(batch)
                    batch = []

        if batch:
            batch_num += 1
            batch_df = normalize_batch(batch)
            batch_path = temp_dir / f'batch_{batch_num:03d}.parquet'
            batch_df.write_parquet(batch_path)
            total_new += len(batch)

        elapsed = time.time() - start_time
        print(f'\nProcessed {total_new} new models in {elapsed:.1f}s')
        print(f'Total failed: {len(failed)}')

        batch_files = sorted(temp_dir.glob('batch_*.parquet'))
        if batch_files:
            dfs = [pl.read_parquet(f) for f in batch_files]
            df = pl.concat(dfs)
            df = df.unique(subset='model_name', keep='first')
            df = df.sort('model_name')
            df.write_parquet(parquet_path)
            print(f'Saved {len(df)} total models to {parquet_path}')

            shutil.rmtree(temp_dir)
        else:
            df = pl.DataFrame(schema=METADATA_SCHEMA)

    # pretrain_org is recomputed from pretrain_config for every model, also for those already
    # stored in the local parquet, so that a fix of the organisation parser needs no re-extraction.
    old_org = df['pretrain_org']
    df = refresh_pretrain_org(df)
    n_ds = df['pretrain_dataset'].null_count()
    df = refresh_pretrain_dataset(df)
    # pretrain_ft as an ordered list of fine-tuning datasets, recomputed from pretrain_config for every model
    df = refresh_pretrain_ft(df)
    print(f'pretrain_dataset filled for {n_ds - df["pretrain_dataset"].null_count()} models')
    n_changed = int((df['pretrain_org'].fill_null('') != old_org.fill_null('')).sum())
    print(f'pretrain_org changed for {n_changed} models')

    # Derived columns (pretrain_tasks, paradigm) are recomputed for every model,
    # also for those already stored in the local parquet.
    df = add_derived_columns(df)
    df.write_parquet(parquet_path)

    print(f'DataFrame shape: {df.shape}')
    print(f'Columns: {df.columns}')

    if args.no_push:
        print('--no-push: local registry written, nothing pushed.')
        return

    api = HfApi(token=get_token())
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)

    # Only the registry parquet is uploaded (not the whole folder: no backups, temp batches or other copies).
    api.upload_file(
        repo_id=repo_id,
        repo_type='dataset',
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_path.name,
        commit_message='Add/update model metadata registry',
    )

    generate_model_registry_readme(
        repo_dir=local_path,
        repo_id=repo_id,
        num_models=len(df),
        push_online=True,
        commit_message='Update README',
    )

    print(f'\nDone! Dataset available at: https://huggingface.co/datasets/{repo_id}')


if __name__ == '__main__':
    main()
