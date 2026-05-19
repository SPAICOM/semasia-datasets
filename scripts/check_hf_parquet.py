"""
Check whether parquet files on Hugging Face are readable via polars.

For each dataset, uses collect_models_by_split to discover available
splits and models, then tries pl.scan_parquet over the hf:// URI.

Results are written incrementally (after each model) to:
    check/<dataset>_working.parquet  — rows: dataset, model, split
    check/<dataset>_failing.parquet  — rows: dataset, model, error

Usage:
    uv run scripts/check_hf_parquet.py
    uv run scripts/check_hf_parquet.py datasets=[cifar10,cifar100]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src import collect_models_by_split


def _save(path: Path, records: list[dict]) -> None:
    if not records:
        return
    pl.DataFrame(records).write_parquet(path)


@hydra.main(
    config_path='../configs/hydra/',
    config_name='check_hf',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    check_dir = Path('check')
    check_dir.mkdir(parents=True, exist_ok=True)

    for dataset in tqdm(cfg.datasets, desc='Datasets'):
        repo_id = f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{dataset}'
        print(f'\n[INFO] {repo_id}')

        models_by_split = collect_models_by_split(repo_id=repo_id)
        if not models_by_split:
            print(f'[WARN] No models found for {repo_id}, skipping.')
            continue

        # All models across all splits
        all_models: set[str] = set()
        for models in models_by_split.values():
            all_models.update(models)

        working_path = check_dir / f'{dataset}_working.parquet'
        failing_path = check_dir / f'{dataset}_failing.parquet'

        # Load previously saved results to avoid re-checking
        working: list[dict] = (
            pl.read_parquet(working_path).to_dicts() if working_path.exists() else []
        )
        failing: list[dict] = (
            pl.read_parquet(failing_path).to_dicts() if failing_path.exists() else []
        )

        already_checked_models: set[str] = {r['model'] for r in working} | {
            r['model'] for r in failing
        }

        models_to_check = sorted(all_models - already_checked_models)

        for model in tqdm(models_to_check, desc=f'{dataset} models', leave=False):
            model_ok = True
            model_error: str | None = None

            for split, split_models in models_by_split.items():
                if model not in split_models:
                    continue

                uri = f'hf://datasets/{repo_id}/{split}/{model}/*.parquet'
                try:
                    pl.scan_parquet(uri).limit(1).collect()
                    working.append({'dataset': dataset, 'model': model, 'split': split})
                except Exception as e:
                    model_ok = False
                    model_error = str(e)
                    print(f'[FAIL] {dataset}/{split}/{model}: {e}')

            if not model_ok:
                failing.append({'dataset': dataset, 'model': model, 'error': model_error})

            _save(working_path, working)
            _save(failing_path, failing)

        print(
            f'[DONE] {dataset}: {len(working)} working entries, '
            f'{len(failing)} failing models'
        )


if __name__ == '__main__':
    main()
