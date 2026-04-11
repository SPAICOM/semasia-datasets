"""
Push local model folders to Hugging Face with retry logic.

This script uploads preprocessed model folders to a Hugging Face dataset
repository. It handles rate limiting (HTTP 429) with automatic backoff and
retries transient network errors with exponential backoff.

Usage:
    uv run scripts/push_to_hf.py dataset=cifar10
"""

import random
import sys
import time
from pathlib import Path

import httpcore
import httpx

sys.path.append(str(Path(sys.path[0]).parent))


import hydra
from huggingface_hub.errors import HfHubHTTPError
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src import collect_unloaded_model_folders, push_folder_to_hub

RATE_LIMIT_SLEEP = 3600 + 60  # 1h + 60s safety margin
NETWORK_RETRY_BASE = 30  # seconds
NETWORK_RETRY_MAX = 10 * 60  # 10 minutes
MAX_RETRIES_PER_MODEL = 5


@hydra.main(
    config_path='../configs/hydra/',
    config_name='push_to_hf',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """
    Push unloaded model folders to Hugging Face with retry logic.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing dataset and HuggingFace settings.
    """
    dataset_name = cfg.dataset.name.split('/')[-1]
    repo_id: str = f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{dataset_name}'

    dataset_path: Path = Path('data') / dataset_name

    model_folders_to_push = collect_unloaded_model_folders(
        dataset_path=dataset_path,
        repo_id=repo_id,
    )

    completed = 0
    retry_counts: dict[Path, int] = {}

    with tqdm(total=len(model_folders_to_push), desc='Pushing models') as pbar:
        while model_folders_to_push:
            model_folder = model_folders_to_push.pop()
            split = model_folder.parent.name
            model_name = model_folder.name

            try:
                push_folder_to_hub(
                    local_folder=model_folder,
                    path_in_repo=f'{split}/{model_name}',
                    repo_id=repo_id,
                    private=cfg.hf.private,
                    commit_message=cfg.hf.commit_message,
                )
                completed += 1
                pbar.update(1)

            except HfHubHTTPError as e:
                if '429' in str(e) or 'Too Many Requests' in str(e):
                    print(
                        f'Rate limit hit while pushing {split}/{model_name}. '
                        f'Sleeping for {RATE_LIMIT_SLEEP // 60} minutes...'
                    )
                    model_folders_to_push.add(model_folder)
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue

                raise

            except (httpx.ReadTimeout, httpcore.ReadTimeout, httpx.TransportError):
                retries = retry_counts.get(model_folder, 0) + 1
                retry_counts[model_folder] = retries

                if retries > MAX_RETRIES_PER_MODEL:
                    print(
                        f'Giving up on {split}/{model_name} after {retries} retries '
                        f'due to repeated network timeouts.'
                    )
                    continue

                sleep_time = min(
                    NETWORK_RETRY_BASE * (2 ** (retries - 1)),
                    NETWORK_RETRY_MAX,
                )
                sleep_time += random.uniform(0, 5)

                print(
                    f'Network timeout while pushing {split}/{model_name} '
                    f'(retry {retries}/{MAX_RETRIES_PER_MODEL}). '
                    f'Sleeping {int(sleep_time)}s...'
                )

                model_folders_to_push.add(model_folder)
                time.sleep(sleep_time)
                continue

            try:
                model_folders_to_push = collect_unloaded_model_folders(
                    dataset_path=dataset_path,
                    repo_id=repo_id,
                )
            except (httpx.ReadTimeout, httpcore.ReadTimeout, httpx.TransportError):
                continue

            pbar.total = completed + len(model_folders_to_push)
            pbar.refresh()

            print(f'Pushed {split}/{model_name}')


if __name__ == '__main__':
    main()
