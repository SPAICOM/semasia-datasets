""""""

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

from src.huggingface import (
    collect_unloaded_model_folders,
    push_folder_to_hub,
)

RATE_LIMIT_SLEEP = 3600 + 60  # 1h + 60s safety delta
NETWORK_RETRY_SLEEP_BASE = 30  # seconds
NETWORK_RETRY_SLEEP_MAX = 10 * 60  # 10 minutes
MAX_RETRIES_PER_MODEL = 5


@hydra.main(
    config_path='../configs/hydra/',
    config_name='push_to_hf',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """"""
    CURRENT: Path = Path('.')
    DATA_PATH: Path = CURRENT / 'data'
    DATASET_PATH: Path = DATA_PATH / cfg.dataset.name

    # Variables
    repo_id: str = f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{cfg.dataset.name}'

    model_folders_to_push = collect_unloaded_model_folders(
        dataset_path=DATASET_PATH,
        repo_id=repo_id,
    )

    i = 0
    retry_counts: dict[Path, int] = {}
    with tqdm(total=len(model_folders_to_push), desc='Pushing models') as pbar:
        while model_folders_to_push:
            model_folder = model_folders_to_push.pop()
            split = model_folder.parent.name
            model_name = model_folder.name

            try:
                # Push the entire model folder
                push_folder_to_hub(
                    local_folder=model_folder,
                    path_in_repo=f'{split}/{model_name}',
                    repo_id=repo_id,
                    private=cfg.hf.private,
                    commit_message=cfg.hf.commit_message,
                )

                # One unit of work completed
                i += 1
                pbar.update(1)

            except HfHubHTTPError as e:
                # Detect rate limit
                if '429' in str(e) or 'Too Many Requests' in str(e):
                    print(
                        f'Rate limit hit while pushing {split}/{model_name}. '
                        f'Sleeping for {RATE_LIMIT_SLEEP // 60} minutes...'
                    )

                    # Re-add the model so it will be retried
                    model_folders_to_push.add(model_folder)

                    time.sleep(RATE_LIMIT_SLEEP)
                    continue

                # Any other HF error should be raised
                raise
            except (httpx.ReadTimeout, httpcore.ReadTimeout, httpx.TransportError):
                # ---- Transient network errors ----
                retries = retry_counts.get(model_folder, 0) + 1
                retry_counts[model_folder] = retries

                if retries > MAX_RETRIES_PER_MODEL:
                    print(
                        f'❌ Giving up on {split}/{model_name} after {retries} retries '
                        f'due to repeated network timeouts.'
                    )
                    continue  # skip permanently

                sleep_time = min(
                    NETWORK_RETRY_SLEEP_BASE * (2 ** (retries - 1)),
                    NETWORK_RETRY_SLEEP_MAX,
                )

                # Add small jitter to avoid sync retries
                sleep_time += random.uniform(0, 5)

                print(
                    f'⚠️ Network timeout while pushing {split}/{model_name} '
                    f'(retry {retries}/{MAX_RETRIES_PER_MODEL}). '
                    f'Sleeping {int(sleep_time)}s before retry...'
                )

                model_folders_to_push.add(model_folder)
                time.sleep(sleep_time)
                continue

            # Recompute unloaded files AFTER upload
            model_folders_to_push = collect_unloaded_model_folders(
                dataset_path=DATASET_PATH,
                repo_id=repo_id,
            )

            # Enforce invariant: total = completed + remaining
            pbar.total = i + len(model_folders_to_push)
            pbar.refresh()

            print(f'Pushed {split}/{model_name}')

    return None


if __name__ == '__main__':
    main()
