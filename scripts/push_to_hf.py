""""""

import sys
import time
from pathlib import Path

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
            except HfHubHTTPError as e:
                # Detect rate limit
                if '429' in str(e) or 'Too Many Requests' in str(e):
                    print(
                        f'Rate limit hit while pushing {split}/{model_name}. '
                        f'Sleeping for {RATE_LIMIT_SLEEP // 60} minutes...'
                    )

                    # Re-add the model so it will be retried
                    model_folders_to_push.append(model_folder)

                    time.sleep(RATE_LIMIT_SLEEP)
                    continue

                # Any other HF error should be raised
                raise

            # One unit of work completed
            i += 1
            pbar.update(1)

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
