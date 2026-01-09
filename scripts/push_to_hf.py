""""""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.huggingface import (
    collect_unloaded_model_folders,
    push_folder_to_hub,
)


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
