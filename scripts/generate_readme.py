# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hydra-core",
#     "datasets",
# ]
# ///
""""""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))


import hydra
from omegaconf import DictConfig

from src.huggingface import (
    generate_readme_with_configs,
)


@hydra.main(
    config_path='../configs/hydra/',
    config_name='config',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    CURRENT: Path = Path('.').resolve()
    EXPORTS_ROOT: Path = CURRENT / 'data'  # Where Parquet trees will be written
    EXPORTS_ROOT.mkdir(exist_ok=True, parents=True)
    repo_id: str = f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{cfg.dataset.name}'
    dataset_dir = EXPORTS_ROOT / cfg.dataset.name

    generate_readme_with_configs(
        repo_dir=dataset_dir,
        dataset_name=cfg.dataset.name,
        repo_id=repo_id,
        push_online=True,
    )

    return None


if __name__ == '__main__':
    main()
