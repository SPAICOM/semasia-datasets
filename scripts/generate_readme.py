"""
Generate or update README.md for a dataset repository on Hugging Face.

This script fetches the models present in the online repository and generates
YAML front-matter configs along with a dynamic usage section that reflects
the available dataset splits.

Usage:
    uv run scripts/generate_readme.py dataset=cifar10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig

from src import generate_readme_with_configs


@hydra.main(
    config_path='../configs/hydra/',
    config_name='generate_readme',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """
    Generate README.md for the specified dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing dataset, HuggingFace, and other settings.
    """
    current: Path = Path('.').resolve()
    exports_root: Path = current / 'data'
    exports_root.mkdir(exist_ok=True, parents=True)

    dataset_name = cfg.dataset.name.split('/')[-1]
    repo_id: str = f'{cfg.hf.namespace}/{cfg.hf.repo_prefix}{dataset_name}'
    dataset_dir = exports_root / dataset_name

    generate_readme_with_configs(
        repo_dir=dataset_dir,
        dataset_name=dataset_name,
        repo_id=repo_id,
        original_dataset_id=cfg.dataset.name,
        push_online=True,
    )


if __name__ == '__main__':
    main()
