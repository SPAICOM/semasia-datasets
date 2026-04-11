"""
Utilities for inspecting locally preprocessed datasets.

This module provides helper functions to discover which models have already
been preprocessed and stored in a local dataset export directory. The primary
use case is to support dataset preprocessing pipelines by enabling them to:

- Detect already-processed models per dataset split
- Avoid redundant preprocessing steps
- Validate local preprocessing output structure

Expected dataset layout
-----------------------

All utilities assume the following directory structure::

    dataset_export_root/
        <split>/
            <model_name>/
                part-xxxxx.parquet

Where:
    - ``<split>`` is typically ``train``, ``validation``, or ``test``
    - ``<model_name>`` uniquely identifies a processed model
    - one or more parquet shards may exist per model

Only files with the ``.parquet`` extension are considered. Any files or
directories not matching this structure are ignored.

Typical usage
-------------

    from pathlib import Path
    from src.utils.local import collect_local_models_by_split

    dataset_export_root = Path("/data/datasets/my_dataset")
    models_by_split = collect_local_models_by_split(dataset_export_root)

Design principles
-----------------

- Structure-driven: model and split detection is inferred from the directory
  layout rather than hard-coded conventions.
- Robustness: multiple shards per model are supported transparently.
- Minimal assumptions: split names are not enforced and are discovered
  dynamically.

This module is intentionally lightweight and suitable for use in large-scale
data preprocessing pipelines.
"""

import shutil
from collections import defaultdict
from pathlib import Path


def collect_local_models_by_split(dataset_export_root: Path) -> dict[str, set[str]]:
    """
    Collect the set of preprocessed models available in each dataset split
    from a local dataset export directory.

    The expected directory layout is identical to the Hugging Face dataset
    repository structure::

        dataset_export_root/
            <split>/
                <model_name>/
                    part-xxxxx.parquet

    Examples of valid files::
        train/model_a/part-00000.parquet
        test/model_b/part-00003.parquet

    Any files not ending in `.parquet` or not matching the expected
    directory depth are ignored.

    Parameters
    ----------
    dataset_export_root : pathlib.Path
        Root directory of the local dataset export.

    Returns
    -------
    Dict[str, Set[str]]
        A dictionary mapping each split name to the set of model names
        already preprocessed for that split.

        Example::
            {
                'train': {'model_a', 'model_b'},
                'test': {'model_a'},
                'validation': set(),
            }

    Raises
    ------
    ValueError
        If `dataset_export_root` does not exist or is not a directory.
    """
    if not dataset_export_root.exists() or not dataset_export_root.is_dir():
        raise ValueError(f'Invalid dataset_export_root: {dataset_export_root}')

    models_by_split: dict[str, set[str]] = defaultdict(set)

    # Recursively scan parquet files
    for parquet_file in dataset_export_root.rglob('*.parquet'):
        try:
            # Expect: split/model/file.parquet
            split, model, _ = parquet_file.relative_to(dataset_export_root).parts[:3]
        except ValueError:
            # Path not relative or too shallow
            continue

        models_by_split[split].add(model)

    return dict(models_by_split)


def remove_matching(
    path: Path | str,
    pattern: str = None,
) -> None:
    """
    Remove all files/directories in `path` matching `pattern`.
    If pattern is None, remove everything in `path`.

    Parameters
    ----------
        path: Base directory to clean.
        pattern: Glob pattern (e.g. "model--timm--*").

    Returns
    -------
        None
    """
    base = Path(path).expanduser().resolve()

    if not base.is_dir():
        raise ValueError(f'{base} is not a directory')

    # If no pattern is given, match everything
    pattern = pattern or '*'

    # IMPORTANT: sort deepest paths first so directories are empty when removed
    for item in sorted(base.rglob(pattern), key=lambda p: len(p.parts), reverse=True):
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except FileNotFoundError:
            # Already removed as part of a parent directory
            pass

    return None


if __name__ == '__main__':
    pass
