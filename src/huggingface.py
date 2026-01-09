"""
Utilities for uploading local datasets to the Hugging Face Hub
with strict correctness and incremental updates.
"""

import re
import textwrap
from collections import defaultdict
from pathlib import Path, PurePosixPath

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import get_token


def collect_models_by_split(
    repo_id: str,
) -> dict[str, set[str]]:
    """
    Collect the set of processed models available in each dataset split
    of a Hugging Face dataset repository.

    The expected repository layout is::

        <split>/<model_name>/part-xxxxx.parquet

    where:
        - <split> is typically "train", "validation", or "test"
        - <model_name> is the identifier of the processed model
        - multiple parquet parts may exist per model

    Any non-parquet files (e.g. `.gitattributes`) or files not following
    the expected directory depth are ignored.

    Parameters
    ----------
    repo_id : str
        The dataset repository ID (e.g. "org_name/dataset_name").

    Returns
    -------
    Dict[str, Set[str]]
        A dictionary mapping each split name to the set of model names
        that have already been uploaded for that split.

        Example::
            {
                'train': {'model_a', 'model_b'},
                'test': {'model_a'},
                'validation': set(),
            }
    """
    token = get_token()
    if token is None:
        print('[WARN] No HF token found. Run `huggingface-cli login` or set HF_TOKEN.')
        return

    api = HfApi(token=token)

    files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')

    models_by_split = defaultdict(set)

    for f in files:
        path = PurePosixPath(f)

        # Expect: split/model/part-xxxxx.parquet
        if len(path.parts) < 3:
            continue

        split, model = path.parts[0], path.parts[1]

        # Optional: only count parquet files
        if path.suffix != '.parquet':
            continue

        models_by_split[split].add(model)

    return dict(models_by_split)


def push_folder_to_hub(
    local_folder: Path | str,
    path_in_repo: Path | str,
    repo_id: str,
    private: bool = False,
    commit_message: str = 'Incremental update',
) -> None:
    """
    Create (if needed) and incrementally synchronize a local folder
    with a Hugging Face dataset repository.

    Parameters
    ----------
    local_folder : pathlib.Path or str
        Path to the local folder to synchronize.
    path_in_repo: pathlib.Path or str
        Hugging Face folder where to upload.
    repo_id : str
        Full Hugging Face repository ID in the form
        ``"<namespace>/<dataset_name>"``.
    private : bool, optional
        Whether the dataset repository should be private.
        Defaults to ``False``.
    commit_message : str, optional
        Commit message used when changes are detected.
        Defaults to ``"Incremental update"``.

    Returns
    -------
    None
        This function is called for its side effects only.
    """
    token = get_token()
    if token is None:
        print('[WARN] No HF token found. Run `huggingface-cli login` or set HF_TOKEN.')
        return

    local_folder = Path(local_folder)
    api = HfApi(token=token)

    # Upload changed / new files
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_folder,
        path_in_repo=path_in_repo,
        repo_type='dataset',
        commit_message=commit_message,
    )

    print(f'[OK] Synced dataset https://huggingface.co/datasets/{repo_id} ')

    return None


def collect_unloaded_parquet_files(
    dataset_path: Path,
    repo_id: str,
) -> set[Path]:
    """
    Collect local parquet files that have not yet been uploaded to Hugging Face.

    A parquet file is considered already uploaded if its corresponding
    (split, model) pair exists in the Hugging Face repository.

    The expected directory structure is::

        dataset_path/
            split_name/
                model_name/
                    *.parquet

    Parameters
    ----------
    dataset_path : pathlib.Path
        Root path of the local dataset.
    repo_id : str
        Hugging Face repository ID used to check which models are already loaded.

    Returns
    -------
    set[pathlib.Path]
        Set of local parquet file paths that are not yet present on
        Hugging Face.
    """
    # Get models already loaded on Hugging Face, grouped by split
    already_loaded_models: dict[str, set[str]] = collect_models_by_split(
        repo_id=repo_id
    )

    # Collect all local parquet files
    local_files: set[Path] = set(dataset_path.rglob('*.parquet'))

    unloaded_files: set[Path] = set()

    for path in local_files:
        # Expected structure: .../<split>/<model>/<file>.parquet
        split_name = path.parent.parent.name
        model_name = path.parent.name

        loaded_models_for_split = already_loaded_models.get(split_name, set())

        if model_name not in loaded_models_for_split:
            unloaded_files.add(path)

    return unloaded_files


def collect_unloaded_model_folders(
    dataset_path: Path,
    repo_id: str,
) -> set[Path]:
    """
    Collect local model folders that have not yet been uploaded to Hugging Face.

    A model folder is considered already uploaded if its corresponding
    (split, model) pair exists in the Hugging Face repository.

    The expected directory structure is::

        dataset_path/
            split_name/
                model_name/
                    *.parquet

    Parameters
    ----------
    dataset_path : pathlib.Path
        Root path of the local dataset.
    repo_id : str
        Hugging Face repository ID used to check which models are already loaded.

    Returns
    -------
    set[pathlib.Path]
        Set of local model folder paths that are not yet present on
        Hugging Face.
    """
    # Get models already loaded on Hugging Face, grouped by split
    already_loaded_models: dict[str, set[str]] = collect_models_by_split(
        repo_id=repo_id
    )

    # Collect all local model folders containing parquet files
    model_folders: set[Path] = {
        parquet_path.parent for parquet_path in dataset_path.rglob('*.parquet')
    }

    unloaded_model_folders: set[Path] = set()

    for model_folder in model_folders:
        # Expected structure: .../<split>/<model>
        split_name = model_folder.parent.name
        model_name = model_folder.name

        loaded_models_for_split = already_loaded_models.get(split_name, set())

        if model_name not in loaded_models_for_split:
            unloaded_model_folders.add(model_folder)

    return unloaded_model_folders


def _split_front_matter(readme_text: str) -> tuple[dict, str]:
    """Return (yaml_dict, body_markdown).
    If no front matter, returns ({}, original_text).
    """
    _FRONT_MATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)
    m = _FRONT_MATTER_RE.match(readme_text)
    if not m:
        return {}, readme_text

    yaml_block = m.group(1)
    body = readme_text[m.end() :]
    try:
        data = yaml.safe_load(yaml_block) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return data, body


def _discover_models_and_splits_from_hub(
    api: HfApi,
    repo_id: str,
    *,
    allowed_splits: tuple[str, ...] = ('train', 'test', 'validation', 'val'),
) -> dict[str, set[str]]:
    """
    Scan repo files and return models_by_split:
      { split_name: {model1, model2, ...}, ... }
    Only counts models that have at least one *.parquet file.
    """
    files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')

    models_by_split: dict[str, set[str]] = {s: set() for s in allowed_splits}

    for f in files:
        if not f.endswith('.parquet'):
            continue
        parts = f.split('/')
        # Expect: split/model/part-xxxxx.parquet
        if len(parts) < 3:
            continue
        split, model = parts[0], parts[1]
        if split not in models_by_split:
            continue
        models_by_split[split].add(model)

    # Drop empty splits for cleanliness
    models_by_split = {k: v for k, v in models_by_split.items() if v}
    return models_by_split


def _build_config_entry(model: str, present_splits: set[str]) -> dict:
    """Build a HF dataset config entry for a model, only for splits that exist online."""
    data_files = []
    if 'train' in present_splits:
        data_files.append({'split': 'train', 'path': f'train/{model}/*.parquet'})
    if 'validation' in present_splits:
        data_files.append(
            {'split': 'validation', 'path': f'validation/{model}/*.parquet'}
        )
    if 'val' in present_splits:
        # Some repos use val/ instead of validation/
        data_files.append({'split': 'validation', 'path': f'val/{model}/*.parquet'})
    if 'test' in present_splits:
        data_files.append({'split': 'test', 'path': f'test/{model}/*.parquet'})

    return {'config_name': model, 'data_files': data_files}


def generate_readme_with_configs(
    repo_dir: Path,
    dataset_name: str,
    *,
    repo_id: str,
    pretty_name: str | None = None,
    extra_markdown: str = '',
    preserve_existing_body: bool = True,
    push_online: bool = True,
    commit_message: str = 'Update README configs (online models only)',
) -> None:
    """
    Update README.md with YAML front-matter configs, but first:
      - fetch models present in the online repo
      - fetch and parse the online README.md (if exists)
      - merge new configs for models missing from the online README

    The resulting README is written to repo_dir/README.md.
    If push_online=True and repo_id is provided, uploads ONLY README.md.
    """
    token = get_token()
    if token is None:
        raise RuntimeError(
            'No HF token found. Run `huggingface-cli login` or set HF_TOKEN.'
        )

    api = HfApi(token=token)

    repo_dir = Path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    # 1) Discover what exists online
    models_by_split = _discover_models_and_splits_from_hub(api, repo_id)
    if not models_by_split:
        raise RuntimeError(
            f'No parquet files found in {repo_id}. '
            'Cannot build configs if nothing is uploaded yet.'
        )

    # Build model -> present_splits map
    model_to_splits: dict[str, set[str]] = {}
    for split, models in models_by_split.items():
        for m in models:
            model_to_splits.setdefault(m, set()).add(split)

    online_models = sorted(model_to_splits.keys())

    # 2) Fetch existing README to optionally preserve body
    existing_body = None
    if preserve_existing_body:
        try:
            readme_path = hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                filename='README.md',
                token=token,
            )
            existing_text = Path(readme_path).read_text(encoding='utf-8')
            _, existing_body = _split_front_matter(existing_text)
            if existing_body is not None and not existing_body.strip():
                existing_body = None
        except Exception:
            existing_body = None

    # 3) Build configs ONLY for online models
    config_entries = [
        _build_config_entry(model=m, present_splits=model_to_splits[m])
        for m in online_models
    ]

    yaml_top = {
        'pretty_name': pretty_name or f'Latents for {dataset_name} (timm)',
        'configs': config_entries,
    }
    yaml_text = '---\n' + yaml.safe_dump(yaml_top, sort_keys=False) + '---\n'

    # 4) Build markdown body (preserve or default)
    if existing_body is not None:
        md_body = existing_body
    else:
        example_model = online_models[0] if online_models else 'resnet50'
        md_body = textwrap.dedent(f"""
        # {pretty_name or f'Latents for {dataset_name} (timm)'}

        This repository hosts **precomputed embeddings** for `{dataset_name}` across many `timm` models.
        Each dataset **config** corresponds to a single model; only that model’s Parquet files are read on `load_dataset`.

        ## Usage

        ```python
        from datasets import load_dataset

        ds_train = load_dataset("{repo_id}", "{example_model}", split="train")
        ds_test  = load_dataset("{repo_id}", "{example_model}", split="test")
        ```

        ## Notes
        - Configs are generated from what is actually uploaded on the Hub (parquet presence).
        {extra_markdown}
        """)

    # 5) Write local README
    out_path = repo_dir / 'README.md'
    out_path.write_text(yaml_text + md_body, encoding='utf-8')
    print(
        f'[OK] Wrote {out_path} with {len(config_entries)}'
        ' configs (online models only).'
    )

    # 6) Push only README
    if push_online:
        api.upload_file(
            repo_id=repo_id,
            repo_type='dataset',
            path_or_fileobj=str(out_path),
            path_in_repo='README.md',
            commit_message=commit_message,
        )
        print(f'[OK] Updated README online: https://huggingface.co/datasets/{repo_id}')


if __name__ == '__main__':
    pass
