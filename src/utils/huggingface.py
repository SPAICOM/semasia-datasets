"""
Utilities for uploading local datasets to the Hugging Face Hub
with strict correctness and incremental updates.
"""

import textwrap
from collections import defaultdict
from pathlib import Path, PurePosixPath

import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token


def collect_models_by_split(
    repo_id: str,
) -> dict[str, set[str]] | None:
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
    dict[str, set[str]] | None
        A dictionary mapping each split name to the set of model names
        that have parquet files in that split, or ``None`` if no token
        is available.
    """
    token = get_token()
    if token is None:
        print('[WARN] No HF token found. Run `huggingface-cli login` or set HF_TOKEN.')
        return None

    api = HfApi(token=token)

    files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')

    models_by_split = defaultdict(set)

    for f in files:
        path = PurePosixPath(f)

        if len(path.parts) < 3:
            continue

        split, model = path.parts[0], path.parts[1]

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


def _build_config_entry(model: str, present_splits: set[str]) -> dict:
    """
    Build a HuggingFace dataset config entry for a model.

    Parameters
    ----------
    model : str
        Name of the model.
    present_splits : set[str]
        Set of split names available for this model.

    Returns
    -------
    dict
        Config entry with config_name and data_files.
    """
    data_files = [
        {'split': split, 'path': f'{split}/{model}/*.parquet'}
        for split in ('train', 'validation', 'valid', 'val', 'test')
        if split in present_splits
    ]

    return {'config_name': model, 'data_files': data_files}


def generate_readme_with_configs(
    repo_dir: Path,
    dataset_name: str,
    *,
    repo_id: str,
    pretty_name: str | None = None,
    extra_markdown: str = '',
    original_dataset_id: str | None = None,
    push_online: bool = True,
    commit_message: str = 'Update README configs (online models only)',
) -> None:
    """
    Update README.md with YAML front-matter configs and a dynamic usage section.

    Parameters
    ----------
    repo_dir : Path
        Local directory where the README.md will be written.
    dataset_name : str
        Name of the dataset (e.g., "cifar10").
    repo_id : str
        Full Hugging Face repository ID (e.g., "org_name/semantic-cifar10").
    pretty_name : str, optional
        Human-readable name for the dataset. If None, a default is generated.
    extra_markdown : str, optional
        Additional markdown content to append to the Notes section.
    original_dataset_id : str, optional
        Original dataset ID to link to (e.g., "uoft-cs/cifar10").
    push_online : bool, optional
        Whether to upload the README to Hugging Face. Defaults to True.
    commit_message : str, optional
        Commit message for the upload. Defaults to a standard message.

    Returns
    -------
    None
        This function is called for its side effects only.
    """
    token = get_token()
    if token is None:
        raise RuntimeError(
            'No HF token found. Run `huggingface-cli login` or set HF_TOKEN.'
        )

    api = HfApi(token=token)

    repo_dir = Path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    models_by_split = collect_models_by_split(repo_id)
    if not models_by_split:
        raise RuntimeError(
            f'No parquet files found in {repo_id}. '
            'Cannot build configs if nothing is uploaded yet.'
        )

    model_to_splits: dict[str, set[str]] = {}
    for split, models in models_by_split.items():
        for m in models:
            model_to_splits.setdefault(m, set()).add(split)

    online_models = sorted(model_to_splits.keys())

    config_entries = [
        _build_config_entry(model=m, present_splits=model_to_splits[m])
        for m in online_models
    ]

    yaml_top = {
        'pretty_name': pretty_name or f'Latents for {dataset_name} (timm)',
        'configs': config_entries,
    }
    yaml_text = '---\n' + yaml.safe_dump(yaml_top, sort_keys=False) + '---\n'

    example_model = online_models[0] if online_models else 'resnet50'
    available_splits = sorted(models_by_split.keys())
    usage_lines = [
        f'    ds_{split} = load_dataset("{repo_id}", '
        f'"{example_model}", split="{split}")'
        for split in available_splits
    ]
    usage_code = '\n'.join(usage_lines)

    original_dataset_link = (
        f'- Based on [{original_dataset_id}]'
        f'(https://huggingface.co/datasets/{original_dataset_id})'
        if original_dataset_id
        else ''
    )

    dataset_desc = f'**precomputed embeddings** for `{dataset_name}` '
    dataset_desc += 'across many `timm` models.'
    md_body = textwrap.dedent(f"""
    # {pretty_name or f'Latents for {dataset_name} (timm)'}

    This repository hosts {dataset_desc}
    Each **config** corresponds to a single model;
    only that model's Parquet files are read on `load_dataset`.

    ## Usage

    ```python
    from datasets import load_dataset

{usage_code}
    ```

    ## Notes
    - Configs are generated from what is actually
      uploaded on the Hub (parquet presence).
    {original_dataset_link}
    {extra_markdown}
    """)

    out_path = repo_dir / 'README.md'
    out_path.write_text(yaml_text + md_body, encoding='utf-8')
    print(
        f'[OK] Wrote {out_path} with {len(config_entries)}'
        ' configs (online models only).'
    )

    if push_online:
        api.upload_file(
            repo_id=repo_id,
            repo_type='dataset',
            path_or_fileobj=str(out_path),
            path_in_repo='README.md',
            commit_message=commit_message,
        )
        print(f'[OK] Updated README online: https://huggingface.co/datasets/{repo_id}')


def generate_model_registry_readme(
    repo_dir: Path,
    repo_id: str,
    *,
    num_models: int,
    push_online: bool = True,
    commit_message: str = 'Update README',
) -> None:
    """
    Generate README.md for the model-registry dataset on HuggingFace.

    Parameters
    ----------
    repo_dir : Path
        Local directory where the README.md will be written.
    repo_id : str
        Full Hugging Face repository ID (e.g., "spaicom-lab/model-registry").
    num_models : int
        Number of models in the registry.
    push_online : bool, optional
        Whether to upload the README to HuggingFace. Defaults to True.
    commit_message : str, optional
        Commit message for the upload. Defaults to "Update README".

    Returns
    -------
    None
        This function is called for its side effects only.
    """
    token = get_token()
    if token is None:
        raise RuntimeError(
            'No HF token found. Run `huggingface-cli login` or set HF_TOKEN.'
        )

    api = HfApi(token=token)

    repo_dir = Path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    yaml_top = {
        'pretty_name': 'Timm Model Metadata Registry',
        'num_models': num_models,
    }
    yaml_text = '---\n' + yaml.safe_dump(yaml_top, sort_keys=False) + '---\n'

    md_body = textwrap.dedent(f"""
    # Timm Model Metadata Registry

    This dataset contains a lookup table of metadata for all pretrained
    models available in the timm library.

    ## Usage

    ```python
    from datasets import load_dataset
    import polars as pl

    ds = load_dataset("spaicom-lab/model-registry")
    df = ds["train"].to_polars()

    # Query by model name
    model = df.filter(pl.col("model_name") == "vit_base_patch16_224")

    # Filter by family
    vit_models = df.filter(pl.col("family") == "ViT")

    # Filter by parameter count
    small_models = df.filter(pl.col("num_parameters") < 50_000_000)
    ```

    ## Fields

    | Field | Description |
    |-------|-------------|
    | model_name | Full timm model identifier |
    | family | Architecture family (e.g., ViT, ConvNeXt, ResNet) |
    | model_version | Architecture generation (e.g., v2, v3) |
    | size | Human-readable size label (e.g., Base, Large) |
    | num_parameters | Total trainable parameters |
    | latent_dim | Output embedding dimension |
    | patch_size | ViT patch size in pixels (if applicable) |
    | input_resolution | Native input resolution |
    | pretrain_org | Organisation responsible for training |
    | pretrain_dataset | Pretraining dataset |
    | pretrain_method | Training objective (e.g., CLIP, MAE, DINO) |
    | pretrain_ft | Fine-tuning dataset (if applicable) |

    ## Notes

    - Metadata is extracted using the `get_model_metadata` function from the
      [semantic-datasets](https://github.com/spaicom-lab/semantic-datasets) library.
    - Some fields may be ``None`` if they cannot be inferred from the model name.
    - Total number of models: **{num_models}**
    """)

    out_path = repo_dir / 'README.md'
    out_path.write_text(yaml_text + md_body, encoding='utf-8')
    print(f'[OK] Wrote {out_path}')

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
