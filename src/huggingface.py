"""
Utilities for uploading local datasets to the Hugging Face Hub
with strict correctness and incremental updates.
"""

from collections import defaultdict
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi
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


if __name__ == '__main__':
    pass
