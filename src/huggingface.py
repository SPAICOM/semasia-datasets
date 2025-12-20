"""
Utilities for uploading local datasets to the Hugging Face Hub
with strict correctness and incremental updates.
"""

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token


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
