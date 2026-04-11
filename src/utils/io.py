"""
Utilities for exporting latent representations to disk.
"""

from pathlib import Path

import polars as pl


def latents_to_parquet_shards(
    df: pl.DataFrame,
    export_path: Path,
    max_rows_per_shard: int,
) -> None:
    """
    Write a Polars DataFrame to multiple Parquet shard files.

    The input DataFrame is split into row-wise shards of at most
    `max_rows_per_shard` rows. Each shard is written as a Parquet file
    named `part-XXXXX.parquet` inside the specified export directory.

    The export directory is created if it does not already exist.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing latent vectors or related data
        to be exported.
    export_path : pathlib.Path
        Directory where the Parquet shard files will be written.
    max_rows_per_shard : int
        Maximum number of rows per Parquet shard.

    Returns
    -------
    None
        This function is called for its side effects only (writing files
        to disk).
    """
    export_path.mkdir(parents=True, exist_ok=True)

    for i, shard in enumerate(df.iter_slices(max_rows_per_shard)):
        shard.write_parquet(export_path / f'part-{i:05d}.parquet')
    return None


if __name__ == '__main__':
    pass
