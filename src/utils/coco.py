"""
Build the COCO 2017 benchmark by joining two datasets of the Hugging Face Hub.

No single Hub dataset has the images, the object annotations and the captions of
COCO 2017, so the benchmark is built by joining two of them on the official COCO
``image_id``:

- the *images* source (``rafaelpadilla/coco2017``) has ``image``, ``image_id`` and
  ``objects`` (``id``, ``area``, ``bbox``, ``label`` and ``iscrowd`` of every object);
- the *captions* source (``phiyodr/coco2017``) has ``image_id``, ``captions``, ``ids``
  (the ids of the captions, NOT of the objects), ``license``, ``coco_url`` and
  ``flickr_url``.
"""

from collections.abc import Mapping
from typing import Any

from datasets import Dataset, load_dataset

# Columns of the joined dataset, in order; `image` is the one to encode and `id` is
# the COCO `image_id`.
COCO_COLUMNS: tuple[str, ...] = (
    'image',
    'id',
    'objects',
    'captions',
    'captions_ids',
    'license',
    'coco_url',
    'flickr_url',
)


def _load_source(
    source: Mapping[str, Any],
    split: str,
    cache_dir: str | None,
) -> Dataset:
    """
    Load one split of one source dataset.

    Parameters
    ----------
    source : Mapping[str, Any]
        Source entry of the dataset config: ``name``, ``revision``, ``data_files``
        (a pattern with a ``{split}`` placeholder) and ``split`` (mapping from the
        split name of SEMASIA to the split name used by the source).
    split : str
        Split name of SEMASIA.
    cache_dir : str, optional
        Cache folder of the Hugging Face datasets.

    Returns
    -------
    datasets.Dataset
        The requested split of the source dataset.
    """
    source_split = source['split'][split]
    return load_dataset(
        source['name'],
        revision=source.get('revision'),
        data_files={split: source['data_files'].format(split=source_split)},
        split=split,
        cache_dir=cache_dir,
        verification_mode='no_checks',
    )


def join_coco2017(images: Dataset, captions: Dataset) -> Dataset:
    """
    Join the images dataset and the captions dataset of COCO 2017 on ``image_id``.

    The rows follow the order of ``images``, so that the images are not touched and
    stay lazily decoded. The join is by ``image_id`` and never by position (the two
    sources have different row orders), and it is an exact one: both sides must have
    the same unique ``image_id``.

    Parameters
    ----------
    images : datasets.Dataset
        Dataset with the columns ``image``, ``image_id`` and ``objects``.
    captions : datasets.Dataset
        Dataset with the columns ``image_id``, ``captions``, ``ids``, ``license``,
        ``coco_url`` and ``flickr_url``.

    Returns
    -------
    datasets.Dataset
        Dataset with the columns of :data:`COCO_COLUMNS`. ``id`` is the COCO
        ``image_id`` and ``captions_ids`` are the ids of the captions (``ids`` of the
        captions source).

    Raises
    ------
    ValueError
        If ``image_id`` is not unique in a source, or if the two sources do not have
        the same images.
    """
    image_ids: list[int] = images['image_id']
    caption_ids: list[int] = captions['image_id']

    if len(set(image_ids)) != len(image_ids):
        raise ValueError('Duplicated image_id in the images source.')
    if len(set(caption_ids)) != len(caption_ids):
        raise ValueError('Duplicated image_id in the captions source.')
    if set(image_ids) != set(caption_ids):
        only_images = len(set(image_ids) - set(caption_ids))
        only_captions = len(set(caption_ids) - set(image_ids))
        raise ValueError(
            'The sources do not have the same images: '
            f'{only_images} only in the images source, '
            f'{only_captions} only in the captions source.'
        )

    # Rows of the captions source in the order of the images source.
    position = {image_id: i for i, image_id in enumerate(caption_ids)}
    aligned = captions.select([position[image_id] for image_id in image_ids])

    joined = images.select_columns(['image', 'image_id', 'objects'])
    joined = joined.rename_column('image_id', 'id')
    for column, source_column in (
        ('captions', 'captions'),
        ('captions_ids', 'ids'),
        ('license', 'license'),
        ('coco_url', 'coco_url'),
        ('flickr_url', 'flickr_url'),
    ):
        joined = joined.add_column(column, aligned[source_column])

    return joined.select_columns(list(COCO_COLUMNS))


def load_coco2017(
    sources: Mapping[str, Mapping[str, Any]],
    splits: list[str],
    cache_dir: str | None = None,
) -> dict[str, Dataset]:
    """
    Load COCO 2017 as the join of its two sources.

    Parameters
    ----------
    sources : Mapping[str, Mapping[str, Any]]
        The ``sources`` entry of the dataset config (``configs/hydra/dataset/
        coco2017.yaml``), with the keys ``images`` and ``captions``.
    splits : list[str]
        Splits of SEMASIA to load, for example ``['train', 'validation']``.
    cache_dir : str, optional
        Cache folder of the Hugging Face datasets. By default, the standard one.

    Returns
    -------
    dict[str, datasets.Dataset]
        Joined dataset of every split (see :func:`join_coco2017`).
    """
    return {
        split: join_coco2017(
            images=_load_source(sources['images'], split, cache_dir),
            captions=_load_source(sources['captions'], split, cache_dir),
        )
        for split in splits
    }
