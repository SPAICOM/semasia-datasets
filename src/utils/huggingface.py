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
    code_repo_url: str | None = None,
    model_registry_repo_id: str | None = None,
    extra_fields: list[str] | None = None,
    badges: list[str] | None = None,
    task_categories: list[str] | None = None,
    tags: list[str] | None = None,
    citation_bibtex: str | None = None,
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
    code_repo_url : str, optional
        URL of the code repository used to generate this dataset. If None,
        the link is omitted.
    model_registry_repo_id : str, optional
        Hugging Face repo ID of the timm model metadata registry
        (e.g., "org_name/semantic-model-registry"). If None, the link
        is omitted.
    extra_fields : list[str], optional
        Names of the non-embedding columns stored alongside `embedding`
        in each Parquet file (e.g., ["label"] or ["fine_label",
        "coarse_label"]), matching this dataset's `extras` config.
        If None, a "Fields" section is not rendered and the usage
        examples fall back to a generic "label" column.
    badges : list[str], optional
        Markdown badge snippets (e.g., shields.io badges) to render
        under the title. If None, no badges are rendered.
    task_categories : list[str], optional
        Hugging Face task category tags (e.g., ["feature-extraction"]).
        If None, the field is omitted from the YAML front-matter.
    tags : list[str], optional
        Free-form Hugging Face discoverability tags (e.g.,
        ["semantic-communication", "explainability"]). If None, the
        field is omitted from the YAML front-matter.
    citation_bibtex : str, optional
        BibTeX citation to include in a "Citation" section. If None,
        the section is omitted.
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
    }
    if task_categories:
        yaml_top['task_categories'] = list(task_categories)
    if tags:
        yaml_top['tags'] = list(tags)
    yaml_top['configs'] = config_entries
    yaml_text = '---\n' + yaml.safe_dump(yaml_top, sort_keys=False) + '---\n'

    example_model = online_models[0]
    available_splits = sorted(models_by_split.keys())
    example_split = (
        'test'
        if 'test' in available_splits
        else available_splits[0]
        if available_splits
        else 'train'
    )

    example_field = extra_fields[0] if extra_fields else 'label'

    torch_usage = '\n'.join(
        [
            '    from datasets import load_dataset',
            '    import torch',
            '',
            '    ds = load_dataset(',
            f'        "{repo_id}",  # repository  →  which benchmark',
            f'        "{example_model}",  # config      →  which model',
            f'        split="{example_split}",  # split       →  which partition',
            '    ).with_format("torch")',
            '',
            '    embeddings = torch.vstack(list(ds["embedding"]))  # (N, d)',
            f'    {example_field} = torch.tensor(ds["{example_field}"])  # (N,)',
        ]
    )

    polars_usage = '\n'.join(
        [
            '    import polars as pl',
            '',
            '    df = pl.read_parquet(',
            f'        "hf://datasets/{repo_id}/{example_split}/'
            f'{example_model}/*.parquet"',
            '    )',
            '',
            '    embeddings = df["embedding"].to_numpy()  # shape (N, d)',
            f'    {example_field} = df["{example_field}"].to_numpy()  # shape (N,)',
        ]
    )

    embedding_dim_note = (
        f'(dimensionality depends on config; see [model registry]'
        f'(https://huggingface.co/datasets/{model_registry_repo_id})).'
        if model_registry_repo_id
        else '(dimensionality depends on config).'
    )
    id_note = (
        f'row order matches the original [{original_dataset_id}]'
        f'(https://huggingface.co/datasets/{original_dataset_id}) split, '
        'so `id` can be used to map a row back to its source sample.'
        if original_dataset_id
        else 'row order matches the corresponding split of the '
        'source dataset, so `id` can be used to map a row back to '
        'its source sample.'
    )
    fields_section = '\n'.join(
        [
            '    ## Fields',
            '',
            '    Columns available in each Parquet file for this dataset:',
            '',
            '    | Field | Description |',
            '    |-------|-------------|',
            '    | `id` | Row index within the shard '
            '(unique per split/model, not across models); the '
            f'{id_note} |',
            '    | `model_name` | `timm` model that produced this '
            "row's embedding (constant within a config). |",
            '    | `embedding` | Precomputed latent representation '
            f'extracted by the model {embedding_dim_note} |',
        ]
        + [
            f'    | `{field}` | Original dataset field, copied '
            'as-is from the source dataset. |'
            for field in (extra_fields or [])
        ]
    )

    model_counts_table = '\n'.join(
        [
            '    | Split | # Models |',
            '    |-------|----------|',
        ]
        + [
            f'    | {split} | {len(models_by_split[split])} |'
            for split in available_splits
        ]
    )

    original_dataset_link = (
        f'- Based on [{original_dataset_id}]'
        f'(https://huggingface.co/datasets/{original_dataset_id})'
        if original_dataset_id
        else ''
    )
    code_repo_link = (
        f'- Code: [{code_repo_url.split("://")[-1]}]({code_repo_url})'
        if code_repo_url
        else ''
    )
    model_registry_link = (
        '- Model metadata (architecture family, parameter count, '
        'embedding dimension, pretraining details, ...) for every model '
        f'in this dataset is available in the [model registry]'
        f'(https://huggingface.co/datasets/{model_registry_repo_id}).'
        if model_registry_repo_id
        else ''
    )

    citation_section = ''
    if citation_bibtex:
        bibtex_lines = [
            '    ' + line for line in citation_bibtex.strip('\n').splitlines()
        ]
        citation_section = '\n'.join(
            [
                '    ## Citation',
                '',
                '    If you use this dataset, please cite:',
                '',
                '    ```bibtex',
            ]
            + bibtex_lines
            + ['    ```']
        )

    badges_block = ''
    if badges:
        badges_block = '\n'.join(
            [
                '    <h5 align="center">',
                '',
                '    ' + '&nbsp;'.join(badges),
                '',
                '     <br>',
                '',
                '    </h5>',
            ]
        )

    dataset_desc = (
        f'**precomputed latent representations (embeddings)** extracted from '
        f'`timm` image-classification backbones on `{dataset_name}`, '
        'released as part of SEMASIA — a large-scale resource for studying '
        'semantic communication, cross-model latent space alignment, '
        'and explainability.'
    )
    md_body = textwrap.dedent(f"""
    # {pretty_name or f'Latents for {dataset_name} (timm)'}

{badges_block}

    This repository hosts {dataset_desc}
    Each **config** corresponds to a single model;
    only that model's Parquet files are read on `load_dataset`.

    ## Usage

    Load with `datasets` and convert to `torch`:

    ```python
{torch_usage}
    ```

    Or read the Parquet files directly with `polars`:

    ```python
{polars_usage}
    ```

{fields_section}

    ## Available Models

    Number of models with precomputed embeddings, per split:

{model_counts_table}

    ## Notes
    - Configs are generated from what is actually
      uploaded on the Hub (parquet presence).
    {original_dataset_link}
    {code_repo_link}
    {model_registry_link}
    {extra_markdown}

{citation_section}
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


# Columns of the model registry, by thematic group, for the generated README.
_MODEL_REGISTRY_FIELDS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        'Identity',
        [
            (
                'model_name',
                'Full `timm` model identifier, e.g., `vit_large_patch14_clip_224.openai_ft_in1k`. '
                'Primary key, joined with the `model_name` column of every SEMASIA dataset.',
            ),
        ],
    ),
    (
        'Architecture',
        [
            (
                'family',
                'High-level class: ViT, ConvNeXt, ResNet, Swin, DeiT, EfficientNet, ...',
            ),
            (
                'macro_family',
                'Coarse architectural category: Convolutional, Vision Transformer, '
                'Hybrid CNN-Transformer, Hierarchical Vision Transformer, MetaFormer, MLP, State Space Model.',
            ),
            (
                'model_version',
                'Refinement for multi-generation families: v2, v3, SE, NF, CSP, ...',
            ),
            (
                'size',
                'Human-readable size label, Tiny to Giant, or B0 to B8 for EfficientNet.',
            ),
            (
                'depth_code',
                'Depth for depth-scaled families, e.g., 50, 101 (ResNet), f0 to f6 (NFNet).',
            ),
            (
                'width_code',
                'Channel or group-width multiplier, e.g., 32x4d (ResNeXt), w44 (HRNet), 075 (MobileNet).',
            ),
            ('patch_size', 'ViT patch size in pixels; null for CNNs.'),
            (
                'input_resolution',
                'Native architectural input resolution, distinct from the pretraining and fine-tuning resolutions.',
            ),
            (
                'window_size',
                'Local-attention window of Swin, MaxViT and CoAtNet; null for global attention.',
            ),
            (
                'stride_code',
                'Stage/stride code of families such as CaiT, ConvFormer and ShViT.',
            ),
        ],
    ),
    (
        'Head and attention',
        [
            (
                'head_type',
                'Pooling before the classifier: GAP (global average), CLS (class token) or CLS+GAP (hybrid).',
            ),
            (
                'num_registers',
                'Number of DINOv2-style register tokens, typically 1 or 4.',
            ),
            (
                'positional_encoding',
                'Non-default scheme: RoPE, RelPos, SinCos, APE; null is the default learned absolute encoding.',
            ),
            (
                'activation',
                'Non-default activation, notably QuickGELU for OpenAI-style CLIP models; null is standard GELU or ReLU.',
            ),
            (
                'pe_scope',
                'Scope tag of the ViT-PE (Perception Encoder) family: Lang, Core, Spatial.',
            ),
        ],
    ),
    (
        'Variant flags (booleans)',
        [
            ('is_distilled', 'Distilled model (e.g., DeiT with a RegNet teacher).'),
            ('is_pruned', 'Structurally pruned model.'),
            ('is_legacy', 'Older model under the `legacy_` prefix.'),
            ('is_gap', 'Shorthand for `head_type == "GAP"`.'),
            ('uses_quickgelu', 'Shorthand for `activation == "QuickGELU"`.'),
            ('uses_rmlp', 'MaxViT/CoAtNet with MLP Log-CPB relative position bias.'),
            (
                'uses_rw',
                '`timm` re-implementation tuned for PyTorch eager-mode efficiency.',
            ),
            ('uses_cr', 'SwinV2 cross-resolution variant.'),
            ('uses_ns', 'SwinV2-CR with LayerNorm at every stage (norm per stage).'),
            ('uses_abswin', 'Hiera with absolute window position embeddings.'),
            ('uses_ts', 'BYOBNet with a tiered three-layer convolutional stem.'),
            ('uses_aa', 'Anti-aliased downsampling.'),
        ],
    ),
    (
        'Pretraining',
        [
            (
                'pretrain_config',
                'Raw configuration string (the part after the dot), kept for reference and debugging.',
            ),
            (
                'pretrain_org',
                'Training organization: Meta, Apple, OpenAI, Google, `timm SBB recipe`, ...',
            ),
            (
                'pretrain_dataset',
                'Corpus: ImageNet-1K/21K/22K, LAION-2B, WebLI, LVD-142M, ...',
            ),
            ('pretrain_dataset_size', 'Corpus size when ambiguous: 400M, s39b, 2.1T.'),
            (
                'pretrain_method',
                'Objective keyword parsed from the identifier: CLIP, MAE, DINO, DINOv2, MIM, FCMAE, AugReg, ...; null for standard supervised training and for several other models.',
            ),
            (
                'pretrain_tasks',
                'List of tasks used to train the model, main objective first (see below).',
            ),
            (
                'paradigm',
                'Learning signal of the main objective: supervised, contrastive, self_supervised, multimodal_generative or unknown (see below).',
            ),
            (
                'pretrain_ft',
                'Subsequent fine-tuning, typically ImageNet-1K, occasionally ImageNet-22K or ImageNet-12K.',
            ),
            ('pretrain_resolution', 'Input resolution at pretraining.'),
            ('pretrain_ft_resolution', 'Input resolution at fine-tuning.'),
            ('pretrain_epochs', 'Compute budget in epochs, e.g., e200 -> 200.'),
            (
                'pretrain_tokens',
                'Token budget for token-based recipes such as SAM2, e.g., 2pt1 -> 2.1T tokens.',
            ),
            (
                'pretrain_aug',
                'Augmentation: AugReg, AdvProp, NoisyStudent, AutoAugment, RandAugment.',
            ),
            (
                'pretrain_i18n',
                'Flags SigLIP models pretrained on multilingual WebLI (109 languages).',
            ),
        ],
    ),
    (
        'Capacity (derived by instantiating the model)',
        [
            (
                'num_parameters',
                'Total trainable parameters, obtained by instantiating the model in `timm` (without downloading weights, when feasible).',
            ),
            (
                'latent_dim',
                'Length of the `embedding` vector in the SEMASIA datasets, i.e., the output size of the `timm` model created with `num_classes=0` (`head_hidden_size` when present, else `num_features`).',
            ),
        ],
    ),
]

_MODEL_REGISTRY_TASKS: list[tuple[str, str]] = [
    (
        'image_classification',
        'Supervised class labels (also weakly or semi-supervised, and distilled models).',
    ),
    (
        'image_text_contrastive',
        'Image-text contrastive or sigmoid loss: CLIP, SigLIP, MobileCLIP, Perception Encoder.',
    ),
    (
        'masked_image_modeling',
        'Reconstruction of masked image content (pixels, tokens or features): MAE, FCMAE, BEiT, EVA, I-JEPA.',
    ),
    (
        'joint_embedding',
        'DINO-style self-distillation and clustering (SwAV, used by SEER).',
    ),
    (
        'vision_language_generative',
        'Captioning, autoregressive or next-token objective with a language decoder: AIMv2, PaliGemma towers, InternViT.',
    ),
    (
        'masked_vision_language_modeling',
        'Masked modeling of images and text together: BEiT-3.',
    ),
    ('promptable_segmentation', 'Segmentation from prompts: SAM, SAM 2.'),
    ('vision_language_multitask', 'Several vision-language tasks at once: Florence-2.'),
]


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
        Full Hugging Face repository ID (e.g., "spaicom-lab/semasia-model-registry").
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

    fields_md = '\n\n'.join(
        '\n'.join(
            [
                f'### {group}',
                '',
                '| Field | Description |',
                '|-------|-------------|',
            ]
            + [f'| `{field}` | {desc} |' for field, desc in rows]
        )
        for group, rows in _MODEL_REGISTRY_FIELDS
    )
    tasks_md = '\n'.join(
        ['| Task | Meaning |', '|------|---------|']
        + [f'| `{task}` | {desc} |' for task, desc in _MODEL_REGISTRY_TASKS]
    )

    md_head = textwrap.dedent("""
    # Timm Model Metadata Registry

    This dataset contains a lookup table of metadata for all pretrained
    models available in the timm library. It is the companion of the SEMASIA
    latent-space datasets: the column `model_name` is the join key with the
    `model_name` column of every SEMASIA dataset.

    ## Usage

    ```python
    from datasets import load_dataset
    import polars as pl

    ds = load_dataset("spaicom-lab/semasia-model-registry")
    df = ds["train"].to_polars()

    # Query by model name
    model = df.filter(pl.col("model_name") == "vit_base_patch16_224.augreg_in21k")

    # Filter by family
    vit_models = df.filter(pl.col("family") == "ViT")

    # Filter by parameter count
    small_models = df.filter(pl.col("num_parameters") < 50_000_000)

    # Filter by training paradigm or task (pretrain_tasks is a list column)
    self_supervised = df.filter(pl.col("paradigm") == "self_supervised")
    contrastive = df.filter(pl.col("pretrain_tasks").list.contains("image_text_contrastive"))
    ```

    ## Fields

    The identifier of a model is `{architecture}.{pretrain_config}`: everything
    before the dot encodes the architecture, everything after it encodes how
    and where the checkpoint was pretrained. The columns are grouped
    accordingly. Fields that do not apply to a model are null.

    """)

    md_tail = (
        textwrap.dedent("""

    ## Training tasks and paradigm

    `pretrain_tasks` is a list. The first element is the main objective of the
    pretraining that produced the weights; the other elements are objectives
    of the same recipe, of the initialization, or of a later supervised
    fine-tuning. `paradigm` is the learning signal of the first element.

    @@TASKS@@

    | Paradigm | Tasks it is derived from |
    |----------|--------------------------|
    | `supervised` | `image_classification`, `promptable_segmentation` |
    | `contrastive` | `image_text_contrastive` |
    | `self_supervised` | `masked_image_modeling`, `joint_embedding` |
    | `multimodal_generative` | `vision_language_generative`, `masked_vision_language_modeling`, `vision_language_multitask` |
    | `unknown` | nothing could be inferred |

    Examples:

    - `convnext_base.clip_laion2b_augreg_ft_in1k`: `["image_text_contrastive", "image_classification"]`, paradigm `contrastive`.
    - `convnextv2_base.fcmae_ft_in1k`: `["masked_image_modeling", "image_classification"]`, paradigm `self_supervised`.
    - `vit_base_patch16_siglip_224.v2_webli` (SigLIP 2): `["image_text_contrastive", "vision_language_generative", "joint_embedding", "masked_image_modeling"]`, paradigm `contrastive`.

    These columns are derived from the model name and `pretrain_method` with
    the rules of `infer_pretrain_tasks` (`src/utils/timm_metadata.py`), which
    cites the papers behind the non-obvious cases; they are not read from the
    model cards. `pretrain_method` is an objective keyword parsed from the
    identifier and is null for several models (e.g., most SigLIP models), so
    use `pretrain_tasks` or `paradigm` to select models by objective.

    ## Notes

    - Metadata is extracted using the `get_model_metadata` function from the
      [semantic-datasets](https://github.com/spaicom-lab/semantic-datasets) library.
    - Some fields may be ``None`` if they cannot be inferred from the model name.
    - Total number of models: **@@N@@**
    """)
        .replace('@@TASKS@@', tasks_md)
        .replace('@@N@@', str(num_models))
    )

    md_body = md_head + fields_md + md_tail

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
