# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=4.8.5",
#     "huggingface-hub>=1.15.0",
#     "marimo>=0.23.6",
#     "matplotlib>=3.10.9",
#     "numpy>=2.4.5",
#     "plotly>=6.7.0",
#     "polars>=1.40.1",
#     "scikit-learn>=1.8.0",
#     "torch>=2.12.0",
#     "umap-learn>=0.5.12",
#     "wigglystuff>=0.5.0",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import io

    import marimo as mo
    import numpy as np

    HF_REPO = "spaicom-lab/semasia-"

    CLASS_NAMES: dict[str, dict[str, list[str]]] = {
        "cifar10": {
            "label": ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],
        },
        "cifar100": {
            "coarse_label": [
                "aquatic_mammals","fish","flowers","food_containers","fruit_and_vegetables",
                "household_electrical_devices","household_furniture","insects","large_carnivores",
                "large_man-made_outdoor_things","large_natural_outdoor_scenes",
                "large_omnivores_and_herbivores","medium_mammals","non-insect_invertebrates",
                "people","reptiles","small_mammals","trees","vehicles_1","vehicles_2",
            ],
            "fine_label": [
                "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
                "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
                "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
                "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
                "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle",
                "mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
                "pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum","rabbit","raccoon",
                "ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail",
                "snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank",
                "telephone","television","tiger","tractor","train","trout","tulip","turtle","wardrobe",
                "whale","willow_tree","wolf","woman","worm",
            ],
        },
        "mnist": {
            "label": [str(i) for i in range(10)],
        },
        "fashion_mnist": {
            "label": ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"],
        },
    }

    DATASET_CONFIGS: dict = {
        "cifar10":        {"data": "img",   "extras": ["label"],                     "name": "uoft-cs/cifar10"},
        "cifar100":       {"data": "img",   "extras": ["fine_label","coarse_label"], "name": "uoft-cs/cifar100"},
        "mnist":          {"data": "image", "extras": ["label"],                     "name": "ylecun/mnist"},
        "fashion_mnist":  {"data": "image", "extras": ["label"],                     "name": "zalando-datasets/fashion_mnist"},
        "fashion-mnist":  {"data": "image", "extras": ["label"],                     "name": "zalando-datasets/fashion_mnist"},
        "oxford-flowers": {"data": "image", "extras": ["label"],                     "name": "nkirschi/oxford-flowers"},
        "tiny-imagenet":  {"data": "image", "extras": ["label"],                     "name": "zh-plus/tiny-imagenet"},
    }

    def show_images(pil_images, titles, max_images=10):
        import matplotlib.pyplot as plt
        pil_images = pil_images[:max_images]
        titles = titles[:max_images]
        n = len(pil_images)
        if n == 0:
            return None
        fig, axes = plt.subplots(1, n, squeeze=False)
        fig.set_size_inches(n * 1.6, 2.0)
        for ax, img, title in zip(axes[0], pil_images, titles):
            ax.imshow(img if img.mode == "RGB" else img.convert("RGB"))
            ax.set_title(title, fontsize=7, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout(pad=0.4)
        return fig

    return CLASS_NAMES, DATASET_CONFIGS, HF_REPO, io, mo, np, show_images


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="min-height:340px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:2rem 1rem">
    <h1>SEMASIA</h1>
    <h2>A Large-Scale Dataset of Semantically Structured Latent Representations</h2>
    <br>
    <p>
    <strong>Mario Edoardo Pandolfo* &nbsp;·&nbsp; Enrico Grimaldi* &nbsp;·&nbsp; Lorenzo Marinucci</strong><br>
    <strong>Leonardo Di Nino &nbsp;·&nbsp; Simone Fiorellino &nbsp;·&nbsp; Sergio Barbarossa &nbsp;·&nbsp; Paolo Di Lorenzo</strong><br>
    <em>Sapienza University of Rome — CNIT</em>
    </p>
    <p><small>*Equal contribution</small></p>
    <br>
    <p>🤗 <a href="https://huggingface.co/collections/spaicom-lab/semasia" target="_blank"><code>huggingface.co/collections/spaicom-lab/semasia</code></a> &nbsp;&nbsp;|&nbsp;&nbsp; 🐙 <a href="https://github.com/SPAICOM/semasia-datasets" target="_blank"><code>SPAICOM/semasia-datasets</code></a> &nbsp;&nbsp;|&nbsp;&nbsp; 📄 <a href="https://arxiv.org/abs/2605.09485" target="_blank">arXiv:2605.09485</a></p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(
            "**TL;DR** — We release **Semasia**, a benchmark of latent representations from"
            " **~1,700 pretrained vision models** across **8 image-classification datasets**, paired"
            " with rich architectural and training metadata. We use it to study how model"
            " choices shape embedding geometry, to evaluate latent space alignment methods,"
            " and to perform large-scale regression analysis of representation structure."
        ),
        mo.Html("""
    <div align="center">
    <img src="https://drive.google.com/thumbnail?id=15_vmbozAH-w9X2dwZQJF4gcl3zDcW43X&sz=w1200" height="400">
    </div>
    """),
        mo.md(r"""
    Modern neural networks learn **latent representations** with semantic structure:
    conceptual similarity is reflected in geometric proximity in embedding space.

    $$\mathcal{W} \xrightarrow{f} \mathcal{Z} \xrightarrow{g} \mathcal{Y}$$

    The encoder $f$ maps raw inputs into a **latent manifold** $\mathcal{Z}$,
    where concepts form compact clusters — the **semantic prototypes**.

    ---

    ### Why study latent geometry?

    Two complementary research questions drive this work:

    **1. Statistical analysis of embedding geometry** — How do architectural choices, pretraining data, training regime, and model scale shape the structure of embeddings? Answering this at scale requires comparing hundreds of models across diverse benchmarks.

    **2. Latent space alignment** — Representations learned by different models are often **geometrically incompatible**: even minor sources of stochasticity — weight initialisation, optimisation dynamics, data shuffling — introduce variability, while differences in architecture or training data further amplify discrepancies.

    > Same inputs → semantically equivalent embeddings, but **not geometrically comparable**.

    This geometric incompatibility surfaces in transfer and multitask learning, federated learning, multi-agent systems, and **semantic communication** — where in AI-native 6G systems, latent vectors *are* the transmitted signal.

    > *How does the geometry of latent spaces vary across models and training conditions — and how can we align heterogeneous representations without joint retraining?*

    Answering this at scale requires a large-scale benchmark of standardised latent representations paired with structured metadata — this is exactly what **SEMASIA** provides.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## The SEMASIA Dataset

    SEMASIA is a large-scale collection of latent representations extracted from **~1 700 models**
    from the vision state-of-the-art (`timm` library), across **8 standard benchmarks**.

    | Dimension | Detail |
    |---|---|
    | Models | **~1 700** pretrained architectures (`timm`) |
    | Benchmarks | **8** standard image-classification datasets |
    | Total rows | **> 1 billion** (input, model, embedding) pairs |
    | Storage | Parquet, one file per (model × split) |
    | Access | 🤗 HuggingFace `spaicom-lab/semasia-*` |

    | Dataset | Splits | Raw examples | Total rows |
    |---|---|---|---|
    | `semasia-celeba` | train / valid / test | 100 k / 19 k / 19 k | ≈ 238 M |
    | `semasia-cifar10` | train / test | 50 k / 10 k | ≈ 102 M |
    | `semasia-cifar100` | train / test | 50 k / 10 k | ≈ 102 M |
    | `semasia-fashion_mnist` | train / test | 60 k / 10 k | ≈ 119 M |
    | `semasia-mnist` | train / test | 60 k / 10 k | ≈ 119 M |
    | `semasia-oxford-flowers` | train / test | 7 k / 1 k | ≈ 14 M |
    | `semasia-tiny-imagenet` | train / valid | 100 k / 10 k | ≈ 187 M |
    | `semasia-imagenet-1k` | valid / test | 50 k / 100 k | ≈ 255 M |

    Covered architectures span ConvNets (ResNet, EfficientNet, ConvNeXt), Vision Transformers (ViT, DeiT, Swin, AiMv2), hybrid architectures, and self-supervised backbones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Structure & Format

    Each Parquet file corresponds to one **(model, dataset, split)** triple:

    | Field | Type | Content |
    |---|---|---|
    | `id` | `uint32` | Source-image index — use to align rows across models |
    | `model_name` | `string` | `timm` identifier of the encoder |
    | `embedding` | `array[float]` | Pre-logit latent vector of dimension $d$ |
    | extra columns | varies | Original benchmark dataset columns (e.g. `label`) |

    The `id` field is the key for **alignment studies**: two rows with the same `id` but different `model_name` correspond to the same image — comparing their embeddings directly quantifies semantic mismatch.

    Each model is paired with structured metadata recording architecture family, depth and width, input resolution, pretraining source, training objective, augmentation scheme, parameter count, and latent dimensionality. This enables controlled regression analyses — isolating the effect of a single factor (e.g. *model scale*) while holding all others fixed.

    Latents are extracted at the **last layer before the classification head** — the *semantic bottleneck*: rich enough to carry full semantic content, not yet collapsed onto the discrete label space.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Loading the Data

    Each SEMASIA entry is identified by three coordinates: a **dataset**, a **model**, and a **split**.
    All data is hosted on HuggingFace and accessed through the standard `datasets` library — no
    custom infrastructure required. The dataset name acts as the HuggingFace repository, the model
    name selects the configuration (one Parquet file per model), and the split selects the partition.

    ```python
    from datasets import load_dataset
    import torch

    ds = load_dataset(
        "spaicom-lab/semasia-cifar10",  # repository  →  which benchmark
        "resnet50.a1_in1k",             # config      →  which model
        split="test",                   # split       →  which partition
    ).with_format("torch")
    ```

    Stacking embeddings and labels into tensors takes one line each:

    ```python
    embeddings = torch.vstack(list(ds["embedding"]))  # (N, d)
    labels     = torch.tensor(ds["label"])             # (N,)
    ```

    To compare representations across models, load two configs from the same repository and
    join on `id` — rows sharing the same `id` correspond to the same input image:

    ```python
    ds_a = load_dataset("spaicom-lab/semasia-cifar10", "resnet50.a1_in1k",   split="test").with_format("torch")
    ds_b = load_dataset("spaicom-lab/semasia-cifar10", "vit_base_patch16_224.augreg_in1k", split="test").with_format("torch")

    Z_a = torch.vstack(list(ds_a["embedding"]))  # (N, d_a)
    Z_b = torch.vstack(list(ds_b["embedding"]))  # (N, d_b)
    # rows i in Z_a and Z_b correspond to the same image
    ```

    ---

    ### Alternative: loading with Polars

    > *This is the approach used by this notebook.*

    SEMASIA is stored as Parquet files on HuggingFace and can be read directly with
    [**Polars**](https://docs.pola.rs/) via the `hf://` URI scheme — **no data is written to disk**.
    Unlike `load_dataset`, which caches the full split under `~/.cache/huggingface/datasets/`,
    `pl.read_parquet` loads directly into memory and leaves no local copy behind.

    See the [HuggingFace Parquet documentation](https://huggingface.co/docs/hub/en/datasets-polars) for full details on the `hf://` protocol and authentication options.

    ```python
    import polars as pl

    df = pl.read_parquet(
        "hf://datasets/spaicom-lab/semasia-cifar10/test/resnet50.a1_in1k/*.parquet"
    )

    embeddings = df["embedding"].to_numpy()  # shape (N, d)
    labels     = df["label"].to_numpy()      # shape (N,)
    ```

    To load two models and align them on `id`:

    ```python
    df_a = pl.read_parquet("hf://datasets/spaicom-lab/semasia-cifar10/test/resnet50.a1_in1k/*.parquet")
    df_b = pl.read_parquet("hf://datasets/spaicom-lab/semasia-cifar10/test/vit_base_patch16_224.augreg_in1k/*.parquet")

    aligned = df_a.join(df_b, on="id", suffix="_b")
    Z_a = aligned["embedding"].to_numpy()    # (N, d_a)
    Z_b = aligned["embedding_b"].to_numpy()  # (N, d_b)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Interactive Demo — Latent Space Exploration

    Load latent representations from a chosen model and benchmark, project them to 2D with **PCA / t-SNE / UMAP**, and select points to inspect the original images. A **parallel coordinates** view lets you brush along principal axes to reveal hierarchical semantic structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Select** a model, dataset(s), split, and reduction method from the controls. "
            "Adjust **Samples / dataset** to trade off speed and coverage — data is pre-fetched "
            "at 1 600 samples so changing the slider never triggers a re-download. "
            "**Select points** in the scatter plot to inspect their original images."
            "Switch to the **Parallel Coordinates** view to brush along principal axes and reveal "
            "hierarchical semantic structure."
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    None
    return


@app.cell(hide_code=True)
def _(HF_REPO, mo):
    import os as _os
    from collections import defaultdict
    from pathlib import PurePosixPath
    from huggingface_hub import HfApi

    def collect_models_by_split(repo_id: str) -> dict[str, set[str]]:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
        models_by_split: dict = defaultdict(set)
        for f in files:
            path = PurePosixPath(f)
            if len(path.parts) < 3 or path.suffix != '.parquet':
                continue
            models_by_split[path.parts[0]].add(path.parts[1])
        return dict(models_by_split)

    _ds_options = ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'oxford-flowers', 'tiny-imagenet']

    with mo.status.spinner(title='Fetching available models for all datasets...'):
        l_all_info: dict = {}
        for _ds in _ds_options:
            _mbs = collect_models_by_split(HF_REPO + _ds)
            if _mbs:
                l_all_info[_ds] = _mbs

    l_all_models = sorted(
        set().union(*(set().union(*splits.values()) for splits in l_all_info.values()))
    )
    l_model_ui = mo.ui.dropdown(
        options=l_all_models,
        value=l_all_models[0] if l_all_models else None,
        label='Model',
    )
    l_method_ui = mo.ui.dropdown(options=['PCA', 't-SNE', 'UMAP'], value='t-SNE', label='Reduction')
    l_n_samples_ui = mo.ui.slider(start=200, stop=1600, step=100, value=800, label='Samples / dataset', show_value=True)
    return l_all_info, l_all_models, l_method_ui, l_model_ui, l_n_samples_ui


@app.cell(hide_code=True)
def _(l_all_info: dict, l_model_ui, mo):
    l_split_uis: dict = {}
    for _ds, _mbs in l_all_info.items():
        _avail = sorted(s for s, models in _mbs.items() if l_model_ui.value in models)
        if _avail:
            if _ds == "tiny-imagenet":
                _preferred = "train" if "train" in _avail else _avail[0]
            else:
                _preferred = next(
                    (s for s in ["test", "val", "valid", "validation"] if s in _avail),
                    _avail[0],
                )
            l_split_uis[_ds] = mo.ui.dropdown(
                options=_avail,
                value=_preferred,
                label=_ds,
            )
    l_available_datasets = list(l_split_uis.keys())
    return l_available_datasets, l_split_uis


@app.cell(hide_code=True)
def _(l_available_datasets, mo):
    l_deselect_ui = mo.ui.multiselect(
        options=l_available_datasets,
        value=[],
        label="Exclude datasets",
    )
    return (l_deselect_ui,)


@app.cell(hide_code=True)
def _(
    l_deselect_ui,
    l_method_ui,
    l_model_ui,
    l_n_samples_ui,
    l_split_uis: dict,
    mo,
):
    mo.vstack([
        mo.hstack([l_model_ui, l_method_ui, l_n_samples_ui], gap=2),
        mo.md('**Split per dataset:**'),
        mo.hstack(list(l_split_uis.values()), gap=2, wrap=True),
        l_deselect_ui,
    ])
    return


@app.cell(hide_code=True)
def _(
    HF_REPO,
    l_available_datasets,
    l_deselect_ui,
    l_model_ui,
    l_split_uis: dict,
    mo,
    np,
):
    import polars as _pl

    l_selected_datasets = [
        d for d in l_available_datasets if d not in l_deselect_ui.value
    ]
    mo.stop(
        not l_selected_datasets,
        mo.callout(mo.md('Select at least one dataset.'), kind='warn'),
    )

    _all_embeddings = []
    _all_ds_labels = []
    l_all_sample_info_full = []

    for _ds in l_selected_datasets:
        _split = l_split_uis[_ds].value
        _uri = (
            f'hf://datasets/{HF_REPO}{_ds}/{_split}/{l_model_ui.value}/*.parquet'
        )
        with mo.status.spinner(title=f'Loading {_ds} ({_split})...'):
            _loaded = _pl.scan_parquet(_uri).limit(1600).collect()
        _latent = np.array(_loaded['embedding'].to_list())
        _n = len(_loaded)
        _all_embeddings.append(_latent)
        _all_ds_labels.extend([_ds] * _n)
        l_all_sample_info_full.extend(
            [(_ds, int(i)) for i in _loaded['id'].to_list()]
        )

    l_X_all_full = np.vstack(_all_embeddings)
    l_y_ds_full = np.array(_all_ds_labels)
    return (
        l_X_all_full,
        l_all_sample_info_full,
        l_selected_datasets,
        l_y_ds_full,
    )


@app.cell(hide_code=True)
def _(
    l_X_all_full,
    l_all_sample_info_full,
    l_n_samples_ui,
    l_selected_datasets,
    l_y_ds_full,
    np,
):
    _n = l_n_samples_ui.value
    _idx = []
    for _ds in l_selected_datasets:
        _ds_idx = np.where(l_y_ds_full == _ds)[0][:_n]
        _idx.extend(_ds_idx.tolist())
    _idx = np.array(_idx)
    l_X_all = l_X_all_full[_idx]
    l_y_ds = l_y_ds_full[_idx]
    l_all_sample_info = [l_all_sample_info_full[i] for i in _idx]
    return l_X_all, l_all_sample_info, l_y_ds


@app.cell(hide_code=True)
def _(DATASET_CONFIGS: dict, l_selected_datasets, l_split_uis: dict, mo):
    import polars as _pl_orig
    from huggingface_hub import HfApi as _HfApi

    _api = _HfApi()


    def _orig_parquet_glob(repo_id: str, split: str) -> str:
        files = [
            f
            for f in _api.list_repo_files(repo_id, repo_type='dataset')
            if f.endswith('.parquet')
        ]
        matches = [f for f in files if f.rsplit('/', 1)[-1].startswith(split)]
        if not matches:
            raise FileNotFoundError(f'No parquet for split={split!r} in {repo_id}')
        config_dir = '/'.join(matches[0].split('/')[:-1])
        return f'hf://datasets/{repo_id}/{config_dir}/{split}-*.parquet'


    l_orig_datasets: dict = {}
    for _ds in l_selected_datasets:
        _cfg = DATASET_CONFIGS.get(_ds, {})
        _orig_repo = _cfg.get('name', '')
        if not _orig_repo:
            continue
        _split = l_split_uis[_ds].value
        _orig_split = 'validation' if _split in ('valid', 'val') else _split
        with mo.status.spinner(
            title=f'Loading original images for {_ds} ({_orig_split})...'
        ):
            try:
                _uri = _orig_parquet_glob(_orig_repo, _orig_split)
                _df = _pl_orig.scan_parquet(_uri).limit(1600).collect()
            except Exception:
                _fallback = 'test' if _orig_split != 'test' else 'validation'
                _uri = _orig_parquet_glob(_orig_repo, _fallback)
                _df = _pl_orig.scan_parquet(_uri).limit(1600).collect()
        l_orig_datasets[_ds] = _df
    return (l_orig_datasets,)


@app.cell(hide_code=True)
def _(l_X_all, l_method_ui, mo):
    import umap as _umap
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE

    with mo.status.spinner(title=f"Running {l_method_ui.value} on {len(l_X_all):,} points…"):
        if l_method_ui.value == "PCA":
            l_X2d = _PCA(n_components=2).fit_transform(l_X_all)
        elif l_method_ui.value == "t-SNE":
            l_X2d = _TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(l_X_all)
        else:
            l_X2d = _umap.UMAP(n_components=2, random_state=42).fit_transform(l_X_all)
    return (l_X2d,)


@app.cell(hide_code=True)
def _(l_X2d, l_all_sample_info, l_selected_datasets, l_y_ds, mo, np):
    import plotly.graph_objects as _go

    _PALETTE = [
        '#2A9D8F', '#E9C46A', '#8E6BBE', '#F4A261', '#457B9D',
        '#6D9B3A', '#E76F51', '#9B72AA', '#3ABEFF', '#C5956B',
    ]
    _MARKERS = ['circle', 'square', 'diamond', 'cross', 'star', 'triangle-up', 'triangle-down', 'pentagon', 'hexagram']
    _axis = dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks='')

    l_trace_to_global = []
    _fig = _go.Figure()
    for _i, _ds in enumerate(l_selected_datasets):
        _mask = l_y_ds == _ds
        _global_pts = np.where(_mask)[0]
        l_trace_to_global.append(_global_pts)
        _orig_indices = np.array([l_all_sample_info[g][1] for g in _global_pts])
        _customdata = np.stack([_global_pts, _orig_indices], axis=1)
        _fig.add_trace(_go.Scatter(
            x=l_X2d[_mask, 0], y=l_X2d[_mask, 1], mode='markers', name=_ds,
            marker=dict(size=5, opacity=0.9, color=_PALETTE[_i % len(_PALETTE)], symbol=_MARKERS[_i % len(_MARKERS)]),
            customdata=_customdata,
            hovertemplate=f'<b>{_ds}</b><br>orig_idx: %{{customdata[1]}}<br>C1: %{{x:.3f}}  C2: %{{y:.3f}}<extra></extra>',
        ))

    _fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=_axis, yaxis=_axis,
        showlegend=True,
        legend=dict(title='Dataset', bgcolor='rgba(255,255,255,0.6)', borderwidth=1),
        width=900, height=520, margin=dict(t=10, b=10, l=10, r=10),
    )
    l_chart = mo.ui.plotly(_fig)
    l_chart
    return l_chart, l_trace_to_global


@app.cell(hide_code=True)
def _(
    DATASET_CONFIGS: dict,
    io,
    l_all_sample_info,
    l_chart,
    l_orig_datasets: dict,
    l_selected_datasets,
    l_trace_to_global,
    mo,
    show_images,
):
    from PIL import Image as _PILImage

    _pts = l_chart.value
    mo.stop(
        not _pts,
        mo.callout(
            mo.md(
                '**Try making a selection with your mouse!**\nBox- or lasso-select points on the scatter plot above.'
            ),
            kind='info',
        ),
    )

    _images_by_ds: dict = {}
    for _pt in _pts:
        _curve = int(_pt.get('curveNumber', 0))
        _point_idx = int(_pt.get('pointIndex', _pt.get('pointNumber', 0)))
        if _curve >= len(l_selected_datasets):
            continue
        _ds_name = l_selected_datasets[_curve]
        _global = int(l_trace_to_global[_curve][_point_idx])
        _, _orig_idx = l_all_sample_info[_global]
        if _ds_name not in _images_by_ds:
            _images_by_ds[_ds_name] = []
        _images_by_ds[_ds_name].append(_orig_idx)

    _max_images = 10
    _ds_list = list(_images_by_ds.keys())
    _queues = {ds: list(idxs) for ds, idxs in _images_by_ds.items()}
    _sampled_by_ds: dict = {ds: [] for ds in _ds_list}
    _total = 0
    while _total < _max_images:
        _added = False
        for _ds in _ds_list:
            if _total >= _max_images:
                break
            if _queues[_ds]:
                _sampled_by_ds[_ds].append(_queues[_ds].pop(0))
                _total += 1
                _added = True
        if not _added:
            break

    _pil_images = []
    _titles = []
    for _ds_name, _ds_indices in _sampled_by_ds.items():
        if not _ds_indices or _ds_name not in l_orig_datasets:
            continue
        _cfg = DATASET_CONFIGS.get(_ds_name, {})
        _img_field = _cfg.get('data', 'image')
        _orig_df = l_orig_datasets[_ds_name]
        for _oi in _ds_indices:
            if _oi >= len(_orig_df):
                continue
            _row = _orig_df.row(_oi, named=True)
            _pil_images.append(
                _PILImage.open(io.BytesIO(_row[_img_field]['bytes']))
            )
            _titles.append(f'{_ds_name}\n#{_oi}')

    mo.stop(
        not _pil_images,
        mo.callout(mo.md('No images available for selection.'), kind='warn'),
    )

    _fig = show_images(_pil_images, _titles)
    _buf = io.BytesIO()
    _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=150)
    _buf.seek(0)
    import matplotlib.pyplot as _plt

    _plt.close(_fig)
    mo.vstack(
        [
            mo.md(
                f'**{len(_pts)} points selected** — showing up to {_max_images} images ({len(_ds_list)} dataset(s)):'
            ),
            mo.image(_buf.getvalue()),
        ]
    )
    return


@app.cell(hide_code=True)
def _(l_available_datasets, mo):
    pc_dataset_ui = mo.ui.dropdown(
        options=l_available_datasets,
        value="cifar10" if "cifar10" in l_available_datasets else l_available_datasets[0],
        label="Dataset",
    )
    pc_method_ui = mo.ui.dropdown(
        options=["PCA", "t-SNE", "UMAP"], value="UMAP", label="Reduction"
    )
    return pc_dataset_ui, pc_method_ui


@app.cell(hide_code=True)
def _(mo, pc_dataset_ui):
    _pc_label_map = {
        "cifar10": ["label"],
        "cifar100": ["fine_label", "coarse_label"],
        "mnist": ["label"],
        "fashion_mnist": ["label"],
        "oxford-flowers": ["label"],
        "tiny-imagenet": ["label"],
    }
    _pc_opts = _pc_label_map.get(pc_dataset_ui.value, ["label"])
    _pc_def = "coarse_label" if "coarse_label" in _pc_opts else _pc_opts[0]
    pc_label_ui = mo.ui.dropdown(options=_pc_opts, value=_pc_def, label="Color by")
    return (pc_label_ui,)


@app.cell(hide_code=True)
def _(mo, pc_dataset_ui, pc_label_ui, pc_method_ui):
    _max_dims = 3 if pc_method_ui.value == "t-SNE" else 10
    pc_n_dims_ui = mo.ui.slider(
        start=2, stop=_max_dims, step=1,
        value=min(6, _max_dims), label="Components", show_value=True,
    )
    mo.hstack([mo.md("### Parallel Coordinates"), pc_dataset_ui, pc_label_ui, pc_method_ui, pc_n_dims_ui], gap=2)
    return (pc_n_dims_ui,)


@app.cell(hide_code=True)
def _(HF_REPO, l_model_ui, l_split_uis: dict, mo, pc_dataset_ui):
    import polars as _pl2

    _pc_ds_name = pc_dataset_ui.value
    _pc_split = l_split_uis[_pc_ds_name].value

    _pc_uri = f'hf://datasets/{HF_REPO}{_pc_ds_name}/{_pc_split}/{l_model_ui.value}/*.parquet'
    with mo.status.spinner(title=f'Loading {_pc_ds_name} ({_pc_split})...'):
        pc_raw_full = _pl2.scan_parquet(_pc_uri).limit(1600).collect()
    return (pc_raw_full,)


@app.cell(hide_code=True)
def _(
    CLASS_NAMES: dict[str, dict[str, list[str]]],
    l_n_samples_ui,
    mo,
    np,
    pc_dataset_ui,
    pc_label_ui,
    pc_method_ui,
    pc_n_dims_ui,
    pc_raw_full,
):
    import polars as _pl2r
    import umap as _umap_pc
    from sklearn.decomposition import PCA as _PCA_pc
    from sklearn.manifold import TSNE as _TSNE_pc

    _pc_ds_name = pc_dataset_ui.value
    _pc_label_col = pc_label_ui.value
    _n_pc = pc_n_dims_ui.value
    _method_pc = pc_method_ui.value

    _n = min(l_n_samples_ui.value, len(pc_raw_full))
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(len(pc_raw_full), size=_n, replace=False)
    _pc_sampled = pc_raw_full[_idx]

    _pc_n = len(_pc_sampled)
    _pc_embeddings = np.array(_pc_sampled['embedding'].to_list())
    _pc_labels_sampled = [int(x) for x in _pc_sampled[_pc_label_col].to_list()]
    _pc_name_map = CLASS_NAMES.get(_pc_ds_name, {}).get(_pc_label_col)
    if _pc_name_map:
        _pc_labels_sampled = [_pc_name_map[x] for x in _pc_labels_sampled]

    with mo.status.spinner(title=f'Running {_method_pc} ({_n_pc} components)...'):
        if _method_pc == 'PCA':
            _reducer_pc = _PCA_pc(n_components=_n_pc)
        elif _method_pc == 't-SNE':
            _reducer_pc = _TSNE_pc(
                n_components=_n_pc,
                perplexity=max(5, min(30, _pc_n - 1)),
                random_state=42,
            )
        else:
            _reducer_pc = _umap_pc.UMAP(n_components=_n_pc, random_state=42)
        _pc_components = _reducer_pc.fit_transform(_pc_embeddings)

    pc_label_col = _pc_label_col
    df_pc = _pl2r.DataFrame(
        {f'PC{i + 1}': _pc_components[:, i] for i in range(_n_pc)}
    ).with_columns(_pl2r.Series(pc_label_col, _pc_labels_sampled))
    pc_ds_indices = [(_pc_ds_name, int(i)) for i in _pc_sampled['id'].to_list()]
    return df_pc, pc_ds_indices, pc_label_col


@app.cell(hide_code=True)
def _(df_pc, mo, pc_label_col):
    from wigglystuff import ParallelCoordinates as _ParallelCoordinates

    pc_widget = mo.ui.anywidget(
        _ParallelCoordinates(df_pc, height=480, color_by=pc_label_col)
    )
    pc_widget
    return (pc_widget,)


@app.cell(hide_code=True)
def _(
    DATASET_CONFIGS: dict,
    io,
    l_orig_datasets: dict,
    mo,
    pc_ds_indices,
    pc_widget,
    show_images,
):
    from PIL import Image as _PILImage_pc

    _uids = pc_widget.widget.selected_uids
    mo.stop(
        not _uids,
        mo.callout(
            mo.md('**Brush an axis** — images update as you drag.'), kind='info'
        ),
    )

    _filtered = sorted(int(u) for u in _uids)
    _sample = _filtered[:10]

    _pc_images_by_ds: dict = {}
    for _row_idx in _sample:
        _pc_ds_name, _pc_orig_idx = pc_ds_indices[_row_idx]
        if _pc_ds_name not in _pc_images_by_ds:
            _pc_images_by_ds[_pc_ds_name] = []
        _pc_images_by_ds[_pc_ds_name].append(_pc_orig_idx)

    _pc_pil_images = []
    _pc_titles = []
    for _pc_ds_name, _pc_ds_indices_list in _pc_images_by_ds.items():
        if _pc_ds_name not in l_orig_datasets:
            continue
        _cfg = DATASET_CONFIGS.get(_pc_ds_name, {})
        _img_field = _cfg.get('data', 'image')
        _orig_df = l_orig_datasets[_pc_ds_name]
        for _orig_idx in _pc_ds_indices_list:
            if _orig_idx >= len(_orig_df):
                continue
            _row = _orig_df.row(_orig_idx, named=True)
            _pc_pil_images.append(
                _PILImage_pc.open(io.BytesIO(_row[_img_field]['bytes']))
            )
            _pc_titles.append(f'{_pc_ds_name}\n#{_orig_idx}')

    mo.stop(
        not _pc_pil_images,
        mo.callout(mo.md('No images available for selection.'), kind='warn'),
    )

    _fig_pc = show_images(_pc_pil_images, _pc_titles)
    _buf_pc = io.BytesIO()
    _fig_pc.savefig(_buf_pc, format='png', bbox_inches='tight', dpi=150)
    _buf_pc.seek(0)
    import matplotlib.pyplot as _plt_pc

    _plt_pc.close(_fig_pc)
    mo.vstack(
        [
            mo.md(
                f'**{len(_filtered)} / {len(pc_ds_indices)} selected** — showing first {len(_sample)}:'
            ),
            mo.image(_buf_pc.getvalue()),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Demo - Latent Space Alignment

    Neural networks trained on the same data but with different architectures, initialisations, or training regimes produce embeddings that are **semantically equivalent yet geometrically incompatible** — a phenomenon known as **semantic mismatch**. Two models may assign the same concept to very similar regions of their respective spaces, yet a direct coordinate comparison yields no meaningful signal because each model defines its own arbitrary basis.

    **Semantic alignment** is the problem of finding a mapping between two such spaces that brings semantically corresponding regions into geometric correspondence — ideally without label supervision and without retraining either model.

    Here we study alignment through **semantic prototypes**: cluster centres that summarise the coarse structure of a latent space. Prototype correspondence is measured via **Jaccard similarity** — the sample-level overlap between clusters — and optimal matching is solved by the **Hungarian algorithm**.

    Given prototype matrices $P_A, P_B \in \mathbb{R}^{k \times d}$, we construct a **Parseval frame** $F = UV^\top$ (thin SVD of $P$) and project embeddings into a shared $k$-dimensional analysis space via $X \mapsto XF^\top$. The Hungarian algorithm then finds the permutation $\sigma : [k] \to [k]$, mapping prototype indices of Model A (rows) to prototype indices of Model B (columns), that maximises

    $$\sigma^* = \arg\max_{\sigma \in S_k} \sum_{i=1}^k J\bigl(i,\, \sigma(i)\bigr),$$

    thereby aligning Model B's prototypes to Model A without any supervision.

    > **Note.** In semantic communication a full semantic alignment pipeline would prepend a **pre-whitening** step — standardising each space to zero mean and unit covariance before frame construction — to remove scale and correlation biases. This demo omits pre-whitening to keep the exposition focused on the alignment mechanism itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Select** a dataset and two models to compare, then set the number of prototypes *k*. "
            "Embeddings for both models are fetched automatically. "
            "The **Jaccard heatmap** shows prototype-level overlap before alignment; "
            "the **scatter plots** show how the Hungarian permutation brings the two spaces into correspondence."
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(l_all_info: dict, l_all_models, mo):
    _align_ds_options = sorted(l_all_info.keys())
    _default_a = "vit_base_patch16_224.augreg_in1k"
    _default_b = "vit_small_patch16_224.augreg_in1k"

    align_dataset_ui = mo.ui.dropdown(
        options=_align_ds_options,
        value="cifar10" if "cifar10" in _align_ds_options else _align_ds_options[0],
        label="Dataset",
    )
    align_model_a_ui = mo.ui.dropdown(
        options=l_all_models,
        value=_default_a if _default_a in l_all_models else l_all_models[0],
        label="Model A",
    )
    align_model_b_ui = mo.ui.dropdown(
        options=l_all_models,
        value=_default_b if _default_b in l_all_models else (l_all_models[1] if len(l_all_models) > 1 else l_all_models[0]),
        label="Model B",
    )
    align_n_proto_ui = mo.ui.slider(
        start=2, stop=20, step=1, value=10, label="Prototypes (k)", show_value=True
    )
    mo.hstack([
        mo.vstack([align_dataset_ui, align_n_proto_ui]),
        align_model_a_ui,
        align_model_b_ui,
    ], justify="start", gap=2)
    return (
        align_dataset_ui,
        align_model_a_ui,
        align_model_b_ui,
        align_n_proto_ui,
    )


@app.cell(hide_code=True)
def _(
    HF_REPO,
    align_dataset_ui,
    align_model_a_ui,
    align_model_b_ui,
    l_all_info: dict,
    mo,
    np,
):
    import polars as _pl_align

    _align_ds = align_dataset_ui.value
    _align_model_a = align_model_a_ui.value
    _align_model_b = align_model_b_ui.value

    mo.stop(
        _align_model_a == _align_model_b,
        mo.callout(mo.md("Please select two **different** models."), kind="warn"),
    )

    _align_mbs = l_all_info.get(_align_ds, {})
    _align_splits = sorted(_align_mbs.keys())
    _align_split = (
        "test" if "test" in _align_splits
        else "val" if "val" in _align_splits
        else _align_splits[0] if _align_splits else "test"
    )

    _uri_a = f"hf://datasets/{HF_REPO}{_align_ds}/{_align_split}/{_align_model_a}/*.parquet"
    _uri_b = f"hf://datasets/{HF_REPO}{_align_ds}/{_align_split}/{_align_model_b}/*.parquet"

    with mo.status.spinner(title=f"Loading embeddings for {_align_ds}/{_align_split}..."):
        _df_a = _pl_align.scan_parquet(_uri_a).limit(1600).collect()
        _df_b = _pl_align.scan_parquet(_uri_b).limit(1600).collect()

    _ids_a = set(_df_a["id"].to_list())
    _ids_b = set(_df_b["id"].to_list())
    _common_ids = sorted(_ids_a & _ids_b)

    _df_a = _df_a.filter(_pl_align.col("id").is_in(_common_ids)).sort("id")
    _df_b = _df_b.filter(_pl_align.col("id").is_in(_common_ids)).sort("id")

    align_X_a = np.array(_df_a["embedding"].to_list(), dtype=np.float32)
    align_X_b = np.array(_df_b["embedding"].to_list(), dtype=np.float32)
    align_true_labels = [int(x) for x in _df_a["label"].to_list()]

    mo.callout(
        mo.md(f"Loaded **{len(align_X_a)}** shared samples · split `{_align_split}` · dim `{align_X_a.shape[1]}`"),
        kind="success",
    )
    return align_X_a, align_X_b


@app.cell(hide_code=True)
def _(align_X_a, align_X_b, align_n_proto_ui, mo, np):
    from sklearn.cluster import KMeans as _KMeans_align
    from scipy.optimize import linear_sum_assignment as _lsa_align

    _k = align_n_proto_ui.value

    with mo.status.spinner(title=f"Running K-Means (k={_k}) on both models..."):
        _km_a = _KMeans_align(n_clusters=_k, random_state=42, n_init="auto").fit(align_X_a)
        _km_b = _KMeans_align(n_clusters=_k, random_state=42, n_init="auto").fit(align_X_b)

    align_cluster_a = _km_a.labels_
    align_cluster_b = _km_b.labels_
    align_proto_a = _km_a.cluster_centers_.astype(np.float32)
    align_proto_b = _km_b.cluster_centers_.astype(np.float32)

    # Parseval frame: orthonormal frame from SVD of prototype matrix
    def _parseval_frame(P):
        U, _, Vh = np.linalg.svd(P, full_matrices=False)
        return (U @ Vh).astype(np.float32)

    align_F_a = _parseval_frame(align_proto_a)  # (k, d)
    align_F_b = _parseval_frame(align_proto_b)

    # Project embeddings into k-dim analysis space
    align_coords_a = align_X_a @ align_F_a.T  # (n, k)
    align_coords_b = align_X_b @ align_F_b.T

    # Jaccard similarity between prototype memberships
    _J = np.zeros((_k, _k), dtype=np.float32)
    _ci_a = {i: set(np.where(align_cluster_a == i)[0].tolist()) for i in range(_k)}
    _ci_b = {i: set(np.where(align_cluster_b == i)[0].tolist()) for i in range(_k)}
    for _i in range(_k):
        for _j in range(_k):
            _u = len(_ci_a[_i] | _ci_b[_j])
            _J[_i, _j] = len(_ci_a[_i] & _ci_b[_j]) / _u if _u else 0.0

    # Hungarian matching: find permutation of B that best aligns with A
    _, _perm = _lsa_align(-_J)
    align_perm = _perm

    # Permute B prototypes and recompute
    align_proto_b_matched = align_proto_b[align_perm]
    align_F_b_matched = _parseval_frame(align_proto_b_matched)
    align_coords_b_matched = align_X_b @ align_F_b_matched.T

    # Permute cluster labels for B
    _inv_perm = np.argsort(align_perm)
    align_cluster_b_matched = _inv_perm[align_cluster_b]

    # MSE before and after alignment
    align_jaccard_matrix = _J
    align_mse_before = float(np.mean((align_coords_a - align_coords_b) ** 2))
    align_mse_after = float(np.mean((align_coords_a - align_coords_b_matched) ** 2))
    align_jaccard_mean = float(np.mean(_J.max(axis=1)))
    return (
        align_coords_a,
        align_coords_b,
        align_coords_b_matched,
        align_jaccard_matrix,
        align_jaccard_mean,
        align_mse_after,
        align_mse_before,
    )


@app.cell(hide_code=True)
def _(
    align_jaccard_matrix,
    align_jaccard_mean,
    align_model_a_ui,
    align_model_b_ui,
    align_mse_after,
    align_mse_before,
    align_n_proto_ui,
    mo,
):
    import plotly.graph_objects as _go_align

    _model_a_name = align_model_a_ui.value
    _model_b_name = align_model_b_ui.value
    _k = align_n_proto_ui.value

    _fig_J = _go_align.Figure(
        data=_go_align.Heatmap(
            z=align_jaccard_matrix.tolist(),
            x=[f"B-{j}" for j in range(_k)],
            y=[f"A-{i}" for i in range(_k)],
            colorscale="magma",
            zmin=0, zmax=1,
            colorbar=dict(title="Jaccard"),
        )
    )
    _fig_J.update_layout(
        title="Jaccard Similarity Matrix",
        xaxis_title="Model B prototypes",
        yaxis_title="Model A prototypes",
        width=500,
        height=500,
        margin=dict(l=60, r=20, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(scaleanchor='x', scaleratio=1),
    )

    _delta_mse = (align_mse_before - align_mse_after) / align_mse_before * 100
    _caption_text = (
        r"**Jaccard similarity** $J(i,j) = \frac{|C_A^i \cap C_B^j|}{|C_A^i \cup C_B^j|}$"
        " measures the sample-level overlap between prototype $i$ of Model A and prototype $j$"
        " of Model B. Each entry ranges from 0 (no shared samples) to 1 (identical membership)."
        " A bright diagonal — or a permuted-diagonal pattern — means both models have learned"
        " semantically consistent clusters. The **Hungarian algorithm** finds the permutation on the $j$ indices"
        r" $\sigma$ that maximises $\sum_i J(i, \sigma(i))$, aligning Model B's prototypes"
        " to Model A without any label supervision.  \n"
        f"\n**Jaccard mean (best match per A-prototype):** {align_jaccard_mean:.3f} · "
        f"**MSE before:** {align_mse_before:.3f} · "
        f"**MSE after:** {align_mse_after:.3f} · "
        f"**reduction:** {_delta_mse:.1f}%"
    )
    _caption = mo.md(_caption_text)

    mo.vstack([
        mo.ui.plotly(_fig_J),
        _caption,
    ])
    return


@app.cell(hide_code=True)
def _(
    align_coords_a,
    align_coords_b,
    align_coords_b_matched,
    align_model_a_ui,
    align_model_b_ui,
    align_n_proto_ui,
    mo,
):
    from sklearn.decomposition import PCA as _PCA_align
    import plotly.graph_objects as _go_align2
    from plotly.subplots import make_subplots as _make_subplots_align

    _k = align_n_proto_ui.value
    _model_a_name = align_model_a_ui.value
    _model_b_name = align_model_b_ui.value

    # Fit PCA on model A coords; transform all spaces with the same projection
    _pca2 = _PCA_align(n_components=2).fit(align_coords_a)
    _xy_a = _pca2.transform(align_coords_a)
    _xy_b_before = _pca2.transform(align_coords_b)
    _xy_b_after = _pca2.transform(align_coords_b_matched)

    _color_a = "#636EFA"
    _color_b = "#EF553B"

    _transparent = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    _fig_scatter = _make_subplots_align(
        rows=1, cols=2,
        subplot_titles=["Before alignment", "After Hungarian alignment"],
    )

    for _col, (_xy_b, _show) in enumerate([
        (_xy_b_before, True), (_xy_b_after, False)
    ], start=1):
        _fig_scatter.add_trace(
            _go_align2.Scatter(
                x=_xy_a[:, 0].tolist(), y=_xy_a[:, 1].tolist(),
                mode="markers",
                marker=dict(size=3, color=_color_a, opacity=0.5),
                name=_model_a_name,
                legendgroup="A",
                showlegend=_show,
            ),
            row=1, col=_col,
        )
        _fig_scatter.add_trace(
            _go_align2.Scatter(
                x=_xy_b[:, 0].tolist(), y=_xy_b[:, 1].tolist(),
                mode="markers",
                marker=dict(size=3, color=_color_b, opacity=0.5),
                name=_model_b_name,
                legendgroup="B",
                showlegend=_show,
            ),
            row=1, col=_col,
        )

    _fig_scatter.update_layout(
        height=420,
        title_text="Analysis-space embeddings projected via PCA(2) of Model A",
        legend=dict(itemsizing="constant"),
        margin=dict(l=40, r=20, t=80, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    _fig_scatter.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    _fig_scatter.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    mo.ui.plotly(_fig_scatter)
    return


if __name__ == "__main__":
    app.run()
