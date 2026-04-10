import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import io
    import sys
    import yaml
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datasets import load_dataset

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.huggingface import collect_models_by_split

    HF_REPO = 'spaicom-lab/semantic-'

    # ── class-name registry ───────────────────────────────────────────────────
    CLASS_NAMES: dict[str, list[str]] = {
        'cifar10': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck',
        ],
        'cifar100:fine_label': [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm',
        ],
        'cifar100:coarse_label': [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects',
            'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
            'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates',
            'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2',
        ],
        'mnist': [str(i) for i in range(10)],
        'fashion_mnist': [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
        ],
    }

    DISCRETE_COLORS = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
        '#72B7B2', '#E45756', '#54A24B', '#EECA3B', '#B279A2',
        '#FF9DA7', '#9D755D', '#BAB0AC', '#4C78A8', '#F58518',
    ]

    # ── load dataset configs from YAML ────────────────────────────────────────
    _configs_dir = Path(__file__).parent.parent / 'configs' / 'hydra' / 'dataset'
    DATASET_CONFIGS: dict[str, dict] = {}
    for _p in sorted(_configs_dir.glob('*.yaml')):
        _d = yaml.safe_load(_p.read_text())
        DATASET_CONFIGS[_p.stem] = _d
        # register both underscore and hyphen variants so lookups work either way
        DATASET_CONFIGS[_p.stem.replace('_', '-')] = _d
        DATASET_CONFIGS[_p.stem.replace('-', '_')] = _d

    def get_class_names(dataset: str, col: str) -> list[str]:
        return CLASS_NAMES.get(f'{dataset}:{col}', CLASS_NAMES.get(dataset, []))

    def extract_label(ds, col: str) -> np.ndarray:
        """Return a 1-D integer label array.

        Handles the SVHN struct label {'digit': [d0, d1, ...]} → int(d0d1...).
        Falls back to the standard flatten+astype path for scalar columns.
        """
        sample = ds[0][col]
        if isinstance(sample, dict) and 'digit' in sample:
            return np.array([
                int(''.join(str(int(d.item()) if hasattr(d, 'item') else int(d)) for d in row[col]['digit']))
                for row in ds
            ])
        return np.array(ds[col]).flatten().astype(int)

    # Some original datasets use a different split name than 'test'
    ORIG_SPLITS: dict[str, str] = {
        'imagenet-1k': 'validation',
        'tiny-imagenet': 'valid',
        'tiny_imagenet': 'valid',
    }


@app.cell
def _():
    dataset_ui = mo.ui.dropdown(
        options=[
            'cifar10',
            'cifar100',
            'mnist',
            'fashion_mnist',
            'tiny-imagenet',
            'imagenet-1k',
            'celeba',
            'svhn',
        ],
        value='cifar10',
        label='Dataset',
    )
    method_ui = mo.ui.dropdown(
        options=['PCA', 't-SNE', 'UMAP'],
        value='UMAP',
        label='Reduction',
    )
    n_samples_ui = mo.ui.slider(
        start=200,
        stop=5000,
        step=100,
        value=1000,
        label='Samples',
        show_value=True,
    )
    return dataset_ui, method_ui, n_samples_ui


@app.cell
def _(dataset_ui):
    _repo = HF_REPO + dataset_ui.value
    with mo.status.spinner(title=f'Fetching available splits/models for {_repo}…'):
        models_by_split = collect_models_by_split(_repo)
    mo.stop(
        not models_by_split,
        mo.callout(mo.md(f'Could not fetch repo info for **{_repo}**. Check your HF token.'), kind='danger'),
    )
    _splits = sorted(models_by_split.keys())
    split_ui = mo.ui.dropdown(options=_splits, value=_splits[0], label='Split')
    return models_by_split, split_ui


@app.cell
def _(models_by_split, split_ui):
    _models = sorted(models_by_split.get(split_ui.value, set()))
    mo.stop(
        not _models,
        mo.callout(mo.md(f'No models found for split **{split_ui.value}**.'), kind='warn'),
    )
    model_ui = mo.ui.dropdown(options=_models, value=_models[0], label='Model')
    return (model_ui,)


@app.cell
def _(attr_ui, dataset_ui, method_ui, model_ui, n_samples_ui, split_ui):
    mo.vstack([
        mo.hstack([dataset_ui, split_ui, model_ui, attr_ui], gap=2),
        mo.hstack([method_ui, n_samples_ui], gap=2),
    ])


@app.cell
def _(dataset_ui, model_ui, split_ui):
    _repo = HF_REPO + dataset_ui.value
    with mo.status.spinner(title=f'Loading {_repo} ({split_ui.value} split)…'):
        ds = load_dataset(_repo, model_ui.value, split=split_ui.value).with_format('torch')
    mo.callout(
        mo.md(f'**{dataset_ui.value}** `{split_ui.value}` — {len(ds):,} samples  |  embedding dim: {ds[0]["embedding"].shape[-1]}'),
        kind='success',
    )
    return (ds,)


@app.cell
def _(dataset_ui, ds):
    _yaml_extras = DATASET_CONFIGS.get(dataset_ui.value, {}).get('extras', ['label'])
    _available = [c for c in _yaml_extras if c in ds.column_names]
    if not _available:
        _available = ['label']
    attr_ui = mo.ui.dropdown(options=_available, value=_available[0], label='Color by')
    return (attr_ui,)


@app.cell
def _(dataset_ui, split_ui):
    _cfg = DATASET_CONFIGS.get(dataset_ui.value, {})
    _orig_repo = _cfg.get('name', '')
    with mo.status.spinner(title=f'Loading original images from {_orig_repo}…'):
        orig_ds = load_dataset(_orig_repo, split=split_ui.value)
    return (orig_ds,)


@app.cell
def _(attr_ui, dataset_ui, ds):
    latent = torch.vstack(list(ds['embedding']))

    label = extract_label(ds, attr_ui.value)
    return label, latent


@app.cell
def _(label, latent, method_ui, n_samples_ui):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap as _umap

    n = min(n_samples_ui.value, len(latent))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(latent), size=n, replace=False)

    X = latent[idx].numpy()
    y = label[idx]

    _method = method_ui.value
    with mo.status.spinner(title=f'Running {_method} on {n:,} points…'):
        if _method == 'PCA':
            X2d = PCA(n_components=2).fit_transform(X)
        elif _method == 't-SNE':
            X2d = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(X)
        else:
            X2d = _umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    return X2d, idx, n, y


@app.cell
def _(X2d, attr_ui, dataset_ui, idx, method_ui, n, y):
    import plotly.graph_objects as go

    _ds_name = dataset_ui.value
    _class_names = get_class_names(_ds_name, attr_ui.value)
    _unique = sorted(set(y.tolist()))
    _many_classes = len(_unique) > 20

    # trace_to_global[curve_idx] = array of global point indices for that trace
    # (global = index into X2d / y / idx arrays)
    trace_to_global = []

    _fig = go.Figure()

    if _many_classes:
        # Single trace — pointIndex IS the global index
        trace_to_global.append(np.arange(len(y)))
        _fig.add_trace(go.Scatter(
            x=X2d[:, 0],
            y=X2d[:, 1],
            mode='markers',
            marker=dict(
                size=4,
                opacity=0.7,
                color=y,
                colorscale='Turbo',
                showscale=True,
                colorbar=dict(title='Label'),
            ),
            customdata=np.stack([idx, y], axis=1),
            hovertemplate=(
                'idx: %{customdata[0]}<br>'
                'label: %{customdata[1]}<br>'
                'C1: %{x:.3f}  C2: %{y:.3f}'
                '<extra></extra>'
            ),
            showlegend=False,
        ))
    else:
        for _lbl in _unique:
            _mask = y == _lbl
            _global_pts = np.where(_mask)[0]
            trace_to_global.append(_global_pts)
            _name = _class_names[_lbl] if _lbl < len(_class_names) else str(_lbl)
            _fig.add_trace(go.Scatter(
                x=X2d[_mask, 0],
                y=X2d[_mask, 1],
                mode='markers',
                name=_name,
                marker=dict(
                    size=5,
                    opacity=0.75,
                    color=DISCRETE_COLORS[_lbl % len(DISCRETE_COLORS)],
                ),
                customdata=idx[_global_pts].reshape(-1, 1),
                hovertemplate=(
                    f'<b>{_name}</b><br>'
                    'idx: %{customdata[0]}<br>'
                    'C1: %{x:.3f}  C2: %{y:.3f}'
                    '<extra></extra>'
                ),
            ))

    _fig.update_layout(
        title=f'{method_ui.value} — {_ds_name} test ({n:,} samples)',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        legend_title='Class',
        height=580,
        margin=dict(t=50, b=10),
    )

    chart = mo.ui.plotly(_fig)
    chart
    return chart, trace_to_global


@app.function
def show_images(pil_images, titles, max_images=10):
    pil_images = pil_images[:max_images]
    titles = titles[:max_images]
    n = len(pil_images)
    if n == 0:
        return None
    fig, axes = plt.subplots(1, n, squeeze=False)
    fig.set_size_inches(n * 1.6, 2.0)
    for ax, img, title in zip(axes[0], pil_images, titles):
        ax.imshow(img if img.mode == 'RGB' else img.convert('RGB'))
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=0.4)
    return fig


@app.cell
def _(X2d, attr_ui, chart, dataset_ui, idx, orig_ds, trace_to_global, y):
    import polars as pl

    _ds_name = dataset_ui.value
    _class_names = get_class_names(_ds_name, attr_ui.value)
    _img_field = DATASET_CONFIGS.get(_ds_name, {}).get('data', 'image')

    _pts = chart.value
    mo.stop(not _pts, mo.callout(
        mo.md('**Try making a selection with your mouse!**  \nBox- or lasso-select points on the scatter plot above.'),
        kind='info',
    ))

    # ── resolve selection via curveNumber + pointIndex ────────────────────────
    _rows = []
    _ds_indices = []
    for _pt in _pts:
        _curve = int(_pt.get('curveNumber', 0))
        _pt_idx = int(_pt.get('pointIndex', _pt.get('pointNumber', 0)))
        _global = int(trace_to_global[_curve][_pt_idx])
        _ds_idx = int(idx[_global])
        _lbl = int(y[_global])
        _ds_indices.append(_ds_idx)
        _rows.append({
            'dataset_idx': _ds_idx,
            'class': _class_names[_lbl] if 0 <= _lbl < len(_class_names) else str(_lbl),
            'label': _lbl,
            'component_1': round(float(X2d[_global, 0]), 4),
            'component_2': round(float(X2d[_global, 1]), 4),
        })

    # ── fetch first 10 original images ───────────────────────────────────────
    _selected = orig_ds.select(_ds_indices[:10])
    _pil_images = [_row[_img_field] for _row in _selected]
    _titles = [f'{r["class"]}\n#{r["dataset_idx"]}' for r in _rows[:10]]
    _fig = show_images(_pil_images, _titles)

    _buf = io.BytesIO()
    _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=150)
    _buf.seek(0)
    plt.close(_fig)

    # ── render ────────────────────────────────────────────────────────────────
    _df = pl.DataFrame(_rows)
    mo.vstack([
        mo.md(f'**Here\'s a preview of the images you\'ve selected** ({len(_rows)} total):'),
        mo.image(_buf.getvalue()),
        mo.ui.table(_df, selection=None),
    ])
    return


@app.cell
def _():
    pc_method_ui = mo.ui.dropdown(
        options=['PCA', 't-SNE', 'UMAP'],
        value='UMAP',
        label='Reduction',
    )
    return (pc_method_ui,)


@app.cell
def _(pc_method_ui):
    _max = 3 if pc_method_ui.value == 't-SNE' else 15
    n_dims_ui = mo.ui.slider(
        start=2,
        stop=_max,
        step=1,
        value=min(6, _max),
        label='Components',
        show_value=True,
    )
    mo.hstack([mo.md('### Parallel Coordinates'), pc_method_ui, n_dims_ui], gap=2)
    return (n_dims_ui,)


@app.cell
def _(attr_ui, dataset_ui, idx, latent, n_dims_ui, pc_method_ui, y):
    import umap as _umap_mod
    import polars as _pl
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE

    _ds_name = dataset_ui.value
    _class_names = get_class_names(_ds_name, attr_ui.value)
    _method = pc_method_ui.value
    _n = n_dims_ui.value
    _X_raw = latent[idx].numpy()

    with mo.status.spinner(title=f'Running {_method} ({_n} components) for parallel coords…'):
        if _method == 'PCA':
            _reducer = _PCA(n_components=_n)
        elif _method == 't-SNE':
            _reducer = _TSNE(
                n_components=_n,
                perplexity=max(5, min(30, len(idx) - 1)),
                random_state=42,
            )
        else:
            _reducer = _umap_mod.UMAP(n_components=_n, random_state=42)
        _components = _reducer.fit_transform(_X_raw)

    pc_label_col = attr_ui.value
    df_pc = _pl.DataFrame(
        {f'PC{i + 1}': _components[:, i] for i in range(_n)}
    ).with_columns(_pl.Series(pc_label_col, [int(lbl) for lbl in y]))
    sel_ds_indices = [int(i) for i in idx]
    return df_pc, pc_label_col, sel_ds_indices


@app.cell
def _(df_pc, pc_label_col):
    from wigglystuff import ParallelCoordinates as _ParallelCoordinates

    pc_widget = mo.ui.anywidget(_ParallelCoordinates(df_pc, height=500, color_by=pc_label_col))
    pc_widget
    return (pc_widget,)


@app.cell
def _(dataset_ui, df_pc, orig_ds, pc_label_col, pc_widget, sel_ds_indices):
    # selected_uids is synced (sync=True) and updates live as you brush axes.
    # filtered_indices is NOT synced, so marimo cannot observe it.
    _uids = pc_widget.widget.selected_uids
    mo.stop(
        not _uids,
        mo.callout(
            mo.md('**Brush an axis** — images update as you drag.'),
            kind='info',
        ),
    )

    _filtered = sorted(int(u) for u in _uids)
    _img_field = DATASET_CONFIGS.get(dataset_ui.value, {}).get('data', 'image')
    _sample = _filtered[:10]
    _ds_indices = [sel_ds_indices[i] for i in _sample]

    _selected = orig_ds.select(_ds_indices)
    _pil_images = [_row[_img_field] for _row in _selected]
    _pc_class_names = get_class_names(dataset_ui.value, pc_label_col)
    _titles = [
        f'{_pc_class_names[df_pc[pc_label_col][_sample[j]]] if df_pc[pc_label_col][_sample[j]] < len(_pc_class_names) else str(df_pc[pc_label_col][_sample[j]])}\n#{_ds_indices[j]}'
        for j in range(len(_ds_indices))
    ]
    _fig = show_images(_pil_images, _titles)

    _buf = io.BytesIO()
    _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=150)
    _buf.seek(0)
    plt.close(_fig)

    mo.vstack([
        mo.md(f'**{len(_filtered)} / {len(sel_ds_indices)} selected** — showing first {len(_sample)}:'),
        mo.image(_buf.getvalue()),
    ])


if __name__ == "__main__":
    app.run()
