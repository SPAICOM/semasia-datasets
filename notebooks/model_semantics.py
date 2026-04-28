import marimo

__generated_with = '0.23.0'
app = marimo.App(width='columns')


@app.cell
def _():
    import io
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import yaml
    from datasets import load_dataset

    _root = mo.notebook_dir().parent
    sys.path.insert(0, str(_root))
    from src import collect_models_by_split

    HF_REPO = 'spaicom-lab/semantic-'

    DATASETS = [
        'cifar10',
        'cifar100',
        'mnist',
        'fashion_mnist',
        'tiny-imagenet',
        'imagenet-1k',
        'celeba',
        'svhn',
        'oxford-flowers',
    ]

    DISCRETE_COLORS = [
        '#636EFA',
        '#EF553B',
        '#00CC96',
        '#AB63FA',
        '#FFA15A',
        '#19D3F3',
        '#FF6692',
        '#B6E880',
        '#FF97FF',
        '#FECB52',
        '#72B7B2',
        '#E45756',
        '#54A24B',
    ]

    _configs_dir = _root / 'configs' / 'hydra' / 'dataset'
    DATASET_CONFIGS: dict = {}
    for _p in sorted(_configs_dir.glob('*.yaml')):
        _d = yaml.safe_load(_p.read_text())
        DATASET_CONFIGS[_p.stem] = _d
        DATASET_CONFIGS[_p.stem.replace('_', '-')] = _d
        DATASET_CONFIGS[_p.stem.replace('-', '_')] = _d

    DATASET_COLOR = {
        ds: DISCRETE_COLORS[i % len(DISCRETE_COLORS)] for i, ds in enumerate(DATASETS)
    }

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

    return (
        DATASETS,
        DATASET_COLOR,
        DATASET_CONFIGS,
        HF_REPO,
        collect_models_by_split,
        io,
        load_dataset,
        mo,
        np,
        plt,
        show_images,
        torch,
    )


@app.cell(hide_code=True)
def _(DATASETS, HF_REPO, collect_models_by_split, mo):
    with mo.status.spinner(title='Fetching available models for all datasets…'):
        all_info: dict = {}
        for _ds in DATASETS:
            _mbs = collect_models_by_split(HF_REPO + _ds)
            if _mbs:
                all_info[_ds] = _mbs

    mo.stop(
        not all_info,
        mo.callout(
            mo.md('Could not fetch dataset info. Check your HF token.'),
            kind='danger',
        ),
    )

    all_models = sorted(
        set().union(*(set().union(*splits.values()) for splits in all_info.values()))
    )
    return all_info, all_models


@app.cell(hide_code=True)
def _(all_models, mo):
    model_ui = mo.ui.dropdown(
        options=all_models,
        value=all_models[0] if all_models else None,
        label='Model',
    )
    method_ui = mo.ui.dropdown(
        options=['PCA', 't-SNE', 'UMAP'],
        value='UMAP',
        label='Reduction',
    )
    n_samples_ui = mo.ui.slider(
        start=100,
        stop=2000,
        step=100,
        value=500,
        label='Samples / dataset',
        show_value=True,
    )
    return method_ui, model_ui, n_samples_ui


@app.cell(hide_code=True)
def _(all_info: dict, mo, model_ui):
    split_uis: dict = {}
    for _ds, _mbs in all_info.items():
        _avail = sorted(s for s, models in _mbs.items() if model_ui.value in models)
        if _avail:
            _preferred = next(
                (s for s in ['test', 'val', 'validation'] if s in _avail),
                _avail[0],
            )
            split_uis[_ds] = mo.ui.dropdown(
                options=_avail,
                value=_preferred,
                label=_ds,
            )
    available_datasets = list(split_uis.keys())
    return available_datasets, split_uis


@app.cell(hide_code=True)
def _(available_datasets, mo):
    deselect_ui = mo.ui.multiselect(
        options=available_datasets,
        value=['svhn', 'imagenet-1k'],
        label='Deselect datasets',
    )
    return (deselect_ui,)


@app.cell(hide_code=True)
def _(deselect_ui, method_ui, mo, model_ui, n_samples_ui, split_uis: dict):
    mo.vstack(
        [
            mo.md('## Model Semantics Explorer'),
            mo.hstack([model_ui, method_ui, n_samples_ui], gap=2),
            mo.md('**Split per dataset:**'),
            mo.hstack(list(split_uis.values()), gap=2, wrap=True),
            deselect_ui,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    HF_REPO,
    available_datasets,
    deselect_ui,
    load_dataset,
    mo,
    model_ui,
    n_samples_ui,
    np,
    split_uis: dict,
    torch,
):
    selected_datasets = [d for d in available_datasets if d not in deselect_ui.value]

    mo.stop(
        not selected_datasets,
        mo.callout(mo.md('Select at least one dataset.'), kind='warn'),
    )

    _rng = np.random.default_rng(42)
    _all_embeddings = []
    _all_ds_labels = []
    all_sample_info = []

    for _ds in selected_datasets:
        _split = split_uis[_ds].value
        _repo = HF_REPO + _ds
        with mo.status.spinner(title=f'Loading {_ds} ({_split})…'):
            _loaded = load_dataset(_repo, model_ui.value, split=_split).with_format(
                'torch'
            )
        _latent = torch.vstack(list(_loaded['embedding']))
        _n = min(n_samples_ui.value, len(_latent))
        _idx = _rng.choice(len(_latent), size=_n, replace=False)
        _all_embeddings.append(_latent[_idx].numpy())
        _all_ds_labels.extend([_ds] * _n)
        all_sample_info.extend([(_ds, int(i)) for i in _idx])

    X_all = np.vstack(_all_embeddings)
    y_ds = np.array(_all_ds_labels)
    return X_all, all_sample_info, selected_datasets, y_ds


@app.cell(hide_code=True)
def _(
    DATASET_CONFIGS: dict,
    load_dataset,
    mo,
    selected_datasets,
    split_uis: dict,
):
    orig_datasets: dict = {}
    for _ds in selected_datasets:
        _cfg = DATASET_CONFIGS.get(_ds, {})
        _orig_repo = _cfg.get('name', '')
        if not _orig_repo:
            continue
        _split = split_uis[_ds].value
        with mo.status.spinner(title=f'Loading original images for {_ds} ({_split})…'):
            print(_split)
            orig_datasets[_ds] = load_dataset(_orig_repo, split=_split)
    return (orig_datasets,)


@app.cell(hide_code=True)
def _(X_all, method_ui, mo):
    import umap as _umap
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE

    _method = method_ui.value
    with mo.status.spinner(title=f'Running {_method} (2D)…'):
        if _method == 'PCA':
            X2d = _PCA(n_components=2).fit_transform(X_all)
        elif _method == 't-SNE':
            X2d = _TSNE(
                n_components=2, perplexity=30, random_state=42, n_jobs=-1
            ).fit_transform(X_all)
        else:
            X2d = _umap.UMAP(n_components=2, random_state=42).fit_transform(X_all)
    return (X2d,)


@app.cell(hide_code=True)
def _(
    DATASET_COLOR,
    X2d,
    all_sample_info,
    method_ui,
    mo,
    model_ui,
    np,
    selected_datasets,
    y_ds,
):
    import plotly.graph_objects as _go

    _fig = _go.Figure()
    for _ds in selected_datasets:
        _mask = y_ds == _ds
        _global_pts = np.where(_mask)[0]
        _color = DATASET_COLOR[_ds]
        _orig_indices = np.array([all_sample_info[g][1] for g in _global_pts])
        _customdata = np.stack([_global_pts, _orig_indices], axis=1)
        _fig.add_trace(
            _go.Scatter(
                x=X2d[_mask, 0],
                y=X2d[_mask, 1],
                mode='markers',
                name=_ds,
                marker=dict(size=4, opacity=0.7, color=_color),
                customdata=_customdata,
                hovertemplate=f'<b>{_ds}</b><br>orig_idx: %{{customdata[1]}}<br>C1: %{{x:.3f}}  C2: %{{y:.3f}}<extra></extra>',
            )
        )

    _fig.update_layout(
        title=f'{method_ui.value} — {model_ui.value}',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        legend_title='Dataset',
        height=560,
        margin={'t': 50, 'b': 10},
    )
    scatter_chart = mo.ui.plotly(_fig)
    mo.vstack([mo.md('### Scatter Plot'), scatter_chart])
    return (scatter_chart,)


@app.cell(hide_code=True)
def _(
    DATASET_CONFIGS: dict,
    all_sample_info,
    io,
    mo,
    orig_datasets: dict,
    plt,
    scatter_chart,
    selected_datasets,
    show_images,
):
    _pts = scatter_chart.value
    mo.stop(
        not _pts,
        mo.callout(
            mo.md(
                '**Try making a selection with your mouse!**  \n'
                'Box- or lasso-select points on the scatter plot above.'
            ),
            kind='info',
        ),
    )

    _ds_offsets: dict = {}
    for _i, (_info_ds, _) in enumerate(all_sample_info):
        if _info_ds not in _ds_offsets:
            _ds_offsets[_info_ds] = _i

    _images_by_ds: dict = {}
    for _pt in _pts[:10]:
        _curve = _pt.get('curveNumber')
        _point_idx = _pt.get('pointIndex')
        if _curve is None or _point_idx is None:
            continue
        if _curve >= len(selected_datasets):
            continue
        _ds_name = selected_datasets[_curve]
        _global = _ds_offsets.get(_ds_name, 0) + _point_idx
        if _global >= len(all_sample_info):
            continue
        _, _orig_idx = all_sample_info[_global]
        if _ds_name not in _images_by_ds:
            _images_by_ds[_ds_name] = []
        _images_by_ds[_ds_name].append(_orig_idx)

    _pil_images = []
    _titles = []
    for _ds_name, _ds_indices in _images_by_ds.items():
        _cfg = DATASET_CONFIGS.get(_ds_name, {})
        _img_field = _cfg.get('data', 'image')
        if _ds_name not in orig_datasets:
            continue
        _selected = orig_datasets[_ds_name].select(_ds_indices)
        for _row, _oi in zip(_selected, _ds_indices):
            _pil_images.append(_row[_img_field])
            _titles.append(f'{_ds_name}\n#{_oi}')

    mo.stop(
        not _pil_images,
        mo.callout(mo.md('No images available for selection.'), kind='warn'),
    )

    _fig = show_images(_pil_images, _titles)
    _buf = io.BytesIO()
    _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=150)
    _buf.seek(0)
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(f'**Selected {len(_pts)} point(s)** — showing up to 10 images:'),
            mo.image(_buf.getvalue()),
        ]
    )
    return


@app.cell(hide_code=True)
def _(X2d, all_sample_info, mo, model_ui, y_ds):
    import io as _io

    import polars as _pl

    scatter_df = _pl.DataFrame(
        {
            'dataset': list(y_ds),
            'orig_idx': [int(oi) for _, oi in all_sample_info],
            'comp_1': X2d[:, 0].tolist(),
            'comp_2': X2d[:, 1].tolist(),
        }
    )

    _buf = _io.BytesIO()
    scatter_df.write_parquet(_buf)

    mo.vstack(
        [
            mo.md(f'**Scatter data** — {len(scatter_df):,} rows'),
            mo.download(
                data=_buf.getvalue(),
                filename=f'scatter_{model_ui.value}.parquet',
                mimetype='application/octet-stream',
                label='Download parquet',
            ),
            mo.ui.table(scatter_df.head(10)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    pc_method_ui = mo.ui.dropdown(
        options=['PCA', 't-SNE', 'UMAP'],
        value='UMAP',
        label='Reduction',
    )
    return (pc_method_ui,)


@app.cell(hide_code=True)
def _(mo, pc_method_ui):
    _max_dims = 3 if pc_method_ui.value == 't-SNE' else 15
    n_dims_ui = mo.ui.slider(
        start=2,
        stop=_max_dims,
        step=1,
        value=min(6, _max_dims),
        label='Components',
        show_value=True,
    )
    mo.vstack(
        [
            mo.md('### Parallel Coordinates'),
            mo.hstack([pc_method_ui, n_dims_ui], gap=2),
        ]
    )
    return (n_dims_ui,)


@app.cell(hide_code=True)
def _(X_all, mo, n_dims_ui, pc_method_ui, y_ds):
    import polars as _pl
    import umap as _umap_pc
    from sklearn.decomposition import PCA as _PCA_pc
    from sklearn.manifold import TSNE as _TSNE_pc

    _method_pc = pc_method_ui.value
    _n_pc = n_dims_ui.value

    with mo.status.spinner(
        title=f'Running {_method_pc} ({_n_pc}D) for parallel coords…'
    ):
        if _method_pc == 'PCA':
            _comps = _PCA_pc(n_components=_n_pc).fit_transform(X_all)
        elif _method_pc == 't-SNE':
            _comps = _TSNE_pc(
                n_components=_n_pc,
                perplexity=max(5, min(30, len(X_all) - 1)),
                random_state=42,
            ).fit_transform(X_all)
        else:
            _comps = _umap_pc.UMAP(n_components=_n_pc, random_state=42).fit_transform(
                X_all
            )

    df_pc = _pl.DataFrame(
        {f'PC{i + 1}': _comps[:, i] for i in range(_n_pc)}
    ).with_columns(_pl.Series('dataset', list(y_ds)))
    return (df_pc,)


@app.cell(hide_code=True)
def _(df_pc, mo):
    from wigglystuff import ParallelCoordinates as _ParallelCoordinates

    pc_widget = mo.ui.anywidget(
        _ParallelCoordinates(df_pc, height=500, color_by='dataset')
    )
    pc_widget
    return (pc_widget,)


@app.cell(hide_code=True)
def _(mo):

    fmt_ui = mo.ui.dropdown(
        options=['pdf', 'png', 'svg'],
        value='pdf',
        label='Format',
    )
    fmt_ui
    return (fmt_ui,)


@app.cell(hide_code=True)
def _(df_pc, mo):
    _ds_options = sorted(df_pc['dataset'].unique().to_list())
    class_select_ui = mo.ui.multiselect(
        options=_ds_options,
        value=_ds_options,
        label='Classes',
    )
    class_select_ui
    return (class_select_ui,)


@app.cell(hide_code=True)
def _(class_select_ui, df_pc, fmt_ui, mo, n_dims_ui, pc_method_ui):
    import importlib as _importlib
    import io as _io_pc

    import matplotlib.pyplot as _plt_pc

    import src.visualizations.parallel_coordinates as _pc_mod

    _importlib.reload(_pc_mod)
    make_parallel_coordinates = _pc_mod.make_parallel_coordinates

    _fmt_mime = {
        'pdf': 'application/pdf',
        'png': 'image/png',
        'svg': 'image/svg+xml',
    }

    def _make_pc_download():
        _fmt = fmt_ui.value
        _fig = make_parallel_coordinates(
            df_pc,
            pc_method_ui.value,
            n_dims_ui.value,
            selected_labels=class_select_ui.value if class_select_ui.value else None,
        )
        _buf = _io_pc.BytesIO()
        _fig.savefig(_buf, format=_fmt, bbox_inches='tight', dpi=150)
        _buf.seek(0)
        _plt_pc.close(_fig)
        return _buf.read()

    def _make_pc_pdf():
        _fig = make_parallel_coordinates(
            df_pc,
            pc_method_ui.value,
            n_dims_ui.value,
            selected_labels=class_select_ui.value if class_select_ui.value else None,
        )
        _data = _pc_mod.to_pdf(_fig)
        _plt_pc.close(_fig)
        return _data

    # preview
    _selected = class_select_ui.value if class_select_ui.value else None
    _fig_prev = make_parallel_coordinates(
        df_pc, pc_method_ui.value, n_dims_ui.value, selected_labels=_selected
    )
    _buf_prev = _io_pc.BytesIO()
    _fig_prev.savefig(_buf_prev, format='png', bbox_inches='tight', dpi=100)
    _buf_prev.seek(0)
    _plt_pc.close(_fig_prev)

    _fname = f'parallel_coordinates_{pc_method_ui.value}_{n_dims_ui.value}d'

    mo.vstack(
        [
            mo.image(_buf_prev.getvalue()),
            mo.hstack(
                [
                    mo.download(
                        data=_make_pc_pdf,
                        filename=f'{_fname}.pdf',
                        mimetype='application/pdf',
                        label='Save as PDF',
                    ),
                    mo.download(
                        data=_make_pc_download,
                        filename=f'{_fname}.{fmt_ui.value}',
                        mimetype=_fmt_mime[fmt_ui.value],
                        label='Save as…',
                    ),
                ],
                align='center',
                gap=1,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    DATASET_CONFIGS: dict,
    all_sample_info,
    io,
    mo,
    orig_datasets: dict,
    pc_widget,
    plt,
    show_images,
):
    _uids = pc_widget.widget.selected_uids
    mo.stop(
        not _uids,
        mo.callout(
            mo.md('**Brush an axis** — images update as you drag.'),
            kind='info',
        ),
    )

    _filtered = sorted(int(u) for u in _uids)
    _sample = _filtered[:10]

    _pc_images_by_ds: dict = {}
    for _row_idx in _sample:
        _pc_ds_name, _pc_orig_idx = all_sample_info[_row_idx]
        if _pc_ds_name not in _pc_images_by_ds:
            _pc_images_by_ds[_pc_ds_name] = []
        _pc_images_by_ds[_pc_ds_name].append(_pc_orig_idx)

    _pc_pil_images = []
    _pc_titles = []
    for _pc_ds_name, _pc_ds_indices in _pc_images_by_ds.items():
        _cfg = DATASET_CONFIGS.get(_pc_ds_name, {})
        _img_field = _cfg.get('data', 'image')
        if _pc_ds_name not in orig_datasets:
            continue
        _selected = orig_datasets[_pc_ds_name].select(_pc_ds_indices)
        for _row, _orig_idx in zip(_selected, _pc_ds_indices):
            _pc_pil_images.append(_row[_img_field])
            _pc_titles.append(f'{_pc_ds_name}\n#{_orig_idx}')

    mo.stop(
        not _pc_pil_images,
        mo.callout(mo.md('No images available for selection.'), kind='warn'),
    )

    _fig = show_images(_pc_pil_images, _pc_titles)
    _buf = io.BytesIO()
    _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=150)
    _buf.seek(0)
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(
                f'**{len(_filtered)} / {len(all_sample_info)} selected** — '
                f'showing first {len(_sample)}:'
            ),
            mo.image(_buf.getvalue()),
        ]
    )
    return


if __name__ == '__main__':
    app.run()
