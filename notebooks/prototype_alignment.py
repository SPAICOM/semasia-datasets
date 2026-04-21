import marimo

__generated_with = '0.23.0'
app = marimo.App(width='columns')


@app.cell
def _():
    import sys

    import marimo as mo
    import numpy as np
    import torch
    import yaml
    from datasets import load_dataset

    _root = mo.notebook_dir().parent
    sys.path.insert(0, str(_root))

    for _mod in list(sys.modules):
        if _mod.startswith('src'):
            del sys.modules[_mod]

    from src import collect_models_by_split
    from src.metrics.alignment import (
        align_prototypes,
        jaccard_prototype_similarity,
    )
    from src.objects import LatentSpace

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

    SINGLE_SPLIT_DATASETS = {'tiny-imagenet'}

    _configs_dir = _root / 'configs' / 'hydra' / 'dataset'
    DATASET_CONFIGS: dict = {}
    for _p in sorted(_configs_dir.glob('*.yaml')):
        _d = yaml.safe_load(_p.read_text())
        DATASET_CONFIGS[_p.stem] = _d
        DATASET_CONFIGS[_p.stem.replace('_', '-')] = _d
        DATASET_CONFIGS[_p.stem.replace('-', '_')] = _d
    return (
        DATASETS,
        HF_REPO,
        LatentSpace,
        SINGLE_SPLIT_DATASETS,
        align_prototypes,
        collect_models_by_split,
        jaccard_prototype_similarity,
        load_dataset,
        mo,
        np,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    This notebook compares **prototype representations** between two models on the same dataset.

    **Prototype Clustering**: Each model's latent space $\mathcal{L} \subset \mathbb{R}^d$ is partitioned 
    into $k$ clusters via $k$-means, yielding prototype vectors $\mathbf{p}_1, \ldots, \mathbf{p}_k \in \mathbb{R}^d$.

    **Hungarian Matching**: Given a Jaccard similarity matrix $J_{ij} = \frac{|C_i \cap C_j|}{|C_i \cup C_j|}$ 
    between model A clusters $C_i$ and model B clusters $C_j$, we solve the optimal permutation via the Hungarian algorithm:
    $$\sigma^* = \arg\max_\sigma \sum_{i} J_{i,\sigma(i)}$$

    **Mean Squared Error (MSE)**: Measures the average squared distance between aligned prototype representations:
    $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(a_i - b_i)^2$$

    **Artificial Prototypes**: We apply model A's cluster structure to model B's latent space, then project 
    both models into their respective prototype spaces for comparison.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(DATASETS, mo):
    dataset_ui = mo.ui.multiselect(
        options=DATASETS,
        value=['cifar10'],
        label='Datasets',
    )
    return (dataset_ui,)


@app.cell(hide_code=True)
def _(HF_REPO, collect_models_by_split, dataset_ui, mo):
    _all_models_by_dataset = {}
    _all_splits = set()

    for _ds_name in dataset_ui.value:
        _repo = HF_REPO + _ds_name
        with mo.status.spinner(title=f'Fetching models for {_repo}…'):
            models_by_split = collect_models_by_split(_repo)
        if models_by_split:
            _all_models_by_dataset[_ds_name] = models_by_split
            _all_splits.update(models_by_split.keys())
        else:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        f'Could not fetch repo info for **{_repo}**. Check your HF token.'
                    ),
                    kind='danger',
                ),
            )

    _common_models = None
    if _all_models_by_dataset:
        _common_models = set.intersection(
            *(
                set().union(*models_by_split.values())
                for models_by_split in _all_models_by_dataset.values()
            )
        )

    available_splits = sorted(_all_splits)
    all_models = sorted(_common_models) if _common_models else []
    splits_by_dataset = {
        ds: list(ms.keys()) for ds, ms in _all_models_by_dataset.items()
    }
    return all_models, available_splits, splits_by_dataset


@app.cell(hide_code=True)
def _(all_models, mo):
    model_a_ui = mo.ui.dropdown(
        options=all_models,
        value='vit_base_patch16_224.augreg_in1k',
        label='Model A (reference)',
    )
    model_b_ui = mo.ui.dropdown(
        options=all_models,
        value='vit_base_patch16_224.augreg_in21k',
        label='Model B (target)',
    )
    n_prototypes_ui = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=20,
        label='N Prototypes',
        show_value=True,
    )
    return model_a_ui, model_b_ui, n_prototypes_ui


@app.cell(hide_code=True)
def _(available_splits, mo):
    align_split_ui = mo.ui.dropdown(
        options=available_splits,
        value='train' if 'train' in available_splits else available_splits[0],
        label='Alignment split',
    )
    return (align_split_ui,)


@app.cell(hide_code=True)
def _(SINGLE_SPLIT_DATASETS, align_split_ui, available_splits, dataset_ui, mo):
    _is_single = any(ds in SINGLE_SPLIT_DATASETS for ds in dataset_ui.value)
    if _is_single:
        test_split_ui = mo.ui.dropdown(
            options=available_splits,
            value=available_splits[0],
            label='Split (auto 80/20 split for single-split datasets)',
        )
    else:
        _other = [s for s in available_splits if s != align_split_ui.value]
        if not _other:
            _other = available_splits
        _pref = next(
            (s for s in ['test', 'val', 'validation'] if s in _other),
            _other[0],
        )
        test_split_ui = mo.ui.dropdown(
            options=_other,
            value=_pref,
            label='Test split',
        )
    return (test_split_ui,)


@app.cell(hide_code=True)
def _(
    SINGLE_SPLIT_DATASETS,
    align_split_ui,
    dataset_ui,
    mo,
    model_a_ui,
    model_b_ui,
    n_prototypes_ui,
    test_split_ui,
):
    _is_single = any(ds in SINGLE_SPLIT_DATASETS for ds in dataset_ui.value)
    _single_datasets = [ds for ds in dataset_ui.value if ds in SINGLE_SPLIT_DATASETS]
    _split_note = (
        mo.callout(
            mo.md(
                f'**{", ".join(_single_datasets)}** have only one split — the selected split will be '
                'automatically divided **80% alignment / 20% test**.'
            ),
            kind='info',
        )
        if _single_datasets
        else mo.md('')
    )
    mo.vstack(
        [
            mo.md('## Prototype Alignment Explorer'),
            mo.hstack([dataset_ui, align_split_ui, test_split_ui], gap=2),
            mo.hstack([model_a_ui, model_b_ui, n_prototypes_ui], gap=2),
            _split_note,
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    HF_REPO,
    SINGLE_SPLIT_DATASETS,
    align_split_ui,
    dataset_ui,
    load_dataset,
    mo,
    model_a_ui,
    model_b_ui,
    np,
    splits_by_dataset,
    test_split_ui,
    torch,
):
    _la_align_list, _lb_align_list = [], []
    _la_test_list, _lb_test_list = [], []
    _dataset_info = []

    for _ds_name in dataset_ui.value:
        _repo = HF_REPO + _ds_name
        _is_single = _ds_name in SINGLE_SPLIT_DATASETS

        _ds_splits = splits_by_dataset.get(_ds_name, [])
        _align_split = (
            align_split_ui.value
            if align_split_ui.value in _ds_splits
            else (
                next(
                    (s for s in ['train', 'validation'] if s in _ds_splits),
                    _ds_splits[0],
                )
            )
        )
        _test_split = (
            test_split_ui.value
            if test_split_ui.value in _ds_splits
            else (
                next(
                    (s for s in ['test', 'validation'] if s in _ds_splits),
                    _ds_splits[0],
                )
            )
        )

        with mo.status.spinner(
            title=f'Loading {model_a_ui.value} ({_ds_name}, {_align_split})…'
        ):
            _ds_a = load_dataset(
                _repo, model_a_ui.value, split=_align_split
            ).with_format('torch')
        with mo.status.spinner(
            title=f'Loading {model_b_ui.value} ({_ds_name}, {_align_split})…'
        ):
            _ds_b = load_dataset(
                _repo, model_b_ui.value, split=_align_split
            ).with_format('torch')

        _la_full = torch.vstack(list(_ds_a['embedding'])).float().numpy()
        _lb_full = torch.vstack(list(_ds_b['embedding'])).float().numpy()

        if _is_single:
            _rng = np.random.default_rng(42)
            _idx = _rng.permutation(len(_la_full))
            _split_at = int(0.8 * len(_la_full))
            _train_idx, _test_idx = _idx[:_split_at], _idx[_split_at:]
            _la_align_list.append(_la_full[_train_idx])
            _lb_align_list.append(_lb_full[_train_idx])
            _la_test_list.append(_la_full[_test_idx])
            _lb_test_list.append(_lb_full[_test_idx])
            _dataset_info.append(
                f'{_ds_name}: {_train_idx.shape[0]:,} align / {_test_idx.shape[0]:,} test'
            )
        else:
            _la_align_list.append(_la_full)
            _lb_align_list.append(_lb_full)
            with mo.status.spinner(
                title=f'Loading {model_a_ui.value} ({_ds_name}, {_test_split})…'
            ):
                _ds_a_test = load_dataset(
                    _repo, model_a_ui.value, split=_test_split
                ).with_format('torch')
            with mo.status.spinner(
                title=f'Loading {model_b_ui.value} ({_ds_name}, {_test_split})…'
            ):
                _ds_b_test = load_dataset(
                    _repo, model_b_ui.value, split=_test_split
                ).with_format('torch')
            _la_test_list.append(
                torch.vstack(list(_ds_a_test['embedding'])).float().numpy()
            )
            _lb_test_list.append(
                torch.vstack(list(_ds_b_test['embedding'])).float().numpy()
            )
            _dataset_info.append(
                f'{_ds_name}: {_la_full.shape[0]:,} align / {_la_test_list[-1].shape[0]:,} test'
            )

    la_align = np.concatenate(_la_align_list)
    lb_align = np.concatenate(_lb_align_list)
    la_test = np.concatenate(_la_test_list)
    lb_test = np.concatenate(_lb_test_list)

    _info = mo.callout(
        mo.md(
            f'**Concatenated ({len(dataset_ui.value)} datasets)**: '
            f'{len(la_align):,} align / {len(la_test):,} test | '
            f'dim A: {la_align.shape[1]} | dim B: {lb_align.shape[1]}'
        ),
        kind='success',
    )

    _info
    return la_align, la_test, lb_align, lb_test


@app.cell(hide_code=True)
def _(
    LatentSpace,
    la_align,
    lb_align,
    mo,
    model_a_ui,
    model_b_ui,
    n_prototypes_ui,
    parseval_ui,
    prewhiten_ui,
):
    from sklearn.cluster import KMeans as _KMeans

    _n_proto = n_prototypes_ui.value

    with mo.status.spinner(
        title=f'Computing {_n_proto} prototypes for {model_a_ui.value}…'
    ):
        ls_a = LatentSpace(la_align, seed=42)
        if prewhiten_ui.value:
            ls_a.prewhiten(inplace=True)
        _, cluster_indices_a = ls_a.compute_prototypes(
            n_samples=None,
            clusterer_cls=_KMeans,
            n_clusters=_n_proto,
            apply_parseval=parseval_ui.value,
            return_cluster_indices=True,
        )

    with mo.status.spinner(
        title=f'Computing {_n_proto} prototypes for {model_b_ui.value}…'
    ):
        ls_b = LatentSpace(lb_align, seed=42)
        if prewhiten_ui.value:
            ls_b.prewhiten(inplace=True)
        _, cluster_indices_b = ls_b.compute_prototypes(
            n_samples=None,
            clusterer_cls=_KMeans,
            n_clusters=_n_proto,
            apply_parseval=parseval_ui.value,
            return_cluster_indices=True,
        )

    _op_type = 'Parseval' if parseval_ui.value else 'Raw'
    mo.callout(
        mo.md(
            f'Prototypes computed ({_op_type}) — '
            f'**{model_a_ui.value}**: {ls_a.prototypes.shape} | '
            f'**{model_b_ui.value}**: {ls_b.prototypes.shape}'
        ),
        kind='success',
    )
    return cluster_indices_a, cluster_indices_b, ls_a, ls_b


@app.cell(hide_code=True)
def _(
    cluster_indices_a,
    cluster_indices_b,
    jaccard_prototype_similarity,
    mo,
    model_a_ui,
    model_b_ui,
):
    import plotly.express as _px

    _sim = jaccard_prototype_similarity(cluster_indices_a, cluster_indices_b)

    _fig = _px.imshow(
        _sim,
        color_continuous_scale='Viridis',
        zmin=0,
        zmax=1,
        labels={
            'x': f'{model_b_ui.value} Prototypes',
            'y': f'{model_a_ui.value} Prototypes',
            'color': 'Jaccard',
        },
        aspect='equal',
    )
    _fig.update_layout(
        height=500,
        margin={'t': 20, 'b': 10, 'l': 10, 'r': 10},
        coloraxis_colorbar={
            'title': 'Jaccard',
            'thickness': 14,
            'len': 0.75,
        },
        xaxis={'showticklabels': False, 'title_standoff': 8},
        yaxis={'showticklabels': False, 'title_standoff': 8},
        font={'size': 12},
    )

    _jmean = float(_sim.max(axis=1).mean())
    _good = float((_sim.max(axis=1) >= 0.7).mean())

    mo.vstack(
        [
            mo.md('### Prototype Similarity Heatmap'),
            mo.callout(
                mo.md(
                    f'Mean best-match Jaccard: **{_jmean:.3f}** · '
                    f'Good matches (≥ 0.7): **{_good:.1%}**'
                ),
                kind='info',
            ),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    scatter_method_ui = mo.ui.dropdown(
        options=['PCA', 'UMAP', 't-SNE'],
        value='PCA',
        label='Dim reduction',
    )
    n_scatter_ui = mo.ui.slider(
        start=200,
        stop=5000,
        step=200,
        value=2000,
        label='Samples per model',
        show_value=True,
    )
    parseval_ui = mo.ui.switch(label='Parseval frame', value=True)
    prewhiten_ui = mo.ui.switch(label='Prewhiten (PCA)', value=True)
    show_delta_ui = mo.ui.switch(label='Show alignment delta')
    mo.vstack(
        [
            mo.md('### Latent & Prototype Space'),
            mo.hstack(
                [
                    scatter_method_ui,
                    n_scatter_ui,
                    parseval_ui,
                    prewhiten_ui,
                    show_delta_ui,
                ],
                gap=2,
            ),
        ]
    )
    return (
        n_scatter_ui,
        parseval_ui,
        prewhiten_ui,
        scatter_method_ui,
        show_delta_ui,
    )


@app.cell(hide_code=True)
def _(
    cluster_indices_a,
    cluster_indices_b,
    la_align,
    lb_align,
    mo,
    model_a_ui,
    model_b_ui,
    n_prototypes_ui,
    n_scatter_ui,
    np,
    scatter_method_ui,
):
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _msp

    _rng = np.random.default_rng(0)
    _n = min(n_scatter_ui.value, len(la_align))
    _idx = _rng.choice(len(la_align), size=_n, replace=False)

    # Build prototype label per observation from cluster_indices dicts
    _la_lbl = np.empty(len(la_align), dtype=int)
    for _c, _obs in cluster_indices_a.items():
        _la_lbl[_obs] = _c
    _lb_lbl = np.empty(len(lb_align), dtype=int)
    for _c, _obs in cluster_indices_b.items():
        _lb_lbl[_obs] = _c

    _Xa, _ya = la_align[_idx], _la_lbl[_idx]
    _Xb, _yb = lb_align[_idx], _lb_lbl[_idx]

    _method = scatter_method_ui.value
    with mo.status.spinner(title=f'Running {_method} on alignment latents…'):
        if _method == 'PCA':
            from sklearn.decomposition import PCA as _PCA

            _X2a = _PCA(n_components=2).fit_transform(_Xa)
            _X2b = _PCA(n_components=2).fit_transform(_Xb)
        elif _method == 't-SNE':
            from sklearn.manifold import TSNE as _TSNE

            _X2a = _TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(_Xa)
            _X2b = _TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(_Xb)
        else:
            import umap as _umap

            _X2a = _umap.UMAP(n_components=2, random_state=42).fit_transform(_Xa)
            _X2b = _umap.UMAP(n_components=2, random_state=42).fit_transform(_Xb)

    _n_proto = n_prototypes_ui.value
    _fig = _msp(
        rows=1,
        cols=2,
        subplot_titles=[model_a_ui.value, model_b_ui.value],
        horizontal_spacing=0.06,
    )

    for _col, (_X2, _y) in enumerate([(_X2a, _ya), (_X2b, _yb)], start=1):
        _fig.add_trace(
            _go.Scatter(
                x=_X2[:, 0],
                y=_X2[:, 1],
                mode='markers',
                marker=dict(
                    size=3,
                    opacity=0.65,
                    color=_y,
                    colorscale='Turbo',
                    cmin=0,
                    cmax=_n_proto - 1,
                    showscale=(_col == 2),
                    colorbar=dict(
                        title='Prototype',
                        thickness=12,
                        len=0.8,
                        x=1.01,
                    ),
                ),
                showlegend=False,
            ),
            row=1,
            col=_col,
        )

    _fig.update_layout(
        height=440,
        margin={'t': 35, 'b': 5, 'l': 5, 'r': 60},
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    _fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    _fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    mo.vstack(
        [
            mo.md(
                f'#### Latent space · alignment split · {_n:,} samples · coloured by prototype'
            ),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    align_prototypes,
    cluster_indices_a,
    cluster_indices_b,
    la_test,
    lb_test,
    ls_a,
    ls_b,
    mo,
    model_a_ui,
    model_b_ui,
    n_scatter_ui,
    np,
    prewhiten_ui,
    scatter_method_ui,
    show_delta_ui,
):
    import plotly.graph_objects as _go2

    # Transform test data: prewhiten first (if enabled), then apply prototype operator
    if prewhiten_ui.value:
        _la_test_w = ls_a.apply_whitening_operator(la_test)
        _lb_test_w = ls_b.apply_whitening_operator(lb_test)
    else:
        _la_test_w = la_test
        _lb_test_w = lb_test

    # Align B's prototype dimensions to A's via Hungarian matching on Jaccard similarity
    _perm = align_prototypes(cluster_indices_a, cluster_indices_b)

    _a_proto = ls_a.apply_analysis_operator(_la_test_w)
    _b_proto = ls_b.apply_analysis_operator(_lb_test_w)[:, _perm]

    _mse_real = float(np.mean((_a_proto - _b_proto) ** 2))

    _rng2 = np.random.default_rng(0)
    _n2 = min(n_scatter_ui.value, len(_a_proto))
    _idx2 = _rng2.choice(len(_a_proto), size=_n2, replace=False)

    _Xa2, _Xb2 = _a_proto[_idx2], _b_proto[_idx2]

    _method2 = scatter_method_ui.value
    with mo.status.spinner(
        title=f'Running {_method2} on prototype space (test split)…'
    ):
        _X_cat = np.vstack([_Xa2, _Xb2])
        if _method2 == 'PCA':
            from sklearn.decomposition import PCA as _PCA2

            _X2_cat = _PCA2(n_components=2).fit_transform(_X_cat)
        elif _method2 == 't-SNE':
            from sklearn.manifold import TSNE as _TSNE2

            _X2_cat = _TSNE2(n_components=2, random_state=42, n_jobs=-1).fit_transform(
                _X_cat
            )
        else:
            import umap as _umap2

            _X2_cat = _umap2.UMAP(n_components=2, random_state=42).fit_transform(_X_cat)

    _X2a2 = _X2_cat[:_n2]
    _X2b2 = _X2_cat[_n2:]

    _fig2 = _go2.Figure()

    _fig2.add_trace(
        _go2.Scatter(
            x=_X2a2[:, 0],
            y=_X2a2[:, 1],
            mode='markers',
            marker=dict(size=5, opacity=0.7, symbol='circle', color='#636EFA'),
            name=model_a_ui.value,
        )
    )
    _fig2.add_trace(
        _go2.Scatter(
            x=_X2b2[:, 0],
            y=_X2b2[:, 1],
            mode='markers',
            marker=dict(size=5, opacity=0.7, symbol='cross', color='#EF553B'),
            name=model_b_ui.value,
        )
    )

    if show_delta_ui.value:
        _xs, _ys = [], []
        for _i in range(_n2):
            _xs += [float(_X2a2[_i, 0]), float(_X2b2[_i, 0]), None]
            _ys += [float(_X2a2[_i, 1]), float(_X2b2[_i, 1]), None]
        _fig2.add_trace(
            _go2.Scatter(
                x=_xs,
                y=_ys,
                mode='lines',
                line=dict(color='rgba(120,120,120,0.25)', width=0.8),
                name='delta',
            )
        )

    _fig2.update_layout(
        height=500,
        margin={'t': 20, 'b': 10, 'l': 10, 'r': 10},
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='left',
            x=0,
        ),
    )
    _fig2.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    _fig2.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    mo.vstack(
        [
            mo.md(
                f'#### Prototype space · Hungarian-aligned · test split · {_n2:,} samples · MSE: {_mse_real:.4f}'
            ),
            mo.ui.plotly(_fig2),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    LatentSpace,
    cluster_indices_a,
    la_test,
    lb_align,
    lb_test,
    ls_a,
    mo,
    model_a_ui,
    model_b_ui,
    n_scatter_ui,
    np,
    prewhiten_ui,
    scatter_method_ui,
    show_delta_ui,
):
    import plotly.graph_objects as _go3

    # Artificial prototypes: fit on alignment split using model A's cluster structure,
    # then apply the resulting operator to the test split
    _ls_b_art = LatentSpace(lb_align, seed=42)
    if prewhiten_ui.value:
        _ls_b_art.prewhiten(inplace=True)
    _ls_b_art.compute_artificial_prototypes(cluster_indices_a)

    # Transform test data: prewhiten first (if enabled), then apply prototype operator
    if prewhiten_ui.value:
        _la_test_w = ls_a.apply_whitening_operator(la_test)
        _lb_test_w = _ls_b_art.apply_whitening_operator(lb_test)
    else:
        _la_test_w = la_test
        _lb_test_w = lb_test
    _a_proto_art = ls_a.apply_analysis_operator(_la_test_w)
    _b_proto_art = _ls_b_art.apply_analysis_operator(_lb_test_w)

    _mse_art = float(np.mean((_a_proto_art - _b_proto_art) ** 2))

    _rng3 = np.random.default_rng(0)
    _n3 = min(n_scatter_ui.value, len(_a_proto_art))
    _idx3 = _rng3.choice(len(_a_proto_art), size=_n3, replace=False)

    _Xa3, _Xb3 = _a_proto_art[_idx3], _b_proto_art[_idx3]

    _method3 = scatter_method_ui.value
    with mo.status.spinner(title=f'Running {_method3} on artificial prototype space…'):
        _X_cat3 = np.vstack([_Xa3, _Xb3])
        if _method3 == 'PCA':
            from sklearn.decomposition import PCA as _PCA3

            _X2_cat3 = _PCA3(n_components=2).fit_transform(_X_cat3)
        elif _method3 == 't-SNE':
            from sklearn.manifold import TSNE as _TSNE3

            _X2_cat3 = _TSNE3(n_components=2, random_state=42, n_jobs=-1).fit_transform(
                _X_cat3
            )
        else:
            import umap as _umap3

            _X2_cat3 = _umap3.UMAP(n_components=2, random_state=42).fit_transform(
                _X_cat3
            )

    _X2a3 = _X2_cat3[:_n3]
    _X2b3 = _X2_cat3[_n3:]

    _fig3 = _go3.Figure()

    _fig3.add_trace(
        _go3.Scatter(
            x=_X2a3[:, 0],
            y=_X2a3[:, 1],
            mode='markers',
            marker=dict(size=5, opacity=0.7, symbol='circle', color='#636EFA'),
            name=model_a_ui.value,
        )
    )
    _fig3.add_trace(
        _go3.Scatter(
            x=_X2b3[:, 0],
            y=_X2b3[:, 1],
            mode='markers',
            marker=dict(size=5, opacity=0.7, symbol='cross', color='#EF553B'),
            name=f'{model_b_ui.value} (artificial)',
        )
    )

    if show_delta_ui.value:
        _xs3, _ys3 = [], []
        for _i in range(_n3):
            _xs3 += [float(_X2a3[_i, 0]), float(_X2b3[_i, 0]), None]
            _ys3 += [float(_X2a3[_i, 1]), float(_X2b3[_i, 1]), None]
        _fig3.add_trace(
            _go3.Scatter(
                x=_xs3,
                y=_ys3,
                mode='lines',
                line=dict(color='rgba(120,120,120,0.25)', width=0.8),
                name='delta',
            )
        )

    _fig3.update_layout(
        height=500,
        margin={'t': 20, 'b': 10, 'l': 10, 'r': 10},
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='left',
            x=0,
        ),
    )
    _fig3.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    _fig3.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    mo.vstack(
        [
            mo.md(
                f"#### Prototype space · artificial prototypes (B uses A's cluster structure) "
                f'· test split · {_n3:,} samples · MSE: {_mse_art:.4f}'
            ),
            mo.ui.plotly(_fig3),
        ]
    )
    return


if __name__ == '__main__':
    app.run()
