import marimo

__generated_with = '0.23.0'
app = marimo.App(width='columns')


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from datasets import load_dataset

    return load_dataset, mo, pl


@app.cell(hide_code=True)
def _(load_dataset, pl):
    dataset = load_dataset('spaicom-lab/model-registry', split='train')
    df = pl.from_arrow(dataset.data.table)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Registry Completeness

    Not all columns are filled for every model. This heatmap shows the **% of models
    per family that have a non-null value** for each column -- darker = more complete.

    Columns that are nearly black across all families are densely populated and
    reliable for analysis. Pale columns are sparsely recorded and any plot that uses
    them reflects only the well-documented minority. Pale rows indicate families
    whose metadata is largely missing.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    _meta_cols = [
        'depth_code',
        'width_code',
        'patch_size',
        'input_resolution',
        'window_size',
        'stride_code',
        'head_type',
        'num_registers',
        'positional_encoding',
        'activation',
        'pe_scope',
        'pretrain_config',
        'pretrain_org',
        'pretrain_dataset',
        'pretrain_dataset_size',
        'pretrain_method',
        'pretrain_ft',
        'pretrain_resolution',
        'pretrain_ft_resolution',
        'pretrain_epochs',
        'pretrain_tokens',
        'pretrain_aug',
    ]

    _families_sorted = df['family'].drop_nulls().unique().sort().to_list()

    _completeness = pl.DataFrame(
        {
            'family': _families_sorted,
            **{
                col: [
                    round(
                        df.filter(pl.col('family') == fam)[col].is_not_null().mean()
                        * 100,
                        1,
                    )
                    for fam in _families_sorted
                ]
                for col in _meta_cols
            },
        }
    )

    fig_completeness = px.imshow(
        _completeness.select(_meta_cols).to_numpy(),
        x=_meta_cols,
        y=_families_sorted,
        color_continuous_scale='Blues',
        zmin=0,
        zmax=100,
        title='Metadata Completeness by Family (% non-null)',
        labels={'x': 'Column', 'y': 'Family', 'color': '% filled'},
        aspect='auto',
        height=620,
    )
    fig_completeness.update_layout(xaxis_tickangle=-45)
    fig_completeness
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parameters vs Latent Dim

    Each point is a model. Both axes are log-scaled, so a straight diagonal means a power-law relationship between parameter count and embedding size.

    Look for **family-specific tracks**: models within the same family tend to align on a narrow diagonal as they scale up -- larger models simply widen the embedding rather than deepening the network. Outliers above the diagonal have unusually large latent dims relative to their parameter budget (efficient wide models); outliers below pack more parameters into a narrow embedding (deep but thin architectures).
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    import plotly.express as px

    scatter_df = df.filter(
        pl.col('num_parameters').is_not_null() & pl.col('latent_dim').is_not_null()
    ).to_pandas()

    fig_params_latent = px.scatter(
        scatter_df,
        x='num_parameters',
        y='latent_dim',
        color='family',
        hover_name='model_name',
        hover_data={'size': True, 'pretrain_dataset': True},
        log_x=True,
        log_y=True,
        title='Parameters vs Latent Dim (log-log), colored by family',
        labels={
            'num_parameters': '# Parameters (log)',
            'latent_dim': 'Latent Dim (log)',
        },
        opacity=0.75,
        height=550,
    )
    fig_params_latent.update_traces(marker_size=6)
    fig_params_latent
    return (px,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Models per Family by Size Tier

    A stacked bar showing how many models each family contributes and how that count breaks down across size tiers (Nano -> Giant).

    Tall bars indicate families with broad size coverage; uniform bars indicate families that mostly ship one size. Families dominated by a single tier (e.g. all Base) may reflect a research focus rather than a production lineup.
    """)
    return


@app.cell(hide_code=True)
def _(df, px):
    family_size = (
        df.group_by(['family', 'size'])
        .len()
        .sort(['family', 'len'], descending=[False, True])
    )

    fig_family_size = px.bar(
        family_size.to_pandas(),
        x='family',
        y='len',
        color='size',
        title='Models per Family, broken down by Size tier',
        labels={'family': 'Family', 'len': 'Count', 'size': 'Size'},
        height=480,
        barmode='stack',
    )
    fig_family_size.update_layout(xaxis_tickangle=-45)
    fig_family_size
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pretrain Method x Dataset Heatmap

    Each cell counts how many models were trained with a given **(method, dataset) combination** -- cells with fewer than 2 models are hidden to reduce noise.

    Dark columns reveal datasets used across many methods (e.g. ImageNet-21k is method-agnostic). Dark rows reveal methods that pull from many datasets. Isolated bright cells indicate niche pairings that only one research group has explored.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    heat_df = (
        df.group_by(['pretrain_method', 'pretrain_dataset'])
        .len()
        .filter(pl.col('len') > 1)
        .pivot(on='pretrain_dataset', index='pretrain_method', values='len')
        .fill_null(0)
    )

    fig_heatmap = px.imshow(
        heat_df.select(pl.exclude('pretrain_method')).to_numpy(),
        x=heat_df.columns[1:],
        y=heat_df['pretrain_method'].to_list(),
        color_continuous_scale='Blues',
        title='Pretrain Method x Dataset (count > 1)',
        labels={'x': 'Pretrain Dataset', 'y': 'Pretrain Method', 'color': 'Count'},
        aspect='auto',
        height=420,
    )
    fig_heatmap.update_layout(xaxis_tickangle=-45)
    fig_heatmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Boolean Flags: How Common Is Each Property?

    Each bar shows the percentage of models in the registry that have a given architectural flag set to True.

    Most flags are rare -- the registry is dominated by standard architectures. `is_distilled` and `is_gap` are the most common exceptions. Very low bars (< 1%) indicate niche techniques present in only a handful of models, which can make their effect hard to study in isolation.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    bool_cols = [c for c, t in df.schema.items() if t == pl.Boolean]
    flag_df = pl.DataFrame(
        {
            'flag': bool_cols,
            'pct_true': [round(df[c].mean() * 100, 1) for c in bool_cols],
        }
    ).sort('pct_true', descending=True)

    fig_flags = px.bar(
        flag_df.to_pandas(),
        x='flag',
        y='pct_true',
        title='Boolean Flags — % of Models Where True',
        labels={'flag': 'Flag', 'pct_true': '% True'},
        color='pct_true',
        color_continuous_scale='Teal',
        height=420,
    )
    fig_flags.update_coloraxes(showscale=False)
    fig_flags
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Boolean Flag Co-occurrence

    Pearson correlation between every pair of the 13 boolean flags (cast to 0/1).
    Values near **+1** mean the two flags almost always appear together;
    values near **-1** mean they are mutually exclusive;
    values near **0** mean independent design choices.

    Clusters of correlated flags reveal implicit "flag bundles" -- sub-styles
    that travel together as a package rather than being set independently.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    import numpy as _np2

    _flag_cols = [c for c, t in df.schema.items() if t == pl.Boolean]
    _flag_arr = df.select(_flag_cols).to_numpy().astype(float)
    _flag_corr = _np2.corrcoef(_flag_arr.T)

    fig_flag_corr = px.imshow(
        _flag_corr,
        x=_flag_cols,
        y=_flag_cols,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        text_auto='.2f',
        title='Boolean Flag Co-occurrence (Pearson r)',
        height=560,
    )
    fig_flag_corr.update_layout(coloraxis_colorbar_title='r')
    fig_flag_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Distilled vs Non-Distilled Models

    Overlapping histograms of log10(parameter count) for distilled and non-distilled
    models, shown for the families that have at least one distilled model.

    Distillation is intended to compress models -- if it works as advertised,
    the distilled distribution should skew left (smaller). Families where the
    distributions overlap heavily use distillation for reasons other than size
    reduction (e.g. knowledge transfer at the same scale).
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    _distilled_families = (
        df.filter(pl.col('is_distilled'))['family']
        .value_counts()
        .sort('count', descending=True)
        .head(10)['family']
        .to_list()
    )

    _distilled_df = (
        df.filter(pl.col('family').is_in(_distilled_families))
        .with_columns(
            [
                pl.col('num_parameters').log(base=10).alias('log_params'),
                pl.col('is_distilled')
                .cast(pl.Utf8)
                .replace({'true': 'Distilled', 'false': 'Non-distilled'})
                .alias('distilled_label'),
            ]
        )
        .select(['family', 'log_params', 'distilled_label'])
        .drop_nulls()
    )

    fig_distilled = px.histogram(
        _distilled_df.to_pandas(),
        x='log_params',
        color='distilled_label',
        facet_col='family',
        facet_col_wrap=4,
        barmode='overlay',
        opacity=0.65,
        nbins=20,
        title='Distilled vs Non-distilled: log10(params) by Family',
        labels={'log_params': 'log10(params)', 'distilled_label': ''},
        color_discrete_map={'Distilled': '#e377c2', 'Non-distilled': '#1f77b4'},
        height=520,
    )
    fig_distilled.update_layout(legend_title='')
    fig_distilled
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parameter Count Distribution by Family

    Violin plots (with embedded box plots) show the full distribution of parameter counts within each family on a log y-axis.

    Wide violins mean a family ships models across a broad size range. Narrow violins indicate a family concentrates around one scale. The median line inside the box reveals whether a family skews toward small or large models -- useful for understanding where the bulk of published work sits.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    violin_df = df.filter(pl.col('num_parameters').is_not_null()).to_pandas()

    fig_violin = px.violin(
        violin_df,
        x='family',
        y='num_parameters',
        box=True,
        log_y=True,
        title='Parameter Count Distribution by Family',
        labels={'family': 'Family', 'num_parameters': '# Parameters (log)'},
        color='family',
        height=520,
    )
    fig_violin.update_layout(xaxis_tickangle=-45, showlegend=False)
    fig_violin
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pretrain Resolution vs Fine-Tune Resolution

    Each point is a model that has both a recorded pretraining resolution and a fine-tuning resolution. The **dashed diagonal is the identity line** (pretrain == fine-tune).

    Points **above** the diagonal were fine-tuned at a higher resolution than they were pretrained -- a common trick to boost performance on high-resolution benchmarks. Points **on** the diagonal were evaluated at the same resolution they were trained at. Points **below** are rare and indicate downsampling at fine-tune time.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    res_df = df.filter(
        pl.col('pretrain_resolution').is_not_null()
        & pl.col('pretrain_ft_resolution').is_not_null()
        & (pl.col('pretrain_ft_resolution') > 0)
    ).to_pandas()

    fig_res = px.scatter(
        res_df,
        x='pretrain_resolution',
        y='pretrain_ft_resolution',
        color='family',
        hover_name='model_name',
        title='Pretrain Resolution vs Fine-Tune Resolution',
        labels={
            'pretrain_resolution': 'Pretrain Resolution',
            'pretrain_ft_resolution': 'Fine-Tune Resolution',
        },
        opacity=0.75,
        height=480,
    )
    rmin = res_df['pretrain_resolution'].min()
    rmax = res_df['pretrain_resolution'].max()
    fig_res.add_shape(
        type='line',
        x0=rmin,
        y0=rmin,
        x1=rmax,
        y1=rmax,
        line=dict(color='gray', dash='dash'),
    )
    fig_res
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pearson Correlation Matrix

    Pairwise correlations between four key numeric features after log-transforming `num_parameters` (base 10) and `latent_dim` (base 2). Only rows with all four values present are used.

    The strong positive correlation between `log(params)` and `log(latent)` confirms the power-law scaling visible in the scatter above -- bigger models consistently have wider embeddings. `patch_size` and `input_resolution` are weakly correlated with the others because they apply only to ViT-style models and are absent for the majority.
    """)
    return


@app.cell(hide_code=True)
def _(df, mo, pl, px):
    import numpy as np

    corr_cols = ['num_parameters', 'latent_dim', 'patch_size', 'input_resolution']
    corr_labels = [
        'log10(params)',
        'log2(latent dim)',
        'patch size',
        'input resolution',
    ]

    corr_df = (
        df.select(corr_cols)
        .drop_nulls()
        .with_columns(
            [
                pl.col('num_parameters').log(base=10).alias('num_parameters'),
                pl.col('latent_dim').log(base=2).alias('latent_dim'),
            ]
        )
    )

    arr = corr_df.to_numpy().astype(float)
    corr_matrix = np.corrcoef(arr.T)

    fig_corr = px.imshow(
        corr_matrix,
        x=corr_labels,
        y=corr_labels,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        text_auto='.2f',
        title=f'Pearson Correlation (log-transformed, n={len(corr_df)})',
        height=480,
    )
    fig_corr.update_layout(coloraxis_colorbar_title='r')
    mo.ui.plotly(fig_corr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parallel Coordinates: Architecture Fingerprint

    Each line is a model passing through four architectural axes. Lines are coloured by family. Filtering any axis (click-drag on it) highlights models in the selected range across all other axes simultaneously.

    Families that produce a tight bundle of near-parallel lines have a consistent internal design language. Families with crossing or diverging lines mix design choices -- for example, using multiple patch sizes or input resolutions within the same family. Only models with all four features recorded are included, so ViT-style families dominate this view.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    family_order = sorted(df['family'].drop_nulls().unique().to_list())
    family_map = {f: i for i, f in enumerate(family_order)}

    par_df = (
        df.filter(
            pl.col('num_parameters').is_not_null()
            & pl.col('latent_dim').is_not_null()
            & pl.col('patch_size').is_not_null()
            & pl.col('input_resolution').is_not_null()
        )
        .with_columns(
            [
                pl.col('num_parameters').log(base=10).alias('log_params'),
                pl.col('latent_dim').log(base=2).alias('log2_latent'),
                pl.col('family')
                .replace(family_map)
                .cast(pl.Float64)
                .alias('family_id'),
            ]
        )
        .select(
            [
                'family_id',
                'log_params',
                'log2_latent',
                'patch_size',
                'input_resolution',
                'family',
            ]
        )
    )

    fig_par = px.parallel_coordinates(
        par_df.to_pandas(),
        dimensions=['log_params', 'log2_latent', 'patch_size', 'input_resolution'],
        labels={
            'log_params': 'log10(params)',
            'log2_latent': 'log2(latent dim)',
            'patch_size': 'Patch Size',
            'input_resolution': 'Input Resolution',
        },
        color='family_id',
        color_continuous_scale=px.colors.sequential.Turbo,
        title='Parallel Coordinates: Architecture Fingerprint by Family',
        height=520,
    )
    fig_par.update_coloraxes(
        colorbar_tickvals=list(family_map.values()),
        colorbar_ticktext=list(family_map.keys()),
        colorbar_title='Family',
    )
    fig_par
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Depth x Width Design Space

    Each model with both codes recorded is placed on a grid of
    **depth_code** (y-axis) vs **width_code** (x-axis), both treated as
    ordered numerics. Node size scales with parameter count.

    This reveals the "shape" each family prefers: deep-and-narrow models
    sit in the top-left, shallow-and-wide in the bottom-right. Families
    that span the diagonal scale both dimensions together. Only ~92 models
    have both codes recorded, so this reflects the most structurally
    documented subset of the registry.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    _dw_df = df.filter(
        pl.col('depth_code').is_not_null()
        & pl.col('width_code').is_not_null()
        & pl.col('num_parameters').is_not_null()
    ).with_columns(
        [
            pl.col('depth_code').cast(pl.Int64).alias('depth'),
            pl.col('width_code').cast(pl.Int64).alias('width'),
            pl.col('num_parameters').log(base=10).alias('log_params'),
        ]
    )

    fig_dw = px.scatter(
        _dw_df.to_pandas(),
        x='width',
        y='depth',
        color='family',
        size='log_params',
        size_max=22,
        hover_name='model_name',
        hover_data={'depth_code': True, 'width_code': True},
        title='Architecture Design Space: Depth vs Width (size = log10 params)',
        labels={'depth': 'Depth Code (numeric)', 'width': 'Width Code (numeric)'},
        opacity=0.8,
        height=520,
    )
    fig_dw.update_layout(legend_title='Family')
    fig_dw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Who Trained What: Org -> Family -> Count

    A treemap where the outer level is the training organisation and the inner level is the model family. Rectangle area is proportional to the number of models. Click any outer rectangle to zoom into that organisation.

    This reveals how concentrated model production is: a few organisations (Meta, timm recipes, SAIL) account for a disproportionate share of the registry. It also shows which families are the exclusive output of one lab vs. which have been reproduced or extended by multiple groups.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    # Treemap: pretrain_org -> family -> count
    tree_df = (
        df.filter(pl.col('pretrain_org').is_not_null())
        .group_by(['pretrain_org', 'family'])
        .len()
        .sort('len', descending=True)
    )

    fig_tree = px.treemap(
        tree_df.to_pandas(),
        path=['pretrain_org', 'family'],
        values='len',
        color='len',
        color_continuous_scale='Viridis',
        title='Who Trained What: Org -> Family -> Count',
        height=580,
    )
    fig_tree.update_traces(textinfo='label+value')
    fig_tree
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Scaling Tracks: Latent Dim vs log10(Params)

    Each line connects models within the same family, sorted by latent dim. The x-axis is linear latent dim; the y-axis is log10(parameters).

    Steep lines mean a family adds many parameters for a small gain in embedding width -- depth-heavy scaling. Shallow lines mean the family grows primarily by widening the embedding. Gaps between points on a line indicate discrete size jumps with no intermediate models published. The log-log version below reveals whether the relationship is a clean power law.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    scale_df = (
        df.filter(
            pl.col('num_parameters').is_not_null()
            & pl.col('latent_dim').is_not_null()
            & pl.col('family').is_not_null()
        )
        .with_columns(
            [
                pl.col('num_parameters').log(base=10).alias('log_params'),
            ]
        )
        .sort(['family', 'latent_dim'])
    )

    fig_scale = px.line(
        scale_df.to_pandas(),
        x='latent_dim',
        y='log_params',
        color='family',
        markers=True,
        hover_name='model_name',
        hover_data={'size': True},
        title='Scaling Tracks: Latent Dim vs log10(Params) per Family',
        labels={'latent_dim': 'Latent Dim', 'log_params': 'log10(# Parameters)'},
        height=560,
        line_shape='linear',
    )
    fig_scale.update_traces(marker_size=4, line_width=1.5, opacity=0.8)
    fig_scale
    return (scale_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Scaling Tracks: log-log View

    The same data as above but with the x-axis (latent dim) also on a log scale. On a log-log plot a **straight line means a power law**: `params ~ latent_dim^alpha` where `alpha` is the slope.

    Families whose lines are straight and parallel share the same scaling exponent -- they grow in the same way, just at different scales. Families with curved or kinked lines change their scaling strategy across size tiers, e.g. growing depth early then switching to width for the largest variants.
    """)
    return


@app.cell(hide_code=True)
def _(px, scale_df):
    fig_scale_loglog = px.line(
        scale_df.to_pandas(),
        x='latent_dim',
        y='log_params',
        color='family',
        markers=True,
        hover_name='model_name',
        hover_data={'size': True},
        log_x=True,
        title='Scaling Tracks: log-log (Latent Dim vs log10(Params) per Family)',
        labels={
            'latent_dim': 'Latent Dim (log)',
            'log_params': 'log10(# Parameters)',
        },
        height=560,
        line_shape='linear',
    )
    fig_scale_loglog.update_traces(marker_size=4, line_width=1.5, opacity=0.8)
    fig_scale_loglog
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pretrain Methods: Scale vs Breadth

    Each bubble is a pretraining method. The x-axis is the **median parameter count** of models using that method (log scale); the y-axis is the number of distinct families that have used it; bubble size is the total model count.

    Methods in the **top-right** are both large-scale and broadly adopted across architectures. Methods in the **bottom-left** are niche: small models trained by one or two groups. Large bubbles low on the y-axis indicate prolific but family-specific methods -- a single lab producing many variants without the technique spreading elsewhere.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    method_df = (
        df.filter(
            pl.col('pretrain_method').is_not_null()
            & pl.col('num_parameters').is_not_null()
        )
        .group_by('pretrain_method')
        .agg(
            [
                pl.col('num_parameters').median().alias('median_params'),
                pl.len().alias('count'),
                pl.col('family').n_unique().alias('n_families'),
            ]
        )
        .sort('count', descending=True)
    )

    fig_method = px.scatter(
        method_df.to_pandas(),
        x='median_params',
        y='n_families',
        size='count',
        color='pretrain_method',
        hover_name='pretrain_method',
        hover_data={'count': True},
        log_x=True,
        title='Pretrain Methods: Median Params vs Families Covered (bubble = model count)',
        labels={
            'median_params': 'Median Params (log scale)',
            'n_families': 'Distinct Families',
        },
        height=500,
    )
    fig_method.update_traces(marker_sizemin=6)
    fig_method
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pretraining Augmentation Strategy

    Which augmentation strategies appear in the registry, broken down by
    pretraining method. Only ~9% of models have this field recorded.

    Augmentation is a core part of the training recipe but is rarely
    documented. The five strategies here (RandAugment, AugReg, AutoAugment,
    NoisyStudent, AdvProp) span a spectrum from weak generic aug to
    curriculum-style self-training. Their concentration in specific methods
    (AugReg is definitional; NoisyStudent implies semi-supervised training)
    confirms the field is sparsely but non-randomly populated.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    _aug_df = (
        df.filter(pl.col('pretrain_aug').is_not_null())
        .group_by(['pretrain_aug', 'pretrain_method'])
        .len()
        .sort('len', descending=True)
    )

    fig_aug = px.bar(
        _aug_df.to_pandas(),
        x='pretrain_aug',
        y='len',
        color='pretrain_method',
        barmode='stack',
        title='Pretrain Augmentation Strategy, broken down by Pretrain Method',
        labels={
            'pretrain_aug': 'Augmentation',
            'len': '# Models',
            'pretrain_method': 'Method',
        },
        height=460,
    )
    fig_aug.update_layout(xaxis_tickangle=-20, legend_title='Pretrain Method')
    fig_aug
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pretraining Datasets

    How many models in the registry were trained on each dataset. Datasets with only one model are included to show the long tail of niche training sources.

    The dominance of a few datasets (ImageNet-1k, ImageNet-21k) reflects both their historical centrality and the tendency for new architectures to benchmark on familiar ground. The long tail of single-model datasets shows the diversity of specialised pretraining setups that exist outside the mainstream.
    """)
    return


@app.cell(hide_code=True)
def _(df, px):
    _pretrain_counts = (
        df.group_by('pretrain_dataset')
        .len()
        .sort('len', descending=True)
        .rename({'len': 'count'})
    )

    fig_pretrain_counts = px.bar(
        _pretrain_counts.to_pandas(),
        x='pretrain_dataset',
        y='count',
        title='Model Count per Pretraining Dataset',
        labels={'pretrain_dataset': 'Pretrain Dataset', 'count': '# Models'},
        color='count',
        color_continuous_scale='Blues',
        height=480,
    )
    fig_pretrain_counts.update_layout(
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
    )
    fig_pretrain_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pretrain Epochs vs Model Scale

    Scatter of training epochs against log10(parameter count), coloured by
    pretraining method. Only the ~89 models with epoch data are shown.

    Larger models generally require more compute per epoch, so they tend to
    be trained for fewer epochs under a fixed budget -- a downward trend
    would confirm this. Points far **above** the trend were trained unusually
    long relative to their size; points far **below** were undertrained.
    Different methods cluster at very different epoch ranges, reflecting
    fundamentally different training regimes (e.g. CLIP trains for far fewer
    epochs than supervised recipes on ImageNet).
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    _epochs_df = (
        df.filter(
            pl.col('pretrain_epochs').is_not_null()
            & pl.col('num_parameters').is_not_null()
        )
        .with_columns(pl.col('num_parameters').log(base=10).alias('log_params'))
        .select(
            [
                'model_name',
                'family',
                'pretrain_method',
                'pretrain_epochs',
                'log_params',
            ]
        )
    )

    fig_epochs = px.scatter(
        _epochs_df.to_pandas(),
        x='log_params',
        y='pretrain_epochs',
        color='pretrain_method',
        hover_name='model_name',
        hover_data={'family': True},
        title='Pretrain Epochs vs Model Scale (n=89 models with epoch data)',
        labels={
            'log_params': 'log10(# Parameters)',
            'pretrain_epochs': 'Pretrain Epochs',
            'pretrain_method': 'Method',
        },
        height=500,
        opacity=0.85,
    )
    fig_epochs.update_traces(marker_size=8)
    fig_epochs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Model Similarity Graph

    We build an undirected graph where each **node is a model** and **edges connect architecturally similar models**. The pipeline runs in three stages:

    | Step | What | Output |
    |------|------|--------|
    | 1. Feature engineering | Encode architecture metadata into a numeric vector | `G_X` (1699 x 37) |
    | 2. k-NN graph | Connect each model to its 5 most similar neighbours (cosine > 0.90) | `G` networkx graph |
    | 3. UMAP layout | Embed the same feature matrix in 2D for node positions | `G_pos` (1699 x 2) |

    Nodes that appear close together share similar architecture, pretraining method, and capability flags -- not just parameter count.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    import numpy as _np
    import pandas as _pd
    from sklearn.preprocessing import StandardScaler as _SS

    _bool_cols = [c for c, t in df.schema.items() if t == pl.Boolean]

    _feat_df = df.select(
        [
            'model_name',
            'family',
            'num_parameters',
            'latent_dim',
            'activation',
            'pretrain_method',
            *_bool_cols,
        ]
    ).with_columns(
        [
            pl.col('num_parameters').log(base=10),
            pl.col('latent_dim').log(base=2),
            pl.col('activation').fill_null('unknown'),
            pl.col('pretrain_method').fill_null('unknown'),
            *[pl.col(c).cast(pl.Float64) for c in _bool_cols],
        ]
    )

    _num_block = (
        _feat_df.select(['num_parameters', 'latent_dim', *_bool_cols])
        .to_numpy()
        .astype(float)
    )
    _cat_block = _pd.get_dummies(
        _feat_df.select(['activation', 'pretrain_method']).to_pandas(), dtype=float
    ).to_numpy()

    G_X = _SS().fit_transform(_np.hstack([_num_block, _cat_block]))
    G_meta = _feat_df.select(
        ['model_name', 'family', 'num_parameters', 'latent_dim']
    ).to_pandas()

    print(f'Feature matrix: {G_X.shape}')
    return G_X, G_meta


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 1 -- Feature matrix

    Each model becomes a **37-dimensional vector** built from three blocks:

    - **Numeric** (2 dims, log-transformed): `log10(num_parameters)`, `log2(latent_dim)`
    - **Boolean flags** (13 dims): `is_distilled`, `is_pruned`, `uses_rmlp`, ... cast to 0/1
    - **One-hot categoricals** (22 dims): `activation` function x `pretrain_method`

    Null values in categoricals become an `"unknown"` category. All 37 dimensions are then z-scored with `StandardScaler` so no single feature dominates the distance calculation.
    """)
    return


@app.cell(hide_code=True)
def _(G_X, G_meta):
    import networkx as _nx
    from sklearn.neighbors import NearestNeighbors as _KNN2

    _knn = _KNN2(n_neighbors=6, metric='cosine', algorithm='brute')
    _knn.fit(G_X)
    _distances, _indices = _knn.kneighbors(G_X)

    G = _nx.Graph()
    G.add_nodes_from(range(len(G_meta)))

    for i, (dists, nbrs) in enumerate(zip(_distances, _indices)):
        for d, j in zip(dists[1:], nbrs[1:]):
            sim = 1.0 - d
            if sim > 0.90:
                G.add_edge(i, j, weight=sim)

    print(f'Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}')
    print(f'Connected components: {_nx.number_connected_components(G)}')
    print(
        f'Largest component: {max(len(c) for c in _nx.connected_components(G))} nodes'
    )
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 2 -- k-NN similarity graph

    We use **cosine distance** on the scaled feature matrix: it measures the angle between vectors, so two models are similar if they point in the same direction in feature space regardless of overall magnitude.

    For each model we find its 5 nearest neighbours (k=6 minus self) and add an undirected edge when **similarity = 1 - cosine_distance > 0.90** -- a strict threshold that keeps only strong structural matches.

    The resulting graph has **5,856 edges** across **77 connected components**; the largest cluster contains 500 models, mostly within the same family.
    """)
    return


@app.cell(hide_code=True)
def _(G_X):
    import umap as _umap

    _reducer = _umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=False,
    )
    G_pos = _reducer.fit_transform(G_X)  # shape (1699, 2)
    print(f'UMAP done: {G_pos.shape}')
    return (G_pos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 3 -- UMAP layout

    We run **UMAP** (`n_neighbors=15`, `min_dist=0.1`, cosine metric) on the same feature matrix to get 2D node coordinates:

    - Nodes that are geometrically close share *architectural similarity*, not just neighbourhood in the k-NN graph.
    - The axes have no intrinsic meaning -- only relative distances matter.
    - `random_state=42` makes the layout reproducible.

    The k-NN edges are drawn on top of the UMAP embedding, so tightly clustered nodes with many edges represent strongly cohesive model families.
    """)
    return


@app.cell(hide_code=True)
def _(G, G_meta, G_pos, px):
    import plotly.graph_objects as go

    _ex, _ey = [], []
    for _i, _j in G.edges():
        _ex += [G_pos[_i, 0], G_pos[_j, 0], None]
        _ey += [G_pos[_i, 1], G_pos[_j, 1], None]

    _edge_trace = go.Scatter(
        x=_ex,
        y=_ey,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.25)', width=0.5),
        hoverinfo='none',
        showlegend=False,
    )

    _families = sorted(G_meta['family'].unique())
    _palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    _fam_color = {f: _palette[i % len(_palette)] for i, f in enumerate(_families)}

    _node_traces = []
    for _fam in _families:
        _mask = G_meta['family'] == _fam
        _idx = G_meta.index[_mask].tolist()
        _node_traces.append(
            go.Scatter(
                x=G_pos[_idx, 0],
                y=G_pos[_idx, 1],
                mode='markers',
                name=_fam,
                marker=dict(
                    size=5,
                    color=_fam_color[_fam],
                    opacity=0.85,
                    line=dict(width=0.3, color='white'),
                ),
                text=[
                    f'{G_meta.loc[_i, "model_name"]}<br>family: {_fam}<br>params: {10 ** G_meta.loc[_i, "num_parameters"]:,.0f}'
                    for _i in _idx
                ],
                hoverinfo='text',
            )
        )

    fig_graph = go.Figure(data=[_edge_trace, *_node_traces])
    fig_graph.update_layout(
        title='Model Similarity Graph (cosine k-NN, UMAP layout)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=700,
        legend_title='Family',
    )
    fig_graph
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### k-NN Graph: Force-Directed Layout

    While the UMAP plot above positions nodes by raw feature similarity, this view uses a **Fruchterman-Reingold spring layout** where the graph edges themselves drive the positions: similar models attract each other and dissimilar models repel. The two layouts ask different questions:

    | | UMAP layout | Force-directed layout |
    |---|---|---|
    | Positions driven by | All 37 feature dimensions | Only the edges that survived the 0.90 threshold |
    | Shows | Continuous similarity landscape | Discrete community structure |
    | Best for | Spotting gradual transitions | Spotting hubs, bridges, and isolated clusters |

    **Reading the chart:**

    - **Node size** scales with degree -- large nodes have many strong-similarity neighbours and act as architectural hubs.
    - **Edge opacity** encodes similarity strength: darker edges have cosine similarity > 0.97; lighter edges are borderline (0.90-0.93).
    - **Isolated nodes** at the periphery have no neighbour above the 0.90 threshold -- architecturally unique models with no close relatives in the registry.
    - **Tight clumps** with many dark edges are families where models differ only in scale, not in design choices.
    - **Thin bridges** connecting two clumps are models that happen to share one strong feature (e.g. same pretraining method) despite coming from different families.
    """)
    return


@app.cell(hide_code=True)
def _(G, G_meta, px):
    import networkx as _nx2
    import plotly.graph_objects as _go2

    # Fruchterman-Reingold spring layout â€” edge weights pull similar models together
    _spring_pos = _nx2.spring_layout(G, weight='weight', seed=42, iterations=60, k=0.4)

    # Node degree for sizing: high-degree nodes are hubs with many strong similarities
    _degrees = dict(G.degree())

    # Separate edge traces by similarity band so we can vary opacity per band
    _edges_high_x, _edges_high_y = [], []  # sim > 0.97
    _edges_mid_x, _edges_mid_y = [], []  # 0.93 < sim <= 0.97
    _edges_low_x, _edges_low_y = [], []  # sim <= 0.93

    for _u, _v, _data in G.edges(data=True):
        _w = _data.get('weight', 0.9)
        _x0, _y0 = _spring_pos[_u]
        _x1, _y1 = _spring_pos[_v]
        if _w > 0.97:
            _edges_high_x += [_x0, _x1, None]
            _edges_high_y += [_y0, _y1, None]
        elif _w > 0.93:
            _edges_mid_x += [_x0, _x1, None]
            _edges_mid_y += [_y0, _y1, None]
        else:
            _edges_low_x += [_x0, _x1, None]
            _edges_low_y += [_y0, _y1, None]

    _edge_traces2 = [
        _go2.Scatter(
            x=_edges_high_x,
            y=_edges_high_y,
            mode='lines',
            line=dict(color='rgba(80,80,80,0.65)', width=0.8),
            hoverinfo='none',
            showlegend=False,
            name='sim>0.97',
        ),
        _go2.Scatter(
            x=_edges_mid_x,
            y=_edges_mid_y,
            mode='lines',
            line=dict(color='rgba(120,120,120,0.35)', width=0.5),
            hoverinfo='none',
            showlegend=False,
            name='sim>0.93',
        ),
        _go2.Scatter(
            x=_edges_low_x,
            y=_edges_low_y,
            mode='lines',
            line=dict(color='rgba(180,180,180,0.15)', width=0.4),
            hoverinfo='none',
            showlegend=False,
            name='sim<=0.93',
        ),
    ]

    # One node trace per family so the legend maps colour -> family name
    _families2 = sorted(G_meta['family'].unique())
    _palette2 = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    _fam_color2 = {f: _palette2[i % len(_palette2)] for i, f in enumerate(_families2)}

    _node_traces2 = []
    for _fam2 in _families2:
        _mask2 = G_meta['family'] == _fam2
        _idx2 = G_meta.index[_mask2].tolist()
        _sizes2 = [max(4, min(16, 4 + _degrees[_i] * 1.5)) for _i in _idx2]
        _texts2 = [
            (
                G_meta.loc[_i, 'model_name']
                + '<br>family: '
                + _fam2
                + '<br>degree: '
                + str(_degrees[_i])
                + '<br>params: '
                + f'{10 ** G_meta.loc[_i, "num_parameters"]:,.0f}'
            )
            for _i in _idx2
        ]
        _node_traces2.append(
            _go2.Scatter(
                x=[_spring_pos[_i][0] for _i in _idx2],
                y=[_spring_pos[_i][1] for _i in _idx2],
                mode='markers',
                name=_fam2,
                marker=dict(
                    size=_sizes2,
                    color=_fam_color2[_fam2],
                    opacity=0.85,
                    line=dict(width=0.3, color='white'),
                ),
                text=_texts2,
                hoverinfo='text',
            )
        )

    fig_knn = _go2.Figure(data=[*_edge_traces2, *_node_traces2])
    fig_knn.update_layout(
        title='k-NN Similarity Graph (Fruchterman-Reingold, node size = degree)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=720,
        legend_title='Family',
    )
    fig_knn
    return


if __name__ == '__main__':
    app.run()
