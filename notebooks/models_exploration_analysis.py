import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    ROOT = Path(__file__).resolve().parent.parent
    # Registry: local copy of spaicom-lab/semasia-model-registry (updated registry, 1,699 models).
    REGISTRY = (
        ROOT / 'data' / 'semasia-model-registry' / 'semasia_model_registry.parquet'
    )
    # Every figure of this notebook goes here (results/ layout: registry outputs live in results/registry/).
    OUT = ROOT / 'results' / 'registry' / 'exploration'
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT, REGISTRY, mo, pl


@app.cell(hide_code=True)
def _(REGISTRY, pl):
    df = pl.read_parquet(REGISTRY)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl):
    df.filter(pl.col('latent_dim') < 200)
    return


@app.cell
def _(df):
    df['family'].value_counts().sort('count', descending=True)
    return


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
        'pretrain_tasks',
        'paradigm',
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


@app.cell
def _(OUT, df, pl):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
    import numpy as np
    import pandas as pd

    # ── Reproducible style close to NeurIPS/ICML papers ──────────────────────────
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 7.5,
            'legend.title_fontsize': 8,
            'axes.linewidth': 0.7,
            'xtick.major.width': 0.7,
            'ytick.major.width': 0.7,
            'xtick.minor.width': 0.5,
            'ytick.minor.width': 0.5,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'axes.grid': True,
            'grid.linewidth': 0.4,
            'grid.alpha': 0.4,
            'grid.linestyle': '--',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'pdf.fonttype': 42,  # embeds fonts as TrueType → avoids Type 3 issues
            'ps.fonttype': 42,
        }
    )

    # ── Data prep ─────────────────────────────────────────────────────────────────
    _scatter_df = df.filter(
        pl.col('num_parameters').is_not_null() & pl.col('latent_dim').is_not_null()
    ).to_pandas()

    top_n = 15
    top_families = _scatter_df['family'].value_counts().nlargest(top_n).index.tolist()
    is_top = _scatter_df['family'].isin(top_families)

    # Use a perceptually-distinct palette (avoid tab20's light pastels)
    palette = plt.cm.get_cmap('tab20', top_n)
    color_map = {f: palette(i) for i, f in enumerate(top_families)}

    # ── Figure: NeurIPS single-col width = 5.5 in ────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # 1. Background "other" points — drawn first so they sit below everything
    ax.scatter(
        _scatter_df.loc[~is_top, 'num_parameters'],
        _scatter_df.loc[~is_top, 'latent_dim'],
        color='#cccccc',
        alpha=0.35,
        s=10,
        linewidths=0,
        rasterized=True,  # keeps PDF file size sane for many grey points
        zorder=1,
    )

    # 2. Top-20 families — drawn on top
    for family in top_families:
        mask = _scatter_df['family'] == family
        ax.scatter(
            _scatter_df.loc[mask, 'num_parameters'],
            _scatter_df.loc[mask, 'latent_dim'],
            color=color_map[family],
            alpha=0.80,
            s=18,
            linewidths=0.3,
            edgecolors='white',
            zorder=2,
            label=family,
        )

    # ── Axes ──────────────────────────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Latent Dimension')

    # Clean log tick formatting (e.g. 10³ instead of 1000)
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # ── Legend ────────────────────────────────────────────────────────────────────
    # "Other" proxy handle
    other_handle = mpatches.Patch(color='#cccccc', alpha=0.6, label='Other')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(other_handle)
    labels.append('Other')

    legend = ax.legend(
        handles=handles,
        labels=labels,
        title=f'Model Family (top {top_n})',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # below the axes
        ncol=4,
        frameon=True,
        framealpha=0.9,
        edgecolor='#aaaaaa',
        borderpad=0.6,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=0.8,
        labelspacing=0.35,
    )
    legend.get_title().set_fontweight('bold')

    # ── Export ────────────────────────────────────────────────────────────────────
    plt.savefig(str(OUT / 'params_vs_latent_dim.pdf'), format='pdf')
    plt.show()
    return np, plt


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
    ## Macro-family and Training Paradigm

    `macro_family` groups the families into broad architecture classes (Convolutional, Vision Transformer, ...). `paradigm` is the learning signal of the pretraining, derived from the first element of `pretrain_tasks` (supervised, contrastive, self_supervised, multimodal_generative).

    The bars show how many models fall in each macro-family, split by paradigm. A macro-family dominated by one paradigm means that comparing it against the others also compares training recipes. Models with a null `macro_family` are shown as `unknown`.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, px):
    macro_paradigm = (
        df.with_columns(
            pl.col('macro_family').fill_null('unknown'),
            pl.col('paradigm').fill_null('unknown'),
        )
        .group_by(['macro_family', 'paradigm'])
        .len()
        .sort('len', descending=True)
    )

    fig_macro_paradigm = px.bar(
        macro_paradigm.to_pandas(),
        x='macro_family',
        y='len',
        color='paradigm',
        title='Models per Macro-family, broken down by Paradigm',
        labels={'macro_family': 'Macro-family', 'len': 'Count', 'paradigm': 'Paradigm'},
        height=480,
        barmode='stack',
    )
    fig_macro_paradigm.update_layout(xaxis_tickangle=-30)
    fig_macro_paradigm
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
        df.explode('pretrain_dataset')  # a model with several corpora counts once for each of them
        .group_by(['pretrain_method', 'pretrain_dataset'])
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


@app.cell
def _(df, np, pl, px):
    violin_df = df.filter(
        (pl.col('num_parameters').is_not_null()) & (pl.col('num_parameters') != 0)
    ).to_pandas()
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
    y_min = violin_df['num_parameters'].min()
    y_max = violin_df['num_parameters'].max()
    fig_violin.update_yaxes(range=[np.log10(y_min) - 0.1, np.log10(y_max) + 0.1])
    fig_violin.update_traces(spanmode='hard')
    fig_violin.update_layout(xaxis_tickangle=-45, showlegend=False)
    fig_violin
    fig_violin.show()
    return


@app.cell(hide_code=True)
def _(OUT, df, np, pl, plt):
    def parameter_count_distribution_by_family():
        # ── Configuration ─────────────────────────────────────────────────────────────
        TOP_N = 15
        OUTPUT_PATH = str(OUT / 'param_distribution_by_family.pdf')
        FIGSIZE = (7.0, 4.5)  # NeurIPS single-col width = 5.5; use 7.0 for wider venue

        # ── NeurIPS/ICML rcParams ─────────────────────────────────────────────────────
        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 9,
                'axes.titlesize': 10,
                'axes.labelsize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 7.5,
                'legend.title_fontsize': 8,
                'axes.linewidth': 0.7,
                'xtick.major.width': 0.7,
                'ytick.major.width': 0.7,
                'xtick.minor.width': 0.5,
                'ytick.minor.width': 0.5,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'ytick.minor.visible': True,
                'axes.grid': True,
                'grid.linewidth': 0.4,
                'grid.alpha': 0.4,
                'grid.linestyle': '--',
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.02,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
            }
        )

        # ── Data prep ─────────────────────────────────────────────────────────────────
        violin_df = df.filter(
            pl.col('num_parameters').is_not_null() & (pl.col('num_parameters') != 0)
        ).to_pandas()

        # Top-N families (by count), preserve original appearance order
        top_families = violin_df['family'].value_counts().nlargest(TOP_N).index.tolist()
        other_mask = ~violin_df['family'].isin(top_families)
        violin_df['family_plot'] = violin_df['family'].where(~other_mask, other='Other')

        # x-axis order: top families first, then "Other"
        x_order = top_families + ['Other']
        groups = [
            violin_df.loc[violin_df['family_plot'] == f, 'num_parameters'].values
            for f in x_order
        ]
        n_groups = len(x_order)

        # ── Palette: tab20 for top families, grey for Other ───────────────────────────
        palette = plt.cm.get_cmap('tab20', TOP_N)
        colors = [palette(i) for i in range(TOP_N)] + ['#bbbbbb']

        # ── Figure ────────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for i, (data, color, label) in enumerate(zip(groups, colors, x_order)):
            data = data[data > 0]
            if len(data) < 2:  # skip families with too few points for a violin
                ax.scatter(
                    [i + 1],
                    [np.median(data)],
                    color=color,
                    s=20,
                    zorder=4,
                    marker='D',
                )
                continue

            log_data = np.log10(data)

            # ── Violin ────────────────────────────────────────────────────────────────
            parts = ax.violinplot(
                log_data,
                positions=[i + 1],
                widths=0.7,
                showextrema=False,  # we'll draw our own whiskers below
                showmedians=False,
            )
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('white')
                pc.set_linewidth(0.4)
                pc.set_alpha(0.75)

            # ── Box (IQR) ─────────────────────────────────────────────────────────────
            q1, med, q3 = np.percentile(log_data, [25, 50, 75])
            iqr = q3 - q1
            w_lo = max(log_data[log_data >= q1 - 1.5 * iqr].min(), log_data.min())
            w_hi = min(log_data[log_data <= q3 + 1.5 * iqr].max(), log_data.max())

            box_w = 0.12
            ax.add_patch(
                plt.Rectangle(
                    (i + 1 - box_w / 2, q1),
                    box_w,
                    iqr,
                    facecolor='white',
                    edgecolor='#333333',
                    linewidth=0.8,
                    zorder=3,
                )
            )
            ax.plot(
                [i + 1 - box_w / 2, i + 1 + box_w / 2],
                [med, med],
                color='#333333',
                linewidth=1.2,
                zorder=4,
            )
            ax.plot(
                [i + 1, i + 1], [w_lo, q1], color='#333333', linewidth=0.7, zorder=3
            )
            ax.plot(
                [i + 1, i + 1], [q3, w_hi], color='#333333', linewidth=0.7, zorder=3
            )

            # ── Outliers ──────────────────────────────────────────────────────────────
            outliers = log_data[(log_data < w_lo) | (log_data > w_hi)]
            if len(outliers):
                ax.scatter(
                    np.full_like(outliers, i + 1),
                    outliers,
                    color=color,
                    s=6,
                    alpha=0.6,
                    linewidths=0,
                    zorder=5,
                    rasterized=True,
                )

        # ── Axes & ticks ──────────────────────────────────────────────────────────────
        ax.set_xlim(0.3, n_groups + 0.7)

        y_min_raw = violin_df['num_parameters'].min()
        y_max_raw = violin_df['num_parameters'].max()
        ax.set_ylim(np.log10(y_min_raw) - 0.15, np.log10(y_max_raw) + 0.15)

        # Restore human-readable tick labels (10^k) on the log-transformed y axis
        y_ticks = np.arange(
            np.floor(np.log10(y_min_raw)),
            np.ceil(np.log10(y_max_raw)) + 1,
        )
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([r'$10^{%d}$' % int(t) for t in y_ticks])

        ax.set_xticks(range(1, n_groups + 1))
        ax.set_xticklabels(x_order, rotation=45, ha='right', rotation_mode='anchor')

        ax.set_xlabel('Family')
        ax.set_ylabel('Number of Parameters')

        # Separate "Other" visually with a dashed vertical line
        ax.axvline(TOP_N + 0.5, color='#999999', linewidth=0.7, linestyle=':', zorder=1)
        ax.text(
            TOP_N + 0.55,
            ax.get_ylim()[1] - 0.1,
            'other',
            fontsize=6.5,
            color='#888888',
            va='top',
            style='italic',
        )

        # ── Export ────────────────────────────────────────────────────────────────────
        fig.savefig(OUTPUT_PATH, format='pdf')
        plt.show()

    parameter_count_distribution_by_family()
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
def _(df, mo, np, pl, px):

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
            # torchvision v1 and v2 recipes are the same organisation: merged here (the registry keeps them apart)
            .with_columns(pretrain_org=pl.col('pretrain_org').replace({'torchvision-v2': 'torchvision'}))
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
    return (tree_df,)


@app.cell
def _():
    import kaleido

    return


@app.cell(hide_code=True)
def _(OUT, tree_df):
    def _():
        """Who trained what? Nested circle packing (organisation -> model family).

        Circle area is proportional to the number of checkpoints. Every organisation is labelled directly:
        large ones inside their circle; the ones whose name does not fit are labelled next to the circle.
        Labels are placed only where they do not collide with an already placed label.
        """
        import math

        import circlify
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from types import SimpleNamespace

        from matplotlib.patches import Circle

        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
            }
        )
        tree_pd = tree_df.to_pandas()  # polars -> pandas; columns: pretrain_org, family, len

        # ── palette: the 11 largest organisations get a hand-picked colour, the others a generated one ──────────
        COLOURS = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7',
                   '#882255', '#117733', '#DDCC77', '#6699CC', '#AA4499']
        GREY = '#8C8C8C'

        def rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

        def tint(c, t):
            return tuple(x + (1 - x) * t for x in c)

        def shade(c, s):
            return tuple(x * (1 - s) for x in c)

        def lum(c):
            lin = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])

        # The timm training recipes ("timm D recipe", "timm RA2 recipe", ...) are not organisations: they form one macro-group
        # "timm" (organisation -> recipe -> family); every other organisation is organisation -> family.
        group_of = lambda o: 'TIMM' if o.startswith('timm ') else o
        tree_pd['group'] = tree_pd['pretrain_org'].map(group_of)
        totals = tree_pd.groupby('group')['len'].sum().sort_values(ascending=False)
        orgs = totals.index.tolist()  # groups, largest first
        import colorsys

        def extra(i):  # beyond the 11 hand-picked colours: golden-angle hues, so that neighbours in the ranking differ
            h = (0.05 + 0.618034 * i) % 1.0
            return colorsys.hsv_to_rgb(h, 0.55 if i % 2 else 0.70, 0.80 if i % 3 else 0.68)

        hue = {o: rgb(COLOURS[i]) if i < len(COLOURS) else extra(i - len(COLOURS)) for i, o in enumerate(orgs)}
        rec_tot = tree_pd.groupby('pretrain_org')['len'].sum()
        members = {g: rec_tot[tree_pd[tree_pd['group'] == g]['pretrain_org'].unique()].sort_values(ascending=False).index.tolist() for g in orgs}
        fams = {r: tree_pd[tree_pd['pretrain_org'] == r].sort_values('len', ascending=False).reset_index(drop=True) for r in rec_tot.index}
        total = int(totals.sum())
        n_orgs = tree_pd['pretrain_org'].nunique()
        n_recipes = len(members.get('TIMM', []))

        # ── packing ──────────────────────────────────────────────────────────────────────────────────
        SMALL = 30  # one single packing of all the organisations: the smallest end up at the top left, the biggest at the bottom right
        small_orgs = [o for o in orgs if totals[o] < SMALL]
        big_orgs = [o for o in orgs if totals[o] >= SMALL]
        big_vals = [float(totals[o]) for o in big_orgs]
        bc = list(reversed(circlify.circlify(big_vals, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))))
        pos = {o: SimpleNamespace(x=c.x, y=c.y, r=c.r) for o, c in zip(big_orgs, bc)}
        r_pack = max(math.hypot(c.x, c.y) + c.r for c in bc)
        k = bc[0].r / math.sqrt(big_vals[0])  # same area scale for the small circles
        rs = {o: k * math.sqrt(totals[o]) for o in small_orgs}

        # The small circles are placed around the big packing: each one comes in from the outside along a ray and stops when it touches a circle,
        # then the ray that lets it get closest to the centre wins. So they nestle into the concavities of the border but never sit among the big circles.
        GAP, LABEL_W = 0.012, 0.75
        discs = [(c.x, c.y, c.r) for c in bc]
        for o in big_orgs:  # the name of a macro-group that does not fit inside is written above it: keep that spot free
            if o == 'TIMM':
                discs += [(pos[o].x + dx, pos[o].y + pos[o].r + 0.05, 0.06) for dx in (-0.13, 0.0, 0.13)]
        placed_deg = []
        for o in sorted(small_orgs, key=lambda o: (-totals[o], len(o))):  # innermost first
            r, best = rs[o], None
            for deg in range(0, 360, 2):
                ux, uy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
                d = r_pack + r + 0.5
                STEP = 0.002
                while d > 0:  # never deeper than the outline of the packing: no small circle sits in a concavity between big ones
                    x, y = (d - STEP) * ux, (d - STEP) * uy
                    if any(math.hypot(x - cx, y - cy) < cr + r + GAP for cx, cy, cr in discs):
                        break
                    d -= STEP
                cost = d + 0.05 * sum(1 for q in placed_deg if abs((deg - q + 180) % 360 - 180) < 25)  # nestle in, but spread around the packing
                if best is None or cost < best[0]:
                    best = (cost, d * ux, d * uy)
            pos[o] = SimpleNamespace(x=best[1], y=best[2], r=r)
            discs.append((best[1], best[2], r))
            placed_deg.append(math.degrees(math.atan2(best[2], best[1])) % 360)
        # A tiny circle enclosed by the others (no free ray to the outside) is moved to the border of the packing, towards the top left,
        # where the other small circles are: it comes in from the outside along a ray and stops when it touches a circle.
        def enclosed(o):
            p = pos[o]
            others = [(q.x, q.y, q.r) for k, q in pos.items() if k != o]
            for deg in range(0, 360, 5):
                ux, uy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
                if all(math.hypot(p.x + t * ux - cx, p.y + t * uy - cy) >= cr + p.r for t in [0.01 * k for k in range(1, 260)] for cx, cy, cr in others):
                    return False
            return True

        for o in [o for o in orgs if totals[o] < 10 and False]:  # only the tiny ones: the big circles are meant to be in the middle
            others = [(q.x, q.y, q.r) for k, q in pos.items() if k != o]
            r, best = pos[o].r, None
            for deg in range(0, 360, 2):
                ux, uy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
                d = r_pack + r + 0.5
                while d > 0:
                    if any(math.hypot((d - 0.002) * ux - cx, (d - 0.002) * uy - cy) < cr + r + GAP for cx, cy, cr in others):
                        break
                    d -= 0.002
                diff = abs((math.radians(deg) - math.radians(135) + math.pi) % (2 * math.pi) - math.pi)  # angular distance from the top left
                if best is None or d + 0.25 * diff < best[0]:
                    best = (d + 0.25 * diff, d * ux, d * uy)
            print('moved to the border (it was enclosed):', o)
            pos[o] = SimpleNamespace(x=best[1], y=best[2], r=r)
        # OpenGVLab goes to the bottom left of LAION (nearest free spot to 225 degrees around it), away from the names at the bottom
        if 'OpenGVLab' in pos and 'LAION' in pos:
            _g, _l = pos['OpenGVLab'], pos['LAION']
            _others = [(q.x, q.y, q.r) for k_, q in pos.items() if k_ != 'OpenGVLab']
            _best = None
            for _off in range(0, 90, 2):
                for _sgn in (1, -1):
                    _a = math.radians(225 + _sgn * _off)
                    _x, _y = _l.x + (_l.r + _g.r + GAP) * math.cos(_a), _l.y + (_l.r + _g.r + GAP) * math.sin(_a)
                    if all(math.hypot(_x - cx, _y - cy) >= cr + _g.r + GAP * 0.5 for cx, cy, cr in _others):
                        _best = (_x, _y)
                        break
                if _best:
                    break
            if _best:
                pos['OpenGVLab'] = SimpleNamespace(x=_best[0], y=_best[1], r=_g.r)
                print('OpenGVLab moved to the bottom left of LAION')
        # Alibaba goes to the bottom right of Baidu/PaddlePaddle: next to Shopee its name was read as Shopee's
        if 'Alibaba' in pos and 'Baidu/PaddlePaddle' in pos:
            _g, _l = pos['Alibaba'], pos['Baidu/PaddlePaddle']
            _others = [(q.x, q.y, q.r) for k_, q in pos.items() if k_ != 'Alibaba']
            _best = None
            for _off in range(0, 90, 2):
                for _sgn in (1, -1):
                    _a = math.radians(315 + _sgn * _off)
                    _x, _y = _l.x + (_l.r + _g.r + GAP) * math.cos(_a), _l.y + (_l.r + _g.r + GAP) * math.sin(_a)
                    if all(math.hypot(_x - cx, _y - cy) >= cr + _g.r + GAP * 0.5 for cx, cy, cr in _others):
                        _best = (_x, _y)
                        break
                if _best:
                    break
            if _best:
                pos['Alibaba'] = SimpleNamespace(x=_best[0], y=_best[1], r=_g.r)
                print('Alibaba moved to the bottom right of Baidu/PaddlePaddle')
        outer = [pos[o] for o in orgs]
        small_set = set(small_orgs)
        x_lo = min(c.x - c.r for c in outer) - LABEL_W
        x_hi = max(c.x + c.r for c in outer) + LABEL_W
        y_lo, y_hi = min(c.y - c.r for c in outer) - 0.03, max(c.y + c.r for c in outer) + 0.03
        X_PAD, Y_PAD = 0.0, 0.02
        FIG_W = 12.5
        FIG_H = FIG_W * ((y_hi - y_lo) + 2 * Y_PAD) / ((x_hi - x_lo) + X_PAD) + 0.95  # + title
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(x_lo - 0.02, x_hi + X_PAD)
        ax.set_ylim(y_lo - Y_PAD, y_hi + Y_PAD)
        lim = r_pack
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed = []  # window-space bboxes of the labels already placed

        def free(t, pad=2.0):
            bb = t.get_window_extent(renderer).expanded(1, 1)
            bb = type(bb)(((bb.x0 - pad, bb.y0 - pad), (bb.x1 + pad, bb.y1 + pad)))
            return not any(bb.overlaps(p) for p in placed), bb

        def put(x, y, s, fit_w=None, size=9, min_size=5.0, **kw):
            """Text at (x, y); s = one string or a list of variants (e.g. one and two lines). Shrinks until it is narrower
            than fit_w (data units) and collides with no placed label; returns None if it cannot be placed."""
            halo = kw.pop('halo', None)
            variants = [s] if isinstance(s, str) else s
            # a variant is tried down to 3 pt below the starting size before the next one (e.g. the two-line one) is considered
            for k, v in enumerate(variants):
                floor = min_size if k == len(variants) - 1 else max(min_size, size - 3)
                sz = size
                while sz >= floor:
                    t = ax.text(x, y, v, fontsize=sz, zorder=7, linespacing=0.95, **kw)
                    if halo:
                        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=halo)])
                    w = t.get_window_extent(renderer)
                    inv = ax.transData.inverted()
                    wd = inv.transform((w.x1, w.y0))[0] - inv.transform((w.x0, w.y0))[0]
                    ok, bb = free(t)
                    if ok and (fit_w is None or wd <= fit_w):
                        placed.append(bb)
                        return t
                    t.remove()
                    sz -= 0.5
            return None

        # ── draw circles (labels afterwards, so that every label sees every circle) ─────────────────
        fam_draw, rec_draw = [], []

        def pack(values, x, y, r, shrink=0.94):
            """Circles (x, y, r) packed inside the circle (x, y, r), same order as `values` (sorted descending)."""
            if len(values) == 1:
                return [(x, y, r * 0.92)]
            cs = list(reversed(circlify.circlify(values, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))))
            return [(x + c.x * r * shrink, y + c.y * r * shrink, c.r * r * shrink) for c in cs]

        def draw_families(recipe, x, y, r, base):
            f = fams[recipe]
            n = len(f)
            for (fx, fy, fr), (rank, row) in zip(pack(f['len'].tolist(), x, y, r), f.iterrows()):
                col = tint(base, 0.06 + 0.50 * rank / max(n - 1, 1))
                ax.add_patch(Circle((fx, fy), fr, facecolor=(*col, 0.95), edgecolor='white', linewidth=0.5, zorder=4))
                fam_draw.append((fx, fy, fr, row['family'], int(row['len']), col))

        for circ, org in zip(outer, orgs):
            cx, cy, cr = circ.x, circ.y, circ.r
            base, dark = hue[org], shade(hue[org], 0.35)
            ax.add_patch(Circle((cx, cy), cr, facecolor=(*dark, 0.09), edgecolor=(*dark, 0.75), linewidth=1.1, zorder=2))
            if members[org] == [org]:
                draw_families(org, cx, cy, cr, base)
            else:  # macro-group: one circle per recipe, families inside
                for (rx, ry, rr), rec in zip(pack([float(rec_tot[m]) for m in members[org]], cx, cy, cr, 0.95), members[org]):
                    ax.add_patch(Circle((rx, ry), rr, facecolor=(*tint(base, 0.80), 0.55), edgecolor=(*dark, 0.55), linewidth=0.7, zorder=3))
                    draw_families(rec, rx, ry, rr, base)
                    rec_draw.append((rx, ry, rr, rec, int(rec_tot[rec]), dark))

        # ── recipe labels, before the organisation labels so that they keep priority (inside the recipe circle; "timm " is implied by the group) ──────────────────
        for rx, ry, rr, name, cnt, dark in sorted(rec_draw, key=lambda r: -r[2]):
            short = name.removeprefix('timm ').removesuffix(' recipe')  # "D", "RA2", ...
            put(rx, ry + rr * 0.76, [f'{short}  ({cnt})', f'{short}\n({cnt})'], fit_w=rr * 1.8, size=10, min_size=5.0,
                ha='center', va='center', color=shade(dark, 0.1), fontweight='bold', halo=(1, 1, 1, 0.85))

        # ── organisation labels ─────────────────────────────────────────────────────────────────────
        order_by_size = sorted(range(len(orgs)), key=lambda i: -outer[i].r)
        def around(i, s_, dark):
            """Name of a small circle next to it, on the side where it disturbs the least (it may touch the neighbouring circles a bit,
            but never another label); returns True when placed."""
            c = outer[i]
            out = math.atan2(c.y, c.x)
            inv_ = ax.transData.inverted()
            best = None
            for sz in (9.5, 9, 8.5):
                for deg in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                    ux, uy = math.cos(out + math.radians(deg)), math.sin(out + math.radians(deg))
                    ha = 'left' if ux > 0.45 else 'right' if ux < -0.45 else 'center'
                    va = 'bottom' if (ha == 'center' and uy > 0) else 'top' if (ha == 'center' and uy < 0) else 'center'
                    t = ax.text(c.x + (c.r + 0.008) * ux, c.y + (c.r + 0.008) * uy, s_, ha=ha, va=va, fontsize=sz, fontweight='bold', color=dark, zorder=7)
                    ok, bb = free(t)
                    if ok:
                        (x0, y0), (x1, y1) = inv_.transform((bb.x0, bb.y0)), inv_.transform((bb.x1, bb.y1))
                        pen = sum(1 for j, o in enumerate(outer) if j != i and math.hypot(min(max(o.x, x0), x1) - o.x, min(max(o.y, y0), y1) - o.y) < o.r)
                        pen += 0.05 * abs(deg) / 30
                        if best is None or pen < best[0]:
                            best = (pen, ux, uy, ha, va, sz, bb)
                    t.remove()
                if best is not None:
                    break
            if best is None:
                return False
            _, ux, uy, ha, va, sz, bb = best
            t = ax.text(c.x + (c.r + 0.008) * ux, c.y + (c.r + 0.008) * uy, s_, ha=ha, va=va, fontsize=sz, fontweight='bold', color=dark, zorder=7)
            t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=(1, 1, 1, 0.95))])
            placed.append(bb)
            return True

        ring = []  # small organisations whose name does not fit on their circle: named in the column on the left, with a line
        for i in order_by_size:
            c, org = outer[i], orgs[i]
            one = f'{org}  ({int(totals[org])})'
            two = f'{org}\n({int(totals[org])})'
            dark = shade(hue[org], 0.45)
            t = None
            size0 = 11.5 if c.r > 0.15 else 9.5
            small = org in small_set
            spots = [0.0] if small else [0.78] if members[org] != [org] else [0.78, 0.0]  # a macro-group keeps its centre for the recipes
            for floor in ((8.0,) if small else (7.5, 6.0)):  # first try to keep the name large, then let it shrink; top of the circle first
                for yf in spots:
                    t = put(c.x, c.y + c.r * yf, [one, two], fit_w=c.r * 1.55, size=size0, min_size=floor,
                            ha='center', va='center', color=dark, fontweight='bold', halo=(1, 1, 1, 0.9))
                    if t is not None:
                        break
                if t is not None:
                    break
            if t is None and small:
                if not around(i, one, dark):
                    ring.append(i)
            elif t is None:
                t = ax.text(c.x, c.y + c.r + 0.01, one, ha='center', va='bottom', fontsize=10, fontweight='bold', color=dark, zorder=7)
                placed.append(free(t)[1])

        # ── small organisations: names in two columns beside the packing, in the vertical order of their circles, joined by a thin line ──
        inv = ax.transData.inverted()
        _t = ax.text(0, 0, 'Ag (1)', fontsize=9, fontweight='bold')
        _bb = _t.get_window_extent(renderer)
        row_h = (inv.transform((0, _bb.y1))[1] - inv.transform((0, _bb.y0))[1]) * 1.35
        _t.remove()
        x_right = max(c.x + c.r for c in outer) + 0.06
        x_left = min(c.x - c.r for c in outer) - 0.06
        y_bot = min(c.y - c.r for c in outer)
        y_top_all = max(c.y + c.r for c in outer)
        import random

        def seg_x(p, q, r, s_):
            d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            return d(p, q, r) * d(p, q, s_) < 0 and d(r, s_, p) * d(r, s_, q) < 0

        def inside_len(p, q, skip):
            n_in = 0
            for k in range(1, 40):
                x, y = p[0] + (q[0] - p[0]) * k / 40, p[1] + (q[1] - p[1]) * k / 40
                if any(math.hypot(x - o.x, y - o.y) < o.r for j, o in enumerate(outer) if j != skip):
                    n_in += 1
            return n_in
        for side in (-1,):
            items = list(ring)
            n = len(items)
            if not n:
                continue
            xl = x_right if side > 0 else x_left
            mid = sum(outer[i].y for i in items) / n
            top = max(min(mid + row_h * (n - 1) / 2, y_top_all), y_bot + row_h * (n - 1))
            slots = [top - row_h * k for k in range(n)]
            order = sorted(items, key=lambda i: -outer[i].y)

            P = {i: (outer[i].x, outer[i].y) for i in items}
            base_c, cross_c = {}, {}

            def base(i, k):
                if (i, k) not in base_c:
                    p, q = P[i], (xl, slots[k])
                    base_c[(i, k)] = 0.3 * math.hypot(q[0] - p[0], q[1] - p[1]) + 0.15 * inside_len(p, q, i)
                return base_c[(i, k)]

            def crosses(i, k, j, l):
                key = (i, k, j, l) if (i, k) < (j, l) else (j, l, i, k)
                if key not in cross_c:
                    cross_c[key] = 6.0 if seg_x(P[i], (xl, slots[k]), P[j], (xl, slots[l])) else 0.0
                return cross_c[key]

            def local(order, a_, b_):
                tot = base(order[a_], a_) + base(order[b_], b_) + crosses(order[a_], a_, order[b_], b_)
                for m in range(n):
                    if m != a_ and m != b_:
                        tot += crosses(order[a_], a_, order[m], m) + crosses(order[b_], b_, order[m], m)
                return tot
            rnd = random.Random(0)
            T0, N_IT = 1.0, 60000
            for it in range(N_IT):
                if n < 2:
                    break
                a_, b_ = rnd.sample(range(n), 2)
                before = local(order, a_, b_)
                order[a_], order[b_] = order[b_], order[a_]
                delta = local(order, a_, b_) - before
                T = T0 * (1 - it / N_IT) + 1e-3
                if delta > 0 and rnd.random() > math.exp(-delta / T):
                    order[a_], order[b_] = order[b_], order[a_]
            for i, y in zip(order, slots):
                c, org = outer[i], orgs[i]
                dx, dy = xl - c.x, y - c.y
                d = math.hypot(dx, dy)
                ax.plot([c.x + c.r * dx / d, xl - side * 0.01], [c.y + c.r * dy / d, y], color=(*shade(hue[org], 0.35), 0.8), lw=0.6, zorder=6)
                t = ax.text(xl + side * 0.015, y, f'{org}  ({int(totals[org])})', ha='left' if side > 0 else 'right', va='center', fontsize=9,
                            fontweight='bold', color=shade(hue[org], 0.45), zorder=7)
                placed.append(free(t)[1])

        # ── family labels: biggest circles first, only where there is room and nothing is covered ─────
        for fx, fy, fr, name, cnt, col in sorted(fam_draw, key=lambda r: -r[2]):
            if fr < 0.020:
                continue
            fg = (0.97, 0.97, 0.97) if lum(col) < 0.33 else (0.10, 0.10, 0.10)
            t = put(fx, fy + (fr * 0.12 if fr > 0.04 else 0), name, fit_w=fr * 1.75, size=min(10.5, 4 + fr * 95), min_size=5.5,
                    ha='center', va='center', color=fg)
            if t is not None and fr > 0.04:
                put(fx, fy - fr * 0.34, f'{cnt}', fit_w=fr * 1.2, size=max(5.5, t.get_fontsize() * 0.8), min_size=5.0,
                    ha='center', va='center', color=(*fg, 0.8))

        # ── title ───────────────────────────────────────────────────────────────────────────────────
        fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.9 / FIG_H, bottom=0.05 / FIG_H)
        fig.canvas.draw()
        x_left = min(t.get_window_extent(fig.canvas.get_renderer()).x0 for t in ax.texts)  # leftmost label: the title starts there
        x_title = fig.transFigure.inverted().transform((x_left, 0))[0]
        t_title = fig.text(x_title, 0.99, '(a) Models by training organization', ha='left', va='top', fontsize=20, fontweight='bold', color='#111111')
        t_sub = fig.text(x_title, 1 - 0.52 / FIG_H, 'Pretrained checkpoints by training organisation and model family', ha='left', va='top', fontsize=12.5, color='#555555')

        # crop to what is drawn (circles, labels, title): the axes rectangle would otherwise count in matplotlib's tight box
        from matplotlib.transforms import Bbox

        rend = fig.canvas.get_renderer()
        bbs = [x.get_window_extent(rend) for x in (*ax.patches, *ax.texts, *ax.lines, *fig.texts)]
        box = Bbox.from_extents(min(b.x0 for b in bbs), min(b.y0 for b in bbs), max(b.x1 for b in bbs), max(b.y1 for b in bbs))
        box = box.transformed(fig.dpi_scale_trans.inverted())
        box = Bbox.from_extents(box.x0 - 0.1, box.y0 - 0.1, box.x1 + 0.1, box.y1 + 0.1)
        for ext in ('pdf', 'png'):
            fig.savefig(str(OUT / f'models_per_company_packed.{ext}'), dpi=300, bbox_inches=box, facecolor='white', edgecolor='none')
        print('Saved: models_per_company_packed.pdf  +  models_per_company_packed.png')
        # variant without the subtitle (used by the side-by-side figure): the title moves down and the crop follows it
        t_sub.set_visible(False)
        t_title.set_y(0.99 - 0.40 / FIG_H)
        bbs = [x.get_window_extent(rend) for x in (*ax.patches, *ax.texts, *ax.lines, *fig.texts) if x.get_visible()]
        box = Bbox.from_extents(min(b.x0 for b in bbs), min(b.y0 for b in bbs), max(b.x1 for b in bbs), max(b.y1 for b in bbs)).transformed(fig.dpi_scale_trans.inverted())
        box = Bbox.from_extents(box.x0 - 0.1, box.y0 - 0.1, box.x1 + 0.1, box.y1 + 0.1)
        for ext in ('pdf', 'png'):
            fig.savefig(str(OUT / f'models_per_company_packed_nosub.{ext}'), dpi=300, bbox_inches=box, facecolor='white', edgecolor='none')
        t_sub.set_visible(True)
        t_title.set_y(0.99)
        return fig, {o: tuple(hue[o]) for o in orgs}  # the figure and the colour of every organisation (reused by the Sankey)



    fig_packed, org_hue = _()
    fig_packed
    return


@app.cell(hide_code=True)
def _(OUT, df, pl):
    def _():
        """What was it trained on? Nested circle packing (pretraining dataset -> fine-tuning dataset -> model family).

        Circle area is proportional to the number of checkpoints. Every organisation is labelled directly:
        large ones inside their circle; the small ones (< SMALL checkpoints) are stacked in two columns on the left of the packing,
        each with its name next to it, so that no label needs a leader line.
        Labels are placed only where they do not collide with an already placed label.
        """
        import math

        import circlify
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from types import SimpleNamespace

        from matplotlib.patches import Circle

        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
            }
        )
        # Column "pretrain_org" holds the middle-level key (the stage): the pretraining dataset alone, or "dataset → fine-tuning dataset".
        tree_pd = (
            df.filter(pl.col('pretrain_dataset').is_not_null()).with_columns(pretrain_dataset=pl.col('pretrain_dataset').list.join(' + '))  # run-time key: one unit per checkpoint (pretrain_dataset is a list)
            .with_columns(
                pretrain_org=pl.when(pl.col('pretrain_ft').is_not_null())
                .then(pl.col('pretrain_dataset') + ' → ' + pl.col('pretrain_ft').list.join(' → '))
                .otherwise(pl.col('pretrain_dataset'))
            )
            .group_by(['pretrain_org'])
            .len()
            .to_pandas()
        )

        # ── palette: the 11 largest organisations get a colour, the others a neutral grey ──────────
        COLOURS = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7',
                   '#882255', '#117733', '#DDCC77', '#6699CC', '#AA4499']
        GREY = '#8C8C8C'

        def rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

        def tint(c, t):
            return tuple(x + (1 - x) * t for x in c)

        def shade(c, s):
            return tuple(x * (1 - s) for x in c)

        def lum(c):
            lin = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])

        # Outer circle = pretraining dataset; inside it one circle per fine-tuning stage ("A → B"), then the model families.
        group_of = lambda o: o.split(' → ')[0]
        tree_pd['group'] = tree_pd['pretrain_org'].map(group_of)
        totals = tree_pd.groupby('group')['len'].sum().sort_values(ascending=False)
        orgs = totals.index.tolist()  # groups, largest first
        hue = {o: rgb(COLOURS[i]) if i < len(COLOURS) else rgb(GREY) for i, o in enumerate(orgs)}
        rec_tot = tree_pd.groupby('pretrain_org')['len'].sum()
        members = {g: rec_tot[tree_pd[tree_pd['group'] == g]['pretrain_org'].unique()].sort_values(ascending=False).index.tolist() for g in orgs}
        total = int(totals.sum())
        n_orgs = tree_pd['pretrain_org'].nunique()
        n_ft = sum(1 for o in tree_pd['pretrain_org'].unique() if ' → ' in o)

        # ── packing ──────────────────────────────────────────────────────────────────────────────────
        SMALL = 20  # organisations with fewer checkpoints go to the columns on the left
        small_orgs = [o for o in orgs if totals[o] < SMALL]
        big_orgs = [o for o in orgs if totals[o] >= SMALL]
        big_vals = [float(totals[o]) for o in big_orgs]
        bc = list(reversed(circlify.circlify(big_vals, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))))
        pos = {o: SimpleNamespace(x=c.x, y=c.y, r=c.r) for o, c in zip(big_orgs, bc)}
        r_pack = max(math.hypot(c.x, c.y) + c.r for c in bc)
        k = bc[0].r / math.sqrt(big_vals[0])  # same area scale for the small circles
        rs = {o: k * math.sqrt(totals[o]) for o in small_orgs}

        # two columns, largest first, top to bottom; labels (right-aligned) to the left of each circle
        GAP, LABEL_W, COL_GAP = 0.035, 0.52, 0.07
        height = lambda os_: sum(2 * rs[o] + GAP for o in os_)
        col_b, acc = [], 0.0  # column B: the one next to the big packing
        for o in small_orgs:
            if acc >= (height(small_orgs)) / 2:
                break
            col_b.append(o)
            acc += 2 * rs[o] + GAP
        col_a = [o for o in small_orgs if o not in col_b]
        xb = min(c.x - c.r for c in bc) - COL_GAP - max(rs[o] for o in col_b)
        xa = xb - max(rs[o] for o in col_b) - LABEL_W - COL_GAP - max(rs[o] for o in col_a)
        for col, xc in ((col_b, xb), (col_a, xa)):
            y = height(col) / 2
            for o in col:
                pos[o] = SimpleNamespace(x=xc, y=y - rs[o], r=rs[o])
                y -= 2 * rs[o] + GAP
        outer = [pos[o] for o in orgs]
        small_set = set(small_orgs)
        x_lo = xa - max(rs[o] for o in col_a) - LABEL_W - 0.02
        x_hi = max(c.x + c.r for c in bc)
        y_lo, y_hi = min(c.y - c.r for c in outer), max(c.y + c.r for c in outer)
        X_PAD, Y_PAD = 0.60, 0.06  # room on the right for a label of the big packing, and a small margin (y)
        FIG_W = 12.5
        FIG_H = FIG_W * ((y_hi - y_lo) + 2 * Y_PAD) / ((x_hi - x_lo) + X_PAD) + 1.5  # + title and caption
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=150)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(x_lo - 0.02, x_hi + X_PAD)
        ax.set_ylim(y_lo - Y_PAD, y_hi + Y_PAD)
        lim = r_pack
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed = []  # window-space bboxes of the labels already placed

        def free(t, pad=2.0):
            bb = t.get_window_extent(renderer).expanded(1, 1)
            bb = type(bb)(((bb.x0 - pad, bb.y0 - pad), (bb.x1 + pad, bb.y1 + pad)))
            return not any(bb.overlaps(p) for p in placed), bb

        def put(x, y, s, fit_w=None, size=9, min_size=5.0, **kw):
            """Text at (x, y); s = one string or a list of variants (e.g. one and two lines). Shrinks until it is narrower
            than fit_w (data units) and collides with no placed label; returns None if it cannot be placed."""
            halo = kw.pop('halo', None)
            variants = [s] if isinstance(s, str) else s
            # a variant is tried down to 3 pt below the starting size before the next one (e.g. the two-line one) is considered
            for k, v in enumerate(variants):
                floor = min_size if k == len(variants) - 1 else max(min_size, size - 3)
                sz = size
                while sz >= floor:
                    t = ax.text(x, y, v, fontsize=sz, zorder=7, linespacing=0.95, **kw)
                    if halo:
                        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=halo)])
                    w = t.get_window_extent(renderer)
                    inv = ax.transData.inverted()
                    wd = inv.transform((w.x1, w.y0))[0] - inv.transform((w.x0, w.y0))[0]
                    ok, bb = free(t)
                    if ok and (fit_w is None or wd <= fit_w):
                        placed.append(bb)
                        return t
                    t.remove()
                    sz -= 0.5
            return None

        # ── draw circles (labels afterwards, so that every label sees every circle) ─────────────────
        stage_draw = []

        def pack(values, x, y, r, shrink=0.94):
            """Circles (x, y, r) packed inside the circle (x, y, r), same order as `values` (sorted descending)."""
            if len(values) == 1:
                return [(x, y, r * 0.92)]
            cs = list(reversed(circlify.circlify(values, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))))
            return [(x + c.x * r * shrink, y + c.y * r * shrink, c.r * r * shrink) for c in cs]

        for circ, org in zip(outer, orgs):
            cx, cy, cr = circ.x, circ.y, circ.r
            base, dark = hue[org], shade(hue[org], 0.35)
            if members[org] == [org]:  # no fine-tuned checkpoints: the dataset circle is the leaf
                ax.add_patch(Circle((cx, cy), cr, facecolor=(*tint(base, 0.15), 0.95), edgecolor=(*dark, 0.8), linewidth=1.1, zorder=3))
                continue
            ax.add_patch(Circle((cx, cy), cr, facecolor=(*dark, 0.09), edgecolor=(*dark, 0.75), linewidth=1.1, zorder=2))
            n = len(members[org])
            for rank, ((rx, ry, rr), st) in enumerate(zip(pack([float(rec_tot[m]) for m in members[org]], cx, cy, cr, 0.95), members[org])):
                col = tint(base, 0.10 + 0.45 * rank / max(n - 1, 1))
                ax.add_patch(Circle((rx, ry), rr, facecolor=(*col, 0.95), edgecolor='white', linewidth=0.8, zorder=4))
                stage_draw.append((rx, ry, rr, st, int(rec_tot[st]), col))

        # ── stage labels, before the dataset labels so that they keep priority ──────────────────
        for rx, ry, rr, name, cnt, col in sorted(stage_draw, key=lambda r: -r[2]):
            short = '→ ' + ' → '.join(name.split(' → ')[1:]) if ' → ' in name else 'no fine-tuning'
            fg = (0.97, 0.97, 0.97) if lum(col) < 0.33 else (0.10, 0.10, 0.10)
            abbr = short.replace('ImageNet-', 'IN-').replace('fine-tuning', 'no FT') if short.startswith('→') else 'no FT'
            put(rx, ry, [f'{short}  ({cnt})', f'{short}\n({cnt})', f'{abbr}\n({cnt})'], fit_w=rr * 1.8, size=10, min_size=5.0, ha='center', va='center', color=fg)

        # ── organisation labels ─────────────────────────────────────────────────────────────────────
        order_by_size = sorted(range(len(orgs)), key=lambda i: -outer[i].r)
        ring = []  # big organisations whose name does not fit inside: labelled outside, on the right
        for i in order_by_size:
            c, org = outer[i], orgs[i]
            if org in small_set:
                ax.text(c.x - c.r - 0.03, c.y, f'{org}  ({int(totals[org])})', ha='right', va='center', fontsize=9, fontweight='bold',
                        color=shade(hue[org], 0.45), zorder=7)
                continue
            one = f'{org}  ({int(totals[org])})'
            two = f'{org}\n({int(totals[org])})'
            dark = shade(hue[org], 0.45)
            t = None
            size0 = 11.5 if c.r > 0.15 else 9.5
            spots = [0.78] if members[org] != [org] else [0.78, 0.0]  # a macro-group keeps its centre for the recipes
            for floor in (7.5, 6.0):  # first try to keep the name large, then let it shrink; top of the circle first
                for yf in spots:
                    t = put(c.x, c.y + c.r * yf, [one, two], fit_w=c.r * 1.55, size=size0, min_size=floor,
                            ha='center', va='center', color=dark, fontweight='bold', halo=(1, 1, 1, 0.9))
                    if t is not None:
                        break
                if t is not None:
                    break
            if t is None and c.x < 0:  # no room inside, and the right-hand ring would need a long leader line: outside the circle, no width limit
                for dy, va in ((c.r + 0.01, 'bottom'), (-c.r - 0.01, 'top'), (c.r * 0.55, 'center')):
                    t = put(c.x, c.y + dy, one, size=9, min_size=6.5, ha='center', va=va, color=dark, fontweight='bold', halo=(1, 1, 1, 0.9))
                    if t is not None:
                        break
            if t is None:
                ring.append(i)

        # ring labels: on the left and on the right of the packing, sorted by height, one per row, leader lines to the circles
        skipped = [orgs[i] for i in ring if outer[i].x < 0]
        if skipped:
            print('WARNING: no room for the label of', skipped)
        for side in (+1,):
            items = sorted([i for i in ring if (outer[i].x >= 0) == (side > 0)], key=lambda i: outer[i].y)
            ys = []
            for k, i in enumerate(items):
                y = outer[i].y
                if ys:
                    y = max(y, ys[-1] + 0.062)
                ys.append(y)
            # shift the block down if it ran past the top
            over = (ys[-1] - y_hi) if ys else 0
            if over > 0:
                ys = [y - over for y in ys]
            for i, y in zip(items, ys):
                c, org = outer[i], orgs[i]
                x_lab = side * (math.sqrt(max(r_pack**2 - y**2, 0.0)) + 0.07)
                ang = math.atan2(y - c.y, x_lab - c.x)
                ax.plot([c.x + c.r * math.cos(ang), x_lab - side * 0.012], [c.y + c.r * math.sin(ang), y],
                        color=(*shade(hue[org], 0.35), 0.8), lw=0.7, zorder=1)
                t = ax.text(x_lab, y, f'{org}  ({int(totals[org])})', ha='left' if side > 0 else 'right', va='center', fontsize=8.5,
                            color=shade(hue[org], 0.45), fontweight='bold', zorder=7)
                placed.append(t.get_window_extent(renderer).expanded(1.02, 1.1))

        # ── title / caption ─────────────────────────────────────────────────────────────────────────
        fig.text(0.015, 0.99, 'What was it trained on?', ha='left', va='top', fontsize=20, fontweight='bold', color='#111111')
        fig.text(0.015, 1 - 0.52 / FIG_H, 'Pretrained checkpoints by pretraining dataset and fine-tuning dataset', ha='left', va='top', fontsize=12.5, color='#555555')
        fig.text(0.015, 0.012,
                 f'Circle area is proportional to the number of checkpoints; only the {total:,} checkpoints with a known pretraining dataset '
                 f'({n_orgs} datasets) are shown. Outer circle: pretraining dataset (the 11 largest have their own colour, the others are grey); '
                 f'inside a dataset with fine-tuned checkpoints: one circle per fine-tuning dataset ("→ X", {n_ft} stages) and one for the checkpoints '
                 'without fine-tuning ("no FT"; IN = ImageNet). Names are followed by their number of checkpoints; stages with very few checkpoints are not labelled.',
                 ha='left', va='bottom', fontsize=8.5, color='#666666', style='italic', wrap=True)
        fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.0 / FIG_H, bottom=0.55 / FIG_H)

        for ext in ('pdf', 'png'):
            fig.savefig(str(OUT / f'pretrain_datasets.{ext}'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print('Saved: pretrain_datasets.pdf  +  pretrain_datasets.png')
        return fig

    _()
    return


@app.cell(hide_code=True)
def _(OUT, df, pl):
    def _():
        """Alluvial diagram: pretraining dataset -> fine-tuning dataset -> learning paradigm (checkpoint counts)."""
        import plotly.graph_objects as go

        SMALL = 10  # datasets with fewer checkpoints are merged into "Other"
        d = (
            df.filter(pl.col('pretrain_dataset').is_not_null()).with_columns(pretrain_dataset=pl.col('pretrain_dataset').list.join(' + '))  # run-time key: one unit per checkpoint (pretrain_dataset is a list)
            .with_columns(
                ft=pl.when(pl.col('pretrain_ft').is_not_null()).then(pl.lit('→ ') + pl.col('pretrain_ft').list.join(' → ')).otherwise(pl.lit('no fine-tuning')),
                paradigm=pl.col('paradigm').str.replace_all('_', ' '),
            )
        )
        size = d.group_by('pretrain_dataset').len()
        small = size.filter(pl.col('len') < SMALL)['pretrain_dataset'].to_list()
        d = d.with_columns(
            ds=pl.when(pl.col('pretrain_dataset').is_in(small)).then(pl.lit(f'Other ({len(small)} datasets)')).otherwise(pl.col('pretrain_dataset'))
        )
        ds_order = d.group_by('ds').len().sort('len', descending=True)['ds'].to_list()
        ds_order = [x for x in ds_order if not x.startswith('Other')] + [x for x in ds_order if x.startswith('Other')]
        ft_order = ['no fine-tuning'] + sorted({x for x in d['ft'].unique().to_list() if x != 'no fine-tuning'})
        pa_order = d.group_by('paradigm').len().sort('len', descending=True)['paradigm'].to_list()

        labels = ds_order + ft_order + pa_order
        idx = {('ds', k): i for i, k in enumerate(ds_order)}
        idx |= {('ft', k): len(ds_order) + i for i, k in enumerate(ft_order)}
        idx |= {('pa', k): len(ds_order) + len(ft_order) + i for i, k in enumerate(pa_order)}

        OI = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#882255', '#117733', '#DDCC77', '#6699CC', '#AA4499']
        GREY = '#8C8C8C'
        col_ds = {k: OI[i] if i < len(OI) and not k.startswith('Other') else GREY for i, k in enumerate(ds_order)}
        col_pa = dict(zip(pa_order, ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00'][: len(pa_order)]))
        rgba = lambda h, a: f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"

        src, tgt, val, lcol = [], [], [], []
        for a, b, n in d.group_by('ds', 'ft').len().iter_rows():
            src.append(idx[('ds', a)]); tgt.append(idx[('ft', b)]); val.append(n); lcol.append(rgba(col_ds[a], 0.45))
        for a, b, n in d.group_by('ft', 'paradigm').len().iter_rows():
            src.append(idx[('ft', a)]); tgt.append(idx[('pa', b)]); val.append(n); lcol.append(rgba(col_pa[b], 0.45))
        ncol = [col_ds[k] for k in ds_order] + ['#B8B8B8'] * len(ft_order) + [col_pa[k] for k in pa_order]
        tot = {('ds', k): int(v) for k, v in d.group_by('ds').len().iter_rows()}
        tot |= {('ft', k): int(v) for k, v in d.group_by('ft').len().iter_rows()}
        tot |= {('pa', k): int(v) for k, v in d.group_by('paradigm').len().iter_rows()}
        nlabels = [f'{k}  ({tot[(c, k)]})' for c, ks in (('ds', ds_order), ('ft', ft_order), ('pa', pa_order)) for k in ks]

        fig = go.Figure(go.Sankey(
            arrangement='snap',
            node=dict(label=nlabels, color=ncol, pad=14, thickness=16, line=dict(color='white', width=0.5)),
            link=dict(source=src, target=tgt, value=val, color=lcol),
        ))
        fig.update_layout(
            title=dict(text=f'<b>What was it trained on?</b><br><sup>{d.height:,} checkpoints: pretraining dataset → fine-tuning dataset → paradigm '
                            f'(datasets with fewer than {SMALL} checkpoints merged into "Other")</sup>', x=0.01),
            font=dict(family='Times New Roman, serif', size=13), height=520, width=1250, margin=dict(l=10, r=10, t=90, b=10),
            paper_bgcolor='white',
        )
        fig.write_html(str(OUT / 'pretrain_sankey.html'))
        try:
            for ext in ('pdf', 'png'):
                fig.write_image(str(OUT / f'pretrain_sankey.{ext}'), scale=3 if ext == 'png' else 1)
            print('Saved: pretrain_sankey.html / .pdf / .png')
        except Exception as e:  # kaleido needs a Chrome
            print('Saved: pretrain_sankey.html only;', type(e).__name__, str(e)[:120])
        return fig


    _()
    return


@app.cell(hide_code=True)
def _(OUT, df, pl):
    def _():
        """Static treemap: outer level = fine-tuning stage, inner level = pretraining dataset (checkpoint counts).
        One data unit = one inch, so that the text sizes can be checked against the size of every tile."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'pdf.fonttype': 42, 'ps.fonttype': 42})
        SMALL = 10  # datasets with fewer checkpoints are merged into "Other"
        W, H = 10.0, 4.8
        d = (
            df.filter(pl.col('pretrain_dataset').is_not_null()).with_columns(pretrain_dataset=pl.col('pretrain_dataset').list.join(' + '))  # run-time key: one unit per checkpoint (pretrain_dataset is a list)
            .with_columns(ft=pl.when(pl.col('pretrain_ft').is_not_null()).then(pl.lit('→ ') + pl.col('pretrain_ft').list.join(' → ')).otherwise(pl.lit('no fine-tuning')))
        )
        small = d.group_by('pretrain_dataset').len().filter(pl.col('len') < SMALL)['pretrain_dataset'].to_list()
        d = d.with_columns(ds=pl.when(pl.col('pretrain_dataset').is_in(small)).then(pl.lit(f'Other ({len(small)} datasets)')).otherwise(pl.col('pretrain_dataset')))
        ds_order = d.group_by('ds').len().sort('len', descending=True)['ds'].to_list()
        ds_order = [x for x in ds_order if not x.startswith('Other')] + [x for x in ds_order if x.startswith('Other')]
        OI = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#882255', '#117733', '#DDCC77', '#6699CC', '#AA4499', '#44AA99']
        col = {k: (OI[i] if i < len(OI) and not k.startswith('Other') else '#9A9A9A') for i, k in enumerate(ds_order)}
        rgb = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))

        def lum(c):
            lin = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])

        def squarify(vals, x, y, w, h):
            """Squarified treemap: rectangles (x, y, w, h) for the values (descending), filling the rectangle."""
            tot, out = float(sum(vals)), []
            areas = [v / tot * w * h for v in vals]

            def worst(row, side):
                s = sum(row)
                return max(max(side**2 * r / s**2, s**2 / (side**2 * r)) for r in row)

            i = 0
            while i < len(areas):
                side = min(w, h)
                row = [areas[i]]
                j = i + 1
                while j < len(areas) and worst(row + [areas[j]], side) <= worst(row, side):
                    row.append(areas[j])
                    j += 1
                s = sum(row)
                if w >= h:  # column on the left
                    cw, cy = s / h, y
                    for a in row:
                        out.append((x, cy, cw, a / cw))
                        cy += a / cw
                    x, w = x + cw, w - cw
                else:  # row at the top
                    rh, cx = s / w, x
                    for a in row:
                        out.append((cx, y, a / rh, rh))
                        cx += a / rh
                    y, h = y + rh, h - rh
                i = j
            return out

        TOP, BOT = 0.75, 0.55  # room for the title and for the colour key
        fig = plt.figure(figsize=(W, H + TOP + BOT), dpi=150)
        ax = fig.add_axes([0, BOT / (H + TOP + BOT), 1, H / (H + TOP + BOT)])
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')  # y downwards
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()

        def fit(x, y, w, h, variants, colour, sizes=(10, 9, 8, 7, 6.5, 6, 5.5), bold=False):
            """Centred text in the rectangle: first variant/size that fits; returns the text or None."""
            for v in variants:
                for sz in sizes:
                    t = ax.text(x + w / 2, y + h / 2, v, ha='center', va='center', fontsize=sz, color=colour, fontweight='bold' if bold else 'normal', linespacing=0.95)
                    bb = t.get_window_extent(rend)
                    if bb.width / fig.dpi <= w - 0.05 and bb.height / fig.dpi <= h - 0.04:
                        return t
                    t.remove()
            return None

        ft_tot = {k: int(v) for k, v in d.group_by('ft').len().iter_rows()}
        groups = sorted(ft_tot, key=lambda k: -ft_tot[k])
        failed = []
        PAD, HEAD = 0.05, 0.34
        for ft, (gx, gy, gw, gh) in zip(groups, squarify([ft_tot[g] for g in groups], 0, 0, W, H)):
            ax.add_patch(Rectangle((gx + PAD / 2, gy + PAD / 2), gw - PAD, gh - PAD, facecolor='#EDEDED', edgecolor='none'))
            title = f'{ft}  ({ft_tot[ft]})'
            if fit(gx + PAD / 2, gy + PAD / 2, gw - PAD, HEAD, [title, title.replace('  (', '\n(')], '#333333', sizes=(10, 9, 8, 7, 6.5, 6, 5.5), bold=True) is None:
                failed.append(('group', ft))
            sub = d.filter(pl.col('ft') == ft).group_by('ds').len().sort('len', descending=True)
            names, vals = sub['ds'].to_list(), sub['len'].to_list()
            ix, iy, iw, ih = gx + PAD, gy + PAD / 2 + HEAD, gw - 2 * PAD, gh - PAD - HEAD
            for name, v, (tx, ty, tw, th) in zip(names, vals, squarify(vals, ix, iy, iw, ih)):
                c = rgb(col[name])
                ax.add_patch(Rectangle((tx, ty), tw, th, facecolor=c, edgecolor='white', linewidth=1.2))
                fg = (1, 1, 1) if lum(c) < 0.30 else (0.1, 0.1, 0.1)
                if fit(tx, ty, tw, th, [f'{name}\n{v}', f'{name} {v}', f'{v}'], fg) is None:
                    failed.append((ft, name, v, round(tw, 2), round(th, 2)))
        print('tiles whose text does not fit:', failed)
        Ht = H + TOP + BOT
        fig.text(0.008, 1 - 0.02 / Ht, 'Fine-tuning stage and pretraining dataset', ha='left', va='top', fontsize=16, fontweight='bold', color='#111111')
        fig.text(0.008, 1 - 0.40 / Ht, f'{d.height:,} checkpoints; area proportional to the number of checkpoints. Outer boxes: fine-tuning stage; tiles: pretraining dataset '
                 f'(datasets with fewer than {SMALL} checkpoints are merged into "Other").', ha='left', va='top', fontsize=8.5, color='#555555')
        from matplotlib.patches import Patch

        shown = [k for k in ds_order if d.filter(pl.col('ds') == k).height]
        fig.legend([Patch(facecolor=col[k], edgecolor='none') for k in shown], [k for k in shown], loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=7, frameon=False, fontsize=8.5, handlelength=1.0, handleheight=1.0, columnspacing=1.4, handletextpad=0.5)
        for ext in ('pdf', 'png'):
            fig.savefig(str(OUT / f'treemap_finetuning_dataset.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
        print('Saved: treemap_finetuning_dataset.pdf  +  .png')
        return fig


    fig_tm_ft = _()
    fig_tm_ft
    return


@app.cell(hide_code=True)
def _(OUT, df, pl):
    def _():
        """Circle packing: outer circle = fine-tuning stage, inner circles = pretraining dataset (checkpoint counts).
        One data unit = one inch, so that the text sizes can be checked against the size of every circle."""
        import math

        import numpy as np

        import circlify
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Ellipse, Patch

        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'pdf.fonttype': 42, 'ps.fonttype': 42})
        W = 11.5
        d = (
            df.filter(pl.col('pretrain_dataset').is_not_null()).with_columns(pretrain_dataset=pl.col('pretrain_dataset').list.join(' + '))  # run-time key: one unit per checkpoint (pretrain_dataset is a list)
            .with_columns(ft=pl.when(pl.col('pretrain_ft').is_not_null()).then(pl.lit('→ ') + pl.col('pretrain_ft').list.join(' → ')).otherwise(pl.lit('Single-Stage Training')))
        )
        d = d.with_columns(ds=pl.col('pretrain_dataset'))
        ds_order = d.group_by('ds').len().sort('len', descending=True)['ds'].to_list()
        OI = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#882255', '#117733', '#DDCC77', '#6699CC', '#AA4499', '#44AA99']
        import colorsys

        def extra(i):  # golden-angle hues, alternating lightness
            r, g, b = colorsys.hls_to_rgb((i * 0.618034) % 1, (0.42, 0.58, 0.5)[i % 3], 0.55)
            return '#%02X%02X%02X' % (round(r * 255), round(g * 255), round(b * 255))

        col = {k: (OI[i] if i < len(OI) else extra(i)) for i, k in enumerate(ds_order)}
        rgb = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))

        def lum(c):
            lin = lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])

        def pack(values):
            """Unit-circle packing: circles (x, y, r) in the order of `values` (descending)."""
            if len(values) == 1:
                return [(0.0, 0.0, 1.0)]
            cs = list(reversed(circlify.circlify(values, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1))))
            return [(c.x, c.y, c.r) for c in cs]

        ft_tot = {k: int(v) for k, v in d.group_by('ft').len().iter_rows()}
        groups = sorted(ft_tot, key=lambda k: -ft_tot[k])
        gpos = pack([float(ft_tot[g]) for g in groups])
        INNER, GAP = 0.93, 0.02

        def members(ft):
            sub = d.filter(pl.col('ft') == ft).group_by('ds').len().sort('len', descending=True)
            names, vals = sub['ds'].to_list(), sub['len'].to_list()
            return names, vals, pack([float(v) for v in vals])

        # the biggest group (Single-Stage Training) is an ellipse that hugs its inner circles instead of a circle
        r0 = gpos[0][2]
        pts = []
        for px, py, pr in members(groups[0])[2]:
            t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            pts.append(np.c_[INNER * px + (INNER * pr * 0.97 + 0.03) * np.cos(t), INNER * py + (INNER * pr * 0.97 + 0.03) * np.sin(t)])
        pts = np.vstack(pts)
        best = None
        for a_ in np.linspace(0.7, 1.05, 36):
            if np.abs(pts[:, 0]).max() >= a_:
                continue
            b_ = (np.abs(pts[:, 1]) / np.sqrt(1 - (pts[:, 0] / a_) ** 2)).max()
            if best is None or a_ * b_ < best[0] * best[1]:
                best = (a_, b_)
        EA, EB = best[0] * r0, best[1] * r0
        tt = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        bnd = np.c_[EA * np.cos(tt), EB * np.sin(tt)]

        def place(theta, r, others):
            u = np.array([np.cos(theta), np.sin(theta)])
            for s_ in np.arange(0.0, 6 * r0, 0.02 * r0):
                c = u * s_
                if (c[0] / EA) ** 2 + (c[1] / EB) ** 2 > 1 and np.hypot(*(bnd - c).T).min() >= r + GAP * r0 and all(np.hypot(*(c - o)) >= r + ro + GAP * r0 for o, ro in others):
                    return c
            return None

        small = [g[2] for g in gpos[1:]]  # radii of the fine-tuning groups (chains), by size
        r1 = small[0]
        others = small[1:]
        # the groups form a column on the left of the ellipse: the largest chain at the bottom, the others in rows of two right above it;
        # every title (one line per stage) sits right above its own circle
        xl_c = -(EA + r1 + 0.05 * r0)
        x0, x1 = xl_c - r1, EA
        k0 = W / (x1 - x0)
        # left column: the largest chain (IN-1K) at the bottom; the two smallest chains sit on the upper left and upper right of its circle, on both sides of
        # its title, touching the circle; the two larger chains form one row above them (titles of that row at the same height)
        GP = 0.04 * r0
        r10, r7 = others[2], others[3]
        for _ in range(6):  # the column widens by what the two small circles need beyond the circle of IN-1K (distances in inches -> data units)
            k0 = W / (x1 - x0)
            th_ = 0.5 / k0  # height of a (two-line) title, in data units
            dxl, dxr = 1.05 / k0 + r10, 1.05 / k0 + r7  # the title of IN-1K stays free between the two small circles
            xl_c = -(EA + r1 + max(0.0, dxr + r7 - r1) + 0.05 * r0)
            x0 = min(xl_c - r1, xl_c - dxl - r10, xl_c - 1.85 / k0)
        y1c = -EB + r1 + 0.45 / k0  # the column is lifted a bit above the bottom of the ellipse, away from the legend
        dyl, dyr = [((r1 + r) ** 2 - dx ** 2) ** 0.5 for r, dx in ((r10, dxl), (r7, dxr))]
        dyl, dyr = dyl + 0.15 / k0, dyr + 0.15 / k0  # lifted a little: the two small circles do not touch the circle of IN-1K
        row2 = others[:2]
        ytop_small = max(y1c + dyl + r10, y1c + dyr + r7, y1c + r1) + th_
        ya = ytop_small + GP + max(row2)
        cs = [(xl_c, y1c), (xl_c - 1.0 / k0, ya), (xl_c + 1.0 / k0, ya), (xl_c - dxl, y1c + dyl), (xl_c + dxr, y1c + dyr)]
        tops = [None, r1, max(row2), max(row2), r10, r7]
        y0, y1 = -EB, max(EB, ya + max(row2))
        gpos = [(0.0, 0.0, r0)] + [(c[0], c[1], r) for c, r in zip(cs, small)]
        k = W / (x1 - x0)  # inches per unit
        y1 += 0.5 / k  # room for the group titles
        cx0, cy0 = (x0 + x1) / 2, (y0 + y1) / 2
        H = (y1 - y0) * k

        TOP, BOT = 0.9, 1.45
        fig = plt.figure(figsize=(W, H + TOP + BOT), dpi=150)
        ax = fig.add_axes([0, BOT / (H + TOP + BOT), 1, H / (H + TOP + BOT)])
        ax.set_xlim(-W / 2, W / 2); ax.set_ylim(-H / 2, H / 2); ax.set_aspect('equal'); ax.axis('off')
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()

        def put(x, y, variants, colour, r=None, sizes=(13, 12, 11, 10, 9, 8, 7.5, 7, 6.5, 6), bold=False, halo=None, **kw):
            """Text at (x, y); first variant/size that fits inside a circle of radius r (None: anywhere). Returns the text or None."""
            for v in variants:
                for sz in sizes:
                    t = ax.text(x, y, v, fontsize=sz, color=colour, fontweight='bold' if bold else 'normal', linespacing=0.95, zorder=6, **kw)
                    if halo:
                        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=halo)])
                    bb = t.get_window_extent(rend)
                    if r is None or math.hypot(bb.width / fig.dpi, bb.height / fig.dpi) / 2 <= r * 0.97:
                        return t
                    t.remove()
            return None

        failed, gcirc = [], []
        for n_, (ft, (gx, gy, gr)) in enumerate(zip(groups, gpos)):
            X, Y, R = (gx - cx0) * k, (gy - cy0) * k, gr * k
            if n_ == 0:
                ax.add_patch(Ellipse((X, Y), 2 * EA * k, 2 * EB * k, facecolor='#EDEDED', edgecolor='#B8B8B8', linewidth=0.8, zorder=1))
            else:
                ax.add_patch(Circle((X, Y), R, facecolor='#EDEDED', edgecolor='#B8B8B8', linewidth=0.8, zorder=1))
            names, vals, pk = members(ft)
            inner = R * INNER
            for name, v, (px, py, pr) in zip(names, vals, pk):
                cx, cy, cr = X + px * inner, Y + py * inner, pr * inner * 0.97
                c = rgb(col[name])
                ax.add_patch(Circle((cx, cy), cr, facecolor=c, edgecolor='white', linewidth=1.0, zorder=3))
                fg = (1, 1, 1) if lum(c) < 0.30 else (0.1, 0.1, 0.1)
                if put(cx, cy, [f'{name}\n{v}', f'{name} {v}', f'{v}'], fg, r=cr, ha='center', va='center') is None:
                    failed.append((ft, name, v, round(2 * cr, 2)))
            gcirc.append((ft, X, Y, EB * k if n_ == 0 else tops[n_] * k, R))

        # group titles: centred right above their own shape
        Yb, Bb = gcirc[0][2], gcirc[0][3]
        for ft, X, Y, top, R in gcirc:
            big = R > 1.0
            below = False  # a group under the ellipse: its title goes under it, not into the ellipse
            put(X, Y - top - 0.02 if below else Y + top + 0.02, [f'{ft}  ({ft_tot[ft]})'.replace('ImageNet-', 'IN-')] if big else [f'{ft}\n({ft_tot[ft]})'.replace('ImageNet-', 'IN-')], '#333333', bold=True, halo=(1, 1, 1, 0.9),
                sizes=(16, 15) if big else (14, 13, 12), ha='center', va='top' if below else 'bottom')
        print('circles whose text does not fit:', failed)
        Ht = H + TOP + BOT
        t_title = fig.text(0.008, 1 - 0.02 / Ht, '(b) Pretraining dataset and fine-tuning stage', ha='left', va='top', fontsize=20, fontweight='bold', color='#111111')
        t_sub = fig.text(0.008, 1 - 0.50 / Ht, f'{d.height:,} checkpoints; circle area proportional to the number of checkpoints. Outer circles: fine-tuning stage; inner circles: pretraining dataset '
                 f'(all {len(ds_order)} datasets shown).', ha='left', va='top', fontsize=11, color='#555555')
        shown = [k_ for k_ in ds_order if d.filter(pl.col('ds') == k_).height]
        _leg = fig.legend([Patch(facecolor=col[k_], edgecolor='none') for k_ in shown], shown, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=8, frameon=False, fontsize=11, title='Pretraining Dataset', title_fontsize=17, alignment='left', handlelength=1.0, handleheight=1.0, columnspacing=1.4, handletextpad=0.5)
        _leg.get_title().set_fontweight('bold')
        for ext in ('pdf', 'png'):
            fig.savefig(str(OUT / f'treemap_finetuning_dataset_circles.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
        print('Saved: treemap_finetuning_dataset_circles.pdf  +  .png')
        t_sub.set_visible(False)  # variant without the subtitle (used by the side-by-side figure)
        t_title.set_y(1 - 0.42 / Ht)
        for ext in ('pdf', 'png'):
            fig.savefig(str(OUT / f'treemap_finetuning_dataset_circles_nosub.{ext}'), dpi=300, bbox_inches='tight', facecolor='white')
        t_sub.set_visible(True)
        t_title.set_y(1 - 0.02 / Ht)
        return fig


    fig_circ_ft = _()
    fig_circ_ft
    return


@app.cell(hide_code=True)
def _(OUT):
    import matplotlib.pyplot as _plt
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    _ims = [Image.open(OUT / 'models_per_company_packed_nosub.png').convert('RGB'), Image.open(OUT / 'treemap_finetuning_dataset_circles_nosub.png').convert('RGB')]
    _h = min(im.height for im in _ims)
    _ims = [im.resize((round(im.width * _h / im.height), _h), Image.LANCZOS) for im in _ims]
    _gap = 40
    _canvas = Image.new('RGB', (sum(im.width for im in _ims) + _gap, _h), 'white')
    _canvas.paste(_ims[0], (0, 0))
    _canvas.paste(_ims[1], (_ims[0].width + _gap, 0))
    _canvas.save(OUT / 'who_trained_what_and_finetuning_circles.png')
    _canvas.save(OUT / 'who_trained_what_and_finetuning_circles.pdf', resolution=300)
    fig_side = _plt.figure(figsize=(_canvas.width / 300, _canvas.height / 300))
    _plt.axes([0, 0, 1, 1]).imshow(_canvas)
    _plt.axis('off')
    print('Saved: who_trained_what_and_finetuning_circles.{png,pdf}', _canvas.size)
    fig_side
    return


@app.cell(hide_code=True)
def _(OUT, df, pl):
    def _():
        """Alluvial diagram: pretraining dataset -> fine-tuning dataset -> pretraining tasks (checkpoint counts)."""
        import plotly.graph_objects as go

        TASKS = {
            'image_classification': 'classification', 'image_text_contrastive': 'contrastive', 'masked_image_modeling': 'MIM',
            'vision_language_generative': 'VL-gen', 'joint_embedding': 'joint-emb', 'promptable_segmentation': 'segm.',
            'masked_vision_language_modeling': 'masked VL', 'vision_language_multitask': 'VL multitask',
        }
        SMALL = 20  # datasets with fewer checkpoints are merged into "Other"
        d = (
            df.filter(pl.col('pretrain_dataset').is_not_null()).with_columns(pretrain_dataset=pl.col('pretrain_dataset').list.join(' + '))  # run-time key: one unit per checkpoint (pretrain_dataset is a list)
            .with_columns(
                ft=pl.when(pl.col('pretrain_ft').is_not_null()).then(pl.lit('→ ') + pl.col('pretrain_ft').list.join(' → ')).otherwise(pl.lit('no fine-tuning')),
                paradigm=pl.col('pretrain_tasks').list.eval(pl.element().replace(TASKS)).list.join(' + '),  # last column: pretraining tasks
            )
        )
        size = d.group_by('pretrain_dataset').len()
        small = size.filter(pl.col('len') < SMALL)['pretrain_dataset'].to_list()
        d = d.with_columns(
            ds=pl.when(pl.col('pretrain_dataset').is_in(small)).then(pl.lit(f'Other ({len(small)} datasets)')).otherwise(pl.col('pretrain_dataset'))
        )
        ds_order = d.group_by('ds').len().sort('len', descending=True)['ds'].to_list()
        ds_order = [x for x in ds_order if not x.startswith('Other')] + [x for x in ds_order if x.startswith('Other')]
        ft_order = ['no fine-tuning'] + sorted({x for x in d['ft'].unique().to_list() if x != 'no fine-tuning'})
        pa_size = d.group_by('paradigm').len().sort('len', descending=True)['paradigm'].to_list()  # colours follow this ranking
        # The tiny nodes are interleaved between the bigger ones (smallest first), so that their labels do not overlap on a short figure.
        big, tiny = pa_size[:6], pa_size[6:][::-1]
        pa_order = [k for i in range(max(len(big), len(tiny))) for k in (big[i:i + 1] + tiny[i:i + 1])]

        labels = ds_order + ft_order + pa_order
        idx = {('ds', k): i for i, k in enumerate(ds_order)}
        idx |= {('ft', k): len(ds_order) + i for i, k in enumerate(ft_order)}
        idx |= {('pa', k): len(ds_order) + len(ft_order) + i for i, k in enumerate(pa_order)}

        OI = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#882255', '#117733', '#DDCC77', '#6699CC', '#AA4499']
        GREY = '#8C8C8C'
        col_ds = {k: OI[i] if i < len(OI) and not k.startswith('Other') else GREY for i, k in enumerate(ds_order)}
        TASK_COLS = ['#3B3B98', '#B33771', '#2C9C91', '#E15F41', '#82589F', '#1B9CFC', '#786FA6', '#596275', '#A3A3A3']
        col_pa = {k: TASK_COLS[i % len(TASK_COLS)] for i, k in enumerate(pa_size)}
        rgba = lambda h, a: f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"

        src, tgt, val, lcol = [], [], [], []
        for a, b, n in d.group_by('ds', 'ft').len().iter_rows():
            src.append(idx[('ds', a)]); tgt.append(idx[('ft', b)]); val.append(n); lcol.append(rgba(col_ds[a], 0.45))
        for a, b, n in d.group_by('ft', 'paradigm').len().iter_rows():
            src.append(idx[('ft', a)]); tgt.append(idx[('pa', b)]); val.append(n); lcol.append(rgba(col_pa[b], 0.45))
        ncol = [col_ds[k] for k in ds_order] + ['#B8B8B8'] * len(ft_order) + [col_pa[k] for k in pa_order]
        tot = {('ds', k): int(v) for k, v in d.group_by('ds').len().iter_rows()}
        tot |= {('ft', k): int(v) for k, v in d.group_by('ft').len().iter_rows()}
        tot |= {('pa', k): int(v) for k, v in d.group_by('paradigm').len().iter_rows()}
        nlabels = [f'{k}  ({tot[(c, k)]})' for c, ks in (('ds', ds_order), ('ft', ft_order), ('pa', pa_order)) for k in ks]

        n0 = len(nlabels)
        for k in pa_order:  # invisible sink per task
            src.append(idx[('pa', k)]); tgt.append(n0 + pa_order.index(k)); val.append(tot[('pa', k)]); lcol.append('rgba(0,0,0,0)')
        nlabels += [''] * len(pa_order)
        ncol += ['rgba(0,0,0,0)'] * len(pa_order)

        fig = go.Figure(go.Sankey(
            arrangement='snap',
            node=dict(label=nlabels, color=ncol, pad=10, thickness=14, line=dict(color='white', width=0.5)),
            link=dict(source=src, target=tgt, value=val, color=lcol),
        ))
        fig.update_layout(
            title=dict(text=f'<b>What was it trained on?</b><br><sup>{d.height:,} checkpoints: pretraining dataset → fine-tuning dataset → pretraining tasks '
                            f'(datasets with fewer than {SMALL} checkpoints merged into "Other")</sup>', x=0.01),
            font=dict(family='Times New Roman, serif', size=10), height=600, width=1350, margin=dict(l=10, r=10, t=70, b=5),
            paper_bgcolor='white',
        )
        fig.write_html(str(OUT / 'pretrain_sankey_tasks.html'))
        try:
            for ext in ('pdf', 'png'):
                fig.write_image(str(OUT / f'pretrain_sankey_tasks.{ext}'), scale=3 if ext == 'png' else 1)
            print('Saved: pretrain_sankey_tasks.html / .pdf / .png')
        except Exception as e:  # kaleido needs a Chrome
            print('Saved: pretrain_sankey_tasks.html only;', type(e).__name__, str(e)[:120])
        return fig


    fig_sankey = _()
    fig_sankey
    return


@app.cell(hide_code=True)
def _(OUT, df):
    def _family_plot():
        """
        Combined 1×3 figure:
          Plot 1 – Scatter: num_parameters vs latent_dim (coloured by model family)
          Plot 2 – Violin:  parameter count distribution by family
          Plot 3 – Violin:  latent dimension distribution by family

        Assumptions:
          • `df`  is a Polars DataFrame already in scope with columns:
              "num_parameters", "latent_dim", "family"
          • polars is imported as `pl`
          • matplotlib and numpy are available
        """

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as ticker
        import numpy as np

        # ── Global style (NeurIPS/ICML) ───────────────────────────────────────────────
        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 9,
                'axes.titlesize': 10,
                'axes.labelsize': 9,
                'xtick.labelsize': 7.5,
                'ytick.labelsize': 8,
                'legend.fontsize': 11,
                'legend.title_fontsize': 12,
                'axes.linewidth': 0.7,
                'xtick.major.width': 0.7,
                'ytick.major.width': 0.7,
                'xtick.minor.width': 0.5,
                'ytick.minor.width': 0.5,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.minor.visible': True,
                'ytick.minor.visible': True,
                'axes.grid': True,
                'grid.linewidth': 0.4,
                'grid.alpha': 0.4,
                'grid.linestyle': '--',
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.05,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
            }
        )

        # ── Shared parameters ─────────────────────────────────────────────────────────
        TOP_N = 15

        # ── Data prep: work from pandas for matplotlib ────────────────────────────────
        full_df = df.to_pandas()

        # Top-N families by count (across the full dataset)
        top_families = full_df['family'].value_counts().nlargest(TOP_N).index.tolist()

        # Palette: tab20b gives 20 perceptually distinct, non-grey colours
        palette = plt.cm.get_cmap('tab20b', TOP_N)
        color_map = {f: palette(i) for i, f in enumerate(top_families)}
        OTHER_COLOR = '#bbbbbb'

        # ── Figure layout ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        ax1, ax2, ax3 = axes

        # ═════════════════════════════════════════════════════════════════════════════
        # PLOT 1 – Scatter: num_parameters vs latent_dim
        # ═════════════════════════════════════════════════════════════════════════════
        sc_df = full_df.dropna(subset=['num_parameters', 'latent_dim'])
        is_top = sc_df['family'].isin(top_families)

        # "Other" in the back
        ax1.scatter(
            sc_df.loc[~is_top, 'num_parameters'],
            sc_df.loc[~is_top, 'latent_dim'],
            color=OTHER_COLOR,
            alpha=0.35,
            s=10,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        # Top families on top
        for family in top_families:
            mask = sc_df['family'] == family
            ax1.scatter(
                sc_df.loc[mask, 'num_parameters'],
                sc_df.loc[mask, 'latent_dim'],
                color=color_map[family],
                alpha=0.80,
                s=18,
                linewidths=0.3,
                edgecolors='white',
                zorder=2,
                label=family,
            )

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('Latent Dimension')

        ax1.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

        # ═════════════════════════════════════════════════════════════════════════════
        # Helper: violin + box plot on a given axis
        # ═════════════════════════════════════════════════════════════════════════════
        def draw_violin_ax(ax, column, ylabel):
            """Draw violin+box plot for `column` on `ax`."""
            vdf = full_df.dropna(subset=[column])
            vdf = vdf[vdf[column] > 0].copy()

            other_mask = ~vdf['family'].isin(top_families)
            vdf['family_plot'] = vdf['family'].where(~other_mask, other='Other')

            x_order = top_families + ['Other']
            n_groups = len(x_order)
            colors = [color_map[f] for f in top_families] + [OTHER_COLOR]

            for i, (family, color) in enumerate(zip(x_order, colors)):
                pos = i + 1
                data = vdf.loc[vdf['family_plot'] == family, column].values
                data = data[data > 0]

                if len(data) < 2:
                    if len(data) == 1:
                        ax.scatter(
                            [pos],
                            [np.log10(data[0])],
                            color=color,
                            s=20,
                            zorder=4,
                            marker='D',
                        )
                    continue

                log_data = np.log10(data)

                # Violin
                parts = ax.violinplot(
                    log_data,
                    positions=[pos],
                    widths=0.7,
                    showextrema=False,
                    showmedians=False,
                )
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('white')
                    pc.set_linewidth(0.4)
                    pc.set_alpha(0.75)

                # Box (IQR)
                q1, med, q3 = np.percentile(log_data, [25, 50, 75])
                iqr = q3 - q1
                w_lo = max(log_data[log_data >= q1 - 1.5 * iqr].min(), log_data.min())
                w_hi = min(log_data[log_data <= q3 + 1.5 * iqr].max(), log_data.max())

                box_w = 0.12
                ax.add_patch(
                    plt.Rectangle(
                        (pos - box_w / 2, q1),
                        box_w,
                        iqr,
                        facecolor='white',
                        edgecolor='#333333',
                        linewidth=0.8,
                        zorder=3,
                    )
                )
                ax.plot(
                    [pos - box_w / 2, pos + box_w / 2],
                    [med, med],
                    color='#333333',
                    linewidth=1.2,
                    zorder=4,
                )
                ax.plot(
                    [pos, pos], [w_lo, q1], color='#333333', linewidth=0.7, zorder=3
                )
                ax.plot(
                    [pos, pos], [q3, w_hi], color='#333333', linewidth=0.7, zorder=3
                )

                # Outliers
                outliers = log_data[(log_data < w_lo) | (log_data > w_hi)]
                if len(outliers):
                    ax.scatter(
                        np.full_like(outliers, pos),
                        outliers,
                        color=color,
                        s=6,
                        alpha=0.6,
                        linewidths=0,
                        zorder=5,
                        rasterized=True,
                    )

            # Axes / ticks
            ax.set_xlim(0.3, n_groups + 0.7)

            raw_min = vdf[column].min()
            raw_max = vdf[column].max()
            y_lo = np.log10(raw_min) - 0.15
            y_hi = np.log10(raw_max) + 0.15
            ax.set_ylim(y_lo, y_hi)

            y_ticks = np.arange(
                np.floor(np.log10(raw_min)), np.ceil(np.log10(raw_max)) + 1
            )
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([r'$10^{%d}$' % int(t) for t in y_ticks])

            ax.set_xticks(range(1, n_groups + 1))
            ax.set_xticklabels(x_order, rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlabel('Family')
            ax.set_ylabel(ylabel)

            # Separator before "Other"
            ax.axvline(
                TOP_N + 0.5, color='#999999', linewidth=0.7, linestyle=':', zorder=1
            )
            ax.text(
                TOP_N + 0.55,
                y_hi - 0.05 * (y_hi - y_lo),
                'other',
                fontsize=6.5,
                color='#888888',
                va='top',
                style='italic',
            )

        # ═════════════════════════════════════════════════════════════════════════════
        # PLOT 2 – Violin: parameter count by family
        # ═════════════════════════════════════════════════════════════════════════════
        draw_violin_ax(ax2, column='num_parameters', ylabel='Number of Parameters')

        # ═════════════════════════════════════════════════════════════════════════════
        # PLOT 3 – Violin: latent dimension by family
        # ═════════════════════════════════════════════════════════════════════════════
        draw_violin_ax(ax3, column='latent_dim', ylabel='Latent Dimension')

        # ═════════════════════════════════════════════════════════════════════════════
        # Shared legend (below the whole figure)
        # ═════════════════════════════════════════════════════════════════════════════
        family_handles = [
            mpatches.Patch(color=color_map[f], label=f) for f in top_families
        ]
        other_handle = mpatches.Patch(color=OTHER_COLOR, alpha=0.75, label='Other')
        all_handles = family_handles + [other_handle]

        fig.legend(
            handles=all_handles,
            title=f'Model Family (top {TOP_N})',
            loc='lower center',
            bbox_to_anchor=(0.5, -0.14),
            ncol=8,
            frameon=True,
            framealpha=0.9,
            edgecolor='#aaaaaa',
            borderpad=0.6,
            handlelength=1.2,
            handleheight=0.9,
            columnspacing=0.8,
            labelspacing=0.35,
        ).get_title().set_fontweight('bold')

        plt.tight_layout()
        plt.savefig(str(OUT / 'combined_plots_family.pdf'), format='pdf')
        return plt.show()

    _family_plot()
    return


@app.cell(hide_code=True)
def _(OUT, df):
    def _macrofamily_plot():
        """
        Combined 1×3 figure:
          Plot 1 – Scatter: num_parameters vs latent_dim (coloured by macro_family)
          Plot 2 – Violin:  parameter count distribution by macro_family
          Plot 3 – Violin:  latent dimension distribution by macro_family

        Assumptions:
          • `df`  is a Polars DataFrame already in scope with columns:
              "num_parameters", "latent_dim", "macro_family"
          • polars is imported as `pl`
          • matplotlib and numpy are available
        """

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as ticker
        import numpy as np

        # ── Global style (NeurIPS/ICML) ───────────────────────────────────────────────
        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 9,
                'axes.titlesize': 10,
                'axes.labelsize': 9,
                'xtick.labelsize': 7.5,
                'ytick.labelsize': 8,
                'legend.fontsize': 11,
                'legend.title_fontsize': 12,
                'axes.linewidth': 0.7,
                'xtick.major.width': 0.7,
                'ytick.major.width': 0.7,
                'xtick.minor.width': 0.5,
                'ytick.minor.width': 0.5,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.minor.visible': True,
                'ytick.minor.visible': True,
                'axes.grid': True,
                'grid.linewidth': 0.4,
                'grid.alpha': 0.4,
                'grid.linestyle': '--',
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.05,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
            }
        )

        # ── Data prep ─────────────────────────────────────────────────────────────────
        full_df = df.to_pandas()

        # All unique macro_families, ordered by descending count (no top-N truncation)
        all_families = (
            full_df['macro_family']
            .value_counts()
            .sort_values(ascending=False)
            .index.tolist()
        )
        N_GROUPS = len(all_families)  # e.g. 8

        # ── Palette: one vivid colour per macro_family, no grey reserved ──────────────
        palette = plt.cm.get_cmap('turbo', N_GROUPS)
        color_map = {f: palette(i) for i, f in enumerate(all_families)}
        OTHER_COLOR = '#bbbbbb'  # kept for the scatter "other" points only

        # ── Figure sizing: match the physical per-group width of the 16-group version ──
        # Reference: 18 in wide for 16 violin groups  →  1.125 in / group for the
        # two violin subplots.  Scatter subplot keeps a fixed 6 in width.
        SCATTER_W = 6.0  # fixed width for scatter panel (inches)
        PER_GROUP_W = 18.0 / 16  # physical width each group occupied in the original
        VIOLIN_W = PER_GROUP_W * N_GROUPS  # total width for each violin panel
        FIG_W = SCATTER_W + 2 * VIOLIN_W
        FIG_H = 5.5

        # Violin / box widths scale so they fill ~70 % of each slot (same as original)
        VIOLIN_WIDTH = 0.7  # in data-units (1 slot = 1 data-unit, unchanged)
        BOX_W = 0.12  # matches original

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(FIG_W, FIG_H),
            gridspec_kw={'width_ratios': [SCATTER_W, VIOLIN_W, VIOLIN_W]},
        )
        ax1, ax2, ax3 = axes

        # ═════════════════════════════════════════════════════════════════════════════
        # PLOT 1 – Scatter: num_parameters vs latent_dim
        # ═════════════════════════════════════════════════════════════════════════════
        sc_df = full_df.dropna(subset=['num_parameters', 'latent_dim'])
        is_top = sc_df['macro_family'].isin(all_families)

        ax1.scatter(
            sc_df.loc[~is_top, 'num_parameters'],
            sc_df.loc[~is_top, 'latent_dim'],
            color=OTHER_COLOR,
            alpha=0.35,
            s=10,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        for family in all_families:
            mask = sc_df['macro_family'] == family
            ax1.scatter(
                sc_df.loc[mask, 'num_parameters'],
                sc_df.loc[mask, 'latent_dim'],
                color=color_map[family],
                alpha=0.80,
                s=18,
                linewidths=0.3,
                edgecolors='white',
                zorder=2,
                label=family,
            )

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('Latent Dimension')
        ax1.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

        # ═════════════════════════════════════════════════════════════════════════════
        # Helper: violin + box plot
        # ═════════════════════════════════════════════════════════════════════════════
        def draw_violin_ax(ax, column, ylabel):
            vdf = full_df.dropna(subset=[column])
            vdf = vdf[vdf[column] > 0].copy()

            x_order = all_families  # no "Other" bin — every family is shown
            colors = [color_map[f] for f in x_order]

            for i, (family, color) in enumerate(zip(x_order, colors)):
                pos = i + 1
                data = vdf.loc[vdf['macro_family'] == family, column].values
                data = data[data > 0]

                if len(data) < 2:
                    if len(data) == 1:
                        ax.scatter(
                            [pos],
                            [np.log10(data[0])],
                            color=color,
                            s=20,
                            zorder=4,
                            marker='D',
                        )
                    continue

                log_data = np.log10(data)

                # Violin
                parts = ax.violinplot(
                    log_data,
                    positions=[pos],
                    widths=VIOLIN_WIDTH,
                    showextrema=False,
                    showmedians=False,
                )
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('white')
                    pc.set_linewidth(0.4)
                    pc.set_alpha(0.75)

                # Box (IQR)
                q1, med, q3 = np.percentile(log_data, [25, 50, 75])
                iqr = q3 - q1
                w_lo = max(log_data[log_data >= q1 - 1.5 * iqr].min(), log_data.min())
                w_hi = min(log_data[log_data <= q3 + 1.5 * iqr].max(), log_data.max())

                ax.add_patch(
                    plt.Rectangle(
                        (pos - BOX_W / 2, q1),
                        BOX_W,
                        iqr,
                        facecolor='white',
                        edgecolor='#333333',
                        linewidth=0.8,
                        zorder=3,
                    )
                )
                ax.plot(
                    [pos - BOX_W / 2, pos + BOX_W / 2],
                    [med, med],
                    color='#333333',
                    linewidth=1.2,
                    zorder=4,
                )
                ax.plot(
                    [pos, pos], [w_lo, q1], color='#333333', linewidth=0.7, zorder=3
                )
                ax.plot(
                    [pos, pos], [q3, w_hi], color='#333333', linewidth=0.7, zorder=3
                )

                # Outliers
                outliers = log_data[(log_data < w_lo) | (log_data > w_hi)]
                if len(outliers):
                    ax.scatter(
                        np.full_like(outliers, pos),
                        outliers,
                        color=color,
                        s=6,
                        alpha=0.6,
                        linewidths=0,
                        zorder=5,
                        rasterized=True,
                    )

            # Axes / ticks
            ax.set_xlim(0.3, N_GROUPS + 0.7)

            raw_min = vdf[column].min()
            raw_max = vdf[column].max()
            y_lo = np.log10(raw_min) - 0.15
            y_hi = np.log10(raw_max) + 0.15
            ax.set_ylim(y_lo, y_hi)

            y_ticks = np.arange(
                np.floor(np.log10(raw_min)),
                np.ceil(np.log10(raw_max)) + 1,
            )
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([r'$10^{%d}$' % int(t) for t in y_ticks])

            ax.set_xticks(range(1, N_GROUPS + 1))
            ax.set_xticklabels(x_order, rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlabel('Macro-Family')
            ax.set_ylabel(ylabel)

        # ═════════════════════════════════════════════════════════════════════════════
        # PLOT 2 & 3
        # ═════════════════════════════════════════════════════════════════════════════
        draw_violin_ax(ax2, column='num_parameters', ylabel='Number of Parameters')
        draw_violin_ax(ax3, column='latent_dim', ylabel='Latent Dimension')

        # ═════════════════════════════════════════════════════════════════════════════
        # Shared legend
        # ═════════════════════════════════════════════════════════════════════════════
        handles = [mpatches.Patch(color=color_map[f], label=f) for f in all_families]

        fig.legend(
            handles=handles,
            title='Macro-Family',
            loc='lower center',
            bbox_to_anchor=(0.5, -0.18),
            ncol=N_GROUPS,  # all in one row since there are only ~8
            frameon=True,
            framealpha=0.9,
            edgecolor='#aaaaaa',
            borderpad=0.8,
            handlelength=1.6,
            handleheight=1.2,
            columnspacing=1.2,
            labelspacing=0.5,
        ).get_title().set_fontweight('bold')

        plt.tight_layout()
        plt.savefig(str(OUT / 'combined_plots_macrofamily.pdf'), format='pdf')
        return plt.show()

    _macrofamily_plot()
    return


@app.cell(hide_code=True)
def _(OUT, df):
    def _combined_2x3_plot():
        """
        Combined 2×3 figure that merges the per-family and per-macro_family plots.

        Row 1 (family, top-N + Other):
            (1,1) Scatter: num_parameters vs latent_dim       (coloured by family)
            (1,2) Violin:  num_parameters distribution        by family
            (1,3) Violin:  latent_dim     distribution        by family

        Row 2 (macro_family, all groups):
            (2,1) Scatter: num_parameters vs latent_dim       (coloured by macro_family)
            (2,2) Violin:  num_parameters distribution        by macro_family
            (2,3) Violin:  latent_dim     distribution        by macro_family

        Each column shares the same physical width across the two rows, so
        corresponding subplots line up exactly. The family legend sits ABOVE row 1
        and the macro-family legend sits BELOW row 2.
        """

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as ticker
        import numpy as np

        # ── Global style (NeurIPS/ICML) ───────────────────────────────────────────────
        plt.rcParams.update(
            {
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 18,
                'axes.titlesize': 18,
                'axes.labelsize': 18,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18,
                'legend.title_fontsize': 18,
                'axes.linewidth': 0.7,
                'xtick.major.width': 0.7,
                'ytick.major.width': 0.7,
                'xtick.minor.width': 0.5,
                'ytick.minor.width': 0.5,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.minor.visible': True,
                'ytick.minor.visible': True,
                'axes.grid': True,
                'grid.linewidth': 0.4,
                'grid.alpha': 0.4,
                'grid.linestyle': '--',
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.05,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
            }
        )

        # ── Shared parameters ─────────────────────────────────────────────────────────
        TOP_N = 15
        OTHER_COLOR = '#bbbbbb'
        BOX_W = 0.12
        VIOLIN_W = 0.7

        full_df = df.to_pandas()

        # ── Family (row 1) groupings ──────────────────────────────────────────────────
        top_families = full_df['family'].value_counts().nlargest(TOP_N).index.tolist()
        fam_palette = plt.cm.get_cmap('tab20b', TOP_N)
        fam_color_map = {f: fam_palette(i) for i, f in enumerate(top_families)}
        fam_x_order = top_families + ['Other']
        fam_colors = [fam_color_map[f] for f in top_families] + [OTHER_COLOR]
        N_FAM_GROUPS = len(fam_x_order)  # 16

        # ── Macro-family (row 2) groupings ────────────────────────────────────────────
        all_macro = (
            full_df['macro_family']
            .value_counts()
            .sort_values(ascending=False)
            .index.tolist()
        )
        N_MACRO_GROUPS = len(all_macro)
        macro_palette = plt.cm.get_cmap('turbo', N_MACRO_GROUPS)
        macro_color_map = {f: macro_palette(i) for i, f in enumerate(all_macro)}

        # ═════════════════════════════════════════════════════════════════════════════
        # Figure layout
        # GridSpec rows:
        #   row 0  → reserved band for the family legend (ABOVE row 1)
        #   row 1  → family axes
        #   row 2  → macro_family axes
        #   row 3  → reserved band for the macro_family legend
        # ═════════════════════════════════════════════════════════════════════════════
        SCALE = 1.286

        SCATTER_W = 6.0 * SCALE
        VIOLIN_W_IN = 6.0 * SCALE
        FIG_W = SCATTER_W + 2 * VIOLIN_W_IN
        ROW_H = 5.5 * SCALE
        LEGEND_H = 1.2 * SCALE
        FIG_H = 2 * ROW_H + 2 * LEGEND_H

        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(
            nrows=4,
            ncols=3,
            width_ratios=[SCATTER_W, VIOLIN_W_IN, VIOLIN_W_IN],
            height_ratios=[LEGEND_H, ROW_H, ROW_H, LEGEND_H],
            hspace=0.25,
            wspace=0.22,
        )

        axes = np.empty((2, 3), dtype=object)
        for j in range(3):
            axes[0, j] = fig.add_subplot(gs[1, j])
            axes[1, j] = fig.add_subplot(gs[2, j])

        # Reserved (invisible) axes for the legend bands — one full-width slot each.
        leg_ax_fam = fig.add_subplot(gs[0, :])
        leg_ax_fam.axis('off')
        leg_ax_macro = fig.add_subplot(gs[3, :])
        leg_ax_macro.axis('off')

        # ═════════════════════════════════════════════════════════════════════════════
        # Generic helpers
        # ═════════════════════════════════════════════════════════════════════════════
        def draw_scatter(ax, group_col, group_list, color_map, draw_other=True):
            """Scatter num_parameters vs latent_dim, coloured by `group_col`."""
            sc_df = full_df.dropna(subset=['num_parameters', 'latent_dim'])
            is_top = sc_df[group_col].isin(group_list)

            if draw_other:
                ax.scatter(
                    sc_df.loc[~is_top, 'num_parameters'],
                    sc_df.loc[~is_top, 'latent_dim'],
                    color=OTHER_COLOR,
                    alpha=0.35,
                    s=10,
                    linewidths=0,
                    rasterized=True,
                    zorder=1,
                )

            for family in group_list:
                mask = sc_df[group_col] == family
                ax.scatter(
                    sc_df.loc[mask, 'num_parameters'],
                    sc_df.loc[mask, 'latent_dim'],
                    color=color_map[family],
                    alpha=0.80,
                    s=18,
                    linewidths=0.3,
                    edgecolors='white',
                    zorder=2,
                    label=family,
                )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Number of Parameters')
            ax.set_ylabel('Latent Dimension')
            ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
            ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

        def draw_violin(
            ax,
            column,
            ylabel,
            group_col,
            x_order,
            colors,
            n_groups,
            x_label,
            separator_at=None,
        ):
            """Violin + IQR-box for `column`, grouped by `group_col`."""
            vdf = full_df.dropna(subset=[column])
            vdf = vdf[vdf[column] > 0].copy()

            # Bucket "non-listed" rows into "Other" if "Other" is present in x_order
            if 'Other' in x_order:
                other_mask = ~vdf[group_col].isin([f for f in x_order if f != 'Other'])
                vdf['_plot_group'] = vdf[group_col].where(~other_mask, other='Other')
            else:
                vdf['_plot_group'] = vdf[group_col]

            for i, (family, color) in enumerate(zip(x_order, colors)):
                pos = i + 1
                data = vdf.loc[vdf['_plot_group'] == family, column].values
                data = data[data > 0]

                if len(data) < 2:
                    if len(data) == 1:
                        ax.scatter(
                            [pos],
                            [np.log10(data[0])],
                            color=color,
                            s=20,
                            zorder=4,
                            marker='D',
                        )
                    continue

                log_data = np.log10(data)

                # Violin
                parts = ax.violinplot(
                    log_data,
                    positions=[pos],
                    widths=VIOLIN_W,
                    showextrema=False,
                    showmedians=False,
                )
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('white')
                    pc.set_linewidth(0.4)
                    pc.set_alpha(0.75)

                # Box (IQR)
                q1, med, q3 = np.percentile(log_data, [25, 50, 75])
                iqr = q3 - q1
                w_lo = max(log_data[log_data >= q1 - 1.5 * iqr].min(), log_data.min())
                w_hi = min(log_data[log_data <= q3 + 1.5 * iqr].max(), log_data.max())

                ax.add_patch(
                    plt.Rectangle(
                        (pos - BOX_W / 2, q1),
                        BOX_W,
                        iqr,
                        facecolor='white',
                        edgecolor='#333333',
                        linewidth=0.8,
                        zorder=3,
                    )
                )
                ax.plot(
                    [pos - BOX_W / 2, pos + BOX_W / 2],
                    [med, med],
                    color='#333333',
                    linewidth=1.2,
                    zorder=4,
                )
                ax.plot(
                    [pos, pos], [w_lo, q1], color='#333333', linewidth=0.7, zorder=3
                )
                ax.plot(
                    [pos, pos], [q3, w_hi], color='#333333', linewidth=0.7, zorder=3
                )

                # Outliers
                outliers = log_data[(log_data < w_lo) | (log_data > w_hi)]
                if len(outliers):
                    ax.scatter(
                        np.full_like(outliers, pos),
                        outliers,
                        color=color,
                        s=6,
                        alpha=0.6,
                        linewidths=0,
                        zorder=5,
                        rasterized=True,
                    )

            # Axes / ticks
            ax.set_xlim(0.3, n_groups + 0.7)

            raw_min = vdf[column].min()
            raw_max = vdf[column].max()
            y_lo = np.log10(raw_min) - 0.15
            y_hi = np.log10(raw_max) + 0.15
            ax.set_ylim(y_lo, y_hi)

            y_ticks = np.arange(
                np.floor(np.log10(raw_min)),
                np.ceil(np.log10(raw_max)) + 1,
            )
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([r'$10^{%d}$' % int(t) for t in y_ticks])

            # X-tick labels removed: groups are identified by the row legend.
            ax.set_xticks(range(1, n_groups + 1))
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False)
            # X-axis label is redundant with the legend title — leave it off.
            ax.set_ylabel(ylabel)

            # Optional separator (used for the family row before "Other")
            if separator_at is not None:
                ax.axvline(
                    separator_at,
                    color='#999999',
                    linewidth=0.7,
                    linestyle=':',
                    zorder=1,
                )
                ax.text(
                    separator_at + 0.05,
                    y_hi - 0.05 * (y_hi - y_lo),
                    'other',
                    fontsize=6.5,
                    color='#888888',
                    va='top',
                    style='italic',
                )

        # ═════════════════════════════════════════════════════════════════════════════
        # ROW 1 — family
        # ═════════════════════════════════════════════════════════════════════════════
        ax_fam_scatter, ax_fam_param, ax_fam_latent = axes[0]

        draw_scatter(
            ax_fam_scatter,
            group_col='family',
            group_list=top_families,
            color_map=fam_color_map,
            draw_other=True,
        )

        draw_violin(
            ax_fam_param,
            column='num_parameters',
            ylabel='Number of Parameters',
            group_col='family',
            x_order=fam_x_order,
            colors=fam_colors,
            n_groups=N_FAM_GROUPS,
            x_label='Family',
            separator_at=TOP_N + 0.5,
        )
        draw_violin(
            ax_fam_latent,
            column='latent_dim',
            ylabel='Latent Dimension',
            group_col='family',
            x_order=fam_x_order,
            colors=fam_colors,
            n_groups=N_FAM_GROUPS,
            x_label='Family',
            separator_at=TOP_N + 0.5,
        )

        # ═════════════════════════════════════════════════════════════════════════════
        # ROW 2 — macro_family
        # ═════════════════════════════════════════════════════════════════════════════
        ax_mac_scatter, ax_mac_param, ax_mac_latent = axes[1]

        draw_scatter(
            ax_mac_scatter,
            group_col='macro_family',
            group_list=all_macro,
            color_map=macro_color_map,
            draw_other=False,  # every macro_family is shown explicitly
        )

        macro_colors = [macro_color_map[f] for f in all_macro]
        draw_violin(
            ax_mac_param,
            column='num_parameters',
            ylabel='Number of Parameters',
            group_col='macro_family',
            x_order=all_macro,
            colors=macro_colors,
            n_groups=N_MACRO_GROUPS,
            x_label='Macro-Family',
            separator_at=None,
        )
        draw_violin(
            ax_mac_latent,
            column='latent_dim',
            ylabel='Latent Dimension',
            group_col='macro_family',
            x_order=all_macro,
            colors=macro_colors,
            n_groups=N_MACRO_GROUPS,
            x_label='Macro-Family',
            separator_at=None,
        )

        # ═════════════════════════════════════════════════════════════════════════════
        # Legends
        # Row-1 legend is placed ABOVE row 1 (gs[0, :]).
        # Row-2 legend stays BELOW row 2 (gs[3, :]).
        # ═════════════════════════════════════════════════════════════════════════════

        # ── Legend for row 1 (family) — TOP ───────────────────────────────────────────
        fam_handles = [
            mpatches.Patch(color=fam_color_map[f], label=f) for f in top_families
        ]
        fam_handles.append(mpatches.Patch(color=OTHER_COLOR, alpha=0.75, label='Other'))

        leg1 = leg_ax_fam.legend(
            handles=fam_handles,
            title=f'Model Family (top {TOP_N})',
            loc='center',
            ncol=8,
            frameon=True,
            framealpha=0.9,
            edgecolor='#aaaaaa',
            borderpad=0.6,
            handlelength=1.2,
            handleheight=0.9,
            columnspacing=0.8,
            labelspacing=0.35,
        )
        leg1.get_title().set_fontweight('bold')

        # ── Legend for row 2 (macro_family) — BOTTOM ──────────────────────────────────
        macro_handles = [
            mpatches.Patch(color=macro_color_map[f], label=f) for f in all_macro
        ]

        leg2 = leg_ax_macro.legend(
            handles=macro_handles,
            title='Macro-Family',
            loc='center',
            ncol=N_MACRO_GROUPS,
            frameon=True,
            framealpha=0.9,
            edgecolor='#aaaaaa',
            borderpad=0.8,
            handlelength=1.6,
            handleheight=1.2,
            columnspacing=1.2,
            labelspacing=0.5,
        )
        leg2.get_title().set_fontweight('bold')

        plt.savefig(
            str(OUT / 'combined_plots_2x3_legend_correct.pdf'), format='pdf'
        )
        return plt.show()

    _combined_2x3_plot()
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
    pretraining method. Only ~6% of models have this field recorded.

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
        df.explode('pretrain_dataset')  # a model with several corpora counts once for each of them
        .group_by('pretrain_dataset')
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


if __name__ == "__main__":
    app.run()
