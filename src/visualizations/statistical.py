"""Publication-ready visualizations for statistical regression results."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf

matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
STYLE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / 'configs'
    / 'plotting'
    / 'plt.mplstyle'
)
DPI = 150
ACCENT_COLOR = '#c44e52'
STRONG_COLOR = '#8b0000'
MARGINAL_COLOR = '#f0a500'
MUTED_COLOR = '#8c8c8c'
PALETTE_TREATMENT = {'0': '#7ea6c9', '1': '#c44e52'}
ANALYSIS_LABELS = {
    'dataset_change': 'Dataset Informativity',
    'large_vs_finetuned': 'Specialization',
    'small_vs_finetuned': 'Transfer Learning',
    'augmentation': 'Augmentation',
    'model_scale': 'Model Scale',
}
STAT_CASE_LABELS = {
    'raw': 'Raw Embeddings',
    'proto_no_prewhiten': 'Prototypes (no pre-whitening)',
    'proto_prewhiten': 'Prototypes (pre-whitened)',
}
METRIC_LABELS = {
    'total_spread': 'Total Spread',
    'mean_distance_to_centroid': 'Mean Dist. to Centroid',
    'std_distance_to_centroid': 'Std Dist. to Centroid',
    'density': 'Density',
    'effective_rank': 'Effective Rank',
    'n_components_90pct': r'N Components (90\%)',
    'participation_ratio': 'Participation Ratio',
    'participation_ratio_fast': 'Participation Ratio (fast)',
    'isotropy': 'Isotropy',
    'anisotropy_ratio': 'Anisotropy Ratio',
    'spectral_entropy': 'Spectral Entropy',
    'explained_var_ratio_top1': 'Expl. Var. Ratio (top-1)',
    'explained_var_ratio_top3': 'Expl. Var. Ratio (top-3)',
    'top_eigenvalue_ratio': 'Top Eigenvalue Ratio',
    'probing_accuracy': 'Probing Accuracy',
    'probing_recall': 'Probing Recall',
    'probing_precision': 'Probing Precision',
    'probing_f1': 'Probing F1',
}
METRIC_GROUPS = {
    'Spread / Distance': [
        'total_spread',
        'mean_distance_to_centroid',
        'std_distance_to_centroid',
        'density',
    ],
    'Dimensionality': [
        'effective_rank',
        'n_components_90pct',
        'participation_ratio',
        'participation_ratio_fast',
    ],
    'Spectral / Isotropy': [
        'isotropy',
        'spectral_entropy',
        'explained_var_ratio_top1',
        'explained_var_ratio_top3',
        'top_eigenvalue_ratio',
    ],
    'Probing': [
        'probing_accuracy',
        'probing_recall',
        'probing_precision',
        'probing_f1',
    ],
}
METRIC_ORDER = list(METRIC_LABELS.keys())
ANALYSIS_ORDER = [
    'dataset_change',
    'large_vs_finetuned',
    'small_vs_finetuned',
    'augmentation',
    'model_scale',
]
# Each entry: (list_of_analyses, output_tag).
# Finetuning comparisons share one figure (two columns, shared y-axis).
_PLOT_GROUPS = [
    (['dataset_change'], 'dataset_change'),
    (['large_vs_finetuned', 'small_vs_finetuned'], 'finetuned'),
    (['augmentation'], 'augmentation'),
    (['model_scale'], 'model_scale'),
]
STAT_CASE_ORDER = ['raw', 'proto_no_prewhiten', 'proto_prewhiten']
DATASET_LABELS = {
    'cifar10': 'CIFAR-10',
    'mnist': 'MNIST',
    'cifar100': 'CIFAR-100',
    'fashion_mnist': 'Fashion-MNIST',
    'tiny_imagenet': 'Tiny-ImageNet',
    'svhn': 'SVHN',
    'celeba': 'CelebA',
}


def _apply_style() -> None:
    plt.style.use(str(STYLE_PATH))
    sns.set_theme(style='whitegrid', palette='muted', font_scale=0.9)
    plt.rcParams.update(
        {
            'figure.dpi': DPI,
            'savefig.dpi': DPI,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times'],
        }
    )


def _tex(s: str) -> str:
    """Escape underscores for LaTeX rendering."""
    return s.replace('_', r'\_')


def _sig_style(p_values):
    """Return per-row styling arrays for forest plot elements.

    Four thresholds (BW-robust via color + marker shape + fill):
      p < 0.01  → dark red, filled star     ('*'), thickest line
      p < 0.05  → red,      filled circle   ('o'), thick line
      p < 0.10  → gold,     filled diamond  ('D'), medium line
      p ≥ 0.10  → grey,     hollow square   ('s'), thin line
    """
    line_colors, lws, facecolors, edgecolors, markers, annotations = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for p in p_values:
        if p < 0.01:
            line_colors.append(STRONG_COLOR)
            lws.append(3.5)
            facecolors.append(STRONG_COLOR)
            edgecolors.append(STRONG_COLOR)
            markers.append('*')
            annotations.append('')
        elif p < 0.05:
            line_colors.append(ACCENT_COLOR)
            lws.append(2.5)
            facecolors.append(ACCENT_COLOR)
            edgecolors.append(ACCENT_COLOR)
            markers.append('o')
            annotations.append('')
        elif p < 0.10:
            line_colors.append(MARGINAL_COLOR)
            lws.append(1.8)
            facecolors.append(MARGINAL_COLOR)
            edgecolors.append(MARGINAL_COLOR)
            markers.append('D')
            annotations.append('')
        else:
            line_colors.append(MUTED_COLOR)
            lws.append(1.0)
            facecolors.append('none')
            edgecolors.append(MUTED_COLOR)
            markers.append('s')
            annotations.append('')
    return line_colors, lws, facecolors, edgecolors, markers, annotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_regression_results(results_dir: str | Path) -> pl.DataFrame:
    """Load all regression parquets and concatenate into one DataFrame.

    Filename convention: stat__*{metric}__{stat_case}[___reg_type|__dataset].parquet.
    The stat_case is always immediately after the last "__" before the trailing tag.

    Examples:
        stat__effective_rank__raw.parquet              (legacy, no regression_type)
        stat__effective_rank__raw___pooled.parquet
        stat__effective_rank__raw___interaction.parquet
        stat__effective_rank__raw__cifar10.parquet      (within-dataset)
    """
    results_dir = Path(results_dir)
    frames = []
    for f in sorted(results_dir.glob('stat__*__*.parquet')):
        stem = f.stem  # e.g. "stat__effective_rank__raw___pooled"
        # Parse filename: stat__{metric}__{stat_case}[___type|__dataset]
        # The last "__" separates {stat_case}[___type__dataset] from {metric}
        # But metric may contain "__", so find the SECOND-TO-LAST "__".
        last_sep = stem.rfind('__')
        second_last_sep = stem[:last_sep].rfind('__')

        if second_last_sep < 4:  # "stat__" is 4 chars
            continue

        stat_case_with_tag = stem[last_sep + 2 :]  # e.g. "raw___pooled", "raw__cifar10"
        tag_sep_3 = stat_case_with_tag.rfind('___')

        if tag_sep_3 != -1:
            stat_case = stat_case_with_tag[:tag_sep_3]
            reg_type = stat_case_with_tag[tag_sep_3 + 3 :]
        else:
            tag_sep_2 = stat_case_with_tag.rfind('__')
            if tag_sep_2 != -1:
                stat_case = stat_case_with_tag[:tag_sep_2]
                reg_type = stat_case_with_tag[tag_sep_2 + 2 :]
            elif stat_case_with_tag in ('pooled', 'interaction'):
                stat_case = 'raw'
                reg_type = stat_case_with_tag
            else:
                stat_case = stem[second_last_sep + 2 : last_sep]
                reg_type = 'within'

        metric = stem[6:second_last_sep]

        df = pl.read_parquet(f)
        if 'regression_type' not in df.columns:
            df = df.with_columns(pl.lit(reg_type).alias('regression_type'))
        df = df.with_columns(
            pl.lit(metric).alias('metric'),
            pl.lit(stat_case).alias('stat_case_parsed'),
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame()

    all_cols = set()
    for df in frames:
        all_cols.update(df.columns)

    normalized = []
    for df in frames:
        missing = all_cols - set(df.columns)
        if missing:
            for c in missing:
                df = df.with_columns(pl.lit('').cast(pl.Utf8).alias(c))
        df = df.select(sorted(all_cols))
        for c in all_cols:
            if df.schema[c] in (pl.Utf8, pl.String):
                df = df.with_columns(pl.col(c).cast(pl.Utf8))
        normalized.append(df)
    return pl.concat(normalized)


def _fit_ols(df_pd, metric: str, extra_formula: str = ''):
    """Fit OLS with HC3 robust SEs. Returns the fitted model."""
    formula = f'{metric} ~ is_treatment{extra_formula}'
    return smf.ols(formula, data=df_pd).fit(cov_type='HC3')


# ---------------------------------------------------------------------------
# Figure 1 — Coefficient Forest Plot
# ---------------------------------------------------------------------------


def plot_forest(
    results_dir: str | Path,
    output_dir: str | Path,
    stat_cases: list | None = None,
    metrics: list | None = None,
    regression_type: str | None = None,
    combined: bool = False,
) -> list[Path]:
    """Forest plots of regression coefficients.

    - regression_type='pooled' or 'interaction': one figure per stat_case (or group)
    - regression_type='within': one figure per stat_case per dataset (or group)

    Parameters
    ----------
    results_dir : path to ``results/stat/``
    output_dir : path to save figures
    stat_cases : list of stat_cases to plot. If None, uses all cases present.
    metrics : list of metrics to plot. If None, uses all metrics present.
    regression_type : one of 'pooled', 'within', 'interaction'
    combined : if True, all treatments appear in one wide figure per stat_case;
               if False (default), each treatment group gets its own figure
               (finetuning comparisons share one figure).

    Returns
    -------
    List of saved figure paths.
    """
    if stat_cases is None:
        stat_cases = STAT_CASE_ORDER

    _apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg = load_regression_results(results_dir)
    if reg.is_empty():
        print('[WARN] No regression results found.')
        return []

    if regression_type is not None:
        reg = reg.filter(pl.col('regression_type') == regression_type)
        if reg.is_empty():
            print(f'[WARN] No data for regression_type={regression_type}.')
            return []

    x_label = r'$\hat{\beta}$'

    saved = []

    for case in stat_cases:
        df_case = reg.filter(pl.col('stat_case_parsed') == case)
        if df_case.is_empty():
            continue
        if metrics is not None:
            df_case = df_case.filter(pl.col('metric').is_in(metrics))
        if df_case.is_empty():
            continue

        if regression_type == 'within':
            datasets = sorted(df_case['dataset'].unique().to_list())
        else:
            datasets = [None]

        for dataset in datasets:
            df_plot = (
                df_case.filter(pl.col('dataset') == dataset) if dataset else df_case
            )

            plot_groups = [(ANALYSIS_ORDER, 'all')] if combined else _PLOT_GROUPS
            for analyses, plot_tag in plot_groups:
                has_data = any(
                    not df_plot.filter(pl.col('analysis_type') == a).is_empty()
                    for a in analyses
                )
                if not has_data:
                    continue

                n_cols = len(analyses)
                fig, axes = plt.subplots(
                    1,
                    n_cols,
                    figsize=(4.5 * n_cols + 1, 10),
                    sharey=(n_cols > 1),
                    gridspec_kw={'wspace': 0.12},
                )
                if n_cols == 1:
                    axes = [axes]

                for ax, analysis in zip(axes, analyses):
                    df_a = df_plot.filter(
                        pl.col('analysis_type') == analysis
                    ).to_pandas()

                    if df_a.empty:
                        ax.set_visible(False)
                        continue

                    df_a['metric_cat'] = pl.Series(df_a['metric']).cast(pl.Categorical)
                    present = [m for m in METRIC_ORDER if m in df_a['metric'].values]
                    df_a = df_a.set_index('metric').loc[present].reset_index()

                    # Group boundaries for dividers
                    group_boundaries = []
                    pos = 0
                    for group_metrics in METRIC_GROUPS.values():
                        group_metrics_in_plot = [
                            m for m in group_metrics if m in df_a['metric'].values
                        ]
                        if group_metrics_in_plot:
                            pos += len(group_metrics_in_plot)
                            if pos < len(df_a):
                                group_boundaries.append(pos - 0.5)

                    y_pos = np.arange(len(df_a))
                    p_vals = df_a['p_value'].values
                    # Scale by pooled control SD so all metrics share the same axis.
                    # Dividing by a constant per metric preserves p-values exactly.
                    scale = (
                        df_a['control_std'].values
                        if 'control_std' in df_a.columns
                        else np.ones(len(df_a))
                    )
                    scale = np.where(scale == 0, 1.0, scale)
                    ci_lower = df_a['ci_lower'].values / scale
                    ci_upper = df_a['ci_upper'].values / scale
                    beta = df_a['beta'].values / scale
                    line_colors, lws, facecolors, edgecolors, mrks, annots = _sig_style(
                        p_vals
                    )

                    # Alternate group background shading
                    shade_pos = 0
                    for g_idx, (_, gm) in enumerate(METRIC_GROUPS.items()):
                        gm_in = [m for m in gm if m in df_a['metric'].values]
                        n_gm = len(gm_in)
                        if n_gm:
                            if g_idx % 2 == 0:
                                ax.axhspan(
                                    shade_pos - 0.5,
                                    shade_pos + n_gm - 0.5,
                                    color='#f5f5f5',
                                    zorder=0,
                                    alpha=0.8,
                                )
                            shade_pos += n_gm

                    ax.hlines(
                        y_pos,
                        ci_lower,
                        ci_upper,
                        colors=line_colors,
                        linewidth=lws,
                        zorder=2,
                    )

                    # CI endpoint caps
                    cap_h = 0.13
                    for i, (y, c_lo, c_up) in enumerate(zip(y_pos, ci_lower, ci_upper)):
                        ax.plot(
                            [c_lo, c_lo],
                            [y - cap_h, y + cap_h],
                            color=line_colors[i],
                            linewidth=lws[i],
                            zorder=2,
                            solid_capstyle='butt',
                        )
                        ax.plot(
                            [c_up, c_up],
                            [y - cap_h, y + cap_h],
                            color=line_colors[i],
                            linewidth=lws[i],
                            zorder=2,
                            solid_capstyle='butt',
                        )

                    # Per-row scatter to support distinct marker shapes
                    for i, (b, y, fc, ec, m) in enumerate(
                        zip(beta, y_pos, facecolors, edgecolors, mrks)
                    ):
                        ax.scatter(
                            b,
                            y,
                            facecolors=fc,
                            edgecolors=ec,
                            marker=m,
                            s=65,
                            zorder=3,
                            linewidths=1.5,
                        )

                    ax.axvline(
                        0, color='black', linewidth=0.8, linestyle='--', alpha=0.5
                    )

                    # Horizontal group dividers
                    for y in group_boundaries:
                        ax.axhline(
                            y=y,
                            color='#888888',
                            linewidth=1.5,
                            linestyle='-',
                            alpha=0.7,
                        )

                    labels = [METRIC_LABELS.get(m, m) for m in df_a['metric']]
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(labels, fontsize=20)
                    ax.invert_yaxis()

                    title = ANALYSIS_LABELS.get(analysis, analysis)
                    ax.set_title(title, fontsize=28, pad=14)
                    ax.set_xlabel(x_label, fontsize=24, labelpad=10)
                    ax.tick_params(axis='x', labelsize=18, pad=8)

                    # Expand x-axis, then place significance annotations
                    xlo, xhi = ax.get_xlim()
                    ax.set_xlim(xlo, xhi + 0.14 * abs(xhi - xlo))
                    for i, (y, annot, color) in enumerate(
                        zip(y_pos, annots, line_colors)
                    ):
                        if annot:
                            ax.text(
                                ci_upper[i],
                                y,
                                f' {annot}',
                                va='center',
                                ha='left',
                                fontsize=18,
                                color=color,
                                zorder=4,
                                clip_on=False,
                            )

                from matplotlib.lines import Line2D

                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker='*',
                        color=STRONG_COLOR,
                        markerfacecolor=STRONG_COLOR,
                        markeredgecolor=STRONG_COLOR,
                        markersize=16,
                        linewidth=3.5,
                        label=r'$p < 0.01$',
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker='o',
                        color=ACCENT_COLOR,
                        markerfacecolor=ACCENT_COLOR,
                        markeredgecolor=ACCENT_COLOR,
                        markersize=14,
                        linewidth=2.5,
                        label=r'$p < 0.05$',
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker='D',
                        color=MARGINAL_COLOR,
                        markerfacecolor=MARGINAL_COLOR,
                        markeredgecolor=MARGINAL_COLOR,
                        markersize=12,
                        linewidth=1.8,
                        label=r'$p < 0.10$',
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker='s',
                        color=MUTED_COLOR,
                        markerfacecolor='none',
                        markeredgecolor=MUTED_COLOR,
                        markersize=12,
                        linewidth=1.0,
                        label=r'$p \geq 0.10$',
                    ),
                ]
                fig.legend(
                    handles=legend_elements,
                    loc='upper center',
                    ncol=4,
                    frameon=False,
                    fontsize=24,
                    bbox_to_anchor=(0.5, 1.03),
                )

                tag = f'__{dataset}' if dataset else f'__{regression_type}'
                path = output_dir / f'forest__{case}__{plot_tag}{tag}.pdf'
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)
                saved.append(path)
                print(f'  Saved: {path.name}')

    return saved


# ---------------------------------------------------------------------------
# Figure 2 — Faceted Group Means
# ---------------------------------------------------------------------------


def plot_group_means(
    df: pl.DataFrame,
    metric: str,
    stat_case: str,
    analysis_type: str,
    output_dir: str | Path,
) -> Path:
    """Control vs treatment means per architecture, faceted by arch_key and dataset.

    Parameters
    ----------
    df : raw per-model DataFrame (from the source parquet)
    metric : one of the 14 geometric metrics
    stat_case : 'raw', 'proto_no_prewhiten', or 'proto_prewhiten'
    analysis_type : one of the 4 analysis types
    output_dir : path to save the figure

    Returns
    -------
    Path to saved figure.
    """
    _apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = df.filter(
        (pl.col('stat_case') == stat_case) & (pl.col('analysis_type') == analysis_type)
    ).to_pandas()

    subset['group'] = subset['is_treatment'].map({0: 'Control', 1: 'Treatment'})

    has_dataset = 'dataset' in subset.columns

    archs = sorted(subset['arch_key'].unique())
    n_archs = len(archs)
    ncols = min(6, n_archs)
    nrows = int(np.ceil(n_archs / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        sharey=True,
    )
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    datasets = sorted(subset['dataset'].unique()) if has_dataset else [None]
    palette = {'Control': '#7ea6c9', 'Treatment': ACCENT_COLOR}
    if has_dataset and len(datasets) > 1:
        dataset_palette = {
            ds: sns.color_palette('tab10')[i] for i, ds in enumerate(datasets)
        }

    for i, arch in enumerate(archs):
        ax = axes[i]
        arch_data = subset[subset['arch_key'] == arch]
        hue = 'dataset' if has_dataset and len(datasets) > 1 else 'group'
        sns.pointplot(
            data=arch_data,
            x='group',
            y=metric,
            hue=hue,
            palette=dataset_palette if hue == 'dataset' else palette,
            dodge=False,
            errorbar=('ci', 95),
            capsize=0.15,
            ax=ax,
            legend=False,
        )
        ax.set_title(_tex(arch), fontsize=7, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=7)

    for j in range(n_archs, len(axes)):
        axes[j].set_visible(False)

    metric_label = METRIC_LABELS.get(metric, metric)
    case_label = STAT_CASE_LABELS.get(stat_case, stat_case)
    analysis_label = ANALYSIS_LABELS.get(analysis_type, analysis_type)
    fig.suptitle(
        f'{metric_label} — {analysis_label}\n{case_label}',
        fontsize=12,
        fontweight='bold',
    )
    fig.supylabel(metric_label, fontsize=10)
    fig.tight_layout()

    path = output_dir / f'group_means__{metric}__{stat_case}__{analysis_type}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Figure 3 — Partial Regression (Added Variable) Plot
# ---------------------------------------------------------------------------


def plot_partial_regression(
    df: pl.DataFrame,
    metric: str,
    stat_case: str,
    analysis_type: str,
    output_dir: str | Path,
    extra_formula: str = '',
    regression_type: str | None = None,
) -> Path:
    """Partial regression plot for the treatment effect after removing covariates.

    Parameters
    ----------
    df : raw per-model DataFrame
    metric : outcome variable
    stat_case : representation space
    analysis_type : comparison type
    output_dir : path to save the figure
    extra_formula : extra RHS terms for the regression formula
        (e.g., ' + C(dataset) + C(arch_key)' for pooled)
    regression_type : label for the title (e.g., 'pooled', 'within')

    Returns
    -------
    Path to saved figure.
    """
    _apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = df.filter(
        (pl.col('stat_case') == stat_case) & (pl.col('analysis_type') == analysis_type)
    ).to_pandas()

    model_full = _fit_ols(subset, metric, extra_formula=extra_formula)

    covariate_terms = extra_formula.strip().lstrip('+').strip()
    y_on_covars = smf.ols(f'{metric} ~ {covariate_terms}', data=subset).fit()
    resid_y = y_on_covars.resid

    # Residualize treatment on covariates
    x_on_covars = smf.ols(f'is_treatment ~ {covariate_terms}', data=subset).fit()
    resid_x = x_on_covars.resid

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        resid_x,
        resid_y,
        c=ACCENT_COLOR,
        alpha=0.5,
        s=20,
        edgecolors='white',
        linewidths=0.3,
        zorder=2,
    )

    slope = model_full.params.get('is_treatment', 0)
    x_range = np.array([resid_x.min(), resid_x.max()])
    ax.plot(x_range, slope * x_range, color='black', linewidth=1.2, zorder=3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)

    metric_label = METRIC_LABELS.get(metric, metric)
    case_label = STAT_CASE_LABELS.get(stat_case, stat_case)
    analysis_label = ANALYSIS_LABELS.get(analysis_type, analysis_type)
    rt_label = f' ({regression_type})' if regression_type else ''
    title = f'Partial Regression — {metric_label}\n{analysis_label}{rt_label}'
    title = f'{title} | {case_label}'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'is\_treatment $|$ covariates', fontsize=10)
    ax.set_ylabel(f'{metric_label} $|$ covariates', fontsize=10)

    fig.tight_layout()
    tag = f'__{regression_type}' if regression_type else ''
    path = output_dir / f'partial_reg__{metric}__{stat_case}__{analysis_type}{tag}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Figure 4 — Residual Diagnostics
# ---------------------------------------------------------------------------


def plot_residual_diagnostics(
    df: pl.DataFrame,
    metric: str,
    stat_case: str,
    analysis_type: str,
    output_dir: str | Path,
    extra_formula: str = '',
    regression_type: str | None = None,
) -> Path:
    """Residual diagnostics: (a) residuals vs fitted, (b) residuals by arch_key.

    Parameters
    ----------
    df : raw per-model DataFrame
    metric : outcome variable
    stat_case : representation space
    analysis_type : comparison type
    output_dir : path to save the figure
    extra_formula : extra RHS terms for the regression formula
    regression_type : label for the title (e.g., 'pooled', 'within')

    Returns
    -------
    Path to saved figure.
    """
    _apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = df.filter(
        (pl.col('stat_case') == stat_case) & (pl.col('analysis_type') == analysis_type)
    ).to_pandas()

    model = _fit_ols(subset, metric, extra_formula=extra_formula)
    residuals = model.resid
    fitted = model.fittedvalues

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(
        fitted,
        residuals,
        c=ACCENT_COLOR,
        alpha=0.4,
        s=20,
        edgecolors='white',
        linewidths=0.3,
    )
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fitted Values', fontsize=10)
    ax1.set_ylabel('Residuals', fontsize=10)
    ax1.set_title('(a) Residuals vs Fitted', fontsize=11, fontweight='bold')

    subset_resid = subset.copy()
    subset_resid['residual'] = residuals.values

    arch_var = (
        subset_resid.groupby('arch_key')['residual'].std().sort_values(ascending=False)
    )
    arch_order = arch_var.index.tolist()

    if len(arch_order) > 20:
        top_archs = arch_order[:20]
        subset_resid = subset_resid[subset_resid['arch_key'].isin(top_archs)]
        arch_order = top_archs
        ax2.set_title(
            '(b) Residuals by Architecture (top 20 by std)',
            fontsize=11,
            fontweight='bold',
        )
    else:
        ax2.set_title(
            '(b) Residuals by Architecture',
            fontsize=11,
            fontweight='bold',
        )

    sns.boxplot(
        data=subset_resid,
        x='arch_key',
        y='residual',
        order=arch_order,
        color='#7ea6c9',
        fliersize=2,
        linewidth=0.7,
        ax=ax2,
    )
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Architecture', fontsize=10)
    ax2.set_ylabel('Residuals', fontsize=10)
    ax2.tick_params(axis='x', rotation=90, labelsize=6)

    ticks = ax2.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([_tex(lbl.get_text()) for lbl in ax2.get_xticklabels()])

    threshold = arch_var.median() + 1.5 * arch_var.std()
    high_var = arch_var[arch_var > threshold].index.tolist()
    for label in ax2.get_xticklabels():
        raw_name = label.get_text().replace(r'\_', '_')
        if raw_name in high_var:
            label.set_color(ACCENT_COLOR)
            label.set_fontweight('bold')

    metric_label = METRIC_LABELS.get(metric, metric)
    case_label = STAT_CASE_LABELS.get(stat_case, stat_case)
    analysis_label = ANALYSIS_LABELS.get(analysis_type, analysis_type)
    rt_label = f' ({regression_type})' if regression_type else ''
    title = f'Residual Diagnostics — {metric_label}\n{analysis_label}{rt_label}'
    title = f'{title} | {case_label}'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    fig.tight_layout()
    tag = f'__{regression_type}' if regression_type else ''
    path = output_dir / f'residuals__{metric}__{stat_case}__{analysis_type}{tag}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path
