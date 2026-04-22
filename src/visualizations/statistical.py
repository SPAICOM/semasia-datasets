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
MUTED_COLOR = '#8c8c8c'
PALETTE_TREATMENT = {'0': '#7ea6c9', '1': '#c44e52'}
ANALYSIS_LABELS = {
    'dataset_change': 'Dataset Change',
    'large_vs_finetuned': 'Large vs Fine-tuned',
    'small_vs_finetuned': 'Small vs Fine-tuned',
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
}
METRIC_ORDER = list(METRIC_LABELS.keys())
ANALYSIS_ORDER = [
    'dataset_change',
    'large_vs_finetuned',
    'small_vs_finetuned',
    'augmentation',
    'model_scale',
]
STAT_CASE_ORDER = ['raw', 'proto_no_prewhiten', 'proto_prewhiten']


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_regression_results(results_dir: str | Path) -> pl.DataFrame:
    """Load all regression parquets and concatenate into one DataFrame."""
    results_dir = Path(results_dir)
    frames = []
    for f in sorted(results_dir.glob('stat__*__*.parquet')):
        stem = f.stem  # e.g. stat__effective_rank__raw
        parts = stem.split('__')
        # parts = ['stat', metric..., stat_case]
        stat_case = parts[-1]
        metric = '__'.join(parts[1:-1])
        df = pl.read_parquet(f).with_columns(
            pl.lit(metric).alias('metric'),
            pl.lit(stat_case).alias('stat_case_parsed'),
        )
        frames.append(df)
    return pl.concat(frames)


def _fit_ols(df_pd, metric: str):
    """Fit OLS with HC3 robust SEs. Returns the fitted model."""
    return smf.ols(
        f'{metric} ~ is_treatment + C(arch_key)',
        data=df_pd,
    ).fit(cov_type='HC3')


# ---------------------------------------------------------------------------
# Figure 1 — Coefficient Forest Plot
# ---------------------------------------------------------------------------


def plot_forest(
    results_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Forest plots of regression coefficients, one figure per stat_case.

    Parameters
    ----------
    results_dir : path to ``results/stat/``
    output_dir : path to save figures

    Returns
    -------
    List of saved figure paths.
    """
    _apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg = load_regression_results(results_dir)
    is_standardized = (
        'standardized' in reg.columns
        and reg['standardized'].drop_nulls().to_list()
        and reg['standardized'].drop_nulls()[0]
    )
    x_label = (
        r'$\beta$ (treatment effect, in control SDs)'
        if is_standardized
        else r'$\beta$ (treatment effect)'
    )
    saved = []

    for case in STAT_CASE_ORDER:
        df_case = reg.filter(pl.col('stat_case_parsed') == case)
        if df_case.is_empty():
            continue

        n_analyses = len(ANALYSIS_ORDER)
        fig, axes = plt.subplots(
            1,
            n_analyses,
            figsize=(4.5 * n_analyses, 6),
            sharey=True,
            layout='constrained',
        )
        fig.suptitle(
            f'Treatment Effect Coefficients — {STAT_CASE_LABELS.get(case, case)}',
            fontsize=13,
            fontweight='bold',
            y=1.02,
        )

        for ax, analysis in zip(axes, ANALYSIS_ORDER):
            df_a = df_case.filter(pl.col('analysis_type') == analysis).to_pandas()

            # Order metrics consistently
            df_a['metric'] = pl.Series(df_a['metric']).cast(pl.Categorical)
            present = [m for m in METRIC_ORDER if m in df_a['metric'].values]
            df_a = df_a.set_index('metric').loc[present].reset_index()

            y_pos = np.arange(len(df_a))
            colors = [
                ACCENT_COLOR if p < 0.05 else MUTED_COLOR for p in df_a['p_value']
            ]

            ax.hlines(
                y_pos,
                df_a['ci_lower'],
                df_a['ci_upper'],
                colors=colors,
                linewidth=1.5,
                zorder=1,
            )
            ax.scatter(
                df_a['beta'],
                y_pos,
                c=colors,
                s=40,
                zorder=2,
                edgecolors='white',
                linewidths=0.5,
            )
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

            labels = [METRIC_LABELS.get(m, m) for m in df_a['metric']]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_title(ANALYSIS_LABELS.get(analysis, analysis), fontsize=11)
            ax.set_xlabel(x_label)

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                markerfacecolor=ACCENT_COLOR,
                markersize=8,
                label=r'$p < 0.05$',
            ),
            Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                markerfacecolor=MUTED_COLOR,
                markersize=8,
                label=r'$p \geq 0.05$',
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=2,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.04),
        )

        path = output_dir / f'forest__{case}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
        print(f'  Saved: {path}')

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
    """Control vs treatment means per architecture, faceted by arch_key.

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

    palette = {'Control': '#7ea6c9', 'Treatment': ACCENT_COLOR}

    for i, arch in enumerate(archs):
        ax = axes[i]
        arch_data = subset[subset['arch_key'] == arch]
        sns.pointplot(
            data=arch_data,
            x='group',
            y=metric,
            hue='group',
            palette=palette,
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

    # Hide unused axes
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
) -> Path:
    """Partial regression plot for the treatment effect after removing arch variance.

    Parameters
    ----------
    df : raw per-model DataFrame
    metric : outcome variable
    stat_case : representation space
    analysis_type : comparison type
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

    # Manually compute partial regression residuals
    model_full = _fit_ols(subset, metric)

    # Residualize outcome on arch dummies
    y_on_others = smf.ols(f'{metric} ~ C(arch_key)', data=subset).fit()
    resid_y = y_on_others.resid

    # Residualize treatment on arch dummies
    x_on_others = smf.ols('is_treatment ~ C(arch_key)', data=subset).fit()
    resid_x = x_on_others.resid

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

    # Add regression line through partial residuals
    slope = model_full.params.get('is_treatment', 0)
    x_range = np.array([resid_x.min(), resid_x.max()])
    ax.plot(x_range, slope * x_range, color='black', linewidth=1.2, zorder=3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)

    metric_label = METRIC_LABELS.get(metric, metric)
    case_label = STAT_CASE_LABELS.get(stat_case, stat_case)
    analysis_label = ANALYSIS_LABELS.get(analysis_type, analysis_type)
    ax.set_title(
        f'Partial Regression — {metric_label}\n{analysis_label} | {case_label}',
        fontsize=11,
        fontweight='bold',
    )
    ax.set_xlabel(r'is\_treatment $|$ C(arch\_key)', fontsize=10)
    ax.set_ylabel(f'{metric_label} $|$ C(arch\\_key)', fontsize=10)

    fig.tight_layout()
    path = output_dir / f'partial_reg__{metric}__{stat_case}__{analysis_type}.png'
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
) -> Path:
    """Residual diagnostics: (a) residuals vs fitted, (b) residuals by arch_key.

    Parameters
    ----------
    df : raw per-model DataFrame
    metric : outcome variable
    stat_case : representation space
    analysis_type : comparison type
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

    model = _fit_ols(subset, metric)
    residuals = model.resid
    fitted = model.fittedvalues

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Residuals vs Fitted
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

    # (b) Residuals by arch_key
    subset_resid = subset.copy()
    subset_resid['residual'] = residuals.values

    arch_var = (
        subset_resid.groupby('arch_key')['residual'].std().sort_values(ascending=False)
    )
    arch_order = arch_var.index.tolist()

    # Limit to top 20 architectures by residual variance for readability
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

    # Escape underscores in arch tick labels for LaTeX
    ticks = ax2.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([_tex(lbl.get_text()) for lbl in ax2.get_xticklabels()])

    # Highlight high-variance architectures
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
    fig.suptitle(
        f'Residual Diagnostics — {metric_label}\n{analysis_label} | {case_label}',
        fontsize=12,
        fontweight='bold',
    )

    fig.tight_layout()
    path = output_dir / f'residuals__{metric}__{stat_case}__{analysis_type}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path
