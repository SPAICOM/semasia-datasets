# ruff: noqa: E501, PERF401
"""Tables and claim checks for the semantic-alignment benchmark.

Reads the result directories produced by ``scripts/alignment.py`` and writes CSVs
plus ``tables.md`` to ``results/alignment_report/``. Everything printed in the
report comes from here.

Usage:
    uv run scripts/alignment_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from scipy import stats

from src.utils.alignment_aggregate import aggregate_pairs, load_results, per_pair

MAIN = Path('results/alignment')
EXT_VITL = Path('results/alignment_extended_vitl')
EXT_CNX = Path('results/alignment_extended_convnext')
LAMBDA = Path('results/alignment_lambda')
LSTSQ_CPU = Path('results/alignment_lstsq_cpu')
LSTSQ_GPU = Path('results/alignment_lstsq_gpu')
OUT = Path('results/alignment_report')

METHODS = ['linear', 'cca', 'proto']
NAME = {'linear': 'Linear', 'cca': 'CCA', 'proto': 'PPFE', 'no_mismatch': 'No Mismatch'}
DS = ['cifar10', 'mnist', 'fashion_mnist']
DS_NAME = {'cifar10': 'CIFAR-10', 'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST'}
METRICS = ['accuracy', 'mse', 'mse_per_sample', 'nmse']
DECIMALS = {'accuracy': 3, 'mse': 3, 'mse_per_sample': 1, 'nmse': 3}
HIGHER_BETTER = {'accuracy': True, 'mse': False, 'mse_per_sample': False, 'nmse': False}
KS_DEC = [2, 16, 128]


def md(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for row in df.iter_rows():
        lines.append('| ' + ' | '.join('' if v is None else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def pm(mean: float, half: float, nd: int) -> str:
    return f'{mean:.{nd}f} ± {half:.{nd}f}'


def paper_table(agg: pl.DataFrame, metric: str) -> pl.DataFrame:
    nd = DECIMALS[metric]
    rows = []
    for ds in DS:
        for k in sorted(agg.filter(pl.col('k') > 0)['k'].unique().to_list()):
            row = {'dataset': DS_NAME[ds], 'K': k}
            means = {}
            for m in METHODS:
                r = agg.filter(
                    (pl.col('dataset') == ds)
                    & (pl.col('metric') == metric)
                    & (pl.col('method') == m)
                    & (pl.col('k') == k)
                ).row(0, named=True)
                means[m] = r['mean']
                row[NAME[m]] = pm(r['mean'], (r['ci_high'] - r['ci_low']) / 2, nd)
            best = (max if HIGHER_BETTER[metric] else min)(means, key=means.get)
            row[NAME[best]] = f'**{row[NAME[best]]}**'
            rows.append(row)
    return pl.DataFrame(rows)


def claim_check(
    pairs: pl.DataFrame, metric: str
) -> tuple[pl.DataFrame, pl.DataFrame, int, int]:
    """Per (pair, dataset, K): winner among the three methods."""
    wide = (
        pairs.filter(pl.col('method').is_in(METHODS))
        .pivot(on='method', index=['dataset', 'model_a', 'k'], values=metric)
        .sort(['dataset', 'model_a', 'k'])
    )
    rows = []
    for r in wide.iter_rows(named=True):
        vals = {m: r[m] for m in METHODS}
        best = (max if HIGHER_BETTER[metric] else min)(vals, key=vals.get)
        others = {m: v for m, v in vals.items() if m != best}
        runner = (max if HIGHER_BETTER[metric] else min)(others, key=others.get)
        rows.append(
            {
                'dataset': r['dataset'],
                'model_a': r['model_a'],
                'k': r['k'],
                'winner': best,
                'linear': vals['linear'],
                'winner_value': vals[best],
                'margin_over_linear': abs(vals[best] - vals['linear']),
                'linear_is_best': best == 'linear',
                'runner_up': runner,
            }
        )
    res = pl.DataFrame(rows)
    return (
        res,
        res.filter(~pl.col('linear_is_best')),
        int(res['linear_is_best'].sum()),
        res.height,
    )


def analyze_group(tag: str, data: pl.DataFrame) -> list[str]:
    """Tables and claim checks for a set of pairs that share one receiver."""
    out: list[str] = [f'# Group `{tag}`\n']
    pairs = per_pair(data, METRICS)
    agg = aggregate_pairs(pairs, METRICS)
    agg.write_csv(OUT / f'{tag}_aggregate_across_pairs_all_metrics.csv')
    pairs.filter(pl.col('method').is_in(METHODS)).write_csv(
        OUT / f'{tag}_per_pair_per_k.csv'
    )
    receivers = sorted(data['model_b'].unique().to_list())
    transmitters = sorted(data['model_a'].unique().to_list())
    out.append(
        f'Receiver(s): {receivers}; transmitters ({len(transmitters)}): {transmitters}\n'
    )
    n_pairs = sorted(set(agg['n_pairs'].to_list()))
    out.append(
        f'## Aggregation sanity\n\nn_pairs per cell: {n_pairs}; rows in per-pair table: '
        f'{pairs.filter(pl.col("method").is_in(METHODS)).height} '
        '(= 4 pairs x 3 datasets x 7 K x 3 methods)\n'
    )

    # ---- 2. paper tables --------------------------------------------------
    out.append(
        '## Table 1. Mean ± 95% CI half-width (t interval, n=4 pairs); best mean in bold\n'
    )
    for metric, title in [('accuracy', 'Accuracy'), ('mse', 'MSE (mean over entries)')]:
        t = paper_table(agg, metric)
        t.write_csv(OUT / f'{tag}_table_{metric}.csv')
        out.append(f'### {title}\n\n{md(t)}\n')
    nm = agg.filter(
        (pl.col('method') == 'no_mismatch') & (pl.col('metric') == 'accuracy')
    ).select(
        pl.col('dataset').replace(DS_NAME).alias('dataset'),
        pl.col('mean').round(4).alias('No Mismatch accuracy'),
        pl.col('sd').round(6).alias('SD across pairs'),
    )
    nm.write_csv(OUT / f'{tag}_no_mismatch.csv')
    out.append(
        f'### No Mismatch accuracy (receiver probe on its own test embeddings)\n\n{md(nm)}\n'
    )

    out.append('## Table 2. MSE per sample (squared L2 norm of the error) and NMSE\n')
    for metric, title in [('mse_per_sample', 'MSE per sample'), ('nmse', 'NMSE')]:
        t = paper_table(agg, metric)
        t.write_csv(OUT / f'{tag}_table_{metric}.csv')
        out.append(f'### {title}\n\n{md(t)}\n')
    se = agg.filter(pl.col('k') > 0).select(
        'dataset', 'method', 'k', 'metric', 'mean', 'sd', 'se', 'n_pairs'
    )
    se.write_csv(OUT / f'{tag}_standard_errors.csv')

    # ---- 3. claim check ---------------------------------------------------
    out.append('## Claim check: is Linear the best method?\n')
    summary = []
    for metric in ['accuracy', 'mse']:
        res, fails, n_best, n_all = claim_check(pairs, metric)
        res.write_csv(OUT / f'{tag}_claim_cases_{metric}.csv')
        fails.write_csv(OUT / f'{tag}_claim_exceptions_{metric}.csv')
        summary.append({'metric': metric, 'linear best': f'{n_best} / {n_all}'})
        by_k = (
            res.group_by('k')
            .agg(
                pl.col('linear_is_best').sum().alias('linear_best'),
                pl.len().alias('cases'),
            )
            .sort('k')
        )
        by_ds = (
            res.group_by('dataset')
            .agg(
                pl.col('linear_is_best').sum().alias('linear_best'),
                pl.len().alias('cases'),
            )
            .sort('dataset')
        )
        by_pair = (
            res.group_by('model_a', 'dataset')
            .agg(pl.col('linear_is_best').sum().alias('linear_best_of_7'))
            .sort('model_a', 'dataset')
        )
        out.append(
            f'### {metric}: Linear best in {n_best} of {n_all} (pair, dataset, K) cases\n'
        )
        out.append('By K:\n\n' + md(by_k) + '\n')
        out.append('By dataset:\n\n' + md(by_ds) + '\n')
        out.append('By pair x dataset (of 7 K):\n\n' + md(by_pair) + '\n')
        show = fails.with_columns(
            pl.col('winner').replace(NAME),
            pl.col('runner_up').replace(NAME),
            pl.col('margin_over_linear').round(4),
            pl.col('linear').round(4),
            pl.col('winner_value').round(4),
        ).select(
            'dataset',
            'model_a',
            'k',
            'winner',
            'winner_value',
            'linear',
            'margin_over_linear',
        )
        out.append(f'Cases where Linear is NOT best ({fails.height}):\n\n{md(show)}\n')
        wins = fails.group_by('winner').len().sort('winner')
        out.append('Winners in those cases:\n\n' + md(wins) + '\n')

        # mean level
        best_mean = []
        for ds in DS:
            for k in sorted(agg.filter(pl.col('k') > 0)['k'].unique().to_list()):
                sub = agg.filter(
                    (pl.col('dataset') == ds)
                    & (pl.col('metric') == metric)
                    & (pl.col('k') == k)
                    & pl.col('method').is_in(METHODS)
                )
                s = sub.sort('mean', descending=HIGHER_BETTER[metric])
                best_mean.append(
                    {'dataset': ds, 'k': k, 'best_on_mean': s['method'][0]}
                )
        bm = pl.DataFrame(best_mean)
        bm.write_csv(OUT / f'{tag}_best_on_mean_{metric}.csv')
        piv = bm.pivot(on='dataset', index='k', values='best_on_mean').sort('k')
        out.append(
            f'Best method on the mean across pairs, per dataset and K ({metric}):\n\n{md(piv)}\n'
        )

    # CI overlap
    out.append('## 95% CI overlap of Linear with CCA and PPFE (t interval, n=4)\n')
    ov_rows = []
    for ds in DS:
        for k in sorted(agg.filter(pl.col('k') > 0)['k'].unique().to_list()):
            row = {'dataset': DS_NAME[ds], 'K': k}
            for metric in ['accuracy', 'mse']:
                lin = agg.filter(
                    (pl.col('dataset') == ds)
                    & (pl.col('metric') == metric)
                    & (pl.col('k') == k)
                    & (pl.col('method') == 'linear')
                ).row(0, named=True)
                for other in ['cca', 'proto']:
                    o = agg.filter(
                        (pl.col('dataset') == ds)
                        & (pl.col('metric') == metric)
                        & (pl.col('k') == k)
                        & (pl.col('method') == other)
                    ).row(0, named=True)
                    overlap = max(lin['ci_low'], o['ci_low']) <= min(
                        lin['ci_high'], o['ci_high']
                    )
                    row[f'{metric}: vs {NAME[other]}'] = (
                        'overlap' if overlap else 'separated'
                    )
            ov_rows.append(row)
    ov = pl.DataFrame(ov_rows)
    ov.write_csv(OUT / f'{tag}_ci_overlap_linear.csv')
    out.append(md(ov) + '\n')
    sep_counts = {c: int((ov[c] == 'separated').sum()) for c in ov.columns[2:]}
    out.append(
        f'Number of (dataset, K) cells (of 21) with separated intervals: {sep_counts}\n'
    )

    # ---- paired differences Linear - other (same pairs, so tighter than the CI overlap)
    out.append('## Paired differences (Linear minus other), t interval across pairs\n')
    prow = []
    lp = pairs.filter(pl.col('method').is_in(METHODS))
    for ds in DS:
        for k in sorted(lp['k'].unique().to_list()):
            for metric in ['accuracy', 'mse']:
                sub = lp.filter((pl.col('dataset') == ds) & (pl.col('k') == k)).pivot(
                    on='method', index='model_a', values=metric
                )
                for other in ['cca', 'proto']:
                    d = (sub['linear'] - sub[other]).to_numpy()
                    n = len(d)
                    half = (
                        float(stats.t.ppf(0.975, n - 1)) * float(d.std(ddof=1)) / n**0.5
                    )
                    mean = float(d.mean())
                    better = mean > 0 if HIGHER_BETTER[metric] else mean < 0
                    excl = abs(mean) > half
                    prow.append(
                        {
                            'dataset': ds,
                            'k': k,
                            'metric': metric,
                            'other': NAME[other],
                            'mean_diff': mean,
                            'ci_half': half,
                            'n_pairs': n,
                            'linear_better_on_mean': better,
                            'ci_excludes_0_linear_better': bool(better and excl),
                            'ci_excludes_0_other_better': bool((not better) and excl),
                        }
                    )
    pd_df = pl.DataFrame(prow)
    pd_df.write_csv(OUT / f'{tag}_paired_diff.csv')
    summ = (
        pd_df.group_by('metric', 'other', 'k')
        .agg(
            pl.col('ci_excludes_0_linear_better')
            .sum()
            .alias('cells_linear_sig_better'),
            pl.col('ci_excludes_0_other_better').sum().alias('cells_other_sig_better'),
        )
        .sort('metric', 'other', 'k')
    )
    out.append(
        'Cells (of 3 datasets) per K where the paired 95% interval excludes 0:\n\n'
        + md(summ)
        + '\n'
    )

    # ---- 5. seed variability ---------------------------------------------
    out.append('## PPFE seed variability (SD over 5 seeds, mean over 4 pairs)\n')
    pr = data.filter(pl.col('method') == 'proto')
    sd = (
        pr.group_by('dataset', 'model_a', 'k')
        .agg(
            pl.col('accuracy').std().alias('sd_accuracy'),
            pl.col('mse').std().alias('sd_mse'),
            pl.col('seed').n_unique().alias('n_seeds'),
        )
        .group_by('dataset', 'k')
        .agg(
            pl.col('sd_accuracy').mean(),
            pl.col('sd_mse').mean(),
            pl.col('n_seeds').min().alias('min_n_seeds'),
        )
        .filter(pl.col('k').is_in(KS_DEC))
        .sort('dataset', 'k')
        .with_columns(pl.col('sd_accuracy').round(4), pl.col('sd_mse').round(4))
    )
    sd.write_csv(OUT / f'{tag}_ppfe_seed_sd.csv')
    out.append(md(sd) + '\n')

    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out: list[str] = []

    data = load_results(MAIN)  # core: the 4 pairs of the original benchmark
    groups = {'core4': data}
    if list(EXT_VITL.glob('*.parquet')):
        groups['vitl_all'] = load_results([MAIN, EXT_VITL])
    if list(EXT_CNX.glob('*.parquet')):
        groups['convnext'] = load_results(EXT_CNX)
    allpairs = per_pair(
        load_results([d for d in (MAIN, EXT_VITL, EXT_CNX) if d.exists()]), METRICS
    )
    allpairs.filter(pl.col('method').is_in(METHODS)).write_csv(
        OUT / 'per_pair_per_k.csv'
    )
    for tag, df in groups.items():
        out.extend(analyze_group(tag, df))

    # ---- 4. decoder effect (core 4 pairs only) ------------------------------
    out.append('# Decoder effect and lambda sweep (core 4 pairs)\n')
    out.append('## Decoder effect (accuracy, mean over 4 pairs)\n')
    dec = {}
    main_seed42 = data.filter(pl.col('seed').is_null() | (pl.col('seed') == 42))
    dec['ridge (5 seeds avg)'] = per_pair(data, ['accuracy'])
    dec['ridge (seed 42)'] = per_pair(main_seed42, ['accuracy'])
    for label, d in [('lstsq CPU', LSTSQ_CPU), ('lstsq GPU', LSTSQ_GPU)]:
        if d.exists() and list(d.glob('*.parquet')):
            dec[label] = per_pair(load_results(d), ['accuracy'])
    dec_rows = []
    for ds in DS:
        for k in KS_DEC:
            for m in ['no_mismatch', 'proto', 'cca', 'linear']:
                kk = 0 if m == 'no_mismatch' else k
                row = {
                    'dataset': DS_NAME[ds],
                    'K': '-' if m == 'no_mismatch' else k,
                    'method': NAME[m],
                }
                for label, p in dec.items():
                    v = p.filter(
                        (pl.col('dataset') == ds)
                        & (pl.col('method') == m)
                        & (pl.col('k') == kk)
                    )['accuracy'].mean()
                    row[label] = None if v is None else round(float(v), 4)
                dec_rows.append(row)
    dec_df = pl.DataFrame(dec_rows).unique(maintain_order=True)
    dec_df.write_csv(OUT / 'decoder_effect.csv')
    out.append(md(dec_df.filter(pl.col('dataset') == 'CIFAR-10')) + '\n')
    out.append('All datasets: `decoder_effect.csv`.\n')

    # ---- lambda sweep -----------------------------------------------------
    if LAMBDA.exists():
        out.append('## Lambda sweep (accuracy, seed-averaged, mean over 4 pairs)\n')
        lam = load_results(LAMBDA)
        lp = (
            lam.group_by('dataset', 'model_a', 'method', 'k', 'lam_rel')
            .agg(pl.col('accuracy').mean())
            .group_by('dataset', 'method', 'k', 'lam_rel')
            .agg(pl.col('accuracy').mean())
        )
        lp.write_csv(OUT / 'lambda_sweep.csv')
        lams = sorted(lp['lam_rel'].unique().to_list())
        lam_rows = []
        rank_rows = []
        for ds in DS:
            for k in KS_DEC:
                for m in ['no_mismatch', 'proto', 'cca', 'linear']:
                    kk = 0 if m == 'no_mismatch' else k
                    row = {
                        'dataset': DS_NAME[ds],
                        'K': '-' if m == 'no_mismatch' else k,
                        'method': NAME[m],
                    }
                    for lv in lams:
                        v = lp.filter(
                            (pl.col('dataset') == ds)
                            & (pl.col('method') == m)
                            & (pl.col('k') == kk)
                            & (pl.col('lam_rel') == lv)
                        )['accuracy']
                        row[f'lam={lv:g}'] = round(float(v[0]), 4)
                    lam_rows.append(row)
                order = {}
                for lv in lams:
                    sub = lp.filter(
                        (pl.col('dataset') == ds)
                        & (pl.col('k') == k)
                        & (pl.col('lam_rel') == lv)
                        & pl.col('method').is_in(METHODS)
                    ).sort('accuracy', descending=True)
                    order[f'lam={lv:g}'] = ' > '.join(NAME[x] for x in sub['method'])
                rank_rows.append({'dataset': DS_NAME[ds], 'K': k, **order})
        lt = pl.DataFrame(lam_rows).unique(maintain_order=True)
        lt.write_csv(OUT / 'lambda_sweep_table.csv')
        rk = pl.DataFrame(rank_rows)
        rk.write_csv(OUT / 'lambda_sweep_ranking.csv')
        out.append(md(lt) + '\n')
        out.append(
            'Ranking of the three methods by mean accuracy, per lam_rel:\n\n'
            + md(rk)
            + '\n'
        )
        changed = [
            r
            for r in rank_rows
            if len({v for kx, v in r.items() if kx.startswith('lam=')}) > 1
        ]
        out.append(
            f'(dataset, K) cells (of {len(rank_rows)}) whose ranking changes with lambda: {len(changed)}\n'
        )

    (OUT / 'tables.md').write_text('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
