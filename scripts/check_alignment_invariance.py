"""Sanity checks for the semantic-alignment benchmark.

seeds   : linear and cca transmissions do not depend on the seed; proto does.
compare : two result directories agree to a tolerance (e.g. ridge on CPU vs GPU).

Usage:
    uv run scripts/check_alignment_invariance.py seeds
    uv run scripts/check_alignment_invariance.py compare results/alignment \
        results/alignment_ridge_gpu_check --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl

import alignment as al
from src.utils.alignment_aggregate import load_results

METRICS = ['accuracy', 'mse', 'mse_per_sample', 'nmse']


def check_seeds(dataset: str, model_a: str, model_b: str, ks: list[int]) -> bool:
    path = f'spaicom-lab/semasia-{dataset}'
    la_tr, _ = al._load_split(path, model_a, 'train')
    la_te, _ = al._load_split(path, model_a, 'test')
    lb_tr, _ = al._load_split(path, model_b, 'train')
    ok = True
    for seed in (42, 43):
        svd = al._fit_linear(la_tr, lb_tr, seed)
        if seed == 42:
            ref_lin = {k: al._linear_transmit(la_te, *svd, k) for k in ks}
            ref_cca = {k: al._cca_transmit(la_tr, lb_tr, la_te, k) for k in ks}
            ref_proto = {k: al._proto_transmit(la_tr, lb_tr, la_te, k, 42) for k in ks}
            continue
        for k in ks:
            d_lin = float(
                np.abs(al._linear_transmit(la_te, *svd, k) - ref_lin[k]).max()
            )
            d_cca = float(
                np.abs(al._cca_transmit(la_tr, lb_tr, la_te, k) - ref_cca[k]).max()
            )
            d_same = float(
                np.abs(
                    al._proto_transmit(la_tr, lb_tr, la_te, k, 42) - ref_proto[k]
                ).max()
            )
            d_diff = float(
                np.abs(
                    al._proto_transmit(la_tr, lb_tr, la_te, k, 43) - ref_proto[k]
                ).max()
            )
            print(
                f'K={k:4d}  seed42 vs seed43 max abs diff: linear={d_lin:.3e}'
                f'  cca={d_cca:.3e}'
                f'  | proto: same seed={d_same:.3e}, different seed={d_diff:.3e}'
            )
            ok &= d_lin == 0.0 and d_cca == 0.0 and d_same == 0.0 and d_diff > 0.0
    print('PASS' if ok else 'FAIL')
    return ok


def check_compare(dir_a: str, dir_b: str, seed: int, tol: float) -> bool:
    keys = ['dataset', 'model_a', 'method', 'k']
    frames = []
    for d in (dir_a, dir_b):
        df = load_results(d).filter(pl.col('seed').is_null() | (pl.col('seed') == seed))
        frames.append(df.select(keys + METRICS))
    j = frames[0].join(frames[1], on=keys, how='inner', suffix='_b')
    print(f'rows compared: {j.height} (A={frames[0].height}, B={frames[1].height})')
    ok = j.height == frames[0].height == frames[1].height
    for method in ['no_mismatch', 'linear', 'cca', 'proto']:
        sub = j.filter(pl.col('method') == method)
        diffs = {m: float((sub[m] - sub[f'{m}_b']).abs().max()) for m in METRICS}
        worst = max(diffs.values())
        print(
            f'{method:12s} n={sub.height:3d}  '
            + '  '.join(f'{m}={v:.2e}' for m, v in diffs.items())
        )
        ok &= worst <= tol
    print(f'tolerance {tol:g}: ' + ('PASS' if ok else 'FAIL'))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('seeds')
    s.add_argument('--dataset', default='mnist')
    s.add_argument('--model-a', default='vit_small_patch16_224.augreg_in1k')
    s.add_argument('--model-b', default='vit_large_patch16_224.augreg_in21k')
    s.add_argument('--ks', type=int, nargs='+', default=[2, 16, 128])
    c = sub.add_parser('compare')
    c.add_argument('dir_a')
    c.add_argument('dir_b')
    c.add_argument('--seed', type=int, default=42)
    c.add_argument('--tol', type=float, default=1e-6)
    args = ap.parse_args()
    if args.cmd == 'seeds':
        ok = check_seeds(args.dataset, args.model_a, args.model_b, args.ks)
    else:
        ok = check_compare(args.dir_a, args.dir_b, args.seed, args.tol)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
