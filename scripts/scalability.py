"""Scalability study of SEMASIA: sizes on the Hub, cost of the extraction and GPU-hour estimate.

One entry point for the whole study.  Which benchmarks are included and which analyses run is decided in
``configs/hydra/scalability.yaml`` (or on the command line); every analysis reads and writes parquet files in
``cfg.output_dir`` so that each one can be rerun alone.  Nothing is downloaded in full: Hub sizes come from the
API, row counts from parquet footers read with HTTP range requests, FLOPs from models built with
``pretrained=False``, throughput from dummy inputs.

Analyses (always run in this order; the files each one reads are written by the ones above it):

==========  ================================================================  ===================================
name        what it does                                                      writes (in ``output_dir``)
==========  ================================================================  ===================================
hub         file list and sizes of every ``<repo_id>/<prefix><dataset>``      hub_files, hub_repo_meta
footers     parquet footers (rows, schema) of a random sample of files        hub_footers
columns     bytes per column of a few files, size of the registry repo        column_bytes_sample, hub_registry_files
flops       forward GFLOPs (2 x MACs) and parameters of the registry models   model_flops
throughput  GPU images/s of ~15 models on dummy inputs (CUDA GPU required)    throughput
dataloader  PIL -> RGB -> timm transform -> collate rate, offline HF cache    dataloader_rate_by_dataset
traces      Hub commit history, local embedding files, passes in Hydra logs   hub_commits, local_embedding_files, log_passes
tables      sizes, FLOPs cost, throughput fits, GPU-hour estimates, check      table_*, cost_*, gpu_hours_*, key_numbers
==========  ================================================================  ===================================

Usage::

    uv run scripts/scalability.py                                   # everything in the config
    uv run scripts/scalability.py 'analyses=[flops,tables]'         # some analyses
    uv run scripts/scalability.py 'datasets=[cifar10,mnist]' output_dir=results/scalability_small 'analyses=[hub,footers,tables]'
    just scalability                                                # same as the first line
    just scalability-step flops                                     # one analysis

``tables`` needs the output of hub, footers, flops, throughput, dataloader and traces for the same ``datasets``;
"""

import datetime as dt
import gc
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from omegaconf import DictConfig

# run on the CPU (they do not fit in a 24 GB GPU in fp32): excluded from every GPU-hour figure (FLOPs and sizes still count them)
CPU_MODELS = ['vit_7b_patch16_dinov3.lvd1689m', 'vit_7b_patch16_dinov3.sat493m']
STEPS = [
    'hub',
    'footers',
    'columns',
    'flops',
    'throughput',
    'dataloader',
    'traces',
    'tables',
]


class Ctx:
    """Paths and names shared by all analyses, built from the config."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.R = str(self.out) + '/'
        self.datasets = list(cfg.datasets)
        unknown = [d for d in self.datasets if d not in cfg.benchmarks]
        if unknown:
            raise ValueError(f'datasets without an entry in `benchmarks`: {unknown}')
        self.ENC = {
            d: list(cfg.benchmarks[d].splits) for d in self.datasets
        }  # encoded splits
        self.ORDER = self.datasets

    def repo(self, name: str) -> str:
        return f'{self.cfg.repo_id}/{self.cfg.prefix}{name}'

    def registry(self) -> pl.DataFrame:
        """Model registry; if the local copy is missing it is downloaded (47 KB) from the registry repo on the Hub."""
        path = Path(self.cfg.registry)
        if not path.exists():
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                self.repo('model-registry'),
                path.name,
                repo_type='dataset',
                local_dir=path.parent,
            )
        return pl.read_parquet(path)

    def pq_files(self) -> pl.DataFrame:
        """Embedding parquet files (``<split>/<model>/<shard>``) of the selected datasets, from hub_files."""
        f = pl.read_parquet(self.R + 'hub_files.parquet')
        return f.filter(
            pl.col('is_parquet')
            & pl.col('model').is_not_null()
            & pl.col('dataset').is_in(self.datasets)
        )


# ---------------------------------------------------------------------------------------------------- hub
def step_hub(c: Ctx) -> None:
    """File list and sizes of each dataset repo (``dataset_info(files_metadata=True)``: no download)."""
    api = HfApi()
    rows, meta = [], []
    for n in c.datasets:
        i = api.dataset_info(c.repo(n), files_metadata=True)
        meta.append(
            {
                'dataset': n,
                'used_storage': getattr(i, 'used_storage', None),
                'n_siblings': len(i.siblings),
                'last_modified': str(i.last_modified),
                'sha': i.sha,
            }
        )
        for s in i.siblings:
            p = s.rfilename.split('/')  # <split>/<model>/<shard>.parquet
            rows.append(
                {
                    'dataset': n,
                    'path': s.rfilename,
                    'size': s.size,
                    'split': p[0] if len(p) == 3 else None,
                    'model': p[1] if len(p) == 3 else None,
                    'is_parquet': s.rfilename.endswith('.parquet'),
                }
            )
        print(n, len(i.siblings), flush=True)
    pl.DataFrame(rows).write_parquet(c.R + 'hub_files.parquet')
    pl.DataFrame(meta).write_parquet(c.R + 'hub_repo_meta.parquet')


# ------------------------------------------------------------------------------------------------ footers
def step_footers(c: Ctx) -> None:
    """Row counts and schema from parquet footers, for a random sample of ``footers.per_shard`` files per
    (dataset, split, shard).  Reading all ~30,000 footers hits the Hub rate limit (1000 calls / 5 min), hence the
    sample; every footer of a given (dataset, split, shard) reports the same number of rows (checked in ``tables``).
    Footers already read (hub_footers.parquet) are kept, so an interrupted run resumes."""
    import random

    fc, fs = c.cfg.footers, HfFileSystem()
    P = c.R + 'hub_footers.parquet'
    schema = {
        'dataset': pl.String,
        'path': pl.String,
        'num_rows': pl.Int64,
        'row_groups': pl.Int64,
        'columns': pl.List(pl.String),
        'embedding_type': pl.String,
        'compression': pl.String,
        'created_by': pl.String,
        'err': pl.String,
    }
    good = (
        pl.read_parquet(P).filter(
            pl.col('err').is_null() & pl.col('dataset').is_in(c.datasets)
        )
        if Path(P).exists()
        else pl.DataFrame(schema=schema)
    )
    f = c.pq_files().join(
        good.select('dataset', 'path'), on=['dataset', 'path'], how='anti'
    )
    f = f.with_columns(pl.col('path').str.split('/').list.last().alias('shard'))
    todo = []
    for (d, s, sh), g in f.group_by(['dataset', 'split', 'shard']):
        have = good.filter(
            (pl.col('dataset') == d)
            & pl.col('path').str.starts_with(f'{s}/')
            & pl.col('path').str.ends_with('/' + sh)
        ).height
        rows = g.sort('path').select('dataset', 'path').rows()
        random.Random(fc.seed).shuffle(rows)
        todo += rows if fc.per_shard is None else rows[: max(0, fc.per_shard - have)]
    print('footers to read', len(todo), flush=True)

    def footer(a):
        n, path = a
        err = None
        for k in range(fc.retries):
            try:
                pf = pq.ParquetFile(
                    fs.open(f'datasets/{c.repo(n)}/{path}', block_size=2**20)
                )
                m = pf.metadata
                return {
                    'dataset': n,
                    'path': path,
                    'num_rows': m.num_rows,
                    'row_groups': m.num_row_groups,
                    'columns': pf.schema_arrow.names,
                    'embedding_type': str(pf.schema_arrow.field('embedding').type),
                    'compression': m.row_group(0).column(0).compression,
                    'created_by': m.created_by,
                    'err': None,
                }
            except Exception as e:  # HTTP 429: wait for the rate-limit window
                err = repr(e)[:200]
                time.sleep(5 * 2**k)
        return {
            'dataset': n,
            'path': path,
            'num_rows': None,
            'row_groups': None,
            'columns': None,
            'embedding_type': None,
            'compression': None,
            'created_by': None,
            'err': err,
        }

    with ThreadPoolExecutor(fc.workers) as ex:
        new = pl.DataFrame(list(ex.map(footer, todo)), schema=schema)
    ok = new.filter(pl.col('err').is_null())
    pl.concat([good, ok]).write_parquet(P)
    print('footers ok', good.height + ok.height, 'failed now', new.height - ok.height)


# ------------------------------------------------------------------------------------------------ columns
def step_columns(c: Ctx) -> None:
    """Compressed / uncompressed bytes per column (from the footer of the first file of a few models), and the size of the registry repo."""
    fs, api = HfFileSystem(), HfApi()
    f = c.pq_files()
    rows = []
    for d in c.datasets:
        for m in c.cfg.columns.models:
            sub = f.filter((pl.col('dataset') == d) & (pl.col('model') == m))
            if sub.is_empty():
                continue
            r = sub.sort('path').row(
                0, named=True
            )  # first shard of the first split alphabetically
            md = pq.ParquetFile(
                fs.open(f'datasets/{c.repo(d)}/{r["path"]}', block_size=2**20)
            ).metadata
            for rg in range(md.num_row_groups):
                for k in range(md.num_columns):
                    cc = md.row_group(rg).column(k)
                    rows.append(
                        {
                            'dataset': d,
                            'path': r['path'],
                            'file_size': r['size'],
                            'column': cc.path_in_schema,
                            'compressed': cc.total_compressed_size,
                            'uncompressed': cc.total_uncompressed_size,
                            'compression': cc.compression,
                            'encodings': str(cc.encodings),
                        }
                    )
    pl.DataFrame(rows).write_parquet(c.R + 'column_bytes_sample.parquet')
    i = api.dataset_info(c.repo('model-registry'), files_metadata=True)
    reg = pl.DataFrame([{'path': s.rfilename, 'size': s.size} for s in i.siblings])
    reg.write_parquet(c.R + 'hub_registry_files.parquet')
    print(reg)


# -------------------------------------------------------------------------------------------------- flops
def step_flops(c: Ctx) -> None:
    """FLOPs (2 x MACs, forward, batch 1) and parameters of every registry model; ``pretrained=False, num_classes=0``.

    Input size = ``resolve_data_config({}, model=model)['input_size']``, the one used by encode_dataset_all_timm.py.
    Do NOT wrap the count in ``torch.no_grad()``: FlopCounterMode fails on some models (cait, pit) with it."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.layers import set_fused_attn
    from torch.utils.flop_counter import FlopCounterMode

    torch.set_num_threads(c.cfg.flops.threads)
    set_fused_attn(False)  # explicit matmuls so that attention FLOPs are counted
    reg = c.registry().select(
        'model_name', 'family', 'macro_family', 'latent_dim', 'num_parameters'
    )
    rows = reg.rows(named=True)[: c.cfg.flops.limit]
    out, t0 = [], time.time()

    def count(name, device):
        with torch.device(device):
            m = timm.create_model(name, pretrained=False, num_classes=0)
        m.eval().requires_grad_(False)
        fc = m.head.fc2 if hasattr(getattr(m, 'head', None), 'fc2') else None
        if isinstance(fc, torch.nn.Linear) and fc.out_features == 0:
            m.head.fc2 = (
                torch.nn.Identity()
            )  # same fix as src/models/timm.py:load_model
        cfg = resolve_data_config({}, model=m)
        x = torch.zeros((1, *cfg['input_size']), device=device)
        with FlopCounterMode(display=False) as f:
            y = m(x)
        return {
            'input_c': cfg['input_size'][0],
            'input_h': cfg['input_size'][1],
            'input_w': cfg['input_size'][2],
            'gflops': f.get_total_flops() / 1e9,
            'parameters_encoder': sum(p.numel() for p in m.parameters()),
            'out_dim': int(y.shape[-1]),
            'device': device,
        }

    keys = [
        'input_c',
        'input_h',
        'input_w',
        'gflops',
        'parameters_encoder',
        'out_dim',
        'device',
    ]
    for i, r in enumerate(rows):
        name, res, err, ts = r['model_name'], None, None, time.time()
        for dev in (
            'meta',
            'cpu',
        ):  # meta: no weights allocated; cpu as fallback for the models that need real tensors
            try:
                res = count(name, dev)
                break
            except Exception as e:
                err = f'{dev}: {type(e).__name__}: {str(e)[:200]}'
        row = dict(r) | {'parameters_registry': r['num_parameters']}
        del row['num_parameters']
        row.update(
            res if res else dict.fromkeys(keys),
            error=None if res else err,
            seconds=round(time.time() - ts, 2),
        )
        out.append(row)
        if i % 100 == 0:
            print(
                i,
                name,
                f'{time.time() - t0:.0f}s',
                res and round(res['gflops'], 2),
                flush=True,
            )
    df = pl.DataFrame(out, infer_schema_length=None).select(
        'model_name',
        'family',
        'macro_family',
        'latent_dim',
        *keys[:4],
        'parameters_encoder',
        'parameters_registry',
        'out_dim',
        'device',
        'seconds',
        'error',
    )
    df.write_parquet(
        c.R
        + (
            'model_flops_partial.parquet'
            if c.cfg.flops.limit
            else 'model_flops.parquet'
        )
    )
    print(
        'failed',
        df.filter(pl.col('error').is_not_null()).height,
        'of',
        df.height,
        f'in {time.time() - t0:.0f}s',
    )


# --------------------------------------------------------------------------------------------- throughput
def step_throughput(c: Ctx) -> None:
    """Model-only throughput on dummy inputs: ``pretrained=False, num_classes=0``, eval, ``inference_mode``, fp32, no autocast.

    Same settings as scripts/encode_dataset_all_timm.py (Lightning predict: eval + inference_mode, precision 32-true,
    default float32 matmul precision).  Models: the cheapest, median and dearest of model_flops, ``required_models`` and
    models closest to ``target_gflops`` up to ``n_models``.  Models that run out of memory are retried with
    ``oom_retry_batch_sizes`` until one fits.  Nothing is downloaded."""
    import timm
    import torch
    from timm.data import resolve_data_config

    tc = c.cfg.throughput
    df = pl.read_parquet(c.R + 'model_flops.parquet')
    reg = df['model_name'].to_list()
    s = df.sort('gflops')
    med = s['gflops'].median()
    chosen = [
        s['model_name'][0],
        df.with_columns((pl.col('gflops') - med).abs().alias('d')).sort('d')[
            'model_name'
        ][0],
        s['model_name'][-1],
    ]
    chosen += [m for m in tc.required_models if m in reg and m not in chosen]
    for tgt in tc.target_gflops:
        if len(chosen) >= tc.n_models:
            break
        chosen.append(
            df.filter(~pl.col('model_name').is_in(chosen))
            .with_columns((pl.col('gflops').log() - np.log(tgt)).abs().alias('d'))
            .sort('d')['model_name'][0]
        )
    print('chosen', chosen, flush=True)
    print(shutil.disk_usage('.'), flush=True)

    dev, gpu = 'cuda', torch.cuda.get_device_name(0)
    rows = []

    def measure(name, bs):
        r0 = df.filter(pl.col('model_name') == name).row(0, named=True)
        row = {
            'model_name': name,
            'gflops': r0['gflops'],
            'input_h': r0['input_h'],
            'parameters': r0['parameters_encoder'],
            'batch_size': bs,
            'gpu': gpu,
            'imgs_per_s': None,
            'ms_per_batch': None,
            'peak_mem_gb': None,
            'n_timed': None,
            'error': None,
        }
        m = x = None
        try:
            with torch.device(dev):
                m = timm.create_model(name, pretrained=False, num_classes=0)
            fc2 = getattr(getattr(m, 'head', None), 'fc2', None)
            if isinstance(fc2, torch.nn.Linear) and fc2.out_features == 0:
                m.head.fc2 = torch.nn.Identity()
            m.eval().requires_grad_(False)
            x = torch.randn(
                bs, *resolve_data_config({}, model=m)['input_size'], device=dev
            )
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(tc.warmup):
                    m(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(tc.iters):
                    m(x)
                torch.cuda.synchronize()
                dt_ = time.perf_counter() - t0
            row.update(
                imgs_per_s=bs * tc.iters / dt_,
                ms_per_batch=1000 * dt_ / tc.iters,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                n_timed=tc.iters,
            )
        except torch.OutOfMemoryError as e:
            row['error'] = 'OOM: ' + str(e)[:120]
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {str(e)[:160]}'
        m = x = None
        gc.collect()
        torch.cuda.empty_cache()
        rows.append(row)
        print(
            name,
            bs,
            row['imgs_per_s'] and round(row['imgs_per_s'], 1),
            row['peak_mem_gb'] and round(row['peak_mem_gb'], 2),
            row['error'],
            flush=True,
        )
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            c.R + 'throughput.parquet'
        )
        return row

    for name in chosen:
        for bs in tc.batch_sizes:
            measure(name, bs)
    ok = {(r['model_name'], r['batch_size']) for r in rows if r['error'] is None}
    for name in dict.fromkeys(r['model_name'] for r in rows if r['error'] is not None):
        for bs in tc.oom_retry_batch_sizes:
            if (name, bs) not in ok and measure(name, bs)['error'] is None:
                break
    print(shutil.disk_usage('.'))


# ------------------------------------------------------------------------------------------------ dataloader
def step_dataloader(c: Ctx) -> None:
    """Data-loading rate (PIL -> convert RGB -> timm eval transform at 224 -> collate) per dataset and number of workers,
    from the local HF datasets cache (offline: nothing is downloaded; a dataset missing from the cache is recorded as failed)."""
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    from functools import partial

    import timm
    import torch
    from datasets import load_dataset
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torch.utils.data import DataLoader

    dc = c.cfg.dataloader

    def collate_fn(batch, transform):
        return torch.stack([transform(s.convert('RGB')) for s in batch], dim=0)

    m = timm.create_model('resnet50.a1_in1k', pretrained=False, num_classes=0)
    tr = create_transform(**resolve_data_config({}, model=m))
    del m
    rows = []
    for k in c.datasets:
        b = c.cfg.benchmarks[k]
        try:
            ds = (
                load_dataset(b.hf, b.hf_config, split=b.hf_split)
                if b.hf_config
                else load_dataset(b.hf, split=b.hf_split)
            )
            sub = ds.select(range(min(dc.max_images, len(ds))))[b.image_column]
        except Exception as e:
            rows.append(
                {
                    'dataset': k,
                    'num_workers': None,
                    'imgs_per_s': None,
                    'err': repr(e)[:150],
                    'sample_size': None,
                }
            )
            print(k, 'FAILED', repr(e)[:150], flush=True)
            continue
        for nw in dc.workers:
            dl = DataLoader(
                sub,
                num_workers=nw,
                pin_memory=True,
                batch_size=dc.batch_size,
                collate_fn=partial(collate_fn, transform=tr),
            )
            t0, n = time.perf_counter(), 0
            for batch in dl:
                n += len(batch)
            rows.append(
                {
                    'dataset': k,
                    'num_workers': nw,
                    'imgs_per_s': n / (time.perf_counter() - t0),
                    'err': None,
                    'sample_size': str(sub[0].size),
                }
            )
            print(rows[-1], flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        c.R + 'dataloader_rate_by_dataset.parquet'
    )


# --------------------------------------------------------------------------------------------------- traces
def step_traces(c: Ctx) -> None:
    """Traces of the original extraction: Hub commit history of each repo, mtimes of the local embedding files, and
    the per-(model, split) passes found in the Hydra logs (``Loading pretrained weights`` = start of a pass; the duration
    of a pass = next start - this start, last pass: log mtime - start; includes model load, loader, GPU, parquet write)."""
    api = HfApi()
    rows = []
    for n in [*c.datasets, 'model-registry']:
        for _ in range(6):
            try:
                cs = api.list_repo_commits(c.repo(n), repo_type='dataset')
                break
            except Exception as e:  # 429: wait for the 5-minute window
                print('retry', n, str(e)[:60], flush=True)
                time.sleep(120)
        rows.extend(
            {
                'dataset': n,
                'commit': k.commit_id[:8],
                'date': k.created_at.replace(tzinfo=None),
                'title': k.title[:80],
            }
            for k in cs
        )
        print(
            n,
            len(cs),
            min(k.created_at for k in cs),
            max(k.created_at for k in cs),
            flush=True,
        )
    pl.DataFrame(rows).write_parquet(c.R + 'hub_commits.parquet')

    loc = []
    for p in glob.glob(c.cfg.local_embeddings):
        st, d = os.stat(p), p.split('/')
        if d[1] in c.datasets:
            loc.append(
                {
                    'dataset': d[1],
                    'split': d[2],
                    'model': d[3],
                    'size': st.st_size,
                    'mtime': dt.datetime.fromtimestamp(st.st_mtime),
                }
            )
    pl.DataFrame(loc).write_parquet(c.R + 'local_embedding_files.parquet')

    passes = []
    for d in sorted(glob.glob(c.cfg.hydra_logs)):
        log = d + 'encode_dataset_all_timm.log'
        if not os.path.exists(log):
            continue
        with open(d + '.hydra/overrides.yaml') as f_ov:
            ov = f_ov.read()
        with open(d + '.hydra/config.yaml') as f_cf:
            cf = f_cf.read()
        starts = []
        with open(log, errors='ignore') as f_log:
            for line in f_log:
                m = re.match(
                    r'\[([\d\- :,]+)\]\[timm.*Loading pretrained weights from Hugging Face hub \(timm/(.+)\)',
                    line,
                )
                if m:
                    starts.append(
                        (
                            dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f'),
                            m.group(2),
                        )
                    )
        end = dt.datetime.fromtimestamp(os.path.getmtime(log))
        g = lambda k: (re.search(k + r': (\S+)', cf) or [None, None])[1]  # noqa: E731
        dsname = re.search(r'dataset:\n  name: (\S+)', cf).group(1)
        for i, (t0, mdl) in enumerate(starts):
            t1 = starts[i + 1][0] if i + 1 < len(starts) else end
            passes.append(
                {
                    'run': d[8:-1],
                    'dataset': dsname,
                    'model': mdl,
                    'pass_idx': i,
                    'start': t0,
                    'seconds': (t1 - t0).total_seconds(),
                    'is_last': i + 1 == len(starts),
                    'batch_size': int(g('batch_size')),
                    'num_workers': int(g('num_workers')),
                    'accelerator': g('accelerator'),
                    'overrides': ' '.join(
                        x[2:] for x in ov.split('\n') if x.startswith('- ')
                    ),
                }
            )
    lp = pl.DataFrame(passes)
    lp.write_parquet(c.R + 'log_passes.parquet')
    print(
        lp.group_by('run', 'dataset', 'batch_size', 'num_workers')
        .agg(
            pl.len().alias('passes'),
            pl.col('seconds').sum().alias('sec'),
            pl.col('model').n_unique().alias('models'),
        )
        .sort('run')
        .tail(60)
    )


# -------------------------------------------------------------------------------------------------- tables
def step_tables(c: Ctx) -> None:
    """Derived tables from the parquet files of the previous analyses (every number derived from the other analyses comes from here).

    * sizes per dataset / split / model, expected raw bytes (latent_dim x N x 4) vs real bytes;
    * FLOPs cost = GFLOPs x N images for every (model, dataset) present on the Hub, shares of the top models;
    * throughput fits on the measured models (log-log OLS and latency + rate, both with residual SD);
      GPU-hours = N / throughput, optionally capped by the data-loading rate; low / high = x exp(-+1.96 SD);
    * cross-check of the estimate against the passes measured in the Nov-02 re-encode logs."""
    R, ORDER, ENC = c.R, c.ORDER, c.ENC
    ds_sel = c.datasets
    files = pl.read_parquet(R + 'hub_files.parquet').filter(
        pl.col('dataset').is_in(ds_sel)
    )
    pq_files = c.pq_files()
    foot = pl.read_parquet(R + 'hub_footers.parquet').filter(
        pl.col('err').is_null() & pl.col('dataset').is_in(ds_sel)
    )
    meta = pl.read_parquet(R + 'hub_repo_meta.parquet').filter(
        pl.col('dataset').is_in(ds_sel)
    )
    fl = pl.read_parquet(R + 'model_flops.parquet')
    reg = c.registry().select('model_name', 'latent_dim')
    out = {}

    # ---- rows per (dataset, split) from the footers (shard = file name inside the model folder)
    fj = foot.join(
        pq_files.select('dataset', 'path', 'split', 'model', 'size'),
        on=['dataset', 'path'],
    )
    fj = fj.with_columns(pl.col('path').str.split('/').list.last().alias('shard'))
    sh = fj.group_by('dataset', 'split', 'shard').agg(
        pl.col('num_rows').n_unique().alias('n_distinct_rows'),
        pl.col('num_rows').first().alias('rows'),
        pl.len().alias('footers_read'),
    )
    assert (sh['n_distinct_rows'] == 1).all(), sh.filter(pl.col('n_distinct_rows') > 1)
    rows_split = sh.group_by('dataset', 'split').agg(
        pl.col('rows').sum().alias('rows'),
        pl.col('shard').n_unique().alias('shards'),
        pl.col('footers_read').sum().alias('footers_read'),
    )
    nfiles = pq_files.group_by('dataset', 'split').agg(
        pl.len().alias('parquet_files'),
        pl.col('model').n_unique().alias('models'),
        pl.col('size').sum().alias('bytes'),
    )
    split_tbl = (
        nfiles.join(rows_split, on=['dataset', 'split'])
        .with_columns((pl.col('bytes') / 1e9).alias('GB'))
        .sort('dataset', 'split')
    )
    split_tbl.write_parquet(R + 'table_size_by_split.parquet')
    enc_ok = split_tbl.filter(
        pl.struct('dataset', 'split').map_elements(
            lambda s: s['split'] in ENC[s['dataset']], return_dtype=pl.Boolean
        )
    )
    assert enc_ok.height == split_tbl.height, (
        'Hub has splits other than the encoded ones'
    )
    N_d = split_tbl.group_by('dataset').agg(pl.col('rows').sum().alias('N'))

    # ---- per dataset size table
    ds = (
        split_tbl.group_by('dataset')
        .agg(
            pl.col('parquet_files').sum(),
            pl.col('models').max(),
            pl.col('bytes').sum().alias('parquet_bytes'),
        )
        .join(N_d, on='dataset')
        .join(meta.select('dataset', 'used_storage', 'n_siblings'), on='dataset')
        .join(
            files.group_by('dataset').agg(
                pl.col('size').sum().alias('all_files_bytes')
            ),
            on='dataset',
        )
    )
    ds = (
        ds.with_columns(
            (pl.col('parquet_bytes') / pl.col('models')).alias('bytes_per_model')
        )
        .with_columns(
            pl.col('dataset')
            .replace_strict({d: i for i, d in enumerate(ORDER)}, return_dtype=pl.Int64)
            .alias('o')
        )
        .sort('o')
        .drop('o')
    )
    ds.write_parquet(R + 'table_size_by_dataset.parquet')

    # ---- per-model size across the 8 datasets
    pm = pq_files.group_by('dataset', 'model').agg(pl.col('size').sum().alias('bytes'))
    pm_all = pm.group_by('model').agg(
        pl.col('bytes').sum().alias('bytes_8_datasets'),
        pl.col('dataset').n_unique().alias('n_datasets'),
    )
    pm_all = pm_all.join(reg, left_on='model', right_on='model_name').join(
        fl.select('model_name', 'gflops'), left_on='model', right_on='model_name'
    )
    pm.write_parquet(R + 'table_size_per_model_dataset.parquet')
    pm_all.write_parquet(R + 'table_size_per_model.parquet')

    # ---- expected raw embedding bytes vs real
    exp = (
        pq_files.select('dataset', 'model')
        .unique()
        .join(reg, left_on='model', right_on='model_name')
        .join(
            split_tbl.group_by('dataset').agg(pl.col('rows').sum().alias('N')),
            on='dataset',
        )
        .with_columns((pl.col('latent_dim') * pl.col('N') * 4).alias('raw_bytes'))
        .group_by('dataset')
        .agg(pl.col('raw_bytes').sum())
    )
    chk = (
        ds.select('dataset', 'parquet_bytes')
        .join(exp, on='dataset')
        .with_columns(
            (pl.col('parquet_bytes') / pl.col('raw_bytes')).alias('real_over_raw')
        )
    )
    chk.write_parquet(R + 'table_expected_vs_real.parquet')

    # ---- FLOPs: model x dataset presence
    present = pq_files.select('dataset', 'model').unique()
    cd = (
        present.join(
            fl.select('model_name', 'gflops'), left_on='model', right_on='model_name'
        )
        .join(N_d, on='dataset')
        .with_columns((pl.col('gflops') * 1e9 * pl.col('N')).alias('flops'))
    )
    cd.write_parquet(R + 'cost_model_dataset.parquet')
    cost_ds = cd.group_by('dataset').agg(
        pl.col('flops').sum().alias('flops'),
        pl.len().alias('models'),
        pl.col('N').first().alias('N'),
    )
    cost_model = (
        cd.group_by('model')
        .agg(pl.col('flops').sum().alias('flops'), pl.col('gflops').first())
        .sort('flops', descending=True)
    )
    tot = cost_model['flops'].sum()
    cost_model = cost_model.with_columns((pl.col('flops') / tot).alias('share'))
    cost_model.write_parquet(R + 'cost_per_model.parquet')
    cost_ds.write_parquet(R + 'cost_per_dataset.parquet')
    out['flops_total'] = float(tot)
    out['N_total'] = int(N_d['N'].sum())
    out['top10_share'] = float(cost_model.head(10)['share'].sum())
    inet = cost_ds.filter(pl.col('dataset') == 'imagenet-1k')
    out['imagenet_share'] = (
        float(inet['flops'][0] / tot) if inet.height else None
    )  # None: ImageNet-1k not selected
    out['n_forward_passes'] = int(cd.select(pl.col('N').sum())[0, 0])

    # ---- throughput fit (batch 16; when a model needed a smaller batch the largest measured one is used)
    th = pl.read_parquet(R + 'throughput.parquet').filter(pl.col('error').is_null())
    fit_pts = (
        th.sort('batch_size', descending=True)
        .filter(pl.col('batch_size') <= 16)
        .group_by('model_name')
        .first()
        .sort('gflops')
    )
    x, y = (
        np.log(fit_pts['gflops'].to_numpy()),
        np.log(fit_pts['imgs_per_s'].to_numpy()),
    )
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    r2 = 1 - (res**2).sum() / ((y - y.mean()) ** 2).sum()
    s = float(np.sqrt((res**2).sum() / (len(x) - 2)))
    fit_pts = fit_pts.with_columns(
        pl.Series('fit_imgs_per_s', np.exp(a + b * x)), pl.Series('resid_ln', res)
    )
    fit_pts.write_parquet(R + 'throughput_fit_points.parquet')
    out.update(
        fit_a=float(a), fit_b=float(b), fit_r2=float(r2), fit_s=float(s), fit_n=len(x)
    )
    # interpolation in log-log between the measured points (clamped at the ends)
    from scipy.optimize import least_squares

    _g, _t = fit_pts['gflops'].to_numpy(), 1 / fit_pts['imgs_per_s'].to_numpy()
    _r = least_squares(
        lambda p: np.log(np.exp(p[0]) + np.exp(p[1]) * _g) - np.log(_t),
        [np.log(1e-4), np.log(1e-5)],
    )
    c0, c1 = np.exp(_r.x)
    res_a = _r.fun
    s_a = float(np.sqrt((res_a**2).sum() / (len(_g) - 2)))
    r2_a = float(1 - (res_a**2).sum() / ((np.log(_t) - np.log(_t).mean()) ** 2).sum())
    fit_pts = fit_pts.with_columns(
        pl.Series('aff_imgs_per_s', 1 / (c0 + c1 * _g)),
        pl.Series('aff_resid_ln', res_a),
    )
    fit_pts.write_parquet(R + 'throughput_fit_points.parquet')
    out.update(
        aff_c0_ms=float(c0 * 1e3),
        aff_c1_us_per_gflop=float(c1 * 1e6),
        aff_tflops=float(1 / c1 / 1e3),
        aff_s=s_a,
        aff_r2=r2_a,
    )

    def thr_aff(g):
        return 1 / (c0 + c1 * g)

    def thr_interp(g):
        return np.exp(np.interp(np.log(g), x, y))

    def thr_fit(g):
        return np.exp(a + b * np.log(g))

    # ---- data-loading rate (images/s at 224) per dataset and number of workers; ImageNet-1k = proxy (not measured)
    dl = pl.read_parquet(R + 'dataloader_rate_by_dataset.parquet').filter(
        pl.col('imgs_per_s').is_not_null()
    )
    rate = {
        (r['dataset'], r['num_workers']): r['imgs_per_s']
        for r in dl.iter_rows(named=True)
    }
    for (
        dname,
        proxy,
    ) in (
        c.cfg.dataloader.proxy.items()
    ):  # benchmark not measured -> rate of the proxy (ImageNet-1k: large JPEGs)
        for nw in c.cfg.dataloader.workers:
            if (dname, nw) not in rate and dname in ds_sel and (proxy, nw) in rate:
                rate[(dname, nw)] = rate[(proxy, nw)]

    def hours(model_thr, loader_nw, lo_hi, sd=None):
        """GPU-hours for every (model, dataset) pair present on the Hub; model_thr(g)-> imgs/s, scaled by exp(lo_hi*1.96*s)."""
        k = np.exp(lo_hi * 1.96 * (s if sd is None else sd))
        d = cd.filter(~pl.col('model').is_in(CPU_MODELS)).join(
            fl.select('model_name', 'input_h'), left_on='model', right_on='model_name'
        )
        g, h, n, dsn = (
            d['gflops'].to_numpy(),
            d['input_h'].to_numpy(),
            d['N'].to_numpy(),
            d['dataset'].to_list(),
        )
        thr = model_thr(g) * k
        if loader_nw:
            ld = np.array([rate[(q, loader_nw)] for q in dsn]) * np.minimum(
                1.0, (224.0 / h) ** 2
            )
            thr = np.minimum(thr, ld)
        t = n / thr / 3600
        return d.with_columns(pl.Series('hours', t))

    rows = []
    for meth, f_, sd_ in [
        ('affine: latency + GFLOPs/rate', thr_aff, s_a),
        ('log-log fit', thr_fit, s),
    ]:
        for name, nw in [
            ('model only', 0),
            ('+ loader, 8 workers', 8),
            ('+ loader, 1 worker', 1),
        ]:
            for lab, lh in [('low', +1), ('central', 0), ('high', -1)]:
                r_ = hours(f_, nw, lh, sd_)
                rows.append(
                    dict(
                        scenario=name,
                        estimate=lab,
                        method=meth,
                        hours_total=r_['hours'].sum(),
                        **{
                            q: r_.filter(pl.col('dataset') == q)['hours'].sum()
                            for q in ORDER
                        },
                    )
                )
    for name, nw in [
        ('model only', 0),
        ('+ loader, 8 workers', 8),
        ('+ loader, 1 worker', 1),
    ]:
        r_ = hours(thr_interp, nw, 0)
        rows.append(
            dict(
                scenario=name,
                estimate='central',
                method='interpolation of the 15 points',
                hours_total=r_['hours'].sum(),
                **{q: r_.filter(pl.col('dataset') == q)['hours'].sum() for q in ORDER},
            )
        )
    hrs = pl.DataFrame(rows)
    hrs.write_parquet(R + 'gpu_hours_estimate.parquet')
    cen = hours(thr_aff, 0, 0, s_a)
    cen.write_parquet(R + 'gpu_hours_per_model_dataset_fit_model_only.parquet')
    top = (
        cen.group_by('model').agg(pl.col('hours').sum()).sort('hours', descending=True)
    )
    out['hours_top10_share_fit_model_only'] = float(
        top.head(10)['hours'].sum() / top['hours'].sum()
    )

    # ---- cross-check against the Nov-02 re-encode logs (InceptionNeXt, 5 models, this machine, RTX 4090): measured pass time vs estimate
    lp = pl.read_parquet(R + 'log_passes.parquet').filter(
        pl.col('run').str.starts_with('2026-11-02')
        & pl.col('overrides').str.contains('re_encode=true')
        & ~pl.col('overrides').str.contains('dataset.split')
    )
    DMAP = {b.hf: d for d, b in c.cfg.benchmarks.items()}
    lp = (
        lp.filter(pl.col('dataset').is_in(list(DMAP)))
        .with_columns(pl.col('dataset').replace_strict(DMAP).alias('ds'))
        .filter(pl.col('ds').is_in(ds_sel))
        .sort('run', 'pass_idx')
    )
    chk_rows = []
    for (run, mdl), g in lp.group_by(['run', 'model'], maintain_order=True):
        dsn = g['ds'][0]
        sp = ENC[dsn]
        if g.height != len(sp):
            continue  # incomplete (e.g. interrupted) -> skipped
        rows_total = int(N_d.filter(pl.col('dataset') == dsn)['N'][0])
        gf = float(fl.filter(pl.col('model_name') == mdl)['gflops'][0])
        ih = int(fl.filter(pl.col('model_name') == mdl)['input_h'][0])
        thr = float(thr_aff(gf))
        nw_ = int(g['num_workers'][0])
        ld = (
            rate[(dsn, 8 if nw_ >= 6 else 1)] * min(1.0, (224 / ih) ** 2)
            if nw_ >= 2
            else rate[(dsn, 1)] * min(1.0, (224 / ih) ** 2)
        )
        chk_rows.append(
            {
                'run': run,
                'dataset': dsn,
                'model': mdl,
                'batch_size': int(g['batch_size'][0]),
                'num_workers': nw_,
                'images': rows_total,
                'measured_seconds': float(g['seconds'].sum()),
                'est_model_only_s': rows_total / thr,
                'est_with_loader_s': rows_total / min(thr, ld),
            }
        )
    pl.DataFrame(chk_rows).write_parquet(R + 'crosscheck_nov02_reencode.parquet')

    # ---- traces
    com = pl.read_parquet(R + 'hub_commits.parquet').filter(
        pl.col('dataset').is_in(ds_sel)
    )
    out['commits'] = (
        com.group_by('dataset')
        .agg(
            pl.len().alias('n'),
            pl.col('date').min().alias('first'),
            pl.col('date').max().alias('last'),
        )
        .sort('first')
        .to_dicts()
    )
    pl.DataFrame(
        [{'key': k, 'value_json': json.dumps(v, default=str)} for k, v in out.items()]
    ).write_parquet(R + 'key_numbers.parquet')
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != 'commits'}, indent=1, default=str
        )
    )
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(40)
    print(ds)
    print(chk)
    print(hrs.select('scenario', 'estimate', 'method', 'hours_total'))
    print(pl.read_parquet(R + 'crosscheck_nov02_reencode.parquet').head(30))


STEP_FUNCS = {n: globals()[f'step_{n}'] for n in STEPS}


@hydra.main(
    config_path='../configs/hydra/', config_name='scalability', version_base='1.3'
)
def main(cfg: DictConfig) -> None:
    wanted = list(cfg.analyses)
    bad = [a for a in wanted if a not in STEPS]
    if bad:
        raise ValueError(f'unknown analyses {bad}; choose among {STEPS}')
    logging.getLogger('httpx').setLevel(
        logging.WARNING
    )  # one INFO line per HTTP range request otherwise
    c = Ctx(cfg)
    for name in STEPS:  # canonical order, whatever the order in the config
        if name in wanted:
            print(f'\n===== {name} ({len(c.datasets)} datasets) =====', flush=True)
            t0 = time.time()
            STEP_FUNCS[name](c)
            print(f'===== {name} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
