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
report      ``report.md`` from the parquet files (all 8 benchmarks needed)     report.md
==========  ================================================================  ===================================

Usage::

    uv run scripts/scalability.py                                   # everything in the config
    uv run scripts/scalability.py 'analyses=[flops,tables]'         # some analyses
    uv run scripts/scalability.py 'datasets=[cifar10,mnist]' output_dir=results/scalability_small 'analyses=[hub,footers,tables]'
    just scalability                                                # same as the first line
    just scalability-step flops                                     # one analysis

``tables`` needs the output of hub, footers, flops, throughput, dataloader and traces for the same ``datasets``;
``report`` contains sentences about the full study (1,699 models, 8 benchmarks) and is therefore refused for a subset.
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

REPORT_DATASETS = 8
STEPS = [
    'hub',
    'footers',
    'columns',
    'flops',
    'throughput',
    'dataloader',
    'traces',
    'tables',
    'report',
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
            dict(
                dataset=n,
                used_storage=getattr(i, 'used_storage', None),
                n_siblings=len(i.siblings),
                last_modified=str(i.last_modified),
                sha=i.sha,
            )
        )
        for s in i.siblings:
            p = s.rfilename.split('/')  # <split>/<model>/<shard>.parquet
            rows.append(
                dict(
                    dataset=n,
                    path=s.rfilename,
                    size=s.size,
                    split=p[0] if len(p) == 3 else None,
                    model=p[1] if len(p) == 3 else None,
                    is_parquet=s.rfilename.endswith('.parquet'),
                )
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
                return dict(
                    dataset=n,
                    path=path,
                    num_rows=m.num_rows,
                    row_groups=m.num_row_groups,
                    columns=pf.schema_arrow.names,
                    embedding_type=str(pf.schema_arrow.field('embedding').type),
                    compression=m.row_group(0).column(0).compression,
                    created_by=m.created_by,
                    err=None,
                )
            except Exception as e:  # HTTP 429: wait for the rate-limit window
                err = repr(e)[:200]
                time.sleep(5 * 2**k)
        return dict(
            dataset=n,
            path=path,
            num_rows=None,
            row_groups=None,
            columns=None,
            embedding_type=None,
            compression=None,
            created_by=None,
            err=err,
        )

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
                        dict(
                            dataset=d,
                            path=r['path'],
                            file_size=r['size'],
                            column=cc.path_in_schema,
                            compressed=cc.total_compressed_size,
                            uncompressed=cc.total_uncompressed_size,
                            compression=cc.compression,
                            encodings=str(cc.encodings),
                        )
                    )
    pl.DataFrame(rows).write_parquet(c.R + 'column_bytes_sample.parquet')
    i = api.dataset_info(c.repo('model-registry'), files_metadata=True)
    reg = pl.DataFrame([dict(path=s.rfilename, size=s.size) for s in i.siblings])
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
    reg = pl.read_parquet(c.cfg.registry).select(
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
        return dict(
            input_c=cfg['input_size'][0],
            input_h=cfg['input_size'][1],
            input_w=cfg['input_size'][2],
            gflops=f.get_total_flops() / 1e9,
            parameters_encoder=sum(p.numel() for p in m.parameters()),
            out_dim=int(y.shape[-1]),
            device=device,
        )

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
        row = dict(
            model_name=name,
            gflops=r0['gflops'],
            input_h=r0['input_h'],
            parameters=r0['parameters_encoder'],
            batch_size=bs,
            gpu=gpu,
            imgs_per_s=None,
            ms_per_batch=None,
            peak_mem_gb=None,
            n_timed=None,
            error=None,
        )
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
                dict(
                    dataset=k,
                    num_workers=None,
                    imgs_per_s=None,
                    err=repr(e)[:150],
                    sample_size=None,
                )
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
                dict(
                    dataset=k,
                    num_workers=nw,
                    imgs_per_s=n / (time.perf_counter() - t0),
                    err=None,
                    sample_size=str(sub[0].size),
                )
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
        for k in cs:
            rows.append(
                dict(
                    dataset=n,
                    commit=k.commit_id[:8],
                    date=k.created_at.replace(tzinfo=None),
                    title=k.title[:80],
                )
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
                dict(
                    dataset=d[1],
                    split=d[2],
                    model=d[3],
                    size=st.st_size,
                    mtime=dt.datetime.fromtimestamp(st.st_mtime),
                )
            )
    pl.DataFrame(loc).write_parquet(c.R + 'local_embedding_files.parquet')

    passes = []
    for d in sorted(glob.glob(c.cfg.hydra_logs)):
        log = d + 'encode_dataset_all_timm.log'
        if not os.path.exists(log):
            continue
        ov, cf = (
            open(d + '.hydra/overrides.yaml').read(),
            open(d + '.hydra/config.yaml').read(),
        )
        starts = []
        for line in open(log, errors='ignore'):
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
                dict(
                    run=d[8:-1],
                    dataset=dsname,
                    model=mdl,
                    pass_idx=i,
                    start=t0,
                    seconds=(t1 - t0).total_seconds(),
                    is_last=i + 1 == len(starts),
                    batch_size=int(g('batch_size')),
                    num_workers=int(g('num_workers')),
                    accelerator=g('accelerator'),
                    overrides=' '.join(
                        x[2:] for x in ov.split('\n') if x.startswith('- ')
                    ),
                )
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
    """Derived tables from the parquet files of the previous analyses (all numbers of the report come from here).

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
    reg = pl.read_parquet(c.cfg.registry).select('model_name', 'latent_dim')
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
    out['imagenet_share'] = float(inet['flops'][0] / tot) if inet.height else None  # None: ImageNet-1k not selected
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
        d = cd.join(
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
    out['hours_7b_share'] = float(
        top.filter(pl.col('model').str.contains('vit_7b'))['hours'].sum()
        / top['hours'].sum()
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
            dict(
                run=run,
                dataset=dsn,
                model=mdl,
                batch_size=int(g['batch_size'][0]),
                num_workers=nw_,
                images=rows_total,
                measured_seconds=float(g['seconds'].sum()),
                est_model_only_s=rows_total / thr,
                est_with_loader_s=rows_total / min(thr, ld),
            )
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
        [dict(key=k, value_json=json.dumps(v, default=str)) for k, v in out.items()]
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


# -------------------------------------------------------------------------------------------------- report
def step_report(c: Ctx) -> None:
    """Write ``report.md`` from the parquet files (numbers are read, none typed by hand).  The text refers to the full
    study, so it is refused when ``datasets`` is not the 8 published benchmarks."""
    if len(c.datasets) != REPORT_DATASETS:
        raise ValueError(
            f'report.md describes the full study ({REPORT_DATASETS} benchmarks), got {len(c.datasets)}: drop `report` from `analyses`'
        )
    R, ORDER = c.R, c.ORDER
    K = {
        r['key']: json.loads(r['value_json'])
        for r in pl.read_parquet(R + 'key_numbers.parquet').iter_rows(named=True)
    }
    ds = pl.read_parquet(R + 'table_size_by_dataset.parquet')
    sp = pl.read_parquet(R + 'table_size_by_split.parquet')
    chk = pl.read_parquet(R + 'table_expected_vs_real.parquet')
    pmd = pl.read_parquet(R + 'table_size_per_model_dataset.parquet')
    pm = pl.read_parquet(R + 'table_size_per_model.parquet')
    cds = pl.read_parquet(R + 'cost_per_dataset.parquet')
    cm = pl.read_parquet(R + 'cost_per_model.parquet')
    fl = pl.read_parquet(R + 'model_flops.parquet')
    th = pl.read_parquet(R + 'throughput.parquet')
    fp = pl.read_parquet(R + 'throughput_fit_points.parquet')
    hrs = pl.read_parquet(R + 'gpu_hours_estimate.parquet')
    dlr = pl.read_parquet(R + 'dataloader_rate_by_dataset.parquet')
    xc = pl.read_parquet(R + 'crosscheck_nov02_reencode.parquet')
    com = pl.read_parquet(R + 'hub_commits.parquet')
    lp = pl.read_parquet(R + 'log_passes.parquet')
    hf_reg = pl.read_parquet(R + 'hub_registry_files.parquet')
    cb = pl.read_parquet(R + 'column_bytes_sample.parquet')
    foot = pl.read_parquet(R + 'hub_footers.parquet')
    NICE = {
        'cifar10': 'CIFAR-10',
        'cifar100': 'CIFAR-100',
        'tiny-imagenet': 'Tiny-ImageNet',
        'imagenet-1k': 'ImageNet-1k',
        'mnist': 'MNIST',
        'fashion_mnist': 'Fashion-MNIST',
        'celeba': 'CelebA',
        'oxford-flowers': 'Oxford Flowers',
    }

    def T(h, rows):
        return (
            '| '
            + ' | '.join(h)
            + ' |\n|'
            + '|'.join('---' for _ in h)
            + '|\n'
            + '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
            + '\n'
        )

    gb = lambda b: f'{b / 1e9:,.1f}'
    mb = lambda b: f'{b / 1e6:,.0f}'
    fm = lambda x: f'{x:,.0f}'
    EF = lambda f: f'{f / 1e18:,.1f}'
    o = []
    a = o.append
    d = {r['dataset']: r for r in ds.iter_rows(named=True)}
    tot_b = sum(r['parquet_bytes'] for r in d.values())
    N = K['N_total']
    now = dt.date.today().isoformat()
    a(
        f'# SEMASIA: size, cost and scalability of the extraction\n\nGenerated {now} by `scripts/scalability.py` (config `configs/hydra/scalability.yaml`; run all with `just scalability`). Parquet tables behind every number are in `results/scalability/`.\n'
    )
    a(
        'Labels: **exact** = read directly from the Hub / files / logs; **computed** = deterministic computation from exact inputs; **estimated** = model-based, with its assumptions stated; **NOT FOUND** = cannot be determined from what is available. Sizes use 1 GB = 10^9 bytes, 1 TB = 10^12 bytes.\n'
    )

    # ---------------- 1
    a('## 1. Size of the published data\n')
    a(
        '**How:** `HfApi.dataset_info(files_metadata=True)` on every repo `spaicom-lab/semasia-<dataset>` (sizes of all files, nothing downloaded) [`scalability.py hub` -> `hub_files.parquet`, `hub_repo_meta.parquet`]; row counts and schema from the parquet footers read through HTTP range requests, no data downloaded [`scalability.py footers` -> `hub_footers.parquet`]. Repo state at the time of the query: last modified 2026-11-02 23:20 to 2026-11-03 00:25 UTC (`hub_repo_meta.parquet`).\n'
    )
    a(
        '**Important limit on the row counts.** The Hub rate limit (1000 API calls / 5 min) stopped the full scan of the 30,578 footers: I read **%s footers** (every (dataset, split) is covered; every shard of every split has at least 40 footers read, from different models). All footers of a given (dataset, split, shard) report the same number of rows (checked, no exception), so the row counts are labelled **exact** for the files read and **computed** (by extension, same value for all models) for the others. File sizes and counts are exact for all 30,578 files.\n'
        % f'{foot.height:,}'
    )
    rows = []
    for k in ORDER:
        r = d[k]
        rows.append(
            [
                NICE[k],
                r['models'],
                r['parquet_files'],
                fm(r['N']),
                gb(r['parquet_bytes']),
                mb(r['bytes_per_model']),
            ]
        )
    rows.append(
        [
            '**Total**',
            '1,699 (1,697 on ImageNet-1k)',
            fm(ds['parquet_files'].sum()),
            fm(N),
            gb(tot_b),
            mb(tot_b / ds['models'].sum() * 1) if False else '',
        ]
    )
    a(
        '### 1.1 Per dataset (exact: models, files, bytes; N = sum of the rows of the encoded splits, exact / computed)\n'
    )
    a(
        T(
            [
                'dataset',
                'models',
                'parquet files',
                'rows (images) N_d',
                'size (GB)',
                'MB per model',
            ],
            rows,
        )
    )
    a(
        f'Grand total of the parquet files: **{tot_b / 1e12:,.2f} TB** ({tot_b / 1e9:,.0f} GB, **exact**). Adding the repo metadata files (README, .gitattributes, ~0.35-0.48 MB per repo) changes it by less than 0.001%. The Hub "used storage" figure of the repos (`used_storage`, includes older revisions kept by the Hub) is {sum(r["used_storage"] for r in d.values()) / 1e12:,.2f} TB (**exact**, `hub_repo_meta.parquet`); the difference to the file sizes is {(sum(r["used_storage"] for r in d.values()) - tot_b) / 1e9:,.0f} GB, larger on ImageNet-1k ({(d["imagenet-1k"]["used_storage"] - d["imagenet-1k"]["parquet_bytes"]) / 1e9:,.0f} GB) and Tiny-ImageNet ({(d["tiny-imagenet"]["used_storage"] - d["tiny-imagenet"]["parquet_bytes"]) / 1e9:,.0f} GB), i.e. what a clone of the *current* revision downloads is the file-size figure.\n'
    )
    a(
        'Models per dataset: all 1,699 registry models are present in 7 datasets; **ImageNet-1k has 1,697** (`vit_7b_patch16_dinov3.lvd1689m` and `vit_7b_patch16_dinov3.sat493m` are missing in both splits; **exact**, comparison of the Hub file list with the registry). These two 7B-parameter models (6.7e9 parameters = 26.9 GB in fp32) do not fit in a 24 GB card in fp32 [computed]; how they were encoded on the other 7 datasets is **NOT FOUND** (no log).\n'
    )
    a(
        '### 1.2 Per dataset and split (exact: files, models, bytes; rows exact / computed as above)\n'
    )
    rows = []
    for k in ORDER:
        for r in sp.filter(pl.col('dataset') == k).sort('split').iter_rows(named=True):
            rows.append(
                [
                    NICE[k],
                    r['split'],
                    r['models'],
                    r['parquet_files'],
                    r['shards'],
                    fm(r['rows']),
                    gb(r['bytes']),
                    f'{r["footers_read"]:,}',
                ]
            )
    a(
        T(
            [
                'dataset',
                'split',
                'models',
                'parquet files',
                'shards per model',
                'rows',
                'GB',
                'footers read',
            ],
            rows,
        )
    )
    a(
        'Splits on the Hub are exactly the encoded ones (configs in `configs/hydra/dataset/`): CIFAR-10/100, MNIST, Fashion-MNIST, Oxford Flowers: train + test; Tiny-ImageNet: train + valid; ImageNet-1k: validation + test (the `train` split of ImageNet-1k was **not** encoded); CelebA: train + valid + test (all three, 162,770 + 19,867 + 19,962 = 202,599 images). Shards: `max_rows_per_shard = 100,000` (`configs/hydra/encode_timm.yaml`), so only CelebA-train has two shards (100,000 + 62,770).\n'
    )
    a('### 1.3 Size per model\n')
    a(
        'Mean size of the embeddings of **one model** on each dataset (total / models; **computed** from exact sizes), and distribution over models of the total across the 8 datasets (what one downloads to get one model on all benchmarks):\n'
    )
    pmx = pmd.group_by('dataset').agg(
        pl.col('bytes').mean().alias('mean'),
        pl.col('bytes').median().alias('median'),
        pl.col('bytes').min().alias('min'),
        pl.col('bytes').max().alias('max'),
    )
    rows = [
        [
            NICE[k],
            mb(pmx.filter(pl.col('dataset') == k)['mean'][0]),
            mb(pmx.filter(pl.col('dataset') == k)['median'][0]),
            mb(pmx.filter(pl.col('dataset') == k)['min'][0]),
            mb(pmx.filter(pl.col('dataset') == k)['max'][0]),
        ]
        for k in ORDER
    ]
    a(T(['dataset', 'mean (MB)', 'median (MB)', 'min (MB)', 'max (MB)'], rows))
    b = pm['bytes_8_datasets']
    a(
        f'**One model across the 8 datasets** (exact per model, summary computed over the 1,699 models): mean **{b.mean() / 1e9:,.2f} GB**, median {b.median() / 1e9:,.2f} GB, min {b.min() / 1e9:,.2f} GB, max {b.max() / 1e9:,.2f} GB, 25th-75th percentile {b.quantile(0.25) / 1e9:,.2f}-{b.quantile(0.75) / 1e9:,.2f} GB.\n'
    )
    named = [
        'resnet18.a1_in1k',
        'resnet50.a1_in1k',
        'vit_base_patch16_224.augreg_in21k',
        'vit_large_patch16_224.augreg_in21k',
        'aimv2_1b_patch14_224.apple_pt',
    ]
    rows = []
    for m in named:
        x = pmd.filter(pl.col('model') == m)
        r = pm.filter(pl.col('model') == m)
        rows.append(
            [m, r['latent_dim'][0]]
            + [mb(x.filter(pl.col('dataset') == k)['bytes'][0]) for k in ORDER]
            + [f'{r["bytes_8_datasets"][0] / 1e9:,.2f}']
        )
    big = pm.sort('bytes_8_datasets', descending=True).row(0, named=True)
    small = pm.sort('bytes_8_datasets').row(0, named=True)
    for lab, r in [('largest', big), ('smallest', small)]:
        x = pmd.filter(pl.col('model') == r['model'])
        rows.append(
            [f'{r["model"]} ({lab})', r['latent_dim']]
            + [mb(x.filter(pl.col('dataset') == k)['bytes'][0]) for k in ORDER]
            + [f'{r["bytes_8_datasets"] / 1e9:,.2f}']
        )
    a('Download size of selected models (MB per dataset, GB total; **exact**):\n')
    a(T(['model', 'latent dim'] + [NICE[k] for k in ORDER] + ['total (GB)'], rows))
    rf = hf_reg.filter(pl.col('path').str.ends_with('.parquet'))
    a(
        f'**Model registry file**: `semasia_model_registry.parquet` = **{rf["size"][0]:,} bytes ({rf["size"][0] / 1e3:,.1f} kB)** on the Hub (`spaicom-lab/semasia-model-registry`, **exact**, `hub_registry_files.parquet`); the local copy `data/semasia-model-registry/semasia_model_registry.parquet` has the same size (47,063 bytes, `ls -l`). 1,699 rows x 44 columns.\n'
    )
    a('### 1.4 Check: expected raw size vs real size\n')
    a(
        'Stored format (footers): columns `id` (uint32), the labels/attributes, `model_name` (string) and `embedding` = `fixed_size_list<float>[latent_dim]`, i.e. **float32** (4 bytes/value); parquet compression **ZSTD**, written by Polars (`created_by = Polars`), all footers read agree. Expected raw size = sum over the models of latent_dim x N_d x 4 bytes (**computed**), real = sum of the file sizes (**exact**).\n'
    )
    rows = [
        [
            NICE[k],
            f'{chk.filter(pl.col("dataset") == k)["raw_bytes"][0] / 1e9:,.1f}',
            gb(chk.filter(pl.col('dataset') == k)['parquet_bytes'][0]),
            f'{chk.filter(pl.col("dataset") == k)["real_over_raw"][0]:.3f}',
        ]
        for k in ORDER
    ]
    a(
        T(
            [
                'dataset',
                'expected raw embeddings (GB)',
                'real size on the Hub (GB)',
                'real / expected',
            ],
            rows,
        )
    )
    e = cb.filter(pl.col('column').str.starts_with('embedding'))
    oc = cb.filter(~pl.col('column').str.starts_with('embedding'))
    a(
        f'The real size is **{chk["real_over_raw"].min():.2f}-{chk["real_over_raw"].max():.2f} of the raw float32 size**. Explanation, measured on the column chunks of {cb["path"].n_unique()} files (3 models x 8 datasets, `scalability.py columns` -> `column_bytes_sample.parquet`, **exact** for those files): ZSTD compresses the embedding column to {e["compressed"].sum() / e["uncompressed"].sum():.3f} of its uncompressed size ({e["compressed"].sum() / 1e9:,.2f} of {e["uncompressed"].sum() / 1e9:,.2f} GB in the sample); all other columns (id, labels, model_name, CelebA attributes) add {oc["compressed"].sum() / cb["compressed"].sum() * 100:.2f}% of the compressed bytes. The ratio is above the pure ZSTD factor because of these extra columns and the parquet overhead, and varies between datasets with the dimension mix of the available models (small `latent_dim` -> relatively more overhead from `id`/labels).\n'
    )

    # ---------------- 2
    a('## 2. Number of observations and FLOPs\n')
    a(
        '**Observations** (`N_d`, **exact** / computed from the footers as in 1.2): '
        + ', '.join(f'{NICE[k]} {fm(d[k]["N"])}' for k in ORDER)
        + f'. **N = {fm(N)} images** over the 8 benchmarks ([**computed**] sum of the encoded splits listed above; CelebA = train + valid + test = 202,599). Encoded model-image pairs (forward passes): **{fm(K["n_forward_passes"])}** [computed: sum over datasets of N_d x number of models present].\n'
    )
    a(
        '**FLOPs per forward pass.** Method [`scalability.py flops` -> `model_flops.parquet`]: for each of the 1,699 registry models, `timm.create_model(name, pretrained=False, num_classes=0)` (nothing downloaded; same InceptionNeXt `head.fc2` fix as `src/models/timm.py`), input size from `timm.data.resolve_data_config({}, model=model)["input_size"]` (the config used by `encode_dataset_all_timm.py`, so it is the size seen by the model: resize + center crop to this size), batch 1, `torch.utils.flop_counter.FlopCounterMode` in eval mode, fused attention switched off so that attention matmuls are counted. **FLOPs = 2 x MACs** (the counter counts a multiply-add as 2 FLOPs; timm tables that report "GMACs" are half of these numbers). 1,612 models were counted on the `meta` device (shapes only) and 87 on CPU (meta failed); meta and CPU were checked to give identical counts on 4 models (ResNet-50, ViT-B/16, ConvNeXt-B, Swin-B, equal to 4 decimals). **No model failed**: 1,699/1,699 counted (`error` column empty). The parameter count (encoder with `num_classes=0`) is in `parameters_encoder`; it differs from the registry `num_parameters` in 1,511 models (registry larger, i.e. it includes the classifier head); latent dimension from the forward pass equals the registry `latent_dim` for all 1,699.\n'
    )
    g = fl['gflops']
    a(
        f'Distribution of GFLOPs per forward pass over the 1,699 models (**computed**): min {g.min():.3f}, median {g.median():.2f}, mean {g.mean():.1f}, max {g.max():,.0f} (GFLOPs).\n'
    )
    cheap = fl.sort('gflops').head(5)
    dear = fl.sort('gflops', descending=True).head(5)
    a(
        T(
            ['', 'model', 'input', 'GFLOPs', 'parameters (M)'],
            [
                [
                    'cheapest',
                    r['model_name'],
                    r['input_h'],
                    f'{r["gflops"]:.3f}',
                    f'{r["parameters_encoder"] / 1e6:,.1f}',
                ]
                for r in cheap.iter_rows(named=True)
            ]
            + [
                [
                    'most expensive',
                    r['model_name'],
                    r['input_h'],
                    f'{r["gflops"]:,.0f}',
                    f'{r["parameters_encoder"] / 1e6:,.0f}',
                ]
                for r in dear.iter_rows(named=True)
            ],
        )
    )
    a(
        f'Input sizes over the registry: 224 px for {fl.filter(pl.col("input_h") == 224).height} models, 256 px {fl.filter(pl.col("input_h") == 256).height}, 384 px {fl.filter(pl.col("input_h") == 384).height}, the rest between 106 and 1024 px (`model_flops.parquet`, column `input_h`).\n'
    )
    a(
        f'**Total cost** [**computed**] C = sum over (model, dataset) of GFLOPs(model) x N_d (only the pairs present on the Hub, so the two 7B models are not counted on ImageNet-1k): **C = {K["flops_total"]:.3e} FLOPs = {EF(K["flops_total"])} EFLOPs** (1 EFLOP = 10^18 FLOPs; in MACs half of it).\n'
    )
    rows = [
        [
            NICE[k],
            fm(cds.filter(pl.col('dataset') == k)['N'][0]),
            cds.filter(pl.col('dataset') == k)['models'][0],
            EF(cds.filter(pl.col('dataset') == k)['flops'][0]),
            f'{cds.filter(pl.col("dataset") == k)["flops"][0] / K["flops_total"] * 100:.1f}%',
        ]
        for k in ORDER
    ]
    a(T(['dataset', 'N_d', 'models', 'C_d (EFLOPs)', 'share of C'], rows))
    t10 = cm.head(10)
    a(
        f'Share of the total cost due to the **10 most expensive models: {K["top10_share"] * 100:.1f}%**; due to **ImageNet-1k: {K["imagenet_share"] * 100:.1f}%** (computed). The 10 most expensive in total cost (all datasets):\n'
    )
    a(
        T(
            ['model', 'GFLOPs', 'cost (EFLOPs)', 'share'],
            [
                [
                    r['model'],
                    f'{r["gflops"]:,.0f}',
                    EF(r['flops']),
                    f'{r["share"] * 100:.2f}%',
                ]
                for r in t10.iter_rows(named=True)
            ],
        )
    )

    # ---------------- 3
    a('## 3. Estimate of the GPU hours (estimated)\n')
    a('### 3.1 Extraction settings (read from the code) [exact]\n')
    a(
        'Source: `scripts/encode_dataset_all_timm.py`, `src/models/timm.py`, `src/models/latent.py`, `configs/hydra/encode_timm.yaml`, `configs/hydra/dataset/*.yaml`.\n'
    )
    a(
        '- Model: `timm.create_model(name, pretrained=True, num_classes=0)`, `.eval()`, `requires_grad_(False)`; `LatentExtractor.predict_step` returns `model(batch).detach().cpu()` (pre-logits embedding). Run with Lightning `Trainer.predict` (inference mode, no gradients).\n- **Precision / dtype: float32**, no autocast, no `torch.compile`, no channels-last, no change to the default matmul precision (Lightning default `precision=32-true`). Nothing in the repo selects half precision.\n- Preprocessing: `resolve_data_config({}, model=model)` + `create_transform(**cfg)` (timm eval transform: bicubic/bilinear resize to `input_size / crop_pct`, center crop, normalise); images are converted with `sample.convert("RGB")` in the `collate_fn` of a PyTorch `DataLoader`, on the CPU.\n- DataLoader: `pin_memory: true`; **batch size and workers: the committed defaults are `batch_size: 16`, `num_workers: 1`, but they were overridden on the command line.** The Hydra runs kept in `outputs/` show: May 2026 ImageNet-1k runs with `batch_size=8, num_workers=1` (one run with `batch_size` 512/256/128/16 tests on 2026-05-26), and the November re-encode of InceptionNeXt with `batch_size=128, num_workers=8`, `32/2` and `64/6` (`trainer.accelerator=gpu`). The batch size and workers used for the bulk of the 1,699 models are **NOT FOUND** (no log of that run is in `outputs/`).\n- Trainer: committed default `accelerator: cpu` (overridden with `trainer.accelerator=gpu` or `auto` + `devices: [0]` in the logs); one process, one GPU per run; `parameters_threshold: 1e10` (no model skipped by it).\n- Saved: for each (model, split), parquet shards of `max_rows_per_shard = 100,000` rows with `id`, the dataset labels/attributes (`dataset.extras`), `model_name` and the float32 `embedding`.\n'
    )
    a('### 3.2 Machine and benchmark\n')
    a(
        'GPU used for **all** measurements: **NVIDIA GeForce RTX 4090, 24,564 MiB, driver 590.48.01** (`nvidia-smi`; one GPU on this machine; idle, 0% utilisation before the runs). PyTorch 2.11.0+cu130, timm 1.0.26. The original extraction used two RTX 3090 (24 GB) and two RTX 4090 (8 GB and 24 GB) [as stated by the authors; not verifiable from the repo].\n'
    )
    a(
        "Benchmark [`scalability.py throughput` -> `throughput.parquet`]: 15 models, `pretrained=False`, `num_classes=0`, eval + `torch.inference_mode`, fp32, dummy `randn` inputs of the model's own resolution already on the GPU (so: model-only, no host-to-device copy, no preprocessing), 10 warm-up batches, then 50 timed batches with `torch.cuda.synchronize()`, peak memory from `torch.cuda.max_memory_allocated`. Batch size 16 (the committed default) and 128 (used in the Nov-2026 runs). Models that ran out of memory were re-measured with 8, 4, 2, 1 (`scalability.py throughput`). **Disk**: nothing was downloaded; free space stayed at 462 GB.\n"
    )
    t16 = th.filter(pl.col('batch_size') == 16).sort('gflops')
    rows = []
    for r in t16.iter_rows(named=True):
        r128 = th.filter(
            (pl.col('model_name') == r['model_name']) & (pl.col('batch_size') == 128)
        )
        rows.append(
            [
                r['model_name'],
                r['input_h'],
                f'{r["gflops"]:.3g}',
                f'{r["imgs_per_s"]:,.0f}' if r['imgs_per_s'] else 'OOM (see below)',
                f'{r["peak_mem_gb"]:.2f}' if r['peak_mem_gb'] else '-',
                (
                    f'{r128["imgs_per_s"][0]:,.0f}'
                    if r128.height and r128['imgs_per_s'][0]
                    else 'OOM'
                ),
                (
                    f'{r128["peak_mem_gb"][0]:.2f}'
                    if r128.height and r128['peak_mem_gb'][0]
                    else '-'
                ),
            ]
        )
    a('Measured throughput (images/s, **exact** measurements of this run):\n')
    a(
        T(
            [
                'model',
                'input',
                'GFLOPs',
                'img/s, batch 16',
                'peak GB (16)',
                'img/s, batch 128',
                'peak GB (128)',
            ],
            rows,
        )
    )
    sv = th.filter(
        (pl.col('model_name') == 'samvit_huge_patch16.sa1b') & pl.col('error').is_null()
    )
    a(
        f'`samvit_huge_patch16.sa1b` (5,961 GFLOPs, 1024 px) runs out of memory at batch 16 and 128 and at 8 (explicit fp32 attention at 1024 px); it ran at batch {sv["batch_size"][0]}: {sv["imgs_per_s"][0]:.2f} img/s, {sv["peak_mem_gb"][0]:.1f} GB. `regnety_2560.seer_ft_in1k` ran out of memory at 128 (ran at 16 and 8). The models in the list are the cheapest, the one nearest to the median GFLOPs, the most expensive, ResNet-18/50 (`a1_in1k`), ViT-B/16 and ViT-L/16 `augreg_in21k`, AIMv2-1B 224, and 7 more spread on the log-GFLOPs axis. Small models (<1 GFLOP) are bound by kernel-launch latency, not by FLOPs (e.g. ResNet-18 is faster than predicted by FLOPs alone).\n'
    )
    a('### 3.3 Fit of throughput vs GFLOPs\n')
    a(
        f'**Log-log linear regression** (as requested; batch 16 points, `samvit` at batch 4; 15 points): ln(img/s) = {K["fit_a"]:.3f} {K["fit_b"]:+.3f} ln(GFLOPs), i.e. img/s = {np.exp(K["fit_a"]):,.0f} x GFLOPs^{K["fit_b"]:.3f}; **R^2 = {K["fit_r2"]:.3f}**, residual SD of ln(img/s) = {K["fit_s"]:.2f} (a factor {np.exp(K["fit_s"]):.1f}). The fit is poor at both ends: it predicts 3.9x the measured throughput for the cheapest model and 0.27x for ResNet-18 (residuals in the table). A power law is not appropriate because throughput saturates at low FLOPs (latency floor).\n'
    )
    a(
        f'**Latency + rate model** (my addition, because the log-log fit is poor): time per image = c0 + c1 x GFLOPs, fitted on the same points by least squares on ln(time): c0 = {K["aff_c0_ms"]:.3f} ms/img, c1 = {K["aff_c1_us_per_gflop"]:.1f} us/GFLOP (i.e. an effective **{K["aff_tflops"]:.1f} TFLOP/s** fp32 on the RTX 4090); **R^2 = {K["aff_r2"]:.3f}**, residual SD of ln = {K["aff_s"]:.2f} (a factor {np.exp(K["aff_s"]):.2f}). I use this model for the central estimate because it fits the whole range.\n'
    )
    a('Residuals ln(measured / fitted) of the 15 points (**computed**):\n')
    a(
        T(
            [
                'model',
                'GFLOPs',
                'img/s measured',
                'log-log fit',
                'resid ln',
                'latency+rate fit',
                'resid ln',
            ],
            [
                [
                    r['model_name'],
                    f'{r["gflops"]:.3g}',
                    f'{r["imgs_per_s"]:,.1f}',
                    f'{r["fit_imgs_per_s"]:,.1f}',
                    f'{r["resid_ln"]:+.2f}',
                    f'{r["aff_imgs_per_s"]:,.1f}',
                    f'{r["aff_resid_ln"]:+.2f}',
                ]
                for r in fp.iter_rows(named=True)
            ],
        )
    )
    a('### 3.4 Estimated GPU-hours on one RTX 4090\n')
    a(
        'Time for each (model, dataset) pair present on the Hub = N_d / throughput(model) (1,699 models, 7 datasets with 1,699 and ImageNet-1k with 1,697 models; 1.24e9 forward passes), summed. Throughput of every registry model comes from the fitted curve at its GFLOPs (not measured individually). "Low / high" = fitted throughput x exp(+-1.96 x residual SD), i.e. the range in which about 95% of individual models are expected to fall; applied to the total it is a deliberately wide band (it assumes the errors do not average out). Data-loading scenarios: throughput = min(model throughput, loader throughput), loader throughput measured per dataset at 224 px with the real `collate_fn` + timm transform on the locally cached datasets (offline, nothing downloaded; `scalability.py dataloader`) and scaled by (224/input)^2 for inputs above 224 px (checked against a 896 px measurement: 54 img/s with 1 worker vs 1,000 expected from the 224 px rate / 16, same order). Measured loader rate (img/s, 224 px output):\n'
    )
    a(
        T(
            ['dataset', '1 worker (committed default)', '8 workers'],
            [
                [
                    NICE[k],
                    f'{dlr.filter((pl.col("dataset") == k) & (pl.col("num_workers") == 1))["imgs_per_s"][0]:,.0f}'
                    if k != 'imagenet-1k'
                    else '**not measured** (cache not readable offline); Flowers rate used as proxy',
                    f'{dlr.filter((pl.col("dataset") == k) & (pl.col("num_workers") == 8))["imgs_per_s"][0]:,.0f}'
                    if k != 'imagenet-1k'
                    else 'same proxy',
                ]
                for k in ORDER
            ],
        )
    )
    a(
        'The loader rate depends on the machine (this CPU was shared and loaded by other users during the measurement), so these are indicative. ImageNet-1k (large JPEGs, 150,000 images) was not measured; its rate is assumed equal to Oxford Flowers (500x628 px JPEG), which is an assumption.\n'
    )

    def H(meth, sc, est):
        return hrs.filter(
            (pl.col('method') == meth)
            & (pl.col('scenario') == sc)
            & (pl.col('estimate') == est)
        )

    A = 'affine: latency + GFLOPs/rate'
    rows = []
    for sc in ['model only', '+ loader, 8 workers', '+ loader, 1 worker']:
        rows.append(
            [sc]
            + [
                f'{H(A, sc, e)["hours_total"][0]:,.0f}'
                for e in ['low', 'central', 'high']
            ]
            + [
                f'{H("log-log fit", sc, "central")["hours_total"][0]:,.0f}',
                f'{H("interpolation of the 15 points", sc, "central")["hours_total"][0]:,.0f}',
            ]
        )
    a('**Total GPU-hours, all models x all datasets, one RTX 4090** (**estimated**):\n')
    a(
        T(
            [
                'scenario',
                'low',
                'central',
                'high',
                'central, log-log fit',
                'central, interpolation of the 15 points',
            ],
            rows,
        )
    )
    rows = [
        [NICE[k]]
        + [
            f'{H(A, sc, "central")[k][0]:,.0f}'
            for sc in ['model only', '+ loader, 8 workers', '+ loader, 1 worker']
        ]
        for k in ORDER
    ]
    a(
        'Per dataset, central estimate (latency + rate model), GPU-hours (**estimated**):\n'
    )
    a(T(['dataset', 'model only', '+ loader, 8 workers', '+ loader, 1 worker'], rows))
    lo, ce, hi = (
        H(A, 'model only', 'low')['hours_total'][0],
        H(A, 'model only', 'central')['hours_total'][0],
        H(A, '+ loader, 1 worker', 'high')['hours_total'][0],
    )
    c1w = H(A, '+ loader, 1 worker', 'central')['hours_total'][0]
    c8 = H(A, '+ loader, 8 workers', 'central')['hours_total'][0]
    a(
        f'**Headline (estimated):** about **{ce:,.0f} GPU-hours** (model only) to **{c1w:,.0f} GPU-hours** (with a single data-loading worker), on one RTX 4090 in fp32; the loader costs {(c8 / ce - 1) * 100:.0f}% with 8 workers and {(c1w / ce - 1) * 100:.0f}% with 1. The range, taking the low end of the model-only band and the high end of the 1-worker scenario, is **{lo:,.0f}-{hi:,.0f} GPU-hours**. The three estimation methods agree within a factor 1.4 on the central value (latency + rate {ce:,.0f}, interpolation {H("interpolation of the 15 points", "model only", "central")["hours_total"][0]:,.0f}, log-log {H("log-log fit", "model only", "central")["hours_total"][0]:,.0f}). The 10 most expensive-in-time models account for {K["hours_top10_share_fit_model_only"] * 100:.0f}% of the model-only hours; the two 7B models (which do not fit in 24 GB in fp32) for {K["hours_7b_share"] * 100:.1f}% (included in the estimate by extrapolation).\n'
    )
    x = xc.filter(pl.col('measured_seconds') > 0)
    a(
        f'**Sanity check against real runs** (**exact** measured vs **estimated**): the November 2026 re-encode of the 5 InceptionNeXt models on this machine (`outputs/2026-11-02/*`, RTX 4090 per the `trainer.accelerator=gpu` logs, batch 32-128, 2-8 workers) gives {len(x)} complete (model, dataset) passes, parsed in `scalability.py traces` (a pass starts at the "Loading pretrained weights" line, ends at the next one). Measured total {x["measured_seconds"].sum() / 3600:,.2f} h; the model-only estimate for the same passes is {x["est_model_only_s"].sum() / 3600:,.2f} h and the estimate with loader {x["est_with_loader_s"].sum() / 3600:,.2f} h, so the measured time is {x["measured_seconds"].sum() / x["est_with_loader_s"].sum():.2f}x the estimate with loader (the measured pass also includes the weight download/load, writing the parquet shards and the dataset iteration, which the estimate omits). I therefore add ~20% of fixed overhead to the "with loader" figures in the paper paragraph. This check covers only small models; the large models are extrapolated from the benchmark.\n'
    )
    a(
        '**Cluster wall-clock**: not converted. The relative speed of an RTX 3090 and of the 8 GB card to the RTX 4090 was not measured here and cannot be justified from the repo; the original number of GPUs actually used simultaneously, batch sizes and per-dataset workers are NOT FOUND. Only the single-GPU (RTX 4090) number is given.\n'
    )

    # ---------------- 4
    a('## 4. Traces of the original run\n')
    cs = (
        com.group_by('dataset')
        .agg(
            pl.len().alias('n'),
            pl.col('date').min().alias('first'),
            pl.col('date').max().alias('last'),
        )
        .sort('first')
    )
    a(
        '**Hub commit history** (`HfApi.list_repo_commits`, **exact**; `hub_commits.parquet`). Each model/split is pushed as its own commit (title "Add/update parquet shards and configs"), so the commit dates are the **push dates**:\n'
    )
    a(
        T(
            ['repo', 'commits', 'first commit (UTC)', 'last commit (UTC)'],
            [
                [
                    ('semasia-' + r['dataset']),
                    f'{r["n"]:,}',
                    str(r['first'])[:16],
                    str(r['last'])[:16],
                ]
                for r in cs.iter_rows(named=True)
            ],
        )
    )
    tt = com.group_by('title').len().sort('len', descending=True).head(4)
    a(
        'Commit titles over all repos: '
        + '; '.join(f'"{r["title"]}" x {r["len"]:,}' for r in tt.iter_rows(named=True))
        + '.\n'
    )
    fi, la = (
        com.filter(pl.col('dataset') != 'model-registry')['date'].min(),
        com.filter(pl.col('dataset') != 'model-registry')['date'].max(),
    )
    a(
        f'**Period in which the data were published on the Hub: {str(fi)[:10]} (first commit, `semasia-mnist`) to {str(la)[:10]} (last commit)** ({(la - fi).days} days; **exact**). The last weeks include the re-encoding of the InceptionNeXt models (85 commits "Re-encode InceptionNeXt", 2026-11-02) and README/metadata updates (1,111 "Update README configs" commits). **Reliability:** this is when files were pushed, an upper bound for the end of each encoding, not a measure of compute time: the push times include upload, idle time, and runs were interleaved across datasets and machines. The *compute* time of the original extraction is **NOT FOUND**.\n'
    )
    g0 = lp['start'].min()
    g1 = lp['start'].max()
    a(
        f'**Local Hydra logs** (`outputs/*/*/encode_dataset_all_timm.log`, 29 runs; `log_passes.parquet`): 2026-05-20 to 2026-11-02 ({g0.date()} - {g1.date()}). They are partial: May 2026 ImageNet-1k tests (batch 8, 1 worker; e.g. one run of 2026-05-20/10-13-43 lasting about 10 h, probably including the dataset download; not verified) and the November 2026 InceptionNeXt re-encode (the only complete passes, used in 3.4). They do not cover the bulk of the extraction (the 1,699 models x 8 datasets), so total GPU-time cannot be recovered from them: **NOT FOUND**. No SLURM logs and no `multirun/` outputs of the extraction exist (`multirun/2026-05-19` holds README-generation runs, `generate_readme.log`).\n'
    )
    a(
        '**Git history of the extraction script** (`git log -- scripts/encode_dataset_all_timm.py configs/hydra/encode_timm.yaml`): first commit `c3cf92f` 2025-12-20 ("initial commit"), 24 commits touching these two files (`git log --oneline`), the last on 2026-11-03 (`b9fe458`, registry update). The oldest Hub commit (2025-12-20) coincides with the first repo commit. **Local files**: the local `data/<dataset>/{train,test,...}` folders only hold the re-encoded InceptionNeXt shards and a few models (mtimes 2026-05-20 to 2026-11-02, `local_embedding_files.parquet`); they are not the original run.\n'
    )

    # ---------------- 5
    a('## 5. Multi-object / multi-label images\n')
    a(
        'Checked in `configs/hydra/dataset/*.yaml` (columns saved) and in the parquet schemas on the Hub (`pyarrow` schema of one file per dataset):\n'
    )
    a(
        T(
            [
                'dataset',
                'labels in the published parquet',
                'multi-label / multi-object?',
            ],
            [
                ['CIFAR-10', '`label` (int64)', 'no: one class per image'],
                [
                    'CIFAR-100',
                    '`fine_label`, `coarse_label` (int64)',
                    'no: one fine class (and its coarse superclass)',
                ],
                ['MNIST / Fashion-MNIST', '`label` (int64)', 'no'],
                ['Tiny-ImageNet', '`label` (int64)', 'no (one label per image)'],
                ['Oxford Flowers', '`label` (int64)', 'no (one of 102 flower classes)'],
                [
                    'ImageNet-1k',
                    '`label` (int64)',
                    'single label per image in the dataset, although many ImageNet photos contain several objects; the content of the `test` labels was not checked here',
                ],
                [
                    'CelebA',
                    '`celeb_id` (int64) + 40 attribute columns, each `bool` (e.g. `Smiling`, `Male`, `Wearing_Hat`); 44 columns in total with `id`, `model_name`, `embedding`',
                    '**yes: multi-label** (40 binary attributes per image) plus the identity; the labels are stored as 40 separate boolean columns, not as a list',
                ],
            ],
        )
    )
    a(
        '**Where the extraction is:** `scripts/encode_dataset_all_timm.py` with the Hydra config `configs/hydra/encode_timm.yaml` and one dataset config in `configs/hydra/dataset/<name>.yaml` (fields `name` = HF dataset id, `split` = list of splits, `data` = image column, `extras` = columns copied next to the embedding). Wrappers in `justfile`: `just timm-encode <dataset>` (encode, `hf.push=false`), `just push <dataset>` (`scripts/push_to_hf.py`), `just check-hf` (`scripts/check_hf_parquet.py`).\n'
    )
    a(
        "**Commands for a new dataset** (illustrative, not run):\n\n1. Create `configs/hydra/dataset/mydata.yaml`:\n```yaml\nname: <hf_org>/<hf_dataset>\nsplit: [train, test]\ndata: image            # image column\nextras: [label]        # any columns to keep; lists/strings/booleans are copied as they are\n```\n2. Encode all models on GPU, in this repo's way (settings that were used for the November runs):\n```bash\nuv run scripts/encode_dataset_all_timm.py dataset=mydata hf.push=false trainer.accelerator=gpu trainer.devices=[0] dataloader.num_workers=8 dataloader.batch_size=128\n```\n3. A subset: `models_startwith=<prefix>` (a **single string prefix**, e.g. `models_startwith=resnet`, `vit_base`, `inception_next`); `re_encode=true` to overwrite; `hf.namespace=<user>` to publish to another account; `hf.push=true` to push (needs a Hugging Face token). Output: `data/mydata/<split>/<model>/part-XXXXX.parquet`.\n"
    )
    a(
        '**Does it support multi-label datasets or captions?** Multi-label **yes**: every column listed in `extras` is copied as is (CelebA already uses 40 boolean columns); captions or other per-image text columns would work the same way (string columns in `extras`) because they are only stored next to the embedding. **Not supported / would need code changes:** (a) several captions per image or several images per sample (the loader takes exactly one image column `data`, `collate_fn` converts one PIL image to a tensor; a list-of-strings column should work through Polars but was not tested); (b) per-object embeddings (boxes, crops): the model sees the whole resized/cropped image only; (c) a list of arbitrary models instead of a prefix (only `models_startwith` exists; a `models` list option would be a few lines in `main`); (d) the script returns early with a warning if no HF token is found, and always calls `create_repo(...)` on `<hf.namespace>/semasia-<name>` even with `hf.push=false`, so a token and the right `hf.namespace` are required even for local-only encoding; (e) `configs/hydra/check_hf.yaml` and `configs/hydra/generate_readme.yaml` list the datasets explicitly and must be extended by hand. A text encoder (for captions) would need a separate extraction script, since `timm` models are image-only.\n'
    )

    # ---------------- paragraph
    tot_tb = tot_b / 1e12
    per_model = b.mean() / 1e9
    a('## Ready-to-paste paragraph\n')
    a(
        f"> **Scalability.** SEMASIA contains the embeddings of 1,699 pretrained timm models on 8 benchmarks, {fm(N)} images per model and {K['n_forward_passes'] / 1e9:,.2f} billion model-image forward passes in total (*computed*). The embeddings occupy {tot_tb:,.1f} TB of float32 values stored as compressed parquet on the Hugging Face Hub (*exact*), i.e. {per_model:,.1f} GB per model on average over the 8 datasets (median {b.median() / 1e9:,.1f} GB, range {b.min() / 1e9:,.2f}-{b.max() / 1e9:,.1f} GB; *exact*). Per dataset, one model takes {pmx.filter(pl.col('dataset') == 'cifar10')['mean'][0] / 1e6:,.0f} MB on average on CIFAR-10 and {pmx.filter(pl.col('dataset') == 'celeba')['mean'][0] / 1e9:,.1f} GB on CelebA (*exact*). A forward pass costs between {g.min():.2f} and {g.max():,.0f} GFLOPs (median {g.median():.0f}; FLOPs counted as 2 x multiply-adds at each model's native input resolution, *computed*), so encoding all models on all benchmarks amounts to about {EF(K['flops_total'])} EFLOPs ({K['flops_total']:.2e} FLOPs, *computed*), {K['top10_share'] * 100:.0f}% of which is due to the ten most expensive models. On a single NVIDIA RTX 4090 in fp32 we *estimate* this at about {ce:,.0f} GPU-hours for the models alone and {c1w * 1.2:,.0f} GPU-hours with data-loading and I/O overheads, within a plausible range of {lo:,.0f}-{hi * 1.2:,.0f} GPU-hours (*estimated*). The estimate comes from the throughput measured on 15 models spanning the whole GFLOPs range and from a fit of throughput versus GFLOPs, and it is consistent within about 20% with the measured time of a re-encode of five models; the time of the original extraction was not logged, so it is an estimate and not a measurement. Since the embeddings are published as one parquet folder per model, split and dataset (about {pmx.filter(pl.col('dataset') == 'cifar10')['mean'][0] / 1e6:,.0f} MB per model on CIFAR-10), users can download only the models and benchmarks they need and nobody has to repeat the extraction.\n"
    )
    a(
        'Numbers in the paragraph: sizes, rows and counts are *exact* (rows by extension from the footers read, see section 1); FLOPs are *computed*; GPU-hours are *estimated* (section 3.4; the upper figure includes the +20% overhead observed in the November re-encode; the low end is the model-only estimate).\n'
    )
    (c.out / 'report.md').write_text('\n'.join(o))
    print('written', sum(len(x) for x in o))


STEP_FUNCS = {n: globals()[f'step_{n}'] for n in STEPS}


@hydra.main(
    config_path='../configs/hydra/', config_name='scalability', version_base='1.3'
)
def main(cfg: DictConfig) -> None:
    wanted = list(cfg.analyses)
    bad = [a for a in wanted if a not in STEPS]
    if bad:
        raise ValueError(f'unknown analyses {bad}; choose among {STEPS}')
    logging.getLogger('httpx').setLevel(logging.WARNING)  # one INFO line per HTTP range request otherwise
    c = Ctx(cfg)
    for name in STEPS:  # canonical order, whatever the order in the config
        if name in wanted:
            print(f'\n===== {name} ({len(c.datasets)} datasets) =====', flush=True)
            t0 = time.time()
            STEP_FUNCS[name](c)
            print(f'===== {name} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
