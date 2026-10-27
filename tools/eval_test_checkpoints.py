"""Evaluate the released checkpoints on the CARLA test split or zero-shot on nuScenes.

Each (run, epoch) is evaluated in its own subprocess (clean GPU memory, no
registry / module-level cache leakage between models) using the run's training
config ``config/<run>.py``. Only data paths, test batch size, workers and logging
are overridden for local evaluation (see ``prepare_test_cfg``).

Paths, each overridable by an environment variable:

* ``GEOPILLARS_CKPT_ROOT`` (default ``checkpoints/``): one directory per run with
  ``epoch_<E>.pth``, as downloaded from the Hugging Face release. Training logs
  (``<run>/<timestamp>/<timestamp>.log``) are read from here when present.
* ``GEOPILLARS_CARLA_ROOT`` (default ``data/Processed_data``): the CARLA dataset.
* ``GEOPILLARS_NUSC_ROOT`` (default ``data/nuscenes_zeroshot``): the zero-shot
  nuScenes set.

Per-evaluation results go to ``work_dirs/eval_<dataset>/<run>/epoch_<E>/result.json``
and are merged into one aggregate JSON after every evaluation, so a crash or
Ctrl+C never loses finished work. Finished evaluations are skipped on re-run
unless ``--force`` is given.

``--dataset nusc_zeroshot`` keeps each run's model, use_dim and batch size but
points the test dataloader and GeoPillarsMetric at the nuScenes zero-shot pkl
(4D .npy, intensity overwritten to 1.0, built by
scripts/offline_preprocessing/nuscenes_partial_trainval_infos.py and
nuscenes_to_npy_single_sweep.py --constant_intensity 1.0). Its aggregate
compares against the CARLA test re-evaluation instead of the training logs.

Usage (from the repo root)::

    PYTHONPATH=$(pwd) python tools/eval_test_checkpoints.py              # all runs, epoch 8
    PYTHONPATH=$(pwd) python tools/eval_test_checkpoints.py --runs centerpoint_h100_intensity
    PYTHONPATH=$(pwd) python tools/eval_test_checkpoints.py --epochs 4 5 6 7 8
    PYTHONPATH=$(pwd) python tools/eval_test_checkpoints.py --smoke 256 --runs geo_pillars_h100_intensity
    PYTHONPATH=$(pwd) python tools/eval_test_checkpoints.py --dataset nusc_zeroshot
"""
import argparse
import glob
import hashlib
import json
import math
import os
import os.path as osp
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
CKPT_ROOT = os.environ.get('GEOPILLARS_CKPT_ROOT', osp.join(REPO_ROOT, 'checkpoints'))
CARLA_ROOT = os.environ.get('GEOPILLARS_CARLA_ROOT', osp.join(REPO_ROOT, 'data', 'Processed_data'))
NUSC_ZEROSHOT_ROOT = os.environ.get('GEOPILLARS_NUSC_ROOT',
                                    osp.join(REPO_ROOT, 'data', 'nuscenes_zeroshot'))
RESULTS_DIR = osp.join(REPO_ROOT, 'results')
CARLA_TEST_PKL = 'nuscenes_infos_val.pkl'  # the CARLA test split (30,842 scans)

DATASETS = {
    'carla_test': dict(
        out_root=osp.join(REPO_ROOT, 'work_dirs', 'eval_carla_test'),
        aggregate=osp.join(RESULTS_DIR, 'eval_results_carla_test.json'),
        data_root=CARLA_ROOT,
        ann_file=CARLA_TEST_PKL,
        load_dim=7,
        description='CARLA test split (nuscenes_infos_val.pkl, 30,842 scans) re-evaluation of '
                    'the released checkpoints with the current GeoPillarsMetric '
                    '(mAP, mBEVIoU, mAOE, distance-stratified AP).',
    ),
    'nusc_zeroshot': dict(
        out_root=osp.join(REPO_ROOT, 'work_dirs', 'eval_nusc_zeroshot'),
        aggregate=osp.join(RESULTS_DIR, 'eval_results_nusc_zeroshot.json'),
        data_root=NUSC_ZEROSHOT_ROOT,
        ann_file='nuscenes_infos_zeroshot_npy.pkl',
        load_dim=4,
        description='Zero-shot nuScenes evaluation of the CARLA-trained released checkpoints: '
                    'keyframes of nuScenes trainval lidar blobs 01+02 (6,743 single-sweep scans, '
                    '170 scenes), car class, current GeoPillarsMetric.',
    ),
}

# Paper bookkeeping. Setups are read off the training configs; eval_batch_size
# fits a 16 GB GPU and does not change any metric.
RUNS = {
    'geo_pillars_h100_intensity': dict(
        eval_batch_size=24,
        paper_role='headline',
        setup='Full GeoPillars (NGFM): 0.4 m pillars, 8 pts, max_voxels=(16000,40000), OSF=1'),
    'geo_pillars_h100_nointensity_final': dict(
        eval_batch_size=24,
        paper_role='intensity ablation (GeoPillars)',
        setup='Full GeoPillars (NGFM), xyz only'),
    'centerpoint_h100_intensity_original': dict(
        eval_batch_size=16,
        paper_role='CenterPoint at native budget ("properly configured" baseline)',
        setup='CenterPoint: 0.2 m voxels, 20 pts, max_voxels=(30000,40000), OSF=4'),
    'centerpoint_h100_intensity': dict(
        eval_batch_size=64,
        paper_role='CenterPoint at GeoPillars budget (initial gap)',
        setup='CenterPoint: 0.4 m, 8 pts, max_voxels=(16000,40000), OSF=4'),
    'centerpoint_h100_no_intensity': dict(
        eval_batch_size=64,
        paper_role='intensity ablation (CenterPoint)',
        setup='CenterPoint: 0.4 m, 8 pts, max_voxels=(16000,40000), OSF=4, xyz only'),
    'centerpoint_osf1_ablation': dict(
        eval_batch_size=24,
        paper_role='head-resolution rung',
        setup='CenterPoint: 0.4 m, 8 pts, max_voxels=(16000,40000), OSF=1'),
    'centerpoint_geopillar_encoder_replk_distributed': dict(
        eval_batch_size=24,
        paper_role='NGFM contribution (everything but NGFM)',
        setup='CenterPoint + GeoPillar encoder + RepLK + BEVUpsampler, no NGFM, OSF=1'),
}


# helpers

def _json_safe(obj):
    """NaN/inf -> None and numpy scalars -> python, so the output is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(path, data):
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_json_safe(data), f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(
            ['git', *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _iou_backend_info():
    """Which BEV IoU path GeoPillarsMetric resolves to in this environment."""
    try:
        from src.metrics.geo_pillars_metric import (_HAS_SHAPELY, _SHAPELY_VERSION,
                                                    resolve_iou_backend)
        return dict(resolved=resolve_iou_backend('auto'),
                    rotated=resolve_iou_backend('auto') == 'shapely_rotated',
                    shapely_importable=bool(_HAS_SHAPELY),
                    shapely_version=_SHAPELY_VERSION)
    except Exception as exc:                                   # pragma: no cover
        return dict(error=f'{type(exc).__name__}: {exc}')


def find_config(run):
    path = osp.join(REPO_ROOT, 'config', f'{run}.py')
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    return path


def checkpoint_path(run, epoch):
    return osp.join(CKPT_ROOT, run, f'epoch_{epoch}.pth')


def prepare_test_cfg(cfg, work_dir, data_root=CARLA_ROOT, ann_file=CARLA_TEST_PKL,
                     batch_size=None, num_workers=8, pr_save_dir=None):
    """Turn a training config into a single-GPU test config on local data.

    These are all the local-evaluation overrides: data paths, test batch size and
    workers, launcher and a local-only visualizer. The model, the test pipeline
    and the evaluator settings stay as trained.
    """
    cfg.launcher = 'none'
    # Test only. GeoPillarsWandbHook reads runner.iter after the test epoch,
    # which would otherwise build the train loop and its CARLA train dataset.
    for key in ('train_dataloader', 'train_cfg', 'optim_wrapper', 'param_scheduler',
                'val_dataloader', 'val_cfg', 'val_evaluator'):
        cfg[key] = None
    cfg.work_dir = work_dir
    cfg.visualizer.vis_backends = [dict(type='LocalVisBackend')]
    ds = cfg.test_dataloader.dataset
    if ds.pipeline[0].type != 'LoadPointsFromNpy':
        raise ValueError(f'expected LoadPointsFromNpy first in the test pipeline, '
                         f'got {ds.pipeline[0].type}')
    ds.data_root = data_root
    ds.ann_file = ann_file
    cfg.test_evaluator.ann_file = osp.join(data_root, ann_file)
    cfg.test_evaluator.pr_save_dir = pr_save_dir
    if batch_size is not None:
        cfg.test_dataloader.batch_size = batch_size
    cfg.test_dataloader.num_workers = num_workers
    cfg.test_dataloader.persistent_workers = False
    return cfg


def _rel(path):
    """Repo-relative form of a path inside the repo, for the records."""
    path = osp.abspath(path)
    return osp.relpath(path, REPO_ROOT) if path.startswith(REPO_ROOT + os.sep) else path


def input_features(use_dim, dataset='carla_test'):
    dims = list(range(use_dim)) if isinstance(use_dim, int) else list(use_dim)
    if dims == [0, 1, 2]:
        return 'xyz'
    if dims == [0, 1, 2, 3]:
        if dataset == 'nusc_zeroshot':
            return 'xyz + intensity column (nuScenes intensity overwritten to 1.0, as in CARLA)'
        return 'xyz + intensity column (constant 1.0 in all CARLA scans)'
    return f'use_dim={dims}'


_VAL_LINE = re.compile(r'Epoch\(val\) \[(\d+)\]\[(\d+)/(\d+)\]\s+(.*)$')
_KV = re.compile(r'(GeoPillars/[^:\s]+):\s+(-?[0-9.]+|nan)')


def training_log_val_metrics(run, epoch):
    """Final val metrics for ``epoch`` from the run's training logs.

    Only ``GeoPillars/*`` keys are taken; the ``GeoPillars_*`` keys that
    GeoPillarsWandbHook echoes are from the previous evaluation.
    """
    logs = sorted(glob.glob(osp.join(CKPT_ROOT, run, '2*', '2*.log')))
    found = None
    for log in logs:
        with open(log, errors='replace') as f:
            for line in f:
                m = _VAL_LINE.search(line)
                if m and int(m.group(1)) == epoch and m.group(2) == m.group(3):
                    kv = {k: float(v) for k, v in _KV.findall(m.group(4))}
                    if kv:
                        found = dict(log=osp.relpath(log, REPO_ROOT), metrics=kv)
    return found


# child: one (run, epoch) evaluation

def child_main(args):
    os.environ.setdefault('WANDB_MODE', 'disabled')
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner

    run, epoch = args.run, args.epoch
    cfg_path = find_config(run)
    ckpt = checkpoint_path(run, epoch)
    if not osp.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    dataset = DATASETS[args.dataset]
    cfg = prepare_test_cfg(
        Config.fromfile(cfg_path), args.work_dir,
        data_root=dataset['data_root'], ann_file=dataset['ann_file'],
        batch_size=args.batch_size or RUNS[run]['eval_batch_size'],
        pr_save_dir=osp.join(args.work_dir, 'pr_curves'))
    cfg.load_from = ckpt
    ds_cfg = cfg.test_dataloader.dataset
    ds_cfg.pipeline[0].load_dim = dataset['load_dim']
    if args.smoke:
        ds_cfg.indices = args.smoke
    use_dim = ds_cfg.pipeline[0].get('use_dim')
    ckpt_meta = torch.load(ckpt, map_location='cpu', weights_only=False).get('meta', {})

    runner = Runner.from_cfg(cfg)

    from src.hooks import InferenceProfilerHook
    profiler = InferenceProfilerHook(warmup_iters=10)
    runner.register_hook(profiler)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    metrics = runner.test()
    wall = time.perf_counter() - t0

    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    n_params = sum(p.numel() for p in model.parameters())
    lat = profiler._latencies
    bs = cfg.test_dataloader.batch_size
    n_samples = len(runner.test_dataloader.dataset)

    import numpy as np
    result = dict(
        run=run,
        epoch=epoch,
        checkpoint=osp.relpath(ckpt, REPO_ROOT),
        checkpoint_sha256=_sha256(ckpt),
        checkpoint_meta_epoch=ckpt_meta.get('epoch'),
        checkpoint_meta_iter=ckpt_meta.get('iter'),
        config=osp.relpath(cfg_path, REPO_ROOT),
        smoke_subset=args.smoke or None,
        data=dict(
            dataset=args.dataset,
            data_root=_rel(ds_cfg.data_root),
            ann_file=ds_cfg.ann_file,
            evaluator_ann_file=_rel(cfg.test_evaluator.ann_file),
            num_samples_evaluated=n_samples,
            load_dim=ds_cfg.pipeline[0].load_dim,
            use_dim=use_dim,
            input_features=input_features(use_dim, args.dataset),
            test_max_voxels=cfg.model.data_preprocessor.voxel_layer.max_voxels[1],
        ),
        metric_config=dict(
            type=cfg.test_evaluator.type,
            ap_method=cfg.test_evaluator.get('ap_method', 'r40'),
            iou_thresholds=sorted(cfg.test_evaluator.get('iou_thresholds', [0.7, 0.5])),
            distance_ranges_m=cfg.test_evaluator.get(
                'distance_ranges', [[0, 20], [20, 40], [40, None]]),
            aoe_iou_threshold=min(cfg.test_evaluator.get('iou_thresholds', [0.7, 0.5])),
            note='metric values are rounded to 4 decimals by GeoPillarsMetric; '
                 'AP in [0,1], AOE in radians',
        ),
        metrics=dict(sorted(metrics.items())),
        eval_runtime=dict(
            gpu=torch.cuda.get_device_name(0),
            batch_size=bs,
            num_workers=cfg.test_dataloader.num_workers,
            precision='fp32',
            wall_time_s=round(wall, 1),
            peak_gpu_mem_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20),
            peak_gpu_mem_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20),
            params_total=n_params,
            model_forward_ms_per_batch_mean=round(float(np.mean(lat)), 2) if lat else None,
            model_forward_fps_at_batch=round(bs * 1000.0 / float(np.mean(lat)), 1) if lat else None,
            note='forward timing is data_preprocessor+model+postprocess per batch on this GPU '
                 '(first 10 iters skipped); not single-frame latency',
        ),
        pr_curves=osp.relpath(osp.join(args.work_dir, 'pr_curves', 'pr_curves.json'), REPO_ROOT),
        evaluated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
    )
    _write_json(osp.join(args.work_dir, 'result.json'), result)
    print(f'[eval] wrote {osp.join(args.work_dir, "result.json")}', flush=True)


# parent: loop + aggregate

# ``mBEVIoU`` is the pre-2026-09-20 pooled key and ``mBEVIoU@0.5`` the corrected
# single-threshold one; both are listed so an aggregate can span either metric
# version. Every consumer below reads them with ``.get``.
HEADLINE_KEYS = [
    'GeoPillars/AP@0.7', 'GeoPillars/AP@0.5', 'GeoPillars/mAP',
    'GeoPillars/mBEVIoU', 'GeoPillars/mBEVIoU@0.5',
    'GeoPillars/mAOE', 'GeoPillars/mAOE_folded', 'GeoPillars/heading_flip_rate',
    'GeoPillars/AP@0.7@0-20m', 'GeoPillars/AP@0.7@20-40m', 'GeoPillars/AP@0.7@40m+',
]


def carla_test_metrics(run, epoch):
    """Metrics from this script's own CARLA test re-evaluation, if it exists."""
    path = osp.join(DATASETS['carla_test']['out_root'], run, f'epoch_{epoch}', 'result.json')
    if not osp.isfile(path):
        return None
    with open(path) as f:
        res = json.load(f)
    if res.get('smoke_subset') or not res.get('metrics'):
        return None
    return dict(result=osp.relpath(path, REPO_ROOT), metrics=res['metrics'])


def build_aggregate(runs, epochs, out_path, dataset_name='carla_test'):
    import torch
    dataset = DATASETS[dataset_name]
    zeroshot = dataset_name == 'nusc_zeroshot'
    if zeroshot:
        notes = [
            'nuScenes intensity (0-255) is overwritten with 1.0 for every point, because every CARLA '
            'training scan has intensity exactly 1.0. xyz-only runs never read the column.',
            'GT follows the CARLA test protocol: a car counts only if num_lidar_pts >= 5 and it lies '
            'within 50 m of the sensor (bbox_3d_isvalid). Other cars are ignored as GT, so detections '
            'on them count as false positives, exactly as in the CARLA pkl where they were removed.',
            'Scans are single-sweep keyframes from trainval blobs 01+02, scenes from both the official '
            'train (5,512 scans) and val (1,231 scans) splits; no model saw nuScenes during training.',
            'carla_test holds the same run/epoch on the CARLA test split (work_dirs/eval_carla_test). '
            'drop_pp = (nuScenes - CARLA) * 100; retained_pct = nuScenes / CARLA * 100.',
        ]
    else:
        notes = [
            'CARLA intensity is exactly 1.0 for every point in every scan '
            '(collection script sets atmosphere_attenuation_rate=0). Runs with use_dim=4 '
            'saw xyz plus a constant channel, not real LiDAR intensity.',
            'train_log_val holds the epoch val numbers printed during training (pre-b800b57 metric, '
            'val batch 112 on H100). delta_pp = (re-eval - train_log) * 100.',
        ]
    agg = dict(
        description=dataset['description'],
        generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            git_commit=_git('rev-parse', 'HEAD'),
            metric_file_sha256=_sha256(osp.join(REPO_ROOT, 'src', 'metrics', 'geo_pillars_metric.py')),
            metric_file_modified_vs_HEAD=bool(_git('status', '--porcelain', 'src/metrics/geo_pillars_metric.py')),
            # which BEV IoU implementation the metric will use: every AP and AOE
            # number depends on it, and the fallback is otherwise silent
            iou_backend=_iou_backend_info(),
        ),
        notes=notes,
        runs={},
        summary=[],
    )
    for run in runs:
        entry = dict(RUNS.get(run, {}))
        entry['config'] = osp.relpath(find_config(run), REPO_ROOT)
        entry['epochs'] = {}
        for epoch in epochs:
            res_path = osp.join(dataset['out_root'], run, f'epoch_{epoch}', 'result.json')
            e = {}
            if osp.isfile(res_path):
                with open(res_path) as f:
                    e = json.load(f)
                if e.get('smoke_subset'):
                    e = {}
            if zeroshot:
                ref = carla_test_metrics(run, epoch)
                if ref:
                    e['carla_test'] = dict(
                        result=ref['result'],
                        metrics={k: ref['metrics'].get(k) for k in HEADLINE_KEYS})
                    if e.get('metrics'):
                        e['drop_pp'], e['retained_pct'] = {}, {}
                        for k in HEADLINE_KEYS:
                            new, old = e['metrics'].get(k), ref['metrics'].get(k)
                            if new is None or old is None:
                                continue
                            e['drop_pp'][k] = round((new - old) * 100, 2)
                            if old and 'AOE' not in k:
                                e['retained_pct'][k] = round(new / old * 100, 1)
            else:
                ref = training_log_val_metrics(run, epoch)
                if ref:
                    e['train_log_val'] = ref
                    if e.get('metrics'):
                        e['delta_pp'] = {
                            k: round((e['metrics'][k] - v) * 100, 2)
                            for k, v in ref['metrics'].items()
                            if e['metrics'].get(k) is not None
                        }
            if not e.get('metrics'):
                err = osp.join(dataset['out_root'], run, f'epoch_{epoch}', 'error.txt')
                e['status'] = 'failed' if osp.isfile(err) else 'pending'
            else:
                e['status'] = 'done'
                row = dict(run=run, epoch=epoch, paper_role=entry.get('paper_role'),
                           input_features=e['data']['input_features'])
                row.update({k.split('/', 1)[1]: e['metrics'].get(k) for k in HEADLINE_KEYS})
                if 'delta_pp' in e:
                    row['AP@0.7_delta_vs_train_log_pp'] = e['delta_pp'].get('GeoPillars/AP@0.7')
                if 'carla_test' in e:
                    row['carla_test_AP@0.7'] = e['carla_test']['metrics'].get('GeoPillars/AP@0.7')
                    row['AP@0.7_drop_pp'] = e.get('drop_pp', {}).get('GeoPillars/AP@0.7')
                    row['AP@0.7_retained_pct'] = e.get('retained_pct', {}).get('GeoPillars/AP@0.7')
                agg['summary'].append(row)
            entry['epochs'][str(epoch)] = e
        entry.setdefault('input_features', next(
            (v['data']['input_features'] for v in entry['epochs'].values() if v.get('data')), None))
        agg['runs'][run] = entry
    _write_json(out_path, agg)


def parent_main(args):
    runs = args.runs or list(RUNS)
    for r in runs:
        find_config(r)  # fail fast on typos

    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.setdefault('WANDB_MODE', 'disabled')

    out_root = DATASETS[args.dataset]['out_root']
    out = args.out or DATASETS[args.dataset]['aggregate']
    todo = [(r, e) for r in runs for e in args.epochs]
    for i, (run, epoch) in enumerate(todo, 1):
        sub = f'smoke_{args.smoke}' if args.smoke else f'epoch_{epoch}'
        work_dir = osp.join(out_root, run, sub)
        res = osp.join(work_dir, 'result.json')
        if osp.isfile(res) and not args.force and not args.smoke:
            print(f'[{i}/{len(todo)}] skip {run} epoch {epoch} (done)', flush=True)
            continue
        os.makedirs(work_dir, exist_ok=True)
        err_file = osp.join(work_dir, 'error.txt')
        if osp.isfile(err_file):
            os.remove(err_file)

        cmd = [sys.executable, osp.abspath(__file__), '--child', '--run', run,
               '--epoch', str(epoch), '--work-dir', work_dir, '--dataset', args.dataset]
        if args.batch_size:
            cmd += ['--batch-size', str(args.batch_size)]
        if args.smoke:
            cmd += ['--smoke', str(args.smoke)]
        print(f'[{i}/{len(todo)}] eval {run} epoch {epoch} -> {work_dir}', flush=True)
        t0 = time.time()
        with open(osp.join(work_dir, 'stdout.log'), 'w') as log:
            rc = subprocess.call(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if rc != 0:
            with open(err_file, 'w') as f:
                f.write(f'exit code {rc}; see stdout.log\n')
            print(f'    FAILED (rc={rc}, {dt/60:.1f} min), see {work_dir}/stdout.log', flush=True)
        else:
            print(f'    done in {dt/60:.1f} min', flush=True)
        if not args.smoke:
            build_aggregate(runs if args.runs else list(RUNS), args.epochs, out, args.dataset)

    if not args.smoke:
        build_aggregate(runs if args.runs else list(RUNS), args.epochs, out, args.dataset)
        print(f'aggregate -> {out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', choices=list(RUNS), help='default: all runs')
    p.add_argument('--epochs', nargs='+', type=int, default=[8])
    p.add_argument('--dataset', choices=list(DATASETS), default='carla_test')
    p.add_argument('--out', default=None,
                   help='aggregate JSON path (default: per dataset, in results/)')
    p.add_argument('--batch-size', type=int, default=None,
                   help="test batch size (default: the run's eval_batch_size, sized for 16 GB)")
    p.add_argument('--force', action='store_true', help='re-evaluate finished runs')
    p.add_argument('--smoke', type=int, default=0,
                   help='evaluate only the first N samples into a smoke_N dir (numbers meaningless)')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--run', help=argparse.SUPPRESS)
    p.add_argument('--epoch', type=int, help=argparse.SUPPRESS)
    p.add_argument('--work-dir', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        child_main(args)
    else:
        parent_main(args)


if __name__ == '__main__':
    main()
