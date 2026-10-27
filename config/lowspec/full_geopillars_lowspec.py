"""
Full GeoPillars (NGFM), run A - lowspec smoke test.

The trained config ``config/geo_pillars_h100_intensity.py`` on ``data/Processed_data_mini``:
batch 2, 2 epochs, fp16, reduced voxel cap, local logging only. Fits a ~4 GB GPU.
"""

_base_ = [
    "../geo_pillars_h100_intensity.py",
]

# Switch to mini local dataset
_mini_root = "data/Processed_data_mini"

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    pin_memory=False,
    persistent_workers=False,
    dataset=dict(
        data_root=_mini_root,
        ann_file="nuscenes_infos_train.pkl",
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    pin_memory=False,
    persistent_workers=False,
    dataset=dict(
        data_root=_mini_root,
        ann_file="nuscenes_infos_val.pkl",
    ),
)

test_dataloader = val_dataloader

# Reduced voxel budget
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(max_voxels=(4000, 8000)),
    ),
)

# Short training loop
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=2, val_interval=1, val_begin=0)
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=8e-6, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type="OneCycleLR",
        eta_max=2e-4,
        total_steps=200,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
        three_phase=False,
        by_epoch=False,
    ),
]
auto_scale_lr = dict(enable=False, base_batch_size=2)

# No WandB, local logs only
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=5),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="Det3DVisualizationHook"),
)

# sm_86 cuDNN workaround
import os

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# Evaluator / work dir
val_evaluator = dict(
    ann_file=_mini_root + "/nuscenes_infos_val.pkl",
    pr_save_dir="work_dirs/lowspec/full_geopillars/pr_curves",
)
test_evaluator = val_evaluator
work_dir = "work_dirs/lowspec/full_geopillars"
