"""Shared helpers: build a released checkpoint + its CARLA test dataloader on a subset."""

import os
import os.path as osp
import pickle
import sys

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "tools"))

from eval_test_checkpoints import CARLA_ROOT, CARLA_TEST_PKL  # noqa: E402,F401

# Run letters follow the paper's results table.
RUNS = {
    "A": "geo_pillars_h100_intensity",
    "B": "geo_pillars_h100_nointensity_final",
    "C": "centerpoint_h100_intensity_original",
    "D": "centerpoint_h100_intensity",
    "E": "centerpoint_h100_no_intensity",
    "F": "centerpoint_osf1_ablation",
    "G": "centerpoint_geopillar_encoder_replk_distributed",
}


def build(run, indices=None, batch_size=None, num_workers=None, epoch=8):
    """Return (runner, model, dataloader, cfg) for `run` on the CARLA test split."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from eval_test_checkpoints import RUNS as EVAL_RUNS
    from eval_test_checkpoints import checkpoint_path, find_config, prepare_test_cfg

    cfg = prepare_test_cfg(
        Config.fromfile(find_config(run)),
        osp.join(REPO_ROOT, "work_dirs", "figure_data", run),
        batch_size=batch_size or EVAL_RUNS[run]["eval_batch_size"],
        num_workers=8 if num_workers is None else num_workers,
    )
    cfg.visualizer = dict(
        type="Visualizer", vis_backends=[dict(type="LocalVisBackend")]
    )
    if indices is not None:
        cfg.test_dataloader.dataset.indices = [int(i) for i in indices]

    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, checkpoint_path(run, epoch), map_location="cpu")
    model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)
    return runner, model, loader, cfg


def path_to_sample_idx(pkl=osp.join(CARLA_ROOT, CARLA_TEST_PKL)):
    """Det3DDataset renumbers sample_idx inside a subset; map outputs back by lidar file."""
    with open(pkl, "rb") as f:
        return {
            i["lidar_points"]["lidar_path"]: i["sample_idx"]
            for i in pickle.load(f)["data_list"]
        }
