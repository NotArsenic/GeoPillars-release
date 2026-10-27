"""Quantify the train/test point-truncation mismatch created by ``PointShuffle``.

Hard voxelisation keeps, for each pillar, the points whose *arrival index* is
among the first ``max_num_points`` of that pillar - see
``mmcv/ops/csrc/common/cuda/voxelization_cuda_kernel.cuh:95-136``
(``point_to_voxelidx_kernel``): a point's in-voxel slot is the number of
earlier points sharing its voxel, and it is dropped when that count reaches
``max_points``. There is no sampling.

``PointShuffle`` is the last transform of every run's train pipeline and appears
in no run's test pipeline, so a saturated pillar is filled with a random 8-point
subset during training and with the first 8 points of the raw scan order at
inference. This script measures how different those two retained sets are, and
verifies the kernel's arrival-order semantics against the real voxel layer.

Writes ``figures/data/truncation_mismatch.json`` and, alongside it, ``.npz`` with the
per-comparison arrays (retained-set overlap, signed delta-z and delta-range).
"""

import argparse
import json
import os.path as osp
import pickle
import sys

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from _carla_infer import CARLA_ROOT, CARLA_TEST_PKL  # noqa: E402

PCR = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
VOXEL = np.array([0.4, 0.4, 8.0])
MAX_PTS = 8


def pillar_ids(xyz):
    """Voxel index per point, and the in-range mask, matching mmcv's binning."""
    keep = np.all((xyz >= PCR[:3]) & (xyz < PCR[3:]), axis=1)
    g = np.floor((xyz[keep] - PCR[:3]) / VOXEL).astype(np.int64)
    nx, ny = np.ceil((PCR[3:] - PCR[:3]) / VOXEL).astype(np.int64)[:2]
    return g[:, 0] + nx * g[:, 1] + nx * ny * g[:, 2], keep


def first_n_per_group(ids, order, n):
    """Indices (into ``ids``) kept when points are consumed in ``order``.

    Reproduces the kernel: walking the points in ``order``, a point survives
    iff fewer than ``n`` earlier points share its group.
    """
    ids_o = ids[order]
    # rank of each element within its group, in `order`
    srt = np.argsort(ids_o, kind="stable")
    grp = ids_o[srt]
    new_grp = np.r_[True, grp[1:] != grp[:-1]]
    start = np.flatnonzero(new_grp)
    rank = np.arange(len(grp)) - np.repeat(start, np.diff(np.r_[start, len(grp)]))
    keep_srt = rank < n
    return order[srt[keep_srt]], grp[keep_srt]


def verify_kernel_order(npy_path, device="cuda"):
    """Check the real voxel layer keeps the first ``MAX_PTS`` in arrival order."""
    import torch
    from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape

    pts = np.load(npy_path).astype(np.float32)[:, :4]
    layer = VoxelizationByGridShape(
        max_num_points=MAX_PTS,
        voxel_size=list(VOXEL),
        point_cloud_range=list(PCR),
        max_voxels=(16000, 40000),
        deterministic=True,
    )
    t = torch.from_numpy(pts).to(device)
    voxels, coors, num_points = layer(t)
    voxels = voxels.cpu().numpy()
    coors = coors.cpu().numpy()
    num_points = num_points.cpu().numpy()

    ids, keep = pillar_ids(pts[:, :3])
    pin = pts[keep]
    nx, ny = np.ceil((PCR[3:] - PCR[:3]) / VOXEL).astype(np.int64)[:2]
    order = np.arange(len(ids))
    checked = mismatched = slot_order_ok = 0
    for m in range(len(coors)):
        z, y, x = int(coors[m, 0]), int(coors[m, 1]), int(coors[m, 2])
        vid = x + nx * y + nx * ny * z
        mine = np.flatnonzero(ids == vid)  # already in arrival order
        expect = pin[mine[:MAX_PTS]]
        got = voxels[m, : num_points[m]]
        checked += 1
        if expect.shape != got.shape or not np.allclose(expect, got, atol=1e-5):
            mismatched += 1
        else:
            slot_order_ok += 1
    del order
    return dict(
        scan=osp.basename(npy_path),
        pillars_checked=checked,
        pillars_matching_first_n_in_arrival_order=slot_order_ok,
        pillars_mismatching=mismatched,
        conclusion=(
            "hard voxelisation keeps the first max_num_points "
            "points of each pillar in input order, slot by slot"
            if mismatched == 0
            else "MISMATCH - see counts"
        ),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-scans", type=int, default=200)
    ap.add_argument(
        "--n-draws", type=int, default=5, help="independent PointShuffle draws per scan"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=osp.join(REPO_ROOT, "figures", "data", "truncation_mismatch.json"),
    )
    ap.add_argument("--no-kernel-check", action="store_true")
    args = ap.parse_args()

    with open(osp.join(CARLA_ROOT, CARLA_TEST_PKL), "rb") as f:
        data_list = pickle.load(f)["data_list"]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(data_list), size=args.n_scans, replace=False)
    pick.sort()

    kernel = None
    if not args.no_kernel_check:
        p0 = osp.join(CARLA_ROOT, data_list[pick[0]]["lidar_points"]["lidar_path"])
        try:
            kernel = verify_kernel_order(p0)
            print("[trunc] kernel check:", kernel["conclusion"], flush=True)
        except Exception as e:  # pragma: no cover
            kernel = dict(error=f"{type(e).__name__}: {e}")
            print("[trunc] kernel check failed:", kernel["error"], flush=True)

    # accumulators over saturated pillars
    n_pillars = n_sat = 0
    n_points_total = n_in_range = 0
    diff_any = 0  # retained sets differ at all
    overlap = []  # |A ∩ B| out of 8
    dz_abs, drange_abs, dz_signed, drange_signed = [], [], [], []
    z_noshuf, z_shuf, r_noshuf, r_shuf = [], [], [], []
    sat_counts = []
    per_scan = []

    for si, di in enumerate(pick):
        path = osp.join(CARLA_ROOT, data_list[di]["lidar_points"]["lidar_path"])
        pts = np.load(path).astype(np.float64)
        xyz = pts[:, :3]
        ids, keep = pillar_ids(xyz)
        xin = xyz[keep]
        rng_pt = np.linalg.norm(xin, axis=1)
        n_points_total += len(xyz)
        n_in_range += len(xin)

        uid, counts = np.unique(ids, return_counts=True)
        n_pillars += len(uid)
        sat = uid[counts > MAX_PTS]
        n_sat += len(sat)
        sat_counts.append(counts[counts > MAX_PTS])
        if len(sat) == 0:
            continue
        sat_set = {int(v): k for k, v in enumerate(sat)}

        # unshuffled: file order
        base_idx, base_grp = first_n_per_group(ids, np.arange(len(ids)), MAX_PTS)
        base = {}
        for i, g in zip(base_idx, base_grp):
            if int(g) in sat_set:
                base.setdefault(int(g), []).append(i)

        for d in range(args.n_draws):
            perm = rng.permutation(len(ids))
            s_idx, s_grp = first_n_per_group(ids, perm, MAX_PTS)
            shuf = {}
            for i, g in zip(s_idx, s_grp):
                if int(g) in sat_set:
                    shuf.setdefault(int(g), []).append(i)
            for g, bi in base.items():
                sj = shuf.get(g)
                if sj is None:
                    continue
                a, b = np.asarray(bi), np.asarray(sj)
                inter = len(np.intersect1d(a, b))
                overlap.append(inter)
                diff_any += int(inter != MAX_PTS)
                za, zb = xin[a, 2].mean(), xin[b, 2].mean()
                ra, rb = rng_pt[a].mean(), rng_pt[b].mean()
                dz_abs.append(abs(za - zb))
                dz_signed.append(za - zb)
                drange_abs.append(abs(ra - rb))
                drange_signed.append(ra - rb)
                z_noshuf.append(za)
                z_shuf.append(zb)
                r_noshuf.append(ra)
                r_shuf.append(rb)
        per_scan.append(
            dict(
                sample_idx=int(data_list[di]["sample_idx"]),
                n_points=int(len(xyz)),
                n_in_range=int(len(xin)),
                n_pillars=int(len(uid)),
                n_saturated=int(len(sat)),
            )
        )

    sat_counts = np.concatenate(sat_counts) if sat_counts else np.zeros(0)
    n_cmp = len(overlap)
    ov = np.asarray(overlap, float)
    # analytic expectation: B is a uniform 8-subset of the pillar's n points
    exp_ov = (
        float(np.mean(MAX_PTS * MAX_PTS / sat_counts))
        if len(sat_counts)
        else float("nan")
    )

    out = dict(
        description=(
            "Train/test point-truncation mismatch. PointShuffle is the last "
            "transform of every run train pipeline and is absent from every test "
            "pipeline; hard voxelisation keeps the first max_num_points points of "
            "each pillar in arrival order, so training sees a random 8-point "
            "subset of each saturated pillar and inference sees the first 8 in raw "
            "scan order."
        ),
        source=dict(
            truncation_kernel="modifiedLibs/mmcv-2.1.0/mmcv/ops/csrc/common/cuda/"
            "voxelization_cuda_kernel.cuh:95-136 "
            "(point_to_voxelidx_kernel)",
            deterministic_default="mmcv Voxelization(deterministic=True); no run "
            "config overrides it",
            data_root=CARLA_ROOT,
            ann_file=CARLA_TEST_PKL,
        ),
        settings=dict(
            n_scans=args.n_scans,
            n_shuffle_draws=args.n_draws,
            seed=args.seed,
            max_num_points=MAX_PTS,
            voxel_size=VOXEL.tolist(),
            point_cloud_range=PCR.tolist(),
        ),
        kernel_check=kernel,
        pillar_counts=dict(
            points_total=int(n_points_total),
            points_in_range=int(n_in_range),
            pillars_total=int(n_pillars),
            pillars_saturated=int(n_sat),
            fraction_pillars_saturated=round(n_sat / max(n_pillars, 1), 6),
            points_in_saturated_pillars=int(sat_counts.sum()),
            fraction_in_range_points_discarded=round(
                float((sat_counts - MAX_PTS).sum() / max(n_in_range, 1)), 6
            ),
            saturated_pillar_point_count=dict(
                mean=round(float(sat_counts.mean()), 3) if len(sat_counts) else None,
                median=float(np.median(sat_counts)) if len(sat_counts) else None,
                p90=float(np.percentile(sat_counts, 90)) if len(sat_counts) else None,
                max=int(sat_counts.max()) if len(sat_counts) else None,
            ),
        ),
        retained_set_difference=dict(
            n_comparisons=n_cmp,
            fraction_differing=round(diff_any / max(n_cmp, 1), 6),
            overlap_out_of_8_mean=round(float(ov.mean()), 4) if n_cmp else None,
            overlap_out_of_8_median=float(np.median(ov)) if n_cmp else None,
            overlap_expected_if_uniform_subset=round(exp_ov, 4),
            fraction_identical=(
                round(float((ov == MAX_PTS).mean()), 6) if n_cmp else None
            ),
        ),
        retained_set_geometry=dict(
            note="per saturated pillar, the mean over the 8 retained points; "
            "noshuffle = first 8 in file order, shuffle = 8 after PointShuffle",
            abs_delta_z_m=dict(
                mean=round(float(np.mean(dz_abs)), 4),
                median=round(float(np.median(dz_abs)), 4),
                p90=round(float(np.percentile(dz_abs, 90)), 4),
                max=round(float(np.max(dz_abs)), 4),
            ),
            abs_delta_range_m=dict(
                mean=round(float(np.mean(drange_abs)), 4),
                median=round(float(np.median(drange_abs)), 4),
                p90=round(float(np.percentile(drange_abs, 90)), 4),
                max=round(float(np.max(drange_abs)), 4),
            ),
            signed_delta_z_m_mean=round(float(np.mean(dz_signed)), 5),
            signed_delta_range_m_mean=round(float(np.mean(drange_signed)), 5),
            mean_z_noshuffle=round(float(np.mean(z_noshuf)), 4),
            mean_z_shuffle=round(float(np.mean(z_shuf)), 4),
            mean_range_noshuffle=round(float(np.mean(r_noshuf)), 4),
            mean_range_shuffle=round(float(np.mean(r_shuf)), 4),
        ),
        per_scan=per_scan[:20],
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # per-comparison arrays, for plotting the Delta-z distribution
    npz = osp.splitext(args.out)[0] + ".npz"
    np.savez_compressed(
        npz,
        overlap_out_of_8=np.asarray(overlap, dtype=np.int8),
        delta_z_m=np.asarray(dz_signed, dtype=np.float32),
        delta_range_m=np.asarray(drange_signed, dtype=np.float32),
        mean_z_noshuffle=np.asarray(z_noshuf, dtype=np.float32),
        mean_z_shuffle=np.asarray(z_shuf, dtype=np.float32),
        saturated_pillar_point_count=sat_counts.astype(np.int32),
        note=np.array(
            "one entry per (saturated pillar x shuffle draw); "
            "delta = noshuffle - shuffle of the mean over the 8 "
            "retained points. saturated_pillar_point_count is one "
            "entry per saturated pillar."
        ),
    )
    print(f"[trunc] -> {npz}")
    print(
        json.dumps(
            {
                k: v
                for k, v in out.items()
                if k
                in ("pillar_counts", "retained_set_difference", "retained_set_geometry")
            },
            indent=2,
        )
    )
    print(f"[trunc] -> {args.out}")


if __name__ == "__main__":
    main()
