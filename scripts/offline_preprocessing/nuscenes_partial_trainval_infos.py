#!/usr/bin/env python3
"""
nuscenes_partial_trainval_infos.py

Builds an mmdet3d v2-format (info_version 1.1) info pkl for a *partial*
nuScenes trainval download - e.g. only ``v1.0-trainval01_blobs_lidar`` and
``v1.0-trainval02_blobs_lidar`` plus ``v1.0-trainval_meta``. Used as a
zero-shot test set for the CARLA-trained checkpoints.

mmdet3d's ``create_nuscenes_infos`` can't do this: it calls
``check_file_exist`` on every trainval sample before filtering by scene, so it
aborts on the first scan that isn't downloaded.

Only keyframes whose LIDAR_TOP ``.bin`` exists in one of the blob folders are
kept (train and val scenes alike - nothing here was trained on). Boxes use the
same math as ``tools/dataset_converters/nuscenes_converter.py``: lidar frame,
gravity centre, ``[x, y, z, l, w, h, yaw]``.

GT protocol matches the CARLA test pkl built by ``offline_processor.py``
(>= 5 lidar points; every CARLA car lies within 50 m of the sensor).
``bbox_3d_isvalid`` is ``num_lidar_pts >= --min_lidar_pts`` and
``hypot(x, y) <= --max_dist``; ``GeoPillarsMetric`` ignores invalid GT.
Instances outside the 10 detection classes are dropped, as in CARLA.

``lidar_path`` is relative to ``--data_root``. Feed the output to
``nuscenes_to_npy_single_sweep.py`` to get 4D ``.npy`` points.

Usage:
  python scripts/offline_preprocessing/nuscenes_partial_trainval_infos.py \\
      --data_root data/nuscenes_zeroshot \\
      --out_name nuscenes_infos_zeroshot.pkl
"""

import argparse
import glob
import os
import pickle
from collections import Counter

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from tqdm import tqdm

from mmdet3d.datasets.convert_utils import NuScenesNameMapping

CLASS_NAMES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
CATEGORIES = {name: i for i, name in enumerate(CLASS_NAMES)}


def _se3(rotation, translation):
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', required=True,
                        help='folder holding the *_blobs_lidar dirs and the meta dir')
    parser.add_argument('--meta_dir', default=None,
                        help='dir containing <version>/*.json '
                             '(default: <data_root>/v1.0-trainval_meta)')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--lidar_glob', default='*_blobs_lidar/samples/LIDAR_TOP',
                        help='glob (relative to data_root) for LIDAR_TOP keyframe dirs')
    parser.add_argument('--out_name', default='nuscenes_infos_zeroshot.pkl')
    parser.add_argument('--min_lidar_pts', type=int, default=5)
    parser.add_argument('--max_dist', type=float, default=50.0)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    meta_dir = args.meta_dir or os.path.join(data_root, 'v1.0-trainval_meta')

    # basename -> path relative to data_root
    available = {}
    lidar_dirs = sorted(glob.glob(os.path.join(data_root, args.lidar_glob)))
    for d in lidar_dirs:
        n = 0
        for f in os.listdir(d):
            if f.endswith('.pcd.bin'):
                if f in available:
                    raise RuntimeError(f'{f} found in two blob dirs')
                available[f] = os.path.relpath(os.path.join(d, f), data_root)
                n += 1
        print(f'{os.path.relpath(d, data_root)}: {n} LIDAR_TOP keyframes')
    if not available:
        raise FileNotFoundError(f'no .pcd.bin under {data_root}/{args.lidar_glob}')

    nusc = NuScenes(version=args.version, dataroot=meta_dir, verbose=True)
    scene_split = {name: 'train' for name in splits.train}
    scene_split.update({name: 'val' for name in splits.val})

    samples = []
    for sample in nusc.sample:
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        if os.path.basename(sd['filename']) in available:
            samples.append(sample)
    # deterministic order: scene, then time
    samples.sort(key=lambda s: (nusc.get('scene', s['scene_token'])['name'], s['timestamp']))
    print(f'{len(samples)} of {len(nusc.sample)} {args.version} samples have lidar on disk '
          f'({len(available) - len(samples)} lidar files without a sample)')

    data_list = []
    stats = Counter()
    for sample in tqdm(samples, desc='infos'):
        lidar_token = sample['data']['LIDAR_TOP']
        sd = nusc.get('sample_data', lidar_token)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        scene = nusc.get('scene', sample['scene_token'])
        l2e_r_mat = Quaternion(cs['rotation']).rotation_matrix
        e2g_r_mat = Quaternion(pose['rotation']).rotation_matrix

        # boxes come back in the lidar frame, in sample['anns'] order
        _, boxes, _ = nusc.get_sample_data(lidar_token)
        annotations = [nusc.get('sample_annotation', t) for t in sample['anns']]
        assert len(boxes) == len(annotations)

        instances = []
        for box, anno in zip(boxes, annotations):
            name = NuScenesNameMapping.get(box.name)
            if name not in CATEGORIES:
                stats['dropped_other_category'] += 1
                continue
            x, y, z = box.center
            w, l, h = box.wlh
            yaw = box.orientation.yaw_pitch_roll[0]
            velo = np.array([*nusc.box_velocity(anno['token'])[:2], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velo = np.nan_to_num(velo[:2]).tolist()

            n_pts = int(anno['num_lidar_pts'])
            valid = n_pts >= args.min_lidar_pts and float(np.hypot(x, y)) <= args.max_dist
            label = CATEGORIES[name]
            instances.append(dict(
                bbox_label=label,
                bbox_3d=[float(x), float(y), float(z), float(l), float(w), float(h), float(yaw)],
                bbox_3d_isvalid=valid,
                bbox_label_3d=label,
                num_lidar_pts=n_pts,
                num_radar_pts=int(anno['num_radar_pts']),
                velocity=velo,
            ))
            if name == 'car':
                stats['car_total'] += 1
                stats['car_valid'] += valid
                stats['car_lt_min_pts'] += n_pts < args.min_lidar_pts
                stats['car_beyond_max_dist'] += float(np.hypot(x, y)) > args.max_dist

        data_list.append(dict(
            sample_idx=len(data_list),
            token=sample['token'],
            timestamp=sample['timestamp'] / 1e6,
            ego2global=_se3(pose['rotation'], pose['translation']).tolist(),
            images={},
            lidar_points=dict(
                num_pts_feats=5,
                lidar_path=available[os.path.basename(sd['filename'])],
                lidar2ego=_se3(cs['rotation'], cs['translation']).tolist(),
            ),
            lidar_sweeps=[],
            instances=instances,
            cam_instances={},
            scene_token=sample['scene_token'],
            scene_name=scene['name'],
            nusc_split=scene_split.get(scene['name'], 'unknown'),
        ))
        stats['samples_' + scene_split.get(scene['name'], 'unknown')] += 1

    out = dict(
        metainfo=dict(
            categories=CATEGORIES,
            dataset='nuscenes',
            version=args.version,
            info_version='1.1',
            source='partial trainval: ' + ', '.join(
                os.path.relpath(d, data_root) for d in lidar_dirs),
            num_scenes=len({s['scene_token'] for s in samples}),
            gt_valid_rule=f'num_lidar_pts >= {args.min_lidar_pts} and '
                          f'hypot(x, y) <= {args.max_dist} m (lidar frame)',
        ),
        data_list=data_list,
    )
    out_pkl = os.path.join(data_root, args.out_name)
    with open(out_pkl, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'scenes: {out["metainfo"]["num_scenes"]}')
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')
    print(f'Saved {len(data_list)} samples to {out_pkl}')


if __name__ == '__main__':
    main()
