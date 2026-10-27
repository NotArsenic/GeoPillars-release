"""
Standalone nuScenes data preparation script.
Generates nuscenes_infos_train.pkl and nuscenes_infos_val.pkl
without needing to run from the mmdet3d source tree.

Usage:
    python tools/prepare_nuscenes.py \
        --root-path data/v1.0-mini \
        --out-dir data/v1.0-mini \
        --version v1.0-mini \
        --max-sweeps 9
"""
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare nuScenes info files')
    parser.add_argument('--root-path', required=True,
                        help='Root path of the nuScenes dataset (contains v1.0-mini/ etc.)')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for .pkl files')
    parser.add_argument('--version', default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'])
    parser.add_argument('--max-sweeps', type=int, default=9)
    parser.add_argument('--extra-tag', default='nuscenes',
                        help='Prefix for output pkl filenames')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Import the library function directly - works regardless of CWD
    from tools.dataset_converters.nuscenes_converter import create_nuscenes_infos

    print(f'Generating nuScenes info files')
    print(f'  root-path : {args.root_path}')
    print(f'  out-dir   : {args.out_dir}')
    print(f'  version   : {args.version}')
    print(f'  max-sweeps: {args.max_sweeps}')
    print(f'  prefix    : {args.extra_tag}')
    print()

    create_nuscenes_infos(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        version=args.version,
        max_sweeps=args.max_sweeps,
    )

    # The function writes to root_path by default - move if out_dir differs
    import shutil
    for split in ('train', 'val', 'test'):
        src = os.path.join(args.root_path, f'{args.extra_tag}_infos_{split}.pkl')
        dst = os.path.join(args.out_dir, f'{args.extra_tag}_infos_{split}.pkl')
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)
            print(f'Moved {src} -> {dst}')
        elif os.path.exists(dst):
            print(f'Generated: {dst}')

    print('\nDone.')


if __name__ == '__main__':
    main()
