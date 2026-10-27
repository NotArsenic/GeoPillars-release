#!/usr/bin/env python3
"""Regenerate every paper figure from `figures/data/` with dataset access blocked."""

import glob
import os
import os.path as osp
import runpy
import sys

HERE = osp.dirname(osp.abspath(__file__))
FORBIDDEN = (
    "/Processed_data",
    "nuscenes_zeroshot",
    "v1.0-trainval",
    "LIDAR_TOP",
)


def _guard(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if any(f in path for f in FORBIDDEN):
            raise RuntimeError(f"figure tried to read dataset file: {path}")


def main():
    sys.addaudithook(_guard)
    wanted = sys.argv[1:]
    scripts = sorted(
        glob.glob(osp.join(HERE, "fig[0-9]*.py")),
        key=lambda p: int(osp.basename(p)[3:].split("_")[0]),
    )
    for path in scripts:
        name = osp.basename(path)
        if wanted and not any(name.startswith(w) for w in wanted):
            continue
        print(f"== {name}", flush=True)
        saved = sys.argv
        sys.argv = [path]
        try:
            runpy.run_path(path, run_name="__main__")
        finally:
            sys.argv = saved
            import matplotlib.pyplot as plt

            plt.close("all")
    # the guard itself must work
    try:
        open(osp.join(HERE, "..", "data", "Processed_data", "nuscenes_infos_val.pkl"), "rb")
    except RuntimeError:
        print("dataset guard active: OK")
    else:
        raise SystemExit("dataset guard did not block a dataset read")


if __name__ == "__main__":
    main()
