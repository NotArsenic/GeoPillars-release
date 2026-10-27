#!/usr/bin/env bash
# eval.sh - evaluate one checkpoint on the CARLA test split.
#
# Usage:
#   ./eval.sh <config_file> <checkpoint.pth> [options] [extra test.py args...]
#
# Options:
#   --data-root <path>   Override data_root for the test dataloader and evaluator
#   --batch-size <n>     Override the test batch size. The configs keep the trained
#                        value (112), which needs a large GPU; use 8-24 on 16 GB.
#
# Example (released checkpoint, CARLA dataset under data/Processed_data):
#   ./eval.sh config/geo_pillars_h100_intensity.py \
#       checkpoints/geo_pillars_h100_intensity/epoch_8.pth \
#       --data-root data/Processed_data --batch-size 16
#
# To evaluate every run and write the aggregate JSON, use
# tools/eval_test_checkpoints.py instead.

set -euo pipefail

CONFIG="${1:-}"
CHECKPOINT="${2:-}"

if [[ -z "$CONFIG" || -z "$CHECKPOINT" ]]; then
    echo "Usage: $0 <config_file> <checkpoint.pth> [--data-root <path>] [--batch-size <n>] [extra args...]"
    echo ""
    echo "Available configs:"
    find config -name "*.py" ! -path "*/_base_/*" ! -path "*/__pycache__/*" | sort | sed 's/^/  /'
    exit 1
fi

shift 2  # consume config + checkpoint; parse remaining flags

DATA_ROOT=""
BATCH_SIZE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Build --cfg-options overrides
CFG_OVERRIDES=()
if [[ -n "$DATA_ROOT" ]]; then
    CFG_OVERRIDES+=(
        "test_dataloader.dataset.data_root=${DATA_ROOT}"
        "test_evaluator.ann_file=${DATA_ROOT}/nuscenes_infos_val.pkl"
        "val_dataloader.dataset.data_root=${DATA_ROOT}"
        "val_evaluator.ann_file=${DATA_ROOT}/nuscenes_infos_val.pkl"
    )
fi
if [[ -n "$BATCH_SIZE" ]]; then
    CFG_OVERRIDES+=(
        "test_dataloader.batch_size=${BATCH_SIZE}"
        "test_dataloader.num_workers=0"
        "test_dataloader.persistent_workers=False"
    )
fi

if [[ ${#CFG_OVERRIDES[@]} -gt 0 ]]; then
    EXTRA_ARGS+=("--cfg-options" "${CFG_OVERRIDES[@]}")
fi

PYTHONPATH="$(pwd)" python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
