#!/usr/bin/env bash
# Install PyTorch, the patched OpenMMLab stack and the GeoPillars requirements into
# the active Python environment (create it from environment.yml first).
#
#   bash modifiedLibs/setup_env.sh                 # install everything
#   bash modifiedLibs/setup_env.sh --sources-only  # only prepare the patched sources
#
# mmengine 0.10.7, mmcv 2.1.0, mmdet 3.3.0 and mmdet3d 1.4.0 each come from one of:
#   * a complete source tree in modifiedLibs/<lib>/, installed editable;
#   * otherwise the PyPI source release (checksum-verified), with the patched files
#     kept in modifiedLibs/<lib>/ copied over it before it is built.
#
# Environment overrides:
#   TORCH_INDEX_URL       PyTorch wheel index (default: CUDA 13.0 wheels)
#   TORCH_PKGS            PyTorch packages (default: torch 2.9.0, torchvision 0.24.0, torchaudio 2.9.0)
#   CUDA_HOME             CUDA toolkit used to compile the ops (default: /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST  GPU architectures to compile for, e.g. "12.0" for RTX 50xx
#   MAX_JOBS              parallel compile jobs (default: nproc - 2)
#   BUILD_DIR             where PyPI sources are unpacked (default: modifiedLibs/.build)
#
# RTX 50xx (sm_120) with a CUDA 12.8 toolkit, the setup of the recorded evaluations:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#   TORCH_PKGS="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1" \
#   TORCH_CUDA_ARCH_LIST=12.0 bash modifiedLibs/setup_env.sh
set -euo pipefail
shopt -s inherit_errexit

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TORCH_PKGS="${TORCH_PKGS:-torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
BUILD_DIR="${BUILD_DIR:-$HERE/.build}"

# source_tree <vendored dir> <PyPI name> <version> <package dir>
# Prints the directory to install from.
source_tree() {
    local dir="$HERE/$1" name="$2" version="$3" pkg="$4"
    if [[ -f "$dir/$pkg/__init__.py" ]]; then
        echo "$dir"
        return
    fi
    local dest="$BUILD_DIR/$name-$version"
    if [[ ! -d "$dest" ]]; then
        mkdir -p "$BUILD_DIR"
        python - "$name" "$version" "$BUILD_DIR" <<'PY'
import hashlib, json, sys, tarfile, urllib.request

name, version, out = sys.argv[1:]
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json"))
sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
data = urllib.request.urlopen(sdist["url"]).read()
if hashlib.sha256(data).hexdigest() != sdist["digests"]["sha256"]:
    sys.exit(f"{sdist['filename']}: checksum mismatch")
path = f"{out}/{sdist['filename']}"
with open(path, "wb") as f:
    f.write(data)
with tarfile.open(path) as tar:
    tar.extractall(out)
PY
    fi
    # copy the patched files over the release
    (cd "$dir" && find . -type f -print0) | while IFS= read -r -d '' f; do
        install -D -m 644 "$dir/$f" "$dest/$f"
    done
    echo "$dest"
}

MMENGINE="$(source_tree mmengine-0.10.7 mmengine 0.10.7 mmengine)"
MMCV="$(source_tree mmcv-2.1.0 mmcv 2.1.0 mmcv)"
MMDET="$(source_tree mmdetection-3.3.0 mmdet 3.3.0 mmdet)"
MMDET3D="$(source_tree mmdetection3d-1.4.0 mmdet3d 1.4.0 mmdet3d)"
printf 'sources:\n  %s\n  %s\n  %s\n  %s\n' "$MMENGINE" "$MMCV" "$MMDET" "$MMDET3D"
if [[ "${1:-}" == "--sources-only" ]]; then
    exit 0
fi

ulimit -n 65536 2>/dev/null || true

pip install $TORCH_PKGS --index-url "$TORCH_INDEX_URL"
pip install -r "$REPO/requirements.txt"

(cd "$MMENGINE" && pip install -v -e .)
(cd "$MMCV" && MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v -e . --no-build-isolation)
(cd "$MMDET" && pip install -v -e . --no-build-isolation)
(cd "$MMDET3D" && FORCE_CUDA=1 pip install -v -e . --no-build-isolation)

python - <<'PY'
import torch, mmcv, mmdet, mmdet3d, mmengine
from mmcv.ops import Voxelization  # noqa: F401  (compiled ops are importable)
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
print("mmengine", mmengine.__version__, "| mmcv", mmcv.__version__,
      "| mmdet", mmdet.__version__, "| mmdet3d", mmdet3d.__version__)
PY
