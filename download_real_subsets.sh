#!/bin/bash -l
# ---------------------------------------------------------------------------
# Download ~2GB subsets of TartanAir, Hypersim, NYU Depth V2, KITTI.
# Run on a Unity login node AFTER activating the conda env.
#
# Usage:
#   bash download_real_subsets.sh [tartanair|hypersim|nyu|kitti|all]
# ---------------------------------------------------------------------------
set -euo pipefail

WHICH=${1:-all}

DATA_ROOT="/scratch3/workspace/dakshsanjayk_umass_edu-wenlong_vwn/mde_viewpoint/datasets"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

# ===========================================================================
# 1. TartanAir — 2 trajectories from scenes with varied camera pitch
# ===========================================================================
download_tartanair() {
    echo "=== TartanAir ==="
    local TA_ROOT="$DATA_ROOT/tartanair"
    mkdir -p "$TA_ROOT"
    local BASE="https://tartanair.blob.core.windows.net/tartanair-release1"

    # Pick scenes likely to have varied pitch:
    #   - oldtown: street-level walking (mostly eye-level → category A)
    #   - westerndesert: drone flyovers (varied pitch → B/C/D)
    local SCENES=("oldtown" "westerndesert")
    local DIFFICULTY="Easy"
    local TRAJ="P000"

    for SCENE in "${SCENES[@]}"; do
        echo "[TartanAir] downloading $SCENE/$DIFFICULTY/$TRAJ ..."
        local OUT="$TA_ROOT/$SCENE/$DIFFICULTY/$TRAJ"
        mkdir -p "$OUT"

        # Three artifacts per trajectory: rgb zip, depth zip, pose txt
        for KIND in image_left depth_left; do
            local ZIP_URL="$BASE/$SCENE/$DIFFICULTY/$TRAJ/${KIND}.zip"
            local ZIP_PATH="$OUT/${KIND}.zip"
            if [ ! -f "$ZIP_PATH" ] && [ ! -d "$OUT/$KIND" ]; then
                wget -q --show-progress -O "$ZIP_PATH" "$ZIP_URL" || \
                    { echo "  FAILED $ZIP_URL"; rm -f "$ZIP_PATH"; continue; }
                unzip -q "$ZIP_PATH" -d "$OUT" && rm -f "$ZIP_PATH"
            fi
        done

        # Pose file
        local POSE_URL="$BASE/$SCENE/$DIFFICULTY/$TRAJ/pose_left.txt"
        local POSE_PATH="$OUT/pose_left.txt"
        [ -f "$POSE_PATH" ] || wget -q -O "$POSE_PATH" "$POSE_URL" || \
            { echo "  FAILED $POSE_URL"; rm -f "$POSE_PATH"; }

        # Convert depth .npy is already TartanAir's native format — depths come as
        # PNGs in some releases. Detect and convert if needed.
        if compgen -G "$OUT/depth_left/*.png" >/dev/null; then
            echo "  converting depth PNGs → .npy ..."
            python -c "
import os, glob
import numpy as np
from PIL import Image
for p in glob.glob('$OUT/depth_left/*.png'):
    arr = np.array(Image.open(p), dtype=np.float32) / 100.0  # TA convention
    np.save(p.replace('.png', '.npy'), arr)
" || true
        fi
    done
    echo "[TartanAir] done. Size: $(du -sh "$TA_ROOT" | cut -f1)"
}

# ===========================================================================
# 2. Hypersim — 2 scenes via apple/ml-hypersim helper
# ===========================================================================
download_hypersim() {
    echo "=== Hypersim ==="
    local HS_ROOT="$DATA_ROOT/hypersim"
    mkdir -p "$HS_ROOT"

    # Pick 2 scenes (each ~1GB). ai_001_001 / ai_001_002 are stable choices.
    local SCENES=("ai_001_001" "ai_001_002")
    local BASE="https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1"

    for SCENE in "${SCENES[@]}"; do
        local OUT="$HS_ROOT/$SCENE"
        mkdir -p "$OUT"
        if [ -d "$OUT/_detail" ] && [ -d "$OUT/images" ]; then
            echo "[Hypersim] $SCENE already present — skipping."
            continue
        fi
        echo "[Hypersim] downloading $SCENE ..."
        local URL="$BASE/scenes/$SCENE.zip"
        local ZIP_PATH="$HS_ROOT/${SCENE}.zip"
        wget -q --show-progress -O "$ZIP_PATH" "$URL" || \
            { echo "  FAILED $URL — try the official downloader"; rm -f "$ZIP_PATH"; continue; }
        unzip -q "$ZIP_PATH" -d "$HS_ROOT"
        rm -f "$ZIP_PATH"
    done
    echo "[Hypersim] done. Size: $(du -sh "$HS_ROOT" | cut -f1)"
}

# ===========================================================================
# 3. NYU Depth V2 — labeled .mat file → split into PNG pairs
# ===========================================================================
download_nyu() {
    echo "=== NYU Depth V2 ==="
    local NYU_ROOT="$DATA_ROOT/nyu_depth_v2"
    mkdir -p "$NYU_ROOT"
    local MAT_PATH="$NYU_ROOT/nyu_depth_v2_labeled.mat"

    if [ ! -f "$MAT_PATH" ] && [ ! -d "$NYU_ROOT/nyu_test_rgb" ]; then
        echo "[NYU] downloading 2.8GB labeled .mat ..."
        wget -q --show-progress -O "$MAT_PATH" \
            "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    fi

    if [ ! -d "$NYU_ROOT/nyu_test_rgb" ]; then
        echo "[NYU] extracting RGB / depth pairs ..."
        python -c "
import os, h5py, numpy as np
from PIL import Image

mat_path = '$MAT_PATH'
out_rgb  = '$NYU_ROOT/nyu_test_rgb'
out_dep  = '$NYU_ROOT/nyu_test_depth'
os.makedirs(out_rgb, exist_ok=True)
os.makedirs(out_dep, exist_ok=True)

with h5py.File(mat_path, 'r') as f:
    images = f['images']     # (N, 3, W, H), float
    depths = f['depths']     # (N, W, H), meters
    n = images.shape[0]
    for i in range(n):
        rgb = np.array(images[i], dtype=np.uint8).transpose(2, 1, 0)
        d   = np.array(depths[i], dtype=np.float32).T
        Image.fromarray(rgb).save(f'{out_rgb}/{i:04d}.png')
        # store as 16-bit, depth_meters * 1000 (NYU convention)
        Image.fromarray((np.clip(d, 0, 65) * 1000).astype('uint16')).save(f'{out_dep}/{i:04d}.png')
print(f'wrote {n} pairs')
"
        # Free disk: drop the .mat after extraction
        rm -f "$MAT_PATH"
    fi
    echo "[NYU] done. Size: $(du -sh "$NYU_ROOT" | cut -f1)"
}

# ===========================================================================
# 4. KITTI Eigen test — HuggingFace mirror
# ===========================================================================
download_kitti() {
    echo "=== KITTI ==="
    local KT_ROOT="$DATA_ROOT/kitti"
    mkdir -p "$KT_ROOT"

    if [ -d "$KT_ROOT/kitti_eval_rgb" ]; then
        echo "[KITTI] already present — skipping."
        return
    fi

    echo "[KITTI] downloading Eigen test split via HuggingFace ..."
    python -c "
import os, sys
from pathlib import Path
try:
    from datasets import load_dataset
except ImportError:
    os.system('pip install -q datasets')
    from datasets import load_dataset
from PIL import Image
import numpy as np

ds = load_dataset('Voxel51/KITTI-Multiview', split='test[:700]')
out_rgb = Path('$KT_ROOT/kitti_eval_rgb')
out_dep = Path('$KT_ROOT/kitti_eval_depth')
out_rgb.mkdir(parents=True, exist_ok=True)
out_dep.mkdir(parents=True, exist_ok=True)

n = 0
for i, sample in enumerate(ds):
    img = sample.get('image') or sample.get('rgb') or sample.get('left')
    dep = sample.get('depth') or sample.get('depth_map') or sample.get('lidar_depth')
    if img is None or dep is None:
        continue
    img.save(out_rgb / f'{i:04d}.png')
    if hasattr(dep, 'save'):
        dep.save(out_dep / f'{i:04d}.png')
    else:
        # depth as numpy → store as 16-bit *256 (KITTI convention)
        arr = np.asarray(dep, dtype=np.float32)
        Image.fromarray((np.clip(arr, 0, 250) * 256).astype('uint16')).save(out_dep / f'{i:04d}.png')
    n += 1
print(f'wrote {n} pairs')
" || echo "[KITTI] HF download failed — register at http://www.cvlibs.net/datasets/kitti/eval_depth.php and download manually"

    echo "[KITTI] done. Size: $(du -sh "$KT_ROOT" 2>/dev/null | cut -f1)"
}

# ===========================================================================
# Driver
# ===========================================================================
case "$WHICH" in
    tartanair) download_tartanair ;;
    hypersim)  download_hypersim ;;
    nyu)       download_nyu ;;
    kitti)     download_kitti ;;
    all)
        download_nyu
        download_kitti
        download_tartanair
        download_hypersim
        ;;
    *)
        echo "usage: $0 [tartanair|hypersim|nyu|kitti|all]"
        exit 1
        ;;
esac

echo ""
echo "All downloads complete. Total size:"
du -sh "$DATA_ROOT"/* 2>/dev/null
echo ""
echo "Now symlink into the project root:"
echo "  cd /project/pi_dagarwal_umass_edu/project_19/daksh3/Depth-Estimation"
echo "  ln -sfn $DATA_ROOT/tartanair    datasets/tartanair"
echo "  ln -sfn $DATA_ROOT/hypersim     datasets/hypersim"
echo "  ln -sfn $DATA_ROOT/nyu_depth_v2 datasets/nyu_depth_v2"
echo "  ln -sfn $DATA_ROOT/kitti        datasets/kitti"
