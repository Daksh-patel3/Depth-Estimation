"""Download additional datasets: KITTI (Kaggle), more Hypersim, TartanAir.

Usage (on Unity, with conda env activated):
    python download_extra_datasets.py [kitti|hypersim|tartanair|all]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("/scratch3/workspace/dakshsanjayk_umass_edu-wenlong_vwn/mde_viewpoint/datasets")


def ensure_package(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg])


# =========================================================================
# KITTI via kagglehub
# =========================================================================
def download_kitti():
    print("=== KITTI (Kaggle — Eigen split with GT depth) ===")
    ensure_package("kagglehub")
    import kagglehub
    import numpy as np
    from PIL import Image

    kt_root = DATA_ROOT / "kitti"
    out_rgb = kt_root / "kitti_eval_rgb"
    out_dep = kt_root / "kitti_eval_depth"

    if out_rgb.exists() and len(list(out_rgb.glob("*.png"))) > 100:
        print("[KITTI] already extracted — skipping.")
        return

    # Clean up old bad download if present
    old_cache = Path.home() / ".cache/kagglehub/datasets/klemenko"
    if old_cache.exists():
        print("[KITTI] cleaning up old klemenko cache...")
        subprocess.run(["rm", "-rf", str(old_cache)])

    # Use Eigen split dataset — has paired RGB + GT depth (~1.4 GB)
    print("[KITTI] downloading Eigen split (awsaf49/kitti-eigen-split-dataset)...")
    try:
        raw_path = Path(kagglehub.dataset_download("awsaf49/kitti-eigen-split-dataset"))
    except Exception:
        # Fallback to depth prediction eval dataset
        print("[KITTI] Eigen split failed, trying artemmmtry/kitti-depth-prediction-evaluation...")
        raw_path = Path(kagglehub.dataset_download("artemmmtry/kitti-depth-prediction-evaluation"))

    print(f"[KITTI] downloaded to {raw_path}")

    out_rgb.mkdir(parents=True, exist_ok=True)
    out_dep.mkdir(parents=True, exist_ok=True)

    # Find all depth PNG files then match to RGB by filename/path structure
    depth_files = sorted(raw_path.rglob("*depth*.png")) + sorted(raw_path.rglob("*groundtruth*/*.png"))
    rgb_candidates = sorted(raw_path.rglob("*image*/*.png")) + sorted(raw_path.rglob("*rgb*/*.png"))

    # Build filename → path index for RGB matching
    rgb_index = {}
    for f in rgb_candidates:
        if "depth" not in str(f).lower() and "groundtruth" not in str(f).lower():
            rgb_index[f.name] = f

    n = 0
    # Strategy 1: match depth to RGB by shared filename
    for dep_file in depth_files:
        if n >= 700:
            break
        name = dep_file.name
        rgb_match = rgb_index.get(name)
        if rgb_match is None:
            continue
        arr = np.array(Image.open(dep_file))
        if arr.max() == 0:
            continue
        Image.open(rgb_match).convert("RGB").save(out_rgb / f"{n:04d}.png")
        if arr.dtype == np.uint16:
            Image.fromarray(arr).save(out_dep / f"{n:04d}.png")
        else:
            Image.fromarray((arr.astype(np.float32) * 256).astype(np.uint16)).save(
                out_dep / f"{n:04d}.png"
            )
        n += 1

    if n == 0:
        # Strategy 2: assume parallel directory structure (image/ and depth/ or gt/)
        print("[KITTI] trying parallel directory scan...")
        for dep_file in sorted(raw_path.rglob("*.png"))[:5000]:
            if n >= 700:
                break
            dep_str = str(dep_file)
            if "depth" not in dep_str.lower() and "gt" not in dep_str.lower():
                continue
            # Try swapping 'depth'/'gt' with 'image'/'rgb' in path
            for old, new in [("depth", "image"), ("gt", "image"), ("groundtruth", "image"),
                             ("depth", "rgb"), ("gt", "rgb")]:
                rgb_try = Path(dep_str.replace(old, new, 1))
                if rgb_try.exists() and rgb_try != dep_file:
                    arr = np.array(Image.open(dep_file))
                    if arr.max() > 0:
                        Image.open(rgb_try).convert("RGB").save(out_rgb / f"{n:04d}.png")
                        if arr.dtype == np.uint16:
                            Image.fromarray(arr).save(out_dep / f"{n:04d}.png")
                        else:
                            Image.fromarray((arr.astype(np.float32) * 256).astype(np.uint16)).save(
                                out_dep / f"{n:04d}.png"
                            )
                        n += 1
                    break

    print(f"[KITTI] wrote {n} RGB+depth pairs to {kt_root}")
    _print_size(kt_root)


# =========================================================================
# More Hypersim scenes (~3-4 additional scenes, ~4GB)
# =========================================================================
def download_hypersim():
    print("=== Hypersim (additional scenes) ===")
    hs_root = DATA_ROOT / "hypersim"
    hs_root.mkdir(parents=True, exist_ok=True)

    base_url = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1"

    # Pick 4 more scenes with varied camera angles, each ~0.5-1.5 GB
    scenes = ["ai_003_001", "ai_004_001", "ai_005_001", "ai_008_001"]

    for scene in scenes:
        out = hs_root / scene
        if out.exists() and (out / "_detail").exists() and (out / "images").exists():
            print(f"[Hypersim] {scene} already present — skipping.")
            continue

        url = f"{base_url}/scenes/{scene}.zip"
        zip_path = hs_root / f"{scene}.zip"
        print(f"[Hypersim] downloading {scene}...")
        ret = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(zip_path), url],
            capture_output=False,
        )
        if ret.returncode != 0:
            print(f"  FAILED {url}")
            zip_path.unlink(missing_ok=True)
            continue
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(hs_root)])
        zip_path.unlink(missing_ok=True)

    _print_size(hs_root)


# =========================================================================
# TartanAir via official tartanair Python package
# =========================================================================
def download_tartanair():
    print("=== TartanAir ===")
    ta_root = DATA_ROOT / "tartanair"
    ta_root.mkdir(parents=True, exist_ok=True)

    # Try the official tartanair pip package first
    ensure_package("tartanair")
    try:
        import tartanair as ta
        ta.init(str(ta_root))

        envs = ta.get_available_envs()
        print(f"[TartanAir] available environments: {envs[:10]}...")

        # Download 2 diverse environments (small trajectory counts)
        targets = []
        preferred = ["AbandonedCableExposure", "OldScandinaviaExposure",
                      "AbandonedSchoolExposure", "DesertGasStationExposure"]
        for env in preferred:
            if env in envs:
                targets.append(env)
            if len(targets) >= 2:
                break
        if not targets and envs:
            targets = envs[:2]

        for env in targets:
            print(f"[TartanAir] downloading {env} (image_lcam_front + depth_lcam_front)...")
            try:
                ta.download(
                    env=env,
                    difficulty=["easy"],
                    trajectory_id=["P000"],
                    modality=["image", "depth"],
                    camera_name=["lcam_front"],
                )
            except Exception as e:
                print(f"  tartanair SDK failed for {env}: {e}")
                continue

        _print_size(ta_root)
        return
    except Exception as e:
        print(f"[TartanAir] SDK method failed: {e}")

    # Fallback: try TartanAir V1 with azcopy or direct wget
    print("[TartanAir] trying direct download (V1 URLs)...")
    ensure_package("azure.storage.blob", "azure-storage-blob")
    try:
        from azure.storage.blob import ContainerClient
        container_url = "https://tartanair.blob.core.windows.net/tartanair-release1"
        container = ContainerClient.from_container_url(container_url)

        scenes = ["abandonedfactory", "hospital"]
        difficulty = "Easy"
        traj = "P000"

        for scene in scenes:
            scene_dir = ta_root / scene / difficulty / traj
            if scene_dir.exists() and list(scene_dir.glob("*")):
                print(f"[TartanAir] {scene} already present — skipping.")
                continue

            scene_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{scene}/{difficulty}/{traj}/"
            print(f"[TartanAir] listing blobs under {prefix}...")

            blobs = list(container.list_blobs(name_starts_with=prefix))
            if not blobs:
                print(f"  no blobs found for {scene}")
                continue

            for blob in blobs:
                name = blob.name
                local = ta_root / name
                if local.exists():
                    continue
                local.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith(".zip"):
                    print(f"  downloading {name}...")
                    blob_client = container.get_blob_client(name)
                    with open(local, "wb") as f:
                        stream = blob_client.download_blob()
                        stream.readinto(f)
                    subprocess.run(["unzip", "-q", str(local), "-d", str(local.parent)])
                    local.unlink(missing_ok=True)
                elif name.endswith(".txt"):
                    blob_client = container.get_blob_client(name)
                    with open(local, "wb") as f:
                        stream = blob_client.download_blob()
                        stream.readinto(f)

        _print_size(ta_root)
    except Exception as e:
        print(f"[TartanAir] Azure download also failed: {e}")
        print("  Manual download: https://theairlab.org/tartanair-dataset/")


def _print_size(path):
    result = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
    print(f"  Size: {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="all",
                        choices=["kitti", "hypersim", "tartanair", "all"])
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if args.which in ("kitti", "all"):
        download_kitti()
    if args.which in ("hypersim", "all"):
        download_hypersim()
    if args.which in ("tartanair", "all"):
        download_tartanair()

    print("\n=== All downloads complete ===")
    for d in sorted(DATA_ROOT.iterdir()):
        result = subprocess.run(["du", "-sh", str(d)], capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")

    proj = "/project/pi_dagarwal_umass_edu/project_19/daksh3/Depth-Estimation"
    print(f"\nSymlink into project (if not already done):")
    print(f"  ln -sfn {DATA_ROOT}/kitti     {proj}/datasets/kitti")
    print(f"  ln -sfn {DATA_ROOT}/tartanair {proj}/datasets/tartanair")


if __name__ == "__main__":
    main()
