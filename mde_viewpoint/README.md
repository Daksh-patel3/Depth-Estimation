# MDE Viewpoint Benchmark

Benchmarking monocular depth estimation (MDE) foundation models on diverse
camera viewpoints. The project covers zero-shot evaluation, LoRA fine-tuning
for viewpoint adaptation, and forgetting analysis — each using a dedicated
dataset.

| Task | Dataset |
|------|---------|
| Zero-shot benchmark & LoRA fine-tuning | **Hypersim** |
| Forgetting evaluation | **NYU Depth V2** |

The four models evaluated:

| short name | model | source | depth type |
|------------|-------|--------|------------|
| `dav2` | Depth Anything V2 (Large) | HF `depth-anything/Depth-Anything-V2-Large-hf` | relative |
| `zoedepth` | ZoeDepth (NK head) | HF `Intel/zoedepth-nyu-kitti` | metric |
| `midas` | MiDaS DPT_Large | torch.hub `intel-isl/MiDaS` | relative |
| `marigold` | Marigold (LCM) | diffusers `prs-eth/marigold-depth-lcm-v1-0` | affine-invariant |

## Viewpoint categories

Frames are bucketed by camera pitch angle extracted from pose metadata:

| Category | Pitch | Meaning |
|----------|-------|---------|
| A | 0–15° | Eye-level / horizontal |
| B | 30–45° | Elevated oblique |
| C | 55–65° | Steep oblique |
| D | 75–90° | Near top-down |
| E | any | Non-standard environment (fog, low light) |

## Project layout

```
mde_viewpoint/
├── data/        dataset_builder, dataloader, splits/
├── models/      model_zoo, lora_wrapper
├── eval/        metrics, evaluator
├── analysis/    error_heatmap, attention_maps, depth_histogram
├── train/       lora_finetune
├── experiments/ run_zero_shot_eval, run_failure_analysis,
│                run_lora_adaptation, run_forgetting_eval
├── notebooks/   results_visualization.ipynb
├── configs/     config.yaml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r mde_viewpoint/requirements.txt
```

CUDA is recommended. HuggingFace / torch.hub weights are pulled on first run.

## Datasets

### Hypersim — zero-shot & fine-tuning

Photorealistic indoor synthetic dataset from Apple. Download 2+ scenes:

```
datasets/hypersim/
└── <scene>/           # e.g. ai_001_001
    ├── images/
    │   ├── scene_cam_<k>_final_preview/     frame.<i>.color.jpg
    │   └── scene_cam_<k>_geometry_hdf5/     frame.<i>.depth_meters.hdf5
    └── _detail/
        └── <cam>/     camera_keyframe_orientations.hdf5
```

Official download: <https://github.com/apple/ml-hypersim>

### NYU Depth V2 — forgetting evaluation

Used only in the forgetting eval step to measure whether LoRA fine-tuning on
Hypersim degrades performance on standard indoor benchmarks.

```
datasets/nyu_depth_v2/
├── nyu_test_rgb/      <id>.png   (uint8 RGB)
└── nyu_test_depth/    <id>.png   (uint16, depth_meters × 1000)
```

Download labeled .mat and extract pairs:
```bash
bash download_real_subsets.sh nyu
```

## End-to-end usage

All scripts read paths and hyperparameters from `mde_viewpoint/configs/config.yaml`.

### 1. Build viewpoint-bucketed splits (Hypersim)

```bash
python -m mde_viewpoint.data.dataset_builder \
    --config mde_viewpoint/configs/config.yaml \
    --hypersim_root datasets/hypersim \
    --output_dir mde_viewpoint/data/splits
```

Writes `splits/category_{A,B,C,D,E}.json` plus `summary.json`. Each entry
contains `image_path`, `depth_path`, `pitch_deg`, and `category`.

### 2. Zero-shot benchmark (Hypersim, 4 models × 5 categories)

```bash
python -m mde_viewpoint.experiments.run_zero_shot_eval \
    --config mde_viewpoint/configs/config.yaml
```

Writes per-run JSON, a wide CSV table, and a LaTeX table to `results/zero_shot/`.

### 3. Failure-mode analysis

```bash
python -m mde_viewpoint.experiments.run_failure_analysis \
    --config mde_viewpoint/configs/config.yaml \
    --model dav2 --category D --n_images 50
```

Writes error heatmaps, attention overlays, and depth histograms to
`results/failure_analysis/<model>/<category>/`.

### 4. LoRA fine-tuning on Hypersim (category D, rank 8)

```bash
python -m mde_viewpoint.experiments.run_lora_adaptation \
    --config mde_viewpoint/configs/config.yaml \
    --category D --rank 8
```

Fine-tunes Depth Anything V2 with LoRA on N ∈ {50, 100, 200, 500} Hypersim
samples and plots adaptation curves under `results/lora/category_D_rank8/`.
The best adapter is saved to `best_adapter/`.

### 5. Forgetting evaluation (NYU Depth V2)

```bash
python -m mde_viewpoint.experiments.run_forgetting_eval \
    --config mde_viewpoint/configs/config.yaml \
    --adapter_path results/lora/category_D_rank8/N_500/best_adapter \
    --rank 8
```

Compares base DAV2 vs. LoRA-adapted DAV2 on NYU Depth V2 and reports
AbsRel / RMSE / δ₁ deltas in `results/forgetting/forgetting.json`.

### 6. Visualize results

```bash
jupyter notebook mde_viewpoint/notebooks/results_visualization.ipynb
```

## Implementation notes

- **Pitch extraction.** Hypersim stores `R_c2w` (camera-to-world) matrices with
  Y-up world frame. Pitch = `arcsin(|R[1,2]|)` — the angle between the camera
  look direction (-Z in camera space) and the horizontal plane.
- **Median scaling.** Relative-depth models (DAV2, MiDaS, Marigold) are
  median-scaled per image before metric evaluation; ZoeDepth is metric directly.
- **Loss.** LoRA training uses the scale-invariant log loss (Eigen et al. 2014).
- **LoRA injection.** Adapters target the `query` and `value` projections of
  every DINOv2 attention block (rank 8, alpha 16) via PEFT.
