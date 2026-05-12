# MDE Viewpoint Benchmark

Benchmarking 4 monocular depth-estimation foundation models on non-standard
camera viewpoints, analyzing failure modes, applying LoRA fine-tuning for
viewpoint adaptation, and measuring forgetting on standard benchmarks
(NYU Depth V2, KITTI).

The full project description (steps, viewpoint categories A–E, hyperparameters)
is encoded in `configs/config.yaml`. The four models are:

| short name | model                              | source                                                    | depth type        |
| ---------- | ---------------------------------- | --------------------------------------------------------- | ----------------- |
| `dav2`     | Depth Anything V2 (Large)          | HF `depth-anything/Depth-Anything-V2-Large-hf`            | relative          |
| `zoedepth` | ZoeDepth (NK head)                 | torch.hub `isl-org/ZoeDepth` (`ZoeD_NK`)                  | metric            |
| `midas`    | MiDaS DPT_Large                    | torch.hub `intel-isl/MiDaS` (`DPT_Large`)                 | relative          |
| `marigold` | Marigold (LCM)                     | diffusers `prs-eth/marigold-depth-lcm-v1-0`               | affine-invariant  |

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
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r mde_viewpoint/requirements.txt
```

CUDA is recommended; CPU also works (much slower). The first run will pull
HF / torch.hub weights into the local cache.

## Datasets

This project expects the following on-disk roots (override paths via
`configs/config.yaml`):

| dataset      | download                                                                 | layout used by `dataset_builder.py`                                |
| ------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| TartanAir    | <https://theairlab.org/tartanair-dataset/>                               | `<root>/<scene>/<difficulty>/<traj>/{image_left,depth_left,pose_left.txt}` |
| Hypersim     | <https://github.com/apple/ml-hypersim>                                   | `<root>/<scene>/{images,_detail}` standard Hypersim layout          |
| NYU Depth V2 | <https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html>               | `<root>/{nyu_test_rgb,nyu_test_depth}/<id>.{png,jpg}`              |
| KITTI        | <http://www.cvlibs.net/datasets/kitti/eval_depth.php> (Eigen split)      | `<root>/{kitti_eval_rgb,kitti_eval_depth}/<id>.png`                |

You can place a small subset under `./datasets/` matching the layout above
to smoke-test the pipeline.

## End-to-end usage

All scripts use **seed 42** by default and read paths/hyperparameters from
`configs/config.yaml`.

### 1. Build viewpoint-bucketed splits

```bash
python -m mde_viewpoint.data.dataset_builder \
    --config mde_viewpoint/configs/config.yaml
```

Writes `mde_viewpoint/data/splits/category_{A,B,C,D,E}.json` plus a
`summary.json`.

### 2. Zero-shot benchmark across all 4 models × 5 categories

```bash
python -m mde_viewpoint.experiments.run_zero_shot_eval \
    --config mde_viewpoint/configs/config.yaml
```

Writes per-run JSON metrics, a wide CSV, and a LaTeX table to
`results/zero_shot/`.

### 3. Failure-mode analysis (heatmaps, attention, histograms)

```bash
python -m mde_viewpoint.experiments.run_failure_analysis \
    --model dav2 --category D --n_images 50
```

Writes per-image error heatmaps, an aggregate spatial-bias map, attention
overlays for the last 4 transformer blocks, and a depth histogram to
`results/failure_analysis/<model>/<category>/`.

### 4. LoRA adaptation curves

```bash
python -m mde_viewpoint.experiments.run_lora_adaptation \
    --category D --rank 8
```

Fine-tunes Depth Anything V2 with LoRA on N ∈ {50, 100, 200, 500} samples,
writes `results/lora/category_D_rank8/{adaptation_curve.csv,adaptation_curve.png,...}`,
and saves the best adapter under `best_adapter/`.

### 5. Forgetting evaluation on NYU + KITTI

```bash
python -m mde_viewpoint.experiments.run_forgetting_eval \
    --adapter_path results/lora/category_D_rank8/N_500/best_adapter \
    --rank 8
```

Writes `results/forgetting/forgetting.json` with base vs LoRA metrics on
each benchmark and the forgetting deltas.

### 6. Final figures and tables

```bash
jupyter notebook mde_viewpoint/notebooks/results_visualization.ipynb
```

## Implementation notes

- **Median scaling.** Relative-depth models (DAV2, MiDaS, Marigold) are
  median-scaled per image before metric evaluation; ZoeDepth is treated as
  metric directly. See `eval/metrics.apply_median_scaling`.
- **Loss.** Training uses the scale-invariant log loss from Eigen et al.
  (2014), implemented in `eval/metrics.scale_invariant_loss`.
- **LoRA injection.** Adapters are placed on the **query** and **value**
  linear projections of every DINOv2 attention block via PEFT. See
  `models/lora_wrapper.get_lora_model`. A pure-PyTorch fallback is provided
  if PEFT cannot be imported.
- **Attention extraction.** `analysis/attention_maps.AttentionExtractor`
  registers forward hooks on the last 4 transformer blocks and recovers
  `[CLS]` attention weights, then bilinearly upsamples them to the input
  resolution.
- **Reproducibility.** Every experiment script seeds Python, NumPy and
  PyTorch RNGs to 42 before constructing data / model.
- **Edge cases handled.**
  - Non-finite depth values are masked out at load time and during eval.
  - Empty validation masks return NaN metrics rather than crashing.
  - Prediction shape mismatches are resolved via bilinear upsampling.
  - Missing dataset roots produce empty splits with informative log lines
    instead of hard failures.
