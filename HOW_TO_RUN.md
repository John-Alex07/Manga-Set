# HOW TO RUN: CRAFT Experiment Reproduction Guide

This document contains exact, step-by-step instructions for setting up the environment, running all experiments, and analyzing results. Designed for any agent or person to follow without prior context.

---

## 1. Prerequisites

- **Python 3.10+** (tested with 3.12)
- **pip** (any recent version)
- **Linux environment** recommended (WSL2 on Windows works). All commands below assume bash.
- **Disk space**: ~60 GB free for model downloads (HuggingFace cache)
- **RAM**: 16 GB minimum (32 GB recommended for SDXL Turbo and Flux Schnell)
- **GPU** (optional): NVIDIA GPU with CUDA for faster generation. CPU mode works but is slow (~3-4 min per image for SD 2.1).

## 2. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd Manga-Set

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (exact versions for reproducibility)
pip install -r requirements-benchmark-lock.txt
```

**GPU users**: Instead of the lock file, install PyTorch with CUDA first, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-benchmark.txt
```

## 3. Verify Installation

### 3.1 Check environment

```bash
python -m framework.cli env
```

Expected output: All dependencies show `OK` except `streamlit` (optional, not needed for experiments).

### 3.2 Run the test suite

```bash
python -m unittest discover -s tests -v
```

Expected: **76 tests, all passing**.

### 3.3 Dry-run the experiment matrix

```bash
python scripts/run_full_ablation.py --dry-run
```

Expected: Lists 41 experiment configurations across 5 backbone groups.

## 4. Experiment Matrix Overview

| Backbone | Steps | Conditions | Prompts | Trials | Total Images |
|---|---|---|---|---|---|
| SD 2.1 (`sd21_s4`) | 4 | base, style_prompt, lora_planogram, lora_cartoon | 30 | 3 | 360 |
| SD 2.1 (`sd21_s20`) | 20 | base, style_prompt, lora_planogram, lora_cartoon | 30 | 3 | 360 |
| SD 2.1 + ControlNet (`controlnet`) | 20 | canny | 30 | 3 | 90 |
| SDXL Turbo (`sdxl`) | 4 | base, style_prompt, lora_anime, lora_manga | 30 | 3 | 360 |
| Flux Schnell (`flux`) | 4 | base, style_prompt | 10 | 1 | 20 |
| **Total** | | | | | **1,190** |

## 5. Running Experiments

### 5.1 Run everything (all backbones)

```bash
python scripts/run_full_ablation.py --skip-existing
```

This runs all 41 configs sequentially. `--skip-existing` resumes from where you left off if interrupted.

**Estimated wall-clock time on CPU**: ~60-80 hours (3-4 days continuously).
**Estimated wall-clock time on GPU (A100)**: ~2-4 hours.

### 5.2 Run a single backbone group

```bash
# Run only SD 2.1 at 4 steps (fastest, good for initial testing)
python scripts/run_full_ablation.py --backbone sd21_s4 --skip-existing

# Run only SD 2.1 at 20 steps
python scripts/run_full_ablation.py --backbone sd21_s20 --skip-existing

# Run only ControlNet experiments
python scripts/run_full_ablation.py --backbone controlnet --skip-existing

# Run only SDXL Turbo experiments
python scripts/run_full_ablation.py --backbone sdxl --skip-existing

# Run only Flux Schnell experiments
python scripts/run_full_ablation.py --backbone flux --skip-existing
```

### 5.3 Run a single experiment config

```bash
python -m framework.cli run configs/expanded_base_t0.json
```

### 5.4 Running in the background (recommended for long runs)

```bash
nohup python scripts/run_full_ablation.py --skip-existing > ablation.log 2>&1 &
tail -f ablation.log
```

## 6. Backbone-Specific Notes

### 6.1 SD 2.1 (sd21_s4, sd21_s20)

- **Model**: `sd-research/stable-diffusion-2-1-base`
- **Scheduler**: Euler
- **Adapters**: LoRA cartoon (`dlcvproj/cartoon_sd_lora`), LoRA planogram (`bharadwajkg/finetune-stable-diffusion-2-1-planogram-lora`), style prompt (prompt-based, no model download)
- **Note**: 4-step configs use `guidance_scale=6.0`. 20-step configs also use `guidance_scale=6.0`.
- **First run**: Models download automatically from HuggingFace (~5 GB).

### 6.2 SDXL Turbo (sdxl)

- **Model**: `stabilityai/sdxl-turbo` (~7 GB download)
- **Scheduler**: EulerAncestralDiscrete
- **Guidance scale**: 0.0 (turbo distillation requires no guidance)
- **Adapters**: LoRA anime (`ntc-ai/SDXL-LoRA-slider.anime`), LoRA manga (`artificialguybr/LineAniRedmond-LinearMangaSDXL-V2`), style prompt
- **Note**: Generates 512x512 images. Higher RAM usage than SD 2.1.

### 6.3 Flux Schnell (flux)

- **Model**: `black-forest-labs/FLUX.1-schnell` (~33 GB download, 12B parameters)
- **dtype**: bfloat16
- **Scheduler**: Default (Flow Matching)
- **Guidance scale**: 0.0
- **Note**: No negative prompts (Flux does not support them). Reduced to 10 prompts and 1 trial due to extreme CPU cost. Requires significant RAM (~40 GB on CPU). Only `base` and `style_prompt` conditions.

### 6.4 ControlNet (controlnet)

- **Model**: SD 2.1 base + `thibaud/controlnet-sd21-canny-diffusers`
- **Steps**: 20
- **Note**: Requires `opencv-python-headless` for Canny edge detection. Each prompt includes a `controls.canny` reference image path. The framework generates Canny edge maps automatically from reference images.

## 7. Model Downloads

Models download automatically on first run via HuggingFace Hub. All configs have `local_files_only: false`.

To prefetch models before running experiments:

```bash
# Prefetch SD 2.1
python -m framework.cli prefetch sd-research/stable-diffusion-2-1-base

# Prefetch SDXL Turbo
python -m framework.cli prefetch stabilityai/sdxl-turbo

# Prefetch Flux Schnell
python -m framework.cli prefetch black-forest-labs/FLUX.1-schnell

# Prefetch ControlNet
python -m framework.cli prefetch thibaud/controlnet-sd21-canny-diffusers

# Prefetch LoRA adapters
python -m framework.cli prefetch dlcvproj/cartoon_sd_lora
python -m framework.cli prefetch bharadwajkg/finetune-stable-diffusion-2-1-planogram-lora
python -m framework.cli prefetch ntc-ai/SDXL-LoRA-slider.anime
python -m framework.cli prefetch artificialguybr/LineAniRedmond-LinearMangaSDXL-V2

# Prefetch evaluation models
python -m framework.cli prefetch openai/clip-vit-base-patch32
python -m framework.cli prefetch facebook/dino-vits16
```

## 8. Output Structure

Each experiment produces a directory under `outputs/`:

```
outputs/
  expanded_base_t0/
    experiment_report.json    # Full results (metrics, environment, timings)
    align_rooftop_duel_base_t0.png
    align_library_dialogue_base_t0.png
    ...                       # 30 generated images
  expanded_style_prompt_t0/
    ...
  sdxl_base_t0/
    ...
```

The `experiment_report.json` contains:
- Experiment metadata (name, config, seed)
- Environment snapshot (Python version, package versions, OS, GPU info)
- Per-prompt results with metric scores (CLIP alignment, LPIPS, DINO, latency, etc.)

## 9. Analysis and Metrics

### 9.1 Statistical analysis of experiment results

```bash
python scripts/analyze_results.py
```

Produces per-condition means, standard deviations, and Wilcoxon signed-rank pairwise comparisons.

### 9.2 Distribution-level metrics (FID and CMMD)

```bash
# Requires: pip install clean-fid
python scripts/distribution_metrics.py --all-backbones
```

Computes FID and CMMD for each adapter condition relative to the base condition within each backbone.

### 9.3 Export results to Markdown/CSV

```bash
# Export a single report
python -m framework.cli export outputs/expanded_base_t0/experiment_report.json

# Ablation table comparing conditions
python -m framework.cli ablation-table \
  Base=outputs/expanded_base_t0/experiment_report.json \
  Style=outputs/expanded_style_prompt_t0/experiment_report.json \
  LoRA-Cartoon=outputs/expanded_lora_cartoon_t0/experiment_report.json

# Grouped summary
python -m framework.cli grouped-summary outputs/expanded_base_t0/experiment_report.json
```

## 10. Regenerating Configs

If you need to modify config parameters and regenerate:

```bash
python scripts/generate_expanded_configs.py       # SD 2.1 @ 4 steps
python scripts/generate_sd21_20step_configs.py     # SD 2.1 @ 20 steps
python scripts/generate_sdxl_configs.py            # SDXL Turbo
python scripts/generate_flux_configs.py            # Flux Schnell
python scripts/generate_controlnet_configs.py      # ControlNet Canny
```

## 11. Known Issues

| Issue | Workaround |
|---|---|
| `ImageReward` metric fails with `transformers>=5.0` | Removed from configs. The `image-reward` package depends on `apply_chunking_to_forward` which was removed in newer `transformers`. Do not add `image_reward` to config metrics until `image-reward` is updated. |
| Flux Schnell requires HF authentication | Flux is a gated model. Run `huggingface-cli login` or set `HF_TOKEN` env var before running Flux configs. You must accept the model license at https://huggingface.co/black-forest-labs/FLUX.1-schnell |
| Flux Schnell is extremely slow on CPU | ~30-60 min per image. Only 20 images total. Use GPU if available. |
| SDXL Turbo requires ~16 GB RAM on CPU | May OOM on machines with less. Reduce batch or use GPU. |
| Windows Python + WSL HF cache incompatibility | Run all experiments from within the same OS environment. Do not mix Windows Python with WSL cache paths. |

## 12. File Structure Reference

```
Manga-Set/
├── framework/              # Core framework code
│   ├── models/             # Generation backends (mock, diffusers)
│   ├── adapters/           # Conditioning modules (lora, style_prompt, controlnet)
│   ├── evaluation/         # Metric implementations (CLIP, LPIPS, DINO, etc.)
│   ├── experiments/        # Config-driven experiment runner
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Config loader
│   ├── environment.py      # Environment inspector
│   └── cache.py            # HuggingFace artifact prefetch
├── configs/                # 62 experiment configurations (JSON)
│   ├── expanded_*.json     # SD 2.1 @ 4 steps (12 configs)
│   ├── sd21_s20_*.json     # SD 2.1 @ 20 steps (12 configs)
│   ├── sd21_controlnet_*.json  # ControlNet (3 configs)
│   ├── sdxl_*.json         # SDXL Turbo (12 configs)
│   ├── flux_*.json         # Flux Schnell (2 configs)
│   └── benchmark_*.json    # Legacy benchmark configs (19 configs)
├── prompts/                # Benchmark prompt suites (4 JSON files, 30 prompts)
├── scripts/                # Config generators, runners, analysis
│   ├── run_full_ablation.py         # Main experiment runner
│   ├── analyze_results.py           # Statistical analysis
│   ├── distribution_metrics.py      # FID/CMMD post-hoc metrics
│   ├── generate_*_configs.py        # Config generators (5 scripts)
│   └── validate_*.py               # Validation utilities
├── tests/                  # 76 automated tests
├── outputs/                # Generated images and reports (gitignored)
├── requirements.txt        # Minimum dependencies
├── requirements-benchmark.txt       # Benchmark minimum dependencies
├── requirements-benchmark-lock.txt  # Exact versions for reproducibility
├── FRAMEWORK.md            # Architecture documentation
├── REPRODUCIBILITY.md      # Reproducibility methodology
└── HOW_TO_RUN.md           # This file
```

## 13. Quick Start Checklist

1. [ ] Clone the repository
2. [ ] Create and activate a Python 3.10+ virtual environment
3. [ ] `pip install -r requirements-benchmark-lock.txt`
4. [ ] `python -m framework.cli env` — verify all deps OK
5. [ ] `python -m unittest discover -s tests -v` — verify 76 tests pass
6. [ ] `python scripts/run_full_ablation.py --dry-run` — verify 41 configs listed
7. [ ] `python scripts/run_full_ablation.py --backbone sd21_s4 --skip-existing` — run quickest backbone first
8. [ ] `python scripts/run_full_ablation.py --skip-existing` — run all remaining backbones
9. [ ] `python scripts/analyze_results.py` — analyze results
10. [ ] `python scripts/distribution_metrics.py --all-backbones` — compute FID/CMMD
