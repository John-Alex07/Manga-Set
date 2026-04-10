# CRAFT: Conditioned Reproducible Adapter Framework for Text-to-Image

A modular research framework for controlled adapter ablation studies in manga panel generation with diffusion models.

## Architecture

```
framework/
  models/        Generation backends (mock for tests, diffusers for production)
    diffusion.py   Multi-backbone: SD 2.1, SDXL Turbo, Flux Schnell, ControlNet
  adapters/      Conditioning modules
    style_prompt  Prompt-based style conditioning (fully implemented)
    lora          True LoRA weight loading via diffusers (fully implemented)
    controlnet    ControlNet Canny edge-map conditioning (fully implemented)
    ip_adapter    Interface stub (not wired to IP-Adapter module)
  evaluation/    Metric implementations
    CLIP text-image alignment
    CLIP image-image similarity (via consistency analysis)
    ImageReward human-preference scoring
    LPIPS perceptual distance (character-group aware)
    DINO feature similarity (character-group aware)
    BLIP caption keyword recall
    Image statistics and file integrity
    Latency measurement
  experiments/   Config-driven runner with per-prompt seed control
  environment.py Version-capturing environment inspector with GPU/CUDA detection
  cache.py       HuggingFace artifact prefetch with revision pinning
  refinement/    Refinement stage interface (placeholder)
prompts/         Benchmark prompt suites (30 prompts across 4 suites)
configs/         Experiment configurations (41 configs, revision-pinned)
scripts/         Config generators, runners, and analysis scripts
  generate_expanded_configs.py      SD 2.1 @ 4 steps (original)
  generate_sd21_20step_configs.py   SD 2.1 @ 20 steps
  generate_sdxl_configs.py          SDXL Turbo @ 4 steps
  generate_flux_configs.py          Flux Schnell @ 4 steps (reduced)
  generate_controlnet_configs.py    SD 2.1 + ControlNet Canny @ 20 steps
  run_full_ablation.py              Multi-backbone experiment runner
  distribution_metrics.py           Post-hoc FID and CMMD analysis
tests/           76 automated tests (53 protocol invariant checks + 23 framework unit tests)
```

## Experiment Matrix

| Backbone | Steps | Conditions | Prompts | Trials | Total |
|----------|-------|------------|---------|--------|-------|
| SD 2.1 | 4 | base, style, plano, cartoon | 30 | 3 | 360 |
| SD 2.1 | 20 | base, style, plano, cartoon | 30 | 3 | 360 |
| SD 2.1 | 20 | ControlNet canny | 30 | 3 | 90 |
| SDXL Turbo | 4 | base, style, anime, manga | 30 | 3 | 360 |
| Flux Schnell | 4 | base, style | 10 | 1 | 20 |
| | | | | **Total** | **1,190** |

## Benchmark Suites

| Suite | Prompts | Focus |
|-------|---------|-------|
| Alignment | 7 | Object counting, spatial layout, motion, environment |
| Consistency | 11 | 4 characters (Hiro, Akira, Yuki, Kenji) with distinct visual anchors |
| Story | 6 | Two 3-panel narrative sequences |
| Style Fidelity | 6 | Screentone, speed lines, chibi, shoujo, noir, seinen |

## Quick Start

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run the full multi-backbone ablation study (requires ML runtime):

```bash
python scripts/run_full_ablation.py               # All backbones
python scripts/run_full_ablation.py --backbone sd21_s20  # Single backbone
python scripts/run_full_ablation.py --dry-run      # Preview only
```

Run a single experiment config:

```bash
python -m framework.cli run configs/expanded_base_t0.json
```

Regenerate configs after changes:

```bash
python scripts/generate_expanded_configs.py
python scripts/generate_sd21_20step_configs.py
python scripts/generate_sdxl_configs.py
python scripts/generate_flux_configs.py
python scripts/generate_controlnet_configs.py
```

Post-hoc distribution metrics:

```bash
python scripts/distribution_metrics.py --all-backbones
```

## Evaluation Metrics

| Metric | Type | Scope |
|--------|------|-------|
| CLIP text-image | Inline | Per-image alignment |
| ImageReward | Inline | Per-image human preference |
| CLIP image-image | Post-hoc | Character consistency |
| LPIPS | Inline | Perceptual distance (infra-ready) |
| DINO | Inline | Feature similarity (infra-ready) |
| FID | Post-hoc | Distribution-level distance |
| CMMD | Post-hoc | Distribution-level distance (CLIP-based) |

## Key Design Decisions

1. **Conditioning type separation**: `style_prompt`, `lora`, and `controlnet` adapters have different config types, different code paths, and automated tests preventing mislabeling.
2. **Multi-backbone support**: SD 2.1, SDXL Turbo, and Flux Schnell with pipeline-class-aware loading, scheduler maps, conditional negative prompts, and backbone-specific guidance scales.
3. **Canonical seed control**: All conditions share per-prompt seeds; protocol tests verify seed arithmetic invariants across configs.
4. **Multi-metric evaluation**: CLIP, ImageReward, LPIPS, DINO, FID, and CMMD provide complementary views of quality, alignment, consistency, and distributional shift.
5. **Multi-trial design**: 3 seed offsets (42, 137, 256) per condition enable statistical analysis (mean ± std, Wilcoxon signed-rank tests).
6. **Revision-pinned models**: Every config embeds HuggingFace commit SHAs for the diffusion model, LoRA weights, ControlNet, CLIP, and DINO to prevent silent model drift.
7. **Torch determinism**: Global determinism flags (`use_deterministic_algorithms`, `cudnn.deterministic`) are enforced before every generation.
8. **GPU-aware environment capture**: Every experiment report embeds the full software environment (Python version, package versions, OS, CUDA/GPU properties) for auditability.
9. **LoRA fallback with logging**: Two-path LoRA loading strategy handles both modern and legacy weight formats, with explicit logging of which path was taken.
10. **ControlNet integration**: Full Canny edge-map ControlNet pipeline with configurable thresholds and reference-image preprocessing.
