# Reproducibility Manifest

## Software Environment

All benchmark experiments reported in the paper were executed using the following stack:

- Python 3.10+
- PyTorch >= 2.0.0 (CPU mode for reported experiments)
- Diffusers >= 0.25.0
- Transformers >= 4.35.0
- PEFT >= 0.10.0
- torchvision >= 0.15.0
- lpips >= 0.1.4
- scipy >= 1.10.0
- huggingface_hub >= 0.20.0
- safetensors >= 0.4.0
- accelerate >= 0.25.0

Install benchmark dependencies:

    pip install -r requirements-benchmark.txt

For exact version reproduction, use the pinned lockfile (recorded from the environment used for reported experiments):

    pip install -r requirements-benchmark-lock.txt

## Model Artifacts (Pinned Revisions)

All model artifacts are referenced by HuggingFace repository ID **and** commit SHA to prevent silent drift:

| Artifact | HuggingFace Repo | Revision SHA |
|----------|-----------------|--------------|
| Stable Diffusion 2.1 Base | `sd-research/stable-diffusion-2-1-base` | `0708cecd370b4d1c3a6ff3f7332f5e9aea78896f` |
| Planogram LoRA | `bharadwajkg/finetune-stable-diffusion-2-1-planogram-lora` | `5129789dc19adb2edf3275cfe9c35327bbca8605` |
| Cartoon LoRA | `dlcvproj/cartoon_sd_lora` | `a526614e425b82be098144a862d49aa8e5ded083` |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| DINO ViT-S/16 | `facebook/dino-vits16` | `abe3b354cb6a9b6f146096b14a4a9d7eecbcb4bd` |

Revision SHAs are embedded in every experiment config file and passed through to
`from_pretrained()` and `hf_hub_download()` calls, so the exact model weights are
locked regardless of future repository updates.

Prefetch all model artifacts into the local HuggingFace cache:

    python -m framework.cli prefetch sd-research/stable-diffusion-2-1-base
    python -m framework.cli prefetch openai/clip-vit-base-patch32
    python -m framework.cli prefetch facebook/dino-vits16
    python -m framework.cli prefetch bharadwajkg/finetune-stable-diffusion-2-1-planogram-lora
    python -m framework.cli prefetch dlcvproj/cartoon_sd_lora

## Determinism Controls

### Torch Determinism

The framework enforces deterministic execution before every generation call:

- `torch.manual_seed(0)` — seeds global RNG
- `torch.use_deterministic_algorithms(True)` — forces deterministic op selection
- `torch.backends.cudnn.deterministic = True` — disables cudnn autotuning
- `torch.backends.cudnn.benchmark = False` — prevents non-deterministic kernel selection

These are set in `DiffusersGenerator._enforce_determinism()` (called before each `generate()`).

### Per-prompt Seed Control

Each prompt has a per-prompt `torch.Generator` seeded with `base_seed + trial_seed`:

- Generator device: `"cpu"` (or `"cpu"` when MPS, to avoid MPS generator quirks)
- Seed is passed to the diffusers pipeline via the `generator` kwarg

This ensures the **only** variable across conditions is the adapter configuration.

### Seed Mapping

The expanded study uses base seeds per prompt plus a trial seed offset.
Three trial seeds are used: 42, 137, 256.

Per-prompt seed = base seed + trial seed offset.

Base seed mapping:

| Prompt ID | Base Seed |
|-----------|-----------|
| `align_rooftop_duel` | 1001 |
| `align_library_dialogue` | 1002 |
| `align_alley_chase` | 1003 |
| `align_market_crowd` | 1004 |
| `align_classroom_reveal` | 1005 |
| `align_bridge_standoff` | 1006 |
| `align_dojo_training` | 1007 |
| `consistency_hiro_panel_1` | 1101 |
| `consistency_hiro_panel_2` | 1102 |
| `consistency_hiro_panel_3` | 1103 |
| `consistency_akira_panel_1` | 1104 |
| `consistency_akira_panel_2` | 1105 |
| `consistency_akira_panel_3` | 1106 |
| `consistency_yuki_panel_1` | 1107 |
| `consistency_yuki_panel_2` | 1108 |
| `consistency_yuki_panel_3` | 1109 |
| `consistency_kenji_panel_1` | 1110 |
| `consistency_kenji_panel_2` | 1111 |
| `story_panel_1_setup` | 1201 |
| `story_panel_2_confrontation` | 1202 |
| `story_panel_3_action` | 1203 |
| `story_cafe_1_arrival` | 1204 |
| `story_cafe_2_recognition` | 1205 |
| `story_cafe_3_conversation` | 1206 |
| `style_screentone_gradient` | 1301 |
| `style_speed_lines` | 1302 |
| `style_chibi_emotion` | 1303 |
| `style_dramatic_shadow` | 1304 |
| `style_shoujo_sparkle` | 1305 |
| `style_seinen_detail` | 1306 |

## Benchmark Design

The expanded benchmark comprises **30 prompts** across four suites:

| Suite | Count | Focus |
|-------|-------|-------|
| Alignment | 7 | Object counting, spatial layout, motion, environment |
| Consistency | 11 | Paired/tripled panels for 4 characters (Hiro, Akira, Yuki, Kenji) |
| Story | 6 | Two 3-panel narrative sequences |
| Style Fidelity | 6 | Manga-specific visual elements (screentone, speed lines, chibi, shoujo, noir, seinen) |

## Evaluation Metrics

| Metric | Model | Direction | Character-Group Aware |
|--------|-------|-----------|-----------------------|
| CLIP text-image alignment | `openai/clip-vit-base-patch32` | Higher = better alignment | No |
| CLIP image-image similarity | `openai/clip-vit-base-patch32` | Higher = more consistent | Yes (post-hoc) |
| LPIPS perceptual distance | AlexNet (lpips library) | Reported as 1-LPIPS; higher = more similar | Yes |
| DINO similarity | `facebook/dino-vits16` | Higher = more similar | Yes |

All evaluation models use locally cached weights with `local_files_only: true` and pinned revision SHAs.

LPIPS and DINO metrics automatically use **character group images** from the same experiment run when no explicit reference images are provided. For consistency suite prompts, this means each panel is compared against previously generated panels of the same character.

## Environment Capture

Every `experiment_report.json` includes a full `"environment"` block recording:

- Python version, implementation, OS, and machine architecture
- Installed versions of all core dependencies (torch, diffusers, transformers, peft, etc.)
- Module availability status

This ensures that the exact software stack used for each experiment run is auditable.

## Reproducing Experiments

### Expanded study (30-prompt, 3 trials)

    python scripts/run_expanded_ablation.py

Or run individual conditions:

    python scripts/run_expanded_ablation.py --condition base --trial 0

### Analysis

    python scripts/analyze_results.py               # Analyze expanded data

### Validation

    python -m unittest discover -s tests -v         # Run all 76 tests (53 protocol + 23 framework)
    python scripts/validate_paper_numbers.py        # Verify paper numbers match data

### Protocol Tests

The test suite (`tests/test_benchmark_protocol.py`) enforces the following invariants:

- All 12 expanded configs exist (4 conditions x 3 trials)
- Each config has exactly 30 prompts
- Prompt IDs match across conditions within each trial
- Per-prompt seeds match across conditions within each trial
- Seeds differ across trials
- Seed arithmetic: `actual_seed == base_seed + trial_seed` for every prompt
- Guidance scale and inference steps match across conditions
- `local_files_only=true` for CLIP and DINO evaluation models
- Model revision SHA is present in every config
- CLIP and DINO evaluation model revision SHAs are present
- LoRA adapter revision SHAs are present for LoRA conditions
- LoRA adapters have valid weight paths
- Consistency configs have matching guidance scale and inference steps

## LoRA Loading

The framework uses a two-path LoRA loading strategy with explicit logging:

1. **Primary path**: `pipeline.load_lora_weights()` with `set_adapters()` for scale control
2. **Fallback path**: `pipeline.unet.load_attn_procs()` for legacy key formats, with manual scale application

Both paths log which loading strategy was used. The fallback is needed for older LoRA checkpoints whose key naming conventions predate the modern `diffusers` PEFT integration.
