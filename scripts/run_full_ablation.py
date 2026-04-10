"""Run the complete multi-backbone ablation study.

Usage:
    python scripts/run_full_ablation.py [--dry-run] [--backbone BACKBONE]

Experiment matrix:
  SD 2.1 @ 4 steps:  4 conditions x 3 trials x 30 prompts =  360 images (DONE)
  SD 2.1 @ 20 steps: 4 conditions x 3 trials x 30 prompts =  360 images
  SD 2.1 + ControlNet Canny @ 20 steps: 3 trials x 30     =   90 images
  SDXL Turbo @ 4 steps: 4 conditions x 3 trials x 30      =  360 images
  Flux Schnell @ 4 steps: 2 conditions x 1 trial x 10     =   20 images
                                                     Total = ~1190 images

Estimated CPU time: ~60-80 hours (3-4 days continuously)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.config import load_experiment_config
from framework.experiments.runner import ExperimentRunner

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

BACKBONE_RUNS: dict[str, list[str]] = {
    "sd21_s4": [
        f"expanded_{cond}_t{t}.json"
        for cond in ["base", "style_prompt", "lora_planogram", "lora_cartoon"]
        for t in [0, 1, 2]
    ],
    "sd21_s20": [
        f"sd21_s20_{cond}_t{t}.json"
        for cond in ["base", "style_prompt", "lora_planogram", "lora_cartoon"]
        for t in [0, 1, 2]
    ],
    "controlnet": [
        f"sd21_controlnet_canny_s20_t{t}.json"
        for t in [0, 1, 2]
    ],
    "sdxl": [
        f"sdxl_{cond}_t{t}.json"
        for cond in ["base", "style_prompt", "lora_anime", "lora_manga"]
        for t in [0, 1, 2]
    ],
    "flux": [
        f"flux_{cond}_t0.json"
        for cond in ["base", "style_prompt"]
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full multi-backbone ablation study")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONE_RUNS.keys()) + ["all"],
        default="all",
        help="Run only one backbone (default: all)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output dir already exists with a report")
    args = parser.parse_args()

    backbones = list(BACKBONE_RUNS.keys()) if args.backbone == "all" else [args.backbone]
    configs_to_run: list[tuple[str, str, Path]] = []
    for backbone in backbones:
        for config_name in BACKBONE_RUNS[backbone]:
            config_path = CONFIGS_DIR / config_name
            if not config_path.exists():
                print(f"WARNING: {config_name} not found, skipping")
                continue
            configs_to_run.append((backbone, config_name, config_path))

    print(f"Will run {len(configs_to_run)} experiments across backbones: {', '.join(backbones)}")
    for backbone, name, _ in configs_to_run:
        print(f"  [{backbone}] {name}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    total_start = time.time()
    completed, skipped, failed = 0, 0, 0

    for i, (backbone, config_name, config_path) in enumerate(configs_to_run):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(configs_to_run)}] [{backbone}] Running: {config_name}")
        print(f"{'=' * 60}")

        if args.skip_existing:
            output_dir = config_path.parent.parent / "outputs" / config_name.replace(".json", "")
            report = output_dir / "experiment_report.json"
            if report.exists():
                print(f"  Skipping — report already exists: {report}")
                skipped += 1
                continue

        run_start = time.time()
        try:
            config = load_experiment_config(config_path)
            runner = ExperimentRunner(config, project_root=config_path.parent.parent)
            runner.run()
            elapsed = time.time() - run_start
            print(f"  Completed in {elapsed:.1f}s")
            completed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — {completed} completed, {skipped} skipped, {failed} failed")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 3600:.1f}h)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
