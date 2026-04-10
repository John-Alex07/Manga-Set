"""Run the expanded 30-prompt ablation study across all conditions and trials.

Usage:
    python scripts/run_expanded_ablation.py [--dry-run]

This runs 12 experiment configs (4 conditions x 3 trials) sequentially.
Each config generates 30 images. Total: 360 images.

Estimated time on CPU: ~4-5 hours
Estimated time on GPU: ~30-45 minutes
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

CONDITIONS = ["base", "style_prompt", "lora_planogram", "lora_cartoon"]
TRIALS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanded ablation study")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--condition", choices=CONDITIONS, help="Run only one condition")
    parser.add_argument("--trial", type=int, choices=TRIALS, help="Run only one trial")
    args = parser.parse_args()

    configs_to_run = []
    for cond in CONDITIONS:
        if args.condition and cond != args.condition:
            continue
        for trial in TRIALS:
            if args.trial is not None and trial != args.trial:
                continue
            config_name = f"expanded_{cond}_t{trial}.json"
            config_path = CONFIGS_DIR / config_name
            if not config_path.exists():
                print(f"WARNING: {config_name} not found, skipping")
                continue
            configs_to_run.append((cond, trial, config_path))

    print(f"Will run {len(configs_to_run)} experiments")
    for cond, trial, path in configs_to_run:
        print(f"  {cond} trial {trial}: {path.name}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    total_start = time.time()
    for i, (cond, trial, config_path) in enumerate(configs_to_run):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs_to_run)}] Running: {cond} trial {trial}")
        print(f"{'='*60}")
        run_start = time.time()
        try:
            config = load_experiment_config(config_path)
            runner = ExperimentRunner(config, project_root=config_path.parent.parent)
            runner.run()
            elapsed = time.time() - run_start
            print(f"Completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    total_elapsed = time.time() - total_start
    print(f"\nAll experiments completed in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
