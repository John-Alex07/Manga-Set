"""Smoke test: run 1 prompt from each unique backbone+adapter combination.

Validates every code path (model loading, adapter application, generation,
metric evaluation, report writing) without running the full 30-prompt suite.

Usage:
    python scripts/smoke_test_all.py [--skip-existing] [--only BACKBONE]

Estimated time on CPU:
    SD 2.1 configs:   ~3-5 min each
    SDXL Turbo:       ~5-10 min each
    ControlNet:       ~10-15 min
    Flux Schnell:     ~30-60 min
    Total:            ~2-3 hours on CPU, ~15 min on GPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.config import load_experiment_config
from framework.experiments.runner import ExperimentRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
SMOKE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "_smoke_tests"

SMOKE_CONFIGS: dict[str, list[str]] = {
    "sd21_s4_base": ["expanded_base_t0.json"],
    "sd21_s4_style": ["expanded_style_prompt_t0.json"],
    "sd21_s4_lora_cartoon": ["expanded_lora_cartoon_t0.json"],
    "sd21_s4_lora_planogram": ["expanded_lora_planogram_t0.json"],
    "sd21_s20_base": ["sd21_s20_base_t0.json"],
    "sd21_s20_lora_cartoon": ["sd21_s20_lora_cartoon_t0.json"],
    "controlnet_canny": ["sd21_controlnet_canny_s20_t0.json"],
    "sdxl_base": ["sdxl_base_t0.json"],
    "sdxl_style": ["sdxl_style_prompt_t0.json"],
    "sdxl_lora_anime": ["sdxl_lora_anime_t0.json"],
    "sdxl_lora_manga": ["sdxl_lora_manga_t0.json"],
    "flux_base": ["flux_base_t0.json"],
}


def _trim_to_one_prompt(config_path: Path, smoke_name: str) -> Path:
    """Create a temporary config with only the first prompt."""
    data = json.loads(config_path.read_text())
    data["name"] = f"_smoke_{smoke_name}"
    data["output_dir"] = str(SMOKE_OUTPUT_DIR.relative_to(PROJECT_ROOT))
    data["prompts"] = data["prompts"][:1]
    trimmed_path = SMOKE_OUTPUT_DIR / f"_smoke_{smoke_name}.json"
    trimmed_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed_path.write_text(json.dumps(data, indent=2))
    return trimmed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test all backbone+adapter combos")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only", type=str, default=None, help="Run only this smoke test key")
    args = parser.parse_args()

    tests = SMOKE_CONFIGS
    if args.only:
        if args.only not in tests:
            print(f"Unknown key: {args.only}. Available: {', '.join(tests.keys())}")
            sys.exit(1)
        tests = {args.only: tests[args.only]}

    print(f"=== CRAFT Smoke Test: {len(tests)} backbone+adapter combinations ===\n")

    total_start = time.time()
    results: dict[str, str] = {}

    for smoke_name, config_files in tests.items():
        config_file = config_files[0]
        config_path = CONFIGS_DIR / config_file
        if not config_path.exists():
            print(f"  [{smoke_name}] SKIP — config not found: {config_file}")
            results[smoke_name] = "SKIP (config missing)"
            continue

        output_dir = SMOKE_OUTPUT_DIR / f"_smoke_{smoke_name}"
        report_path = output_dir / "experiment_report.json"
        if args.skip_existing and report_path.exists():
            print(f"  [{smoke_name}] SKIP — already completed")
            results[smoke_name] = "SKIP (existing)"
            continue

        print(f"\n{'='*60}")
        print(f"  [{smoke_name}] Running: {config_file} (1 prompt)")
        print(f"{'='*60}")

        trimmed_config_path = _trim_to_one_prompt(config_path, smoke_name)
        run_start = time.time()
        try:
            config = load_experiment_config(trimmed_config_path)
            runner = ExperimentRunner(config, project_root=PROJECT_ROOT)
            report = runner.run()
            elapsed = time.time() - run_start

            n_results = len(report.get("results", []))
            first_scores = report.get("results", [{}])[0].get("scores", {})
            score_keys = list(first_scores.keys())

            print(f"  PASS — {n_results} image(s), {len(score_keys)} metrics, {elapsed:.1f}s")
            print(f"    Metrics: {', '.join(score_keys)}")
            results[smoke_name] = f"PASS ({elapsed:.1f}s)"
        except Exception as e:
            elapsed = time.time() - run_start
            print(f"  FAIL — {e}")
            import traceback
            traceback.print_exc()
            results[smoke_name] = f"FAIL: {e}"

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v.startswith("PASS"))
    failed = sum(1 for v in results.values() if v.startswith("FAIL"))
    skipped = sum(1 for v in results.values() if v.startswith("SKIP"))
    for name, status in results.items():
        icon = "✓" if status.startswith("PASS") else "✗" if status.startswith("FAIL") else "−"
        print(f"  {icon} {name}: {status}")
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
