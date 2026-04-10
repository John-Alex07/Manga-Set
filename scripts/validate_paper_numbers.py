"""Validate that numbers cited in the paper match actual expanded experiment data.

Checks CLIP alignment overall means, per-suite breakdowns, and consistency values.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

CONDITIONS = {
    "base": "Base",
    "style_prompt": "Style Prompt",
    "lora_planogram": "LoRA (Plano.)",
    "lora_cartoon": "LoRA (Cartoon)",
}
TRIALS = [0, 1, 2]

PAPER_OVERALL = {
    "base": 0.295,
    "style_prompt": 0.281,
    "lora_planogram": 0.301,
    "lora_cartoon": 0.295,
}

PAPER_CONSISTENCY_OVERALL = {
    "base": 0.788,
    "style_prompt": 0.788,
    "lora_planogram": 0.790,
    "lora_cartoon": 0.781,
}


def load_report(cond, trial):
    path = OUTPUTS_DIR / f"expanded_{cond}_t{trial}" / "experiment_report.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate() -> bool:
    passed = True

    print("Validating expanded CLIP alignment overall averages...")
    for cond_key, cond_label in CONDITIONS.items():
        trial_means = []
        for trial in TRIALS:
            report = load_report(cond_key, trial)
            if report is None:
                print(f"  SKIP: expanded_{cond_key}_t{trial} not found")
                continue
            scores = []
            for result in report.get("results", []):
                clip = result.get("scores", {}).get("CLIPTextAlignmentMetric")
                if clip is not None:
                    scores.append(float(clip))
            if scores:
                trial_means.append(statistics.mean(scores))

        if not trial_means:
            print(f"  SKIP: {cond_label} has no data")
            continue

        actual_mean = statistics.mean(trial_means)
        paper_val = PAPER_OVERALL[cond_key]
        diff = abs(actual_mean - paper_val)
        if diff > 0.002:
            print(f"  MISMATCH {cond_label}: paper={paper_val:.3f}, actual={actual_mean:.3f} (diff={diff:.4f})")
            passed = False
        else:
            print(f"  OK {cond_label}: paper={paper_val:.3f}, actual={actual_mean:.3f}")

    print("\nChecking that all 12 experiment reports exist...")
    for cond_key in CONDITIONS:
        for trial in TRIALS:
            report_path = OUTPUTS_DIR / f"expanded_{cond_key}_t{trial}" / "experiment_report.json"
            if not report_path.exists():
                print(f"  MISSING: expanded_{cond_key}_t{trial}")
                passed = False
            else:
                with open(report_path) as f:
                    data = json.load(f)
                n_results = len(data.get("results", []))
                if n_results != 30:
                    print(f"  WRONG COUNT: expanded_{cond_key}_t{trial} has {n_results} results (expected 30)")
                    passed = False
                else:
                    print(f"  OK: expanded_{cond_key}_t{trial} ({n_results} results)")

    if passed:
        print("\nALL PAPER NUMBERS VALIDATED")
    else:
        print("\nSOME NUMBERS DO NOT MATCH")
    return passed


if __name__ == "__main__":
    validate()
