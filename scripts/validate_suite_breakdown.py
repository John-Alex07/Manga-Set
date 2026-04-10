"""Validate per-suite CLIP scores in the expanded ablation table."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

CONDITIONS = {
    "base": "Base",
    "style_prompt": "Style Prompt",
    "lora_planogram": "LoRA (Plano.)",
    "lora_cartoon": "LoRA (Cartoon)",
}
TRIALS = [0, 1, 2]

PAPER_SUITE_VALUES = {
    ("base", "alignment"): 0.287,
    ("base", "consistency"): 0.292,
    ("base", "story"): 0.299,
    ("base", "style_fidelity"): 0.308,
    ("style_prompt", "alignment"): 0.292,
    ("style_prompt", "consistency"): 0.266,
    ("style_prompt", "story"): 0.289,
    ("style_prompt", "style_fidelity"): 0.290,
    ("lora_planogram", "alignment"): 0.300,
    ("lora_planogram", "consistency"): 0.301,
    ("lora_planogram", "story"): 0.296,
    ("lora_planogram", "style_fidelity"): 0.309,
    ("lora_cartoon", "alignment"): 0.300,
    ("lora_cartoon", "consistency"): 0.291,
    ("lora_cartoon", "story"): 0.291,
    ("lora_cartoon", "style_fidelity"): 0.303,
}


def load_report(cond, trial):
    path = OUTPUTS_DIR / f"expanded_{cond}_t{trial}" / "experiment_report.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate() -> bool:
    passed = True
    print("Validating per-suite CLIP alignment (expanded data)...")

    for cond_key, cond_label in CONDITIONS.items():
        for suite in ["alignment", "consistency", "story", "style_fidelity"]:
            trial_means = []
            for trial in TRIALS:
                report = load_report(cond_key, trial)
                scores = []
                for r in report.get("results", []):
                    meta = r.get("metadata", {})
                    if meta.get("suite") == suite:
                        clip = r.get("scores", {}).get("CLIPTextAlignmentMetric")
                        if clip is not None:
                            scores.append(float(clip))
                if scores:
                    trial_means.append(statistics.mean(scores))

            if not trial_means:
                print(f"  SKIP: {cond_label}/{suite} has no data")
                continue

            actual_mean = statistics.mean(trial_means)
            paper_val = PAPER_SUITE_VALUES.get((cond_key, suite))
            if paper_val is None:
                continue

            diff = abs(actual_mean - paper_val)
            ok = diff < 0.002
            status = "OK" if ok else "MISMATCH"
            if not ok:
                passed = False
            print(f"  {status} {cond_label}/{suite}: paper={paper_val:.3f}, actual={actual_mean:.3f}")

    if passed:
        print("\nALL SUITE VALUES VALIDATED")
    else:
        print("\nSOME VALUES MISMATCH")
    return passed


if __name__ == "__main__":
    validate()
