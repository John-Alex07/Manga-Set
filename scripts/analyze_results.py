"""Statistical analysis of ablation experiment results.

Processes experiment_report.json files from the expanded ablation study.
Computes per-condition means, standard deviations, pairwise significance tests,
and metric agreement analysis.

Usage:
    python scripts/analyze_results.py [--existing]

    --existing: Analyze existing quintet/consistency reports (original 5-prompt data)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

QUINTET_REPORTS = {
    "Base": "benchmark_storyboard_sd21_quintet_base_clip",
    "Style Prompt": "benchmark_storyboard_sd21_quintet_clip",
    "LoRA (Planogram)": "benchmark_storyboard_sd21_true_lora_quintet_clip",
    "LoRA (Cartoon)": "benchmark_storyboard_sd21_true_lora_cartoon_quintet_clip",
}

CONSISTENCY_REPORTS = {
    "Base": "benchmark_storyboard_sd21_consistency_base_clip",
    "Style Prompt": "benchmark_storyboard_sd21_style_consistency_clip",
    "LoRA (Planogram)": "benchmark_storyboard_sd21_true_lora_planogram_consistency_clip",
    "LoRA (Cartoon)": "benchmark_storyboard_sd21_true_lora_cartoon_consistency_clip",
}

EXPANDED_CONDITIONS = {
    "Base": "base",
    "Style Prompt": "style_prompt",
    "LoRA (Planogram)": "lora_planogram",
    "LoRA (Cartoon)": "lora_cartoon",
}


def load_report(report_dir: str) -> dict | None:
    path = OUTPUTS_DIR / report_dir / "experiment_report.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_clip_scores(report: dict) -> dict[str, float]:
    scores = {}
    for result in report.get("results", []):
        pid = result["prompt_id"]
        clip_score = result.get("scores", {}).get("CLIPTextAlignmentMetric", None)
        if clip_score is not None:
            scores[pid] = float(clip_score)
    return scores


def extract_scores_by_suite(report: dict) -> dict[str, list[float]]:
    by_suite: dict[str, list[float]] = defaultdict(list)
    for result in report.get("results", []):
        suite = result.get("metadata", {}).get("suite", "unknown")
        clip_score = result.get("scores", {}).get("CLIPTextAlignmentMetric", None)
        if clip_score is not None:
            by_suite[suite].append(float(clip_score))
    return dict(by_suite)


def extract_latencies(report: dict) -> list[float]:
    latencies = []
    for result in report.get("results", []):
        timings = result.get("timings", {})
        gen_time = timings.get("generation", None)
        if gen_time is not None:
            latencies.append(float(gen_time))
    return latencies


def wilcoxon_test(x: list[float], y: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test. Returns (statistic, p-value)."""
    try:
        from scipy.stats import wilcoxon
        if len(x) != len(y) or len(x) < 5:
            return (float("nan"), float("nan"))
        stat, p = wilcoxon(x, y)
        return (float(stat), float(p))
    except ImportError:
        return (float("nan"), float("nan"))


def analyze_existing() -> None:
    """Analyze existing quintet and consistency data."""
    print("=" * 70)
    print("ANALYSIS OF EXISTING QUINTET ABLATION DATA")
    print("=" * 70)

    all_condition_scores: dict[str, list[float]] = {}
    all_condition_by_suite: dict[str, dict[str, list[float]]] = {}

    for cond_label, report_dir in QUINTET_REPORTS.items():
        report = load_report(report_dir)
        if report is None:
            print(f"  WARNING: {report_dir} not found")
            continue

        scores = extract_clip_scores(report)
        by_suite = extract_scores_by_suite(report)
        latencies = extract_latencies(report)

        all_values = list(scores.values())
        all_condition_scores[cond_label] = all_values
        all_condition_by_suite[cond_label] = by_suite

        print(f"\n{cond_label}:")
        print(f"  Overall CLIP: {mean(all_values):.4f} (n={len(all_values)})")
        for suite, suite_scores in sorted(by_suite.items()):
            print(f"  {suite}: {mean(suite_scores):.4f} (n={len(suite_scores)})")
        if latencies:
            print(f"  Latency: {mean(latencies):.1f}s avg")

    print("\n" + "-" * 70)
    print("PAIRWISE COMPARISONS (Wilcoxon signed-rank)")
    print("-" * 70)
    cond_labels = list(all_condition_scores.keys())
    for i, j in combinations(range(len(cond_labels)), 2):
        label_a, label_b = cond_labels[i], cond_labels[j]
        scores_a = all_condition_scores[label_a]
        scores_b = all_condition_scores[label_b]
        if len(scores_a) == len(scores_b):
            stat, p = wilcoxon_test(scores_a, scores_b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {label_a} vs {label_b}: W={stat:.1f}, p={p:.4f} {sig}")
        else:
            print(f"  {label_a} vs {label_b}: SKIPPED (unequal n)")

    print("\n" + "=" * 70)
    print("CONSISTENCY DATA")
    print("=" * 70)
    for cond_label, report_dir in CONSISTENCY_REPORTS.items():
        report = load_report(report_dir)
        if report is None:
            print(f"  WARNING: {report_dir} not found")
            continue
        scores = extract_clip_scores(report)
        print(f"\n{cond_label}:")
        for pid, score in sorted(scores.items()):
            group = "hiro" if "hiro" in pid else "akira" if "akira" in pid else "unknown"
            print(f"  {pid} ({group}): {score:.4f}")


def analyze_expanded() -> None:
    """Analyze expanded multi-trial data."""
    print("=" * 70)
    print("ANALYSIS OF EXPANDED 30-PROMPT MULTI-TRIAL DATA")
    print("=" * 70)

    for cond_label, cond_key in EXPANDED_CONDITIONS.items():
        trial_scores: list[list[float]] = []
        for trial in range(3):
            report_dir = f"expanded_{cond_key}_t{trial}"
            report = load_report(report_dir)
            if report is None:
                continue
            scores = list(extract_clip_scores(report).values())
            trial_scores.append(scores)

        if not trial_scores:
            print(f"\n{cond_label}: No expanded data available")
            continue

        all_means = [mean(ts) for ts in trial_scores]
        print(f"\n{cond_label} ({len(trial_scores)} trials):")
        for i, ts in enumerate(trial_scores):
            print(f"  Trial {i}: {mean(ts):.4f} (n={len(ts)})")
        if len(all_means) >= 2:
            print(f"  Mean of means: {mean(all_means):.4f} +/- {stdev(all_means):.4f}")
        else:
            print(f"  Mean: {all_means[0]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", action="store_true", help="Analyze existing data")
    args = parser.parse_args()

    if args.existing:
        analyze_existing()
    else:
        analyze_expanded()
        print("\n\nFalling back to existing data analysis:")
        analyze_existing()


if __name__ == "__main__":
    main()
