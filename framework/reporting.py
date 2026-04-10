from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("results", []):
        metadata = item.get("metadata", {})
        row = {
            "prompt_id": item.get("prompt_id", ""),
            "backend": metadata.get("backend", ""),
            "model_id": metadata.get("model_id", ""),
            "artifact_count": len(item.get("artifacts", [])),
            "generation_time": item.get("timings", {}).get("generation", ""),
            "refinement_time": item.get("timings", {}).get("refinement", ""),
        }
        _flatten_mapping(prefix="metadata", value=metadata, destination=row)

        for score_name, score_value in item.get("scores", {}).items():
            _flatten_mapping(prefix=score_name, value=score_value, destination=row)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _collect_fieldnames(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _collect_fieldnames(rows)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_rows(rows: list[dict[str, Any]], columns: list[str] | None = None) -> list[dict[str, Any]]:
    selected_columns = columns or default_summary_columns()
    summary: list[dict[str, Any]] = []
    for row in rows:
        summary.append({column: row.get(column, "") for column in selected_columns})
    return summary


def default_summary_columns() -> list[str]:
    return [
        "prompt_id",
        "backend",
        "model_id",
        "metadata.pipeline_class",
        "metadata.seed",
        "metadata.device",
        "metadata.suite",
        "metadata.baseline",
        "generation_time",
        "FileIntegrityMetric",
        "ImageStatisticsMetric.mean_brightness",
        "ImageStatisticsMetric.stddev_brightness",
        "CLIPTextAlignmentMetric",
        "CaptionKeywordRecallMetric",
    ]


def group_rows(
    rows: list[dict[str, Any]],
    *,
    group_by: str | list[str],
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    group_columns = [group_by] if isinstance(group_by, str) else list(group_by)
    metric_columns = metrics or [
        "generation_time",
        "FileIntegrityMetric",
        "ImageStatisticsMetric.mean_brightness",
        "ImageStatisticsMetric.stddev_brightness",
        "CLIPTextAlignmentMetric",
        "CaptionKeywordRecallMetric",
    ]
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in group_columns)
        buckets.setdefault(key, []).append(row)

    grouped: list[dict[str, Any]] = []
    for key_values, bucket in buckets.items():
        summary: dict[str, Any] = {
            "count": len(bucket),
        }
        for column, value in zip(group_columns, key_values):
            summary[column] = value
        for metric in metric_columns:
            numeric_values = [float(item[metric]) for item in bucket if _is_number(item.get(metric))]
            summary[f"{metric}.avg"] = mean(numeric_values) if numeric_values else ""
        grouped.append(summary)
    return grouped


def format_ablation_rows(
    rows: list[dict[str, Any]],
    *,
    label_column: str,
    suite_column: str = "metadata.suite",
    metrics: list[str] | None = None,
    include_overall: bool = True,
) -> list[dict[str, Any]]:
    metric_columns = metrics or [
        "generation_time",
        "CLIPTextAlignmentMetric",
        "CaptionKeywordRecallMetric",
        "ImageStatisticsMetric.mean_brightness",
    ]
    grouped = group_rows(
        rows,
        group_by=[label_column, suite_column],
        metrics=metric_columns,
    )
    ordered_rows: list[dict[str, Any]] = []
    for item in grouped:
        ordered_rows.append(
            {
                "condition": item.get(label_column, ""),
                "suite": item.get(suite_column, ""),
                "count": item.get("count", ""),
                "generation_time.avg": item.get("generation_time.avg", ""),
                "CLIPTextAlignmentMetric.avg": item.get("CLIPTextAlignmentMetric.avg", ""),
                "CaptionKeywordRecallMetric.avg": item.get("CaptionKeywordRecallMetric.avg", ""),
                "ImageStatisticsMetric.mean_brightness.avg": item.get("ImageStatisticsMetric.mean_brightness.avg", ""),
            }
        )

    if include_overall:
        overall_grouped = group_rows(rows, group_by=label_column, metrics=metric_columns)
        for item in overall_grouped:
            ordered_rows.append(
                {
                    "condition": item.get(label_column, ""),
                    "suite": "overall",
                    "count": item.get("count", ""),
                    "generation_time.avg": item.get("generation_time.avg", ""),
                    "CLIPTextAlignmentMetric.avg": item.get("CLIPTextAlignmentMetric.avg", ""),
                    "CaptionKeywordRecallMetric.avg": item.get("CaptionKeywordRecallMetric.avg", ""),
                    "ImageStatisticsMetric.mean_brightness.avg": item.get("ImageStatisticsMetric.mean_brightness.avg", ""),
                }
            )
    return ordered_rows


def _collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _flatten_mapping(prefix: str, value: Any, destination: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_mapping(next_prefix, nested_value, destination)
        return
    destination[prefix] = value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))
