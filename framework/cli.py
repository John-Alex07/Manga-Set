from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import load_experiment_config
from .cache import prefetch_model_artifacts
from .environment import inspect_environment
from .experiments.runner import ExperimentRunner
from .analysis import build_consistency_rows, label_consistency_rows, score_clip_image_similarity
from .reporting import (
    flatten_results,
    format_ablation_rows,
    group_rows,
    load_report,
    summarize_rows,
    write_csv,
    write_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Manga-Set framework experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment configuration.")
    run_parser.add_argument("config", help="Path to a JSON experiment configuration file.")

    env_parser = subparsers.add_parser("env", help="Inspect runtime dependencies.")
    env_parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON output.")

    export_parser = subparsers.add_parser("export", help="Export experiment results to CSV and Markdown.")
    export_parser.add_argument("report", help="Path to experiment_report.json.")
    export_parser.add_argument(
        "--out-dir",
        default="exports",
        help="Output directory for CSV and Markdown exports.",
    )
    summary_parser = subparsers.add_parser("summary", help="Export a concise paper-facing Markdown summary table.")
    summary_parser.add_argument("report", help="Path to experiment_report.json.")
    summary_parser.add_argument(
        "--out-file",
        default="exports/summary.md",
        help="Output Markdown file for the summary table.",
    )
    summary_parser.add_argument(
        "--column",
        action="append",
        dest="columns",
        default=[],
        help="Optional summary column. May be passed multiple times.",
    )
    grouped_parser = subparsers.add_parser("grouped-summary", help="Export grouped paper-facing Markdown summary table.")
    grouped_parser.add_argument("report", help="Path to experiment_report.json.")
    grouped_parser.add_argument(
        "--group-by",
        action="append",
        dest="group_by",
        default=[],
        help="Column to group rows by. May be passed multiple times.",
    )
    grouped_parser.add_argument(
        "--default-group-by",
        default="metadata.suite",
        help="Fallback group column if --group-by is not provided.",
    )
    grouped_parser.add_argument(
        "--out-file",
        default="exports/grouped_summary.md",
        help="Output Markdown file for the grouped summary table.",
    )
    grouped_parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=[],
        help="Optional metric column to average. May be passed multiple times.",
    )
    compare_parser = subparsers.add_parser("compare-reports", help="Combine multiple reports into one grouped comparison table.")
    compare_parser.add_argument("reports", nargs="+", help="Paths to experiment_report.json files.")
    compare_parser.add_argument(
        "--group-by",
        action="append",
        dest="group_by",
        default=[],
        help="Column to group rows by.",
    )
    compare_parser.add_argument(
        "--default-group-by",
        default="metadata.suite",
        help="Fallback group column if --group-by is not provided.",
    )
    compare_parser.add_argument(
        "--out-file",
        default="exports/compare_reports.md",
        help="Output Markdown file for the comparison table.",
    )
    compare_parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=[],
        help="Optional metric column to average. May be passed multiple times.",
    )
    ablation_parser = subparsers.add_parser("ablation-table", help="Export a compact ablation table from labeled reports.")
    ablation_parser.add_argument(
        "inputs",
        nargs="+",
        help="Inputs in the form label=path/to/experiment_report.json",
    )
    ablation_parser.add_argument(
        "--out-file",
        default="exports/ablation_table.md",
        help="Output Markdown file for the ablation table.",
    )
    consistency_parser = subparsers.add_parser("consistency-table", help="Export CLIP image-image consistency table from reports.")
    consistency_parser.add_argument("reports", nargs="+", help="Paths to experiment_report.json files.")
    consistency_parser.add_argument(
        "--out-file",
        default="exports/consistency_table.md",
        help="Output Markdown file for the consistency table.",
    )
    consistency_parser.add_argument(
        "--group-pattern",
        default=r"^(.*?_panel)_\d+$",
        help="Regex used to infer consistency groups from prompt_id when metadata is absent.",
    )
    consistency_parser.add_argument(
        "--model-id",
        default="openai/clip-vit-base-patch32",
        help="CLIP model for image-image similarity.",
    )
    consistency_parser.add_argument(
        "--device",
        default="cpu",
        help="Device for CLIP consistency scoring.",
    )
    consistency_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local model files for CLIP consistency scoring.",
    )
    consistency_ablation_parser = subparsers.add_parser(
        "consistency-ablation-table",
        help="Export a compact labeled consistency table from reports.",
    )
    consistency_ablation_parser.add_argument(
        "inputs",
        nargs="+",
        help="Inputs in the form label=path/to/experiment_report.json",
    )
    consistency_ablation_parser.add_argument(
        "--out-file",
        default="exports/consistency_ablation_table.md",
        help="Output Markdown file for the labeled consistency table.",
    )
    consistency_ablation_parser.add_argument(
        "--group-pattern",
        default=r"^(.*?_panel)_\d+$",
        help="Regex used to infer consistency groups from prompt_id when metadata is absent.",
    )
    consistency_ablation_parser.add_argument(
        "--model-id",
        default="openai/clip-vit-base-patch32",
        help="CLIP model for image-image similarity.",
    )
    consistency_ablation_parser.add_argument(
        "--device",
        default="cpu",
        help="Device for CLIP consistency scoring.",
    )
    consistency_ablation_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local model files for CLIP consistency scoring.",
    )
    prefetch_parser = subparsers.add_parser("prefetch", help="Prefetch Hugging Face model artifacts into cache.")
    prefetch_parser.add_argument("model_id", help="Model repo ID to prefetch.")
    prefetch_parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=[],
        help="Optional allow pattern. May be passed multiple times.",
    )
    prefetch_parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional local target directory. Defaults to Hugging Face cache.",
    )
    prefetch_parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable to read a Hugging Face token from.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.command == "run":
        config = load_experiment_config(project_root / args.config)
        runner = ExperimentRunner(config=config, project_root=project_root)
        report = runner.run()
        print(json.dumps(report, indent=2))
        return

    if args.command == "env":
        report = inspect_environment()
        if args.json_output:
            print(json.dumps(report, indent=2))
        else:
            print(report["summary"])
            for item in report["dependencies"]:
                status = "OK" if item["available"] else "MISSING"
                print(f"- {item['name']}: {status} ({item['required_for']})")
        return

    if args.command == "export":
        report = load_report(project_root / args.report)
        rows = flatten_results(report)
        output_dir = project_root / args.out_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "results.csv"
        md_path = output_dir / "results.md"
        write_csv(rows, csv_path)
        write_markdown(rows, md_path)
        print(json.dumps({"csv": str(csv_path), "markdown": str(md_path)}, indent=2))
        return

    if args.command == "summary":
        report = load_report(project_root / args.report)
        rows = flatten_results(report)
        summary_rows = summarize_rows(rows, columns=args.columns or None)
        output_path = project_root / args.out_file
        write_markdown(summary_rows, output_path)
        print(json.dumps({"markdown": str(output_path)}, indent=2))
        return

    if args.command == "grouped-summary":
        report = load_report(project_root / args.report)
        rows = flatten_results(report)
        group_columns = args.group_by or [args.default_group_by]
        grouped_rows = group_rows(rows, group_by=group_columns, metrics=args.metrics or None)
        output_path = project_root / args.out_file
        write_markdown(grouped_rows, output_path)
        print(json.dumps({"markdown": str(output_path)}, indent=2))
        return

    if args.command == "compare-reports":
        rows: list[dict[str, Any]] = []
        for report_path in args.reports:
            report = load_report(project_root / report_path)
            flattened = flatten_results(report)
            experiment_name = report.get("experiment", {}).get("name", Path(report_path).stem)
            for row in flattened:
                row["experiment_name"] = experiment_name
            rows.extend(flattened)
        group_columns = args.group_by or ["experiment_name", args.default_group_by]
        grouped_rows = group_rows(rows, group_by=group_columns, metrics=args.metrics or None)
        output_path = project_root / args.out_file
        write_markdown(grouped_rows, output_path)
        print(json.dumps({"markdown": str(output_path)}, indent=2))
        return

    if args.command == "ablation-table":
        rows: list[dict[str, Any]] = []
        for raw_input in args.inputs:
            if "=" not in raw_input:
                raise ValueError("Each ablation input must be in the form label=report_path.")
            label, report_path = raw_input.split("=", 1)
            report = load_report(project_root / report_path)
            flattened = flatten_results(report)
            for row in flattened:
                row["condition_label"] = label
            rows.extend(flattened)
        ablation_rows = format_ablation_rows(rows, label_column="condition_label")
        output_path = project_root / args.out_file
        write_markdown(ablation_rows, output_path)
        print(json.dumps({"markdown": str(output_path)}, indent=2))
        return

    if args.command == "consistency-table":
        rows: list[dict[str, Any]] = []
        for report_path in args.reports:
            report = load_report(project_root / report_path)
            flattened = flatten_results(report)
            experiment_name = report.get("experiment", {}).get("name", Path(report_path).stem)
            for row in flattened:
                row["experiment_name"] = experiment_name
            for result, flat in zip(report.get("results", []), flattened):
                artifacts = result.get("artifacts", [])
                if artifacts:
                    flat["artifacts.0.path"] = artifacts[0].get("path", "")
            rows.extend(flattened)
        consistency_rows = build_consistency_rows(rows, group_pattern=args.group_pattern)
        scored_rows = score_clip_image_similarity(
            consistency_rows,
            model_id=args.model_id,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        output_path = project_root / args.out_file
        write_markdown(scored_rows, output_path)
        print(json.dumps({"markdown": str(output_path)}, indent=2))
        return

    if args.command == "consistency-ablation-table":
        rows: list[dict[str, Any]] = []
        for raw_input in args.inputs:
            if "=" not in raw_input:
                raise ValueError("Each consistency ablation input must be in the form label=report_path.")
            label, report_path = raw_input.split("=", 1)
            report = load_report(project_root / report_path)
            flattened = flatten_results(report)
            for row in flattened:
                row["experiment_name"] = label
            for result, flat in zip(report.get("results", []), flattened):
                artifacts = result.get("artifacts", [])
                if artifacts:
                    flat["artifacts.0.path"] = artifacts[0].get("path", "")
            rows.extend(flattened)
        consistency_rows = build_consistency_rows(rows, group_pattern=args.group_pattern)
        for row in consistency_rows:
            row["condition_label"] = row["experiment_name"]
        scored_rows = score_clip_image_similarity(
            consistency_rows,
            model_id=args.model_id,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        for row in scored_rows:
            row["condition_label"] = row["experiment_name"]
        labelled_rows = label_consistency_rows(scored_rows, label_field="condition_label")
        output_path = project_root / args.out_file
        write_markdown(labelled_rows, output_path)
        print(json.dumps({"markdown": str(output_path)}, indent=2))
        return

    if args.command == "prefetch":
        token = os.environ.get(args.token_env, "")
        result = prefetch_model_artifacts(
            args.model_id,
            local_dir=args.local_dir,
            allow_patterns=args.patterns,
            token=token or None,
        )
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
