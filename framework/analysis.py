from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


def infer_consistency_group(row: dict[str, Any], pattern: str = r"^(.*?_panel)_\d+$") -> str:
    metadata_group = row.get("metadata.character_group") or row.get("metadata.character")
    if metadata_group:
        return str(metadata_group)

    prompt_id = str(row.get("prompt_id", ""))
    match = re.match(pattern, prompt_id)
    if match:
        return match.group(1)
    return ""


def build_consistency_rows(
    rows: list[dict[str, Any]],
    *,
    experiment_label_field: str = "experiment_name",
    group_pattern: str = r"^(.*?_panel)_\d+$",
    image_path_field: str = "artifacts.0.path",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        experiment_name = str(row.get(experiment_label_field, ""))
        group_name = infer_consistency_group(row, pattern=group_pattern)
        if not group_name:
            continue
        grouped.setdefault((experiment_name, group_name), []).append(row)

    consistency_rows: list[dict[str, Any]] = []
    for (experiment_name, group_name), bucket in grouped.items():
        image_paths = [Path(str(row.get(image_path_field, ""))) for row in bucket if row.get(image_path_field)]
        consistency_rows.append(
            {
                "experiment_name": experiment_name,
                "consistency_group": group_name,
                "count": len(image_paths),
                "image_paths": image_paths,
            }
        )
    return consistency_rows


def score_clip_image_similarity(
    consistency_rows: list[dict[str, Any]],
    *,
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    if not consistency_rows:
        return []

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("consistency scoring requires torch, PIL, and transformers.") from exc

    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = CLIPModel.from_pretrained(model_id, local_files_only=local_files_only)
    model = model.to(device)
    model.eval()

    scored_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in consistency_rows:
            image_paths = [path for path in row["image_paths"] if path.exists()]
            if len(image_paths) < 2:
                scored_rows.append(
                    {
                        "experiment_name": row["experiment_name"],
                        "consistency_group": row["consistency_group"],
                        "count": len(image_paths),
                        "clip_image_similarity.avg": "",
                    }
                )
                continue

            embeddings = []
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.squeeze(0))

            pair_scores = []
            for left, right in combinations(embeddings, 2):
                pair_scores.append(float(torch.sum(left * right).item()))

            scored_rows.append(
                {
                    "experiment_name": row["experiment_name"],
                    "consistency_group": row["consistency_group"],
                    "count": len(image_paths),
                    "clip_image_similarity.avg": mean(pair_scores) if pair_scores else "",
                }
            )
    return scored_rows


def label_consistency_rows(
    rows: list[dict[str, Any]],
    *,
    label_field: str = "condition_label",
) -> list[dict[str, Any]]:
    labelled: list[dict[str, Any]] = []
    for row in rows:
        labelled.append(
            {
                "condition": row.get(label_field, row.get("experiment_name", "")),
                "consistency_group": row.get("consistency_group", ""),
                "count": row.get("count", ""),
                "clip_image_similarity.avg": row.get("clip_image_similarity.avg", ""),
            }
        )
    return labelled
