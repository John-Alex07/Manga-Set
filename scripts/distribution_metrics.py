"""Post-hoc distribution-level metrics: FID and CMMD.

Computes per-condition FID (Frechet Inception Distance) and CMMD (CLIP Maximum
Mean Discrepancy) using the *base* condition outputs as the reference
distribution.  This lets us measure how much each adapter shifts the image
distribution relative to the unconditioned baseline.

Usage:
    python scripts/distribution_metrics.py [--backbone sd21_s4] [--output results/distribution_metrics.json]

Requires:
    pip install torch torchvision transformers clean-fid  # FID
    pip install open_clip_torch                           # CMMD (inline impl)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

BACKBONE_CONFIGS: dict[str, dict[str, Any]] = {
    "sd21_s4": {
        "pattern": "expanded_{condition}_t{trial}",
        "conditions": ["base", "style_prompt", "lora_planogram", "lora_cartoon"],
        "trials": [0, 1, 2],
        "label": "SD 2.1 @ 4 steps",
    },
    "sd21_s20": {
        "pattern": "sd21_s20_{condition}_t{trial}",
        "conditions": ["base", "style_prompt", "lora_planogram", "lora_cartoon"],
        "trials": [0, 1, 2],
        "label": "SD 2.1 @ 20 steps",
    },
    "sdxl": {
        "pattern": "sdxl_{condition}_t{trial}",
        "conditions": ["base", "style_prompt", "lora_anime", "lora_manga"],
        "trials": [0, 1, 2],
        "label": "SDXL Turbo @ 4 steps",
    },
    "flux": {
        "pattern": "flux_{condition}_t{trial}",
        "conditions": ["base", "style_prompt"],
        "trials": [0],
        "label": "Flux Schnell @ 4 steps",
    },
}


def _collect_images(output_dir: Path) -> list[Path]:
    """Collect all generated .png images from a run directory."""
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.png"))


def collect_condition_images(
    backbone_key: str, condition: str,
) -> list[Path]:
    """Gather all images for a (backbone, condition) across trials."""
    cfg = BACKBONE_CONFIGS[backbone_key]
    images: list[Path] = []
    for trial in cfg["trials"]:
        run_name = cfg["pattern"].format(condition=condition, trial=trial)
        run_dir = OUTPUTS_DIR / run_name
        images.extend(_collect_images(run_dir))
    return images


# ---------- FID ----------


def compute_fid(ref_paths: list[Path], gen_paths: list[Path]) -> float:
    """Compute FID between two sets of images using clean-fid."""
    try:
        from cleanfid import fid
    except ImportError:
        logger.warning("clean-fid not installed; skipping FID.")
        return float("nan")

    import tempfile, shutil

    with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as gen_dir:
        for i, p in enumerate(ref_paths):
            shutil.copy2(p, Path(ref_dir) / f"{i:05d}.png")
        for i, p in enumerate(gen_paths):
            shutil.copy2(p, Path(gen_dir) / f"{i:05d}.png")
        return float(fid.compute_fid(ref_dir, gen_dir))


# ---------- CMMD (inline implementation) ----------


def _load_clip_embeddings(image_paths: list[Path], batch_size: int = 16) -> np.ndarray:
    """Extract CLIP image embeddings for a list of images."""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        logger.warning("transformers/torch not available; cannot compute CMMD.")
        return np.array([])

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()

    all_embeds: list[np.ndarray] = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeds = model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu().numpy())
    return np.concatenate(all_embeds, axis=0) if all_embeds else np.array([])


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian RBF kernel value between two vectors."""
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma ** 2)))


def compute_cmmd(ref_paths: list[Path], gen_paths: list[Path]) -> float:
    """Compute CMMD between reference and generated image distributions.

    Uses CLIP embeddings and an RBF-kernel Maximum Mean Discrepancy estimator.
    """
    ref_embeds = _load_clip_embeddings(ref_paths)
    gen_embeds = _load_clip_embeddings(gen_paths)
    if ref_embeds.size == 0 or gen_embeds.size == 0:
        return float("nan")

    n = len(ref_embeds)
    m = len(gen_embeds)

    sigma = 1.0

    kxx = sum(
        _rbf_kernel(ref_embeds[i], ref_embeds[j], sigma)
        for i in range(n) for j in range(i + 1, n)
    ) * 2.0 / max(n * (n - 1), 1)

    kyy = sum(
        _rbf_kernel(gen_embeds[i], gen_embeds[j], sigma)
        for i in range(m) for j in range(i + 1, m)
    ) * 2.0 / max(m * (m - 1), 1)

    kxy = sum(
        _rbf_kernel(ref_embeds[i], gen_embeds[j], sigma)
        for i in range(n) for j in range(m)
    ) / max(n * m, 1)

    mmd = kxx + kyy - 2 * kxy
    return float(mmd)


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute distribution-level metrics (FID, CMMD)")
    parser.add_argument(
        "--backbone", default="sd21_s4",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="Backbone config to analyze",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--all-backbones", action="store_true", help="Run all backbones")
    args = parser.parse_args()

    backbones = list(BACKBONE_CONFIGS.keys()) if args.all_backbones else [args.backbone]

    all_results: dict[str, Any] = {}

    for backbone_key in backbones:
        cfg = BACKBONE_CONFIGS[backbone_key]
        base_condition = cfg["conditions"][0]  # always "base"
        ref_images = collect_condition_images(backbone_key, base_condition)

        if not ref_images:
            logger.warning("No reference images found for %s/%s — skipping", backbone_key, base_condition)
            continue

        logger.info(
            "Backbone %s: %d reference images from '%s' condition",
            cfg["label"], len(ref_images), base_condition,
        )

        backbone_results: dict[str, dict[str, float]] = {}
        for condition in cfg["conditions"]:
            if condition == base_condition:
                continue
            gen_images = collect_condition_images(backbone_key, condition)
            if not gen_images:
                logger.warning("  No images for condition '%s' — skipping", condition)
                continue

            logger.info("  Condition '%s': %d images", condition, len(gen_images))
            fid_val = compute_fid(ref_images, gen_images)
            cmmd_val = compute_cmmd(ref_images, gen_images)
            backbone_results[condition] = {"fid": fid_val, "cmmd": cmmd_val}
            logger.info("    FID=%.4f  CMMD=%.6f", fid_val, cmmd_val)

        all_results[backbone_key] = {
            "label": cfg["label"],
            "reference_condition": base_condition,
            "reference_image_count": len(ref_images),
            "conditions": backbone_results,
        }

    output_path = args.output or str(
        Path(__file__).resolve().parents[1] / "results" / "distribution_metrics.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
