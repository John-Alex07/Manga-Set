from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from typing import Any

from ..registry import METRIC_REGISTRY
from ..types import GenerationResult


class BaseMetric(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float | dict[str, float]:
        raise NotImplementedError


@METRIC_REGISTRY.register("file_integrity")
class FileIntegrityMetric(BaseMetric):
    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float:
        if not result.artifacts:
            return 0.0
        valid = [artifact.path.exists() and artifact.path.stat().st_size > 0 for artifact in result.artifacts]
        return sum(valid) / len(valid)


@METRIC_REGISTRY.register("image_statistics")
class ImageStatisticsMetric(BaseMetric):
    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> dict[str, float]:
        artifact_paths = [artifact.path for artifact in result.artifacts if artifact.path.exists()]
        if not artifact_paths:
            return {"artifact_count": 0.0, "mean_bytes": 0.0}

        try:
            from PIL import Image, ImageStat
        except ImportError:
            sizes = [float(path.stat().st_size) for path in artifact_paths]
            return {"artifact_count": float(len(artifact_paths)), "mean_bytes": mean(sizes)}

        image_paths = [path for path in artifact_paths if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not image_paths:
            sizes = [float(path.stat().st_size) for path in artifact_paths]
            return {"artifact_count": float(len(artifact_paths)), "mean_bytes": mean(sizes)}

        brightness_means: list[float] = []
        brightness_stddevs: list[float] = []
        for path in image_paths:
            grayscale = Image.open(path).convert("L")
            stat = ImageStat.Stat(grayscale)
            brightness_means.append(stat.mean[0])
            brightness_stddevs.append(stat.stddev[0])
        return {"mean_brightness": mean(brightness_means), "stddev_brightness": mean(brightness_stddevs)}


@METRIC_REGISTRY.register("histogram_consistency")
class HistogramConsistencyMetric(BaseMetric):
    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float:
        artifact_paths = [artifact.path for artifact in result.artifacts if artifact.path.exists()]
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            return self._size_similarity(artifact_paths)

        image_paths = [
            artifact.path
            for artifact in result.artifacts
            if artifact.kind == "image" and artifact.path.exists() and artifact.path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        reference_paths = [Path(path) for path in context.get("reference_images", []) if Path(path).exists()]
        group_paths = [Path(p) for p in context.get("group_images", []) if Path(p).exists()]

        candidates = image_paths + reference_paths + group_paths
        if len(candidates) < 2:
            return self._size_similarity(artifact_paths)

        base_hist = self._normalized_histogram(candidates[0])
        similarities = []
        for path in candidates[1:]:
            other_hist = self._normalized_histogram(path)
            similarities.append(sum(min(a, b) for a, b in zip(base_hist, other_hist)))
        return mean(similarities) if similarities else 1.0

    @staticmethod
    def _normalized_histogram(path: Path) -> list[float]:
        from PIL import Image

        histogram = Image.open(path).convert("RGB").histogram()
        total = math.fsum(histogram) or 1.0
        return [value / total for value in histogram]

    @staticmethod
    def _size_similarity(paths: list[Path]) -> float:
        if len(paths) < 2:
            return 1.0
        sizes = [path.stat().st_size for path in paths]
        baseline = max(sizes) or 1
        distances = [1.0 - abs(baseline - size) / baseline for size in sizes[1:]]
        return mean(distances) if distances else 1.0


@METRIC_REGISTRY.register("latency")
class LatencyMetric(BaseMetric):
    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> dict[str, float]:
        return dict(result.timings)


@METRIC_REGISTRY.register("clip_text_alignment")
class CLIPTextAlignmentMetric(BaseMetric):
    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float:
        image_paths = [
            artifact.path
            for artifact in result.artifacts
            if artifact.kind == "image" and artifact.path.exists() and artifact.path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not image_paths:
            return 0.0

        prompt = str(result.metadata.get("prompt", ""))
        if not prompt:
            return 0.0

        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("clip_text_alignment requires torch, PIL, and transformers.") from exc

        model_id = self.params.get("model_id", "openai/clip-vit-base-patch32")
        device = self.params.get("device", "cpu")
        local_files_only = bool(self.params.get("local_files_only", False))
        revision = self.params.get("revision")

        load_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if revision:
            load_kwargs["revision"] = revision
        model = CLIPModel.from_pretrained(model_id, **load_kwargs)
        processor = CLIPProcessor.from_pretrained(model_id, **load_kwargs)
        model = model.to(device)
        model.eval()

        scores: list[float] = []
        with torch.no_grad():
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                score = torch.sum(image_embeds * text_embeds, dim=-1).item()
                scores.append(float(score))
        return mean(scores) if scores else 0.0


@METRIC_REGISTRY.register("caption_keyword_recall")
class CaptionKeywordRecallMetric(BaseMetric):
    STOPWORDS = {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "with", "for", "from",
        "by", "is", "are", "was", "were", "be", "being", "been", "as", "that", "this", "these",
        "those", "it", "its", "into", "through", "around", "over", "under", "same"
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._model = None
        self._processor = None

    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float:
        image_paths = [
            artifact.path
            for artifact in result.artifacts
            if artifact.kind == "image" and artifact.path.exists() and artifact.path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not image_paths:
            return 0.0

        prompt = str(result.metadata.get("prompt", ""))
        prompt_keywords = self._extract_keywords(prompt)
        if not prompt_keywords:
            return 0.0

        try:
            import torch
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError("caption_keyword_recall requires torch, PIL, and transformers.") from exc

        model_id = self.params.get("model_id", "Salesforce/blip-image-captioning-base")
        device = self.params.get("device", "cpu")
        local_files_only = bool(self.params.get("local_files_only", False))
        max_new_tokens = int(self.params.get("max_new_tokens", 30))

        if self._model is None or self._processor is None:
            self._processor = BlipProcessor.from_pretrained(model_id, local_files_only=local_files_only)
            self._model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=local_files_only)
            self._model = self._model.to(device)
            self._model.eval()

        scores: list[float] = []
        with torch.no_grad():
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
                caption = self._processor.decode(output[0], skip_special_tokens=True)
                caption_keywords = self._extract_keywords(caption)
                matched = prompt_keywords.intersection(caption_keywords)
                scores.append(len(matched) / len(prompt_keywords))
        return mean(scores) if scores else 0.0

    @classmethod
    def _extract_keywords(cls, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
        return {token for token in tokens if token not in cls.STOPWORDS and len(token) > 2}


@METRIC_REGISTRY.register("lpips_consistency")
class LPIPSConsistencyMetric(BaseMetric):
    """Compute LPIPS perceptual distance between generated image and reference/group images.

    Lower LPIPS = more similar. We return 1 - LPIPS so that higher = more consistent,
    matching the direction of other similarity metrics.

    When no explicit reference images are provided, falls back to ``group_images``
    from the evaluation context (populated by the runner for character-group prompts).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._loss_fn = None

    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float:
        image_paths = [
            artifact.path
            for artifact in result.artifacts
            if artifact.kind == "image"
            and artifact.path.exists()
            and artifact.path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        reference_paths = [
            Path(p) for p in context.get("reference_images", []) if Path(p).exists()
        ]
        group_paths = [
            Path(p) for p in context.get("group_images", []) if Path(p).exists()
        ]

        candidates = image_paths + reference_paths + group_paths
        if len(candidates) < 2:
            return 1.0

        try:
            import lpips
            import torch
            from PIL import Image
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "lpips_consistency requires lpips, torch, PIL, and torchvision."
            ) from exc

        device = self.params.get("device", "cpu")
        if self._loss_fn is None:
            self._loss_fn = lpips.LPIPS(net="alex").to(device)
            self._loss_fn.eval()

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        tensors = []
        for path in candidates:
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img).unsqueeze(0).to(device))

        distances: list[float] = []
        with torch.no_grad():
            base = tensors[0]
            for other in tensors[1:]:
                dist = self._loss_fn(base, other).item()
                distances.append(dist)

        avg_distance = mean(distances) if distances else 0.0
        return max(0.0, 1.0 - avg_distance)


@METRIC_REGISTRY.register("image_reward")
class ImageRewardMetric(BaseMetric):
    """Score each generation using ImageReward, a learned human-preference model.

    Returns a scalar reward score per image (higher = better alignment with
    human preferences).  Requires ``pip install image-reward``.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._model: Any = None

    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float:
        image_paths = [
            artifact.path
            for artifact in result.artifacts
            if artifact.kind == "image"
            and artifact.path.exists()
            and artifact.path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not image_paths:
            return 0.0

        prompt = str(result.metadata.get("prompt", ""))
        if not prompt:
            return 0.0

        try:
            import ImageReward as RM  # noqa: N813
        except ImportError as exc:
            raise RuntimeError(
                "image_reward metric requires the image-reward package. "
                "Install with: pip install image-reward"
            ) from exc

        if self._model is None:
            model_name = self.params.get("model_name", "ImageReward-v1.0")
            self._model = RM.load(model_name)

        scores: list[float] = []
        for path in image_paths:
            reward = self._model.score(prompt, [str(path)])
            scores.append(float(reward))
        return mean(scores) if scores else 0.0


@METRIC_REGISTRY.register("dino_similarity")
class DINOSimilarityMetric(BaseMetric):
    """Compute DINO feature similarity between generated image and reference/group images.

    Uses a DINO ViT model to extract CLS token features and compute cosine similarity.

    When no explicit reference images are provided, falls back to ``group_images``
    from the evaluation context (populated by the runner for character-group prompts).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self._model = None
        self._processor = None

    def evaluate(self, result: GenerationResult, context: dict[str, Any]) -> float:
        image_paths = [
            artifact.path
            for artifact in result.artifacts
            if artifact.kind == "image"
            and artifact.path.exists()
            and artifact.path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        reference_paths = [
            Path(p) for p in context.get("reference_images", []) if Path(p).exists()
        ]
        group_paths = [
            Path(p) for p in context.get("group_images", []) if Path(p).exists()
        ]

        candidates = image_paths + reference_paths + group_paths
        if len(candidates) < 2:
            return 1.0

        try:
            import torch
            from PIL import Image
            from transformers import ViTFeatureExtractor, ViTModel
        except ImportError as exc:
            raise RuntimeError(
                "dino_similarity requires torch, PIL, and transformers."
            ) from exc

        model_id = self.params.get("model_id", "facebook/dino-vits16")
        device = self.params.get("device", "cpu")
        local_files_only = bool(self.params.get("local_files_only", False))
        revision = self.params.get("revision")

        if self._model is None or self._processor is None:
            load_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
            if revision:
                load_kwargs["revision"] = revision
            self._processor = ViTFeatureExtractor.from_pretrained(
                model_id, **load_kwargs
            )
            self._model = ViTModel.from_pretrained(
                model_id, **load_kwargs
            )
            self._model = self._model.to(device)
            self._model.eval()

        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for path in candidates:
                image = Image.open(path).convert("RGB")
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                cls_embed = outputs.last_hidden_state[:, 0, :]
                cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
                embeddings.append(cls_embed)

        similarities: list[float] = []
        base_embed = embeddings[0]
        for other_embed in embeddings[1:]:
            sim = torch.sum(base_embed * other_embed, dim=-1).item()
            similarities.append(float(sim))

        return mean(similarities) if similarities else 1.0
