from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..environment import inspect_environment
from ..pipeline import FrameworkPipeline
from ..registry import ADAPTER_REGISTRY, GENERATOR_REGISTRY, METRIC_REGISTRY, REFINER_REGISTRY
from ..types import GenerationRequest


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.output_root = (project_root / config.output_dir / config.name).resolve()

    def build_pipeline(self) -> FrameworkPipeline:
        generator = GENERATOR_REGISTRY.create(self.config.model.backend, self.config.model)
        adapters = [
            ADAPTER_REGISTRY.create(item.type, item)
            for item in self.config.adapters
            if item.enabled
        ]
        refiner = None
        if self.config.refinement and self.config.refinement.enabled:
            refiner = REFINER_REGISTRY.create(self.config.refinement.type, self.config.refinement)
        return FrameworkPipeline(generator=generator, adapters=adapters, refiner=refiner)

    def run(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        pipeline = self.build_pipeline()
        metrics = [
            METRIC_REGISTRY.create(name, self.config.evaluation.params.get(name, {}))
            for name in self.config.evaluation.metrics
        ]
        prompt_reports: list[dict[str, Any]] = []
        character_group_images: dict[str, list[Path]] = {}

        for index, prompt_cfg in enumerate(self.config.prompts, start=1):
            output_dir = self.output_root / f"{index:02d}_{prompt_cfg.id}"
            request = GenerationRequest(
                prompt_id=prompt_cfg.id,
                prompt=prompt_cfg.prompt,
                negative_prompt=prompt_cfg.negative_prompt,
                seed=prompt_cfg.seed if prompt_cfg.seed is not None else self.config.seed + index,
                guidance_scale=prompt_cfg.guidance_scale,
                num_inference_steps=prompt_cfg.num_inference_steps,
                output_name=prompt_cfg.output_name,
                reference_images=[self.project_root / path for path in prompt_cfg.reference_images],
                controls=prompt_cfg.controls,
                metadata=prompt_cfg.metadata,
            )
            result = pipeline.run(request, output_dir=output_dir)

            char_group = prompt_cfg.metadata.get("character_group", "")
            group_images = list(character_group_images.get(char_group, [])) if char_group else []
            metric_scores = self._evaluate(metrics, result, request, group_images=group_images)
            result.scores.update(metric_scores)
            prompt_reports.append(result.to_dict())

            if char_group:
                for artifact in result.artifacts:
                    if artifact.path.exists():
                        character_group_images.setdefault(char_group, []).append(artifact.path)

        report = {
            "experiment": self.config.to_dict(),
            "environment": inspect_environment(),
            "results": prompt_reports,
        }
        report_path = self.output_root / "experiment_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    def _evaluate(
        self,
        metrics: list[Any],
        result: Any,
        request: GenerationRequest,
        group_images: list[Path] | None = None,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "reference_images": request.reference_images,
            "controls": request.controls,
            "group_images": group_images or [],
        }
        scores: dict[str, Any] = {}
        for metric in metrics:
            metric_name = metric.__class__.__name__
            scores[metric_name] = metric.evaluate(result, context)
        return scores
