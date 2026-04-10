from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .types import GenerationRequest, GenerationResult


class FrameworkPipeline:
    def __init__(self, generator: Any, adapters: list[Any] | None = None, refiner: Any | None = None) -> None:
        self.generator = generator
        self.adapters = adapters or []
        self.refiner = refiner

    def run(self, request: GenerationRequest, output_dir: Path) -> GenerationResult:
        context: dict[str, Any] = {
            "applied_adapters": [],
            "output_dir": output_dir,
            "controls": dict(request.controls),
            "reference_images": list(request.reference_images),
        }

        working_request = request
        for adapter in self.adapters:
            start = time.perf_counter()
            working_request = adapter.before_generation(working_request, context)
            context.setdefault("timings", {})[f"adapter:{adapter.name}"] = time.perf_counter() - start

        generation_start = time.perf_counter()
        result = self.generator.generate(working_request, output_dir=output_dir, context=context)
        result.timings["generation"] = time.perf_counter() - generation_start
        result.metadata = {**working_request.metadata, **result.metadata}
        result.metadata["applied_adapters"] = context.get("applied_adapters", [])
        result.metadata["controls"] = context.get("controls", {})

        if self.refiner is not None:
            refine_start = time.perf_counter()
            result = self.refiner.refine(result, context=context)
            result.timings["refinement"] = time.perf_counter() - refine_start

        return result
