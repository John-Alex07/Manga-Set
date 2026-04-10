from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import ModelConfig
from ..registry import GENERATOR_REGISTRY
from ..types import GeneratedArtifact, GenerationRequest, GenerationResult
from .base import BaseGenerator


@GENERATOR_REGISTRY.register("mock")
class MockGenerator(BaseGenerator):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.image_size = tuple(config.params.get("image_size", [768, 768]))

    def generate(
        self,
        request: GenerationRequest,
        output_dir: Path,
        context: dict[str, object],
    ) -> GenerationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = request.output_name or request.prompt_id
        artifact = self._create_artifact(safe_name=safe_name, request=request, output_dir=output_dir, context=context)

        return GenerationResult(
            prompt_id=request.prompt_id,
            artifacts=[artifact],
            metadata={
                "backend": "mock",
                "model_id": self.config.model_id,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
            },
        )

    @staticmethod
    def _background_color(prompt: str) -> tuple[int, int, int]:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        return (digest[0], digest[1], digest[2])

    def _create_artifact(
        self,
        safe_name: str,
        request: GenerationRequest,
        output_dir: Path,
        context: dict[str, object],
    ) -> GeneratedArtifact:
        description = "\n".join(
            [
                f"Prompt: {request.prompt}",
                f"Negative: {request.negative_prompt}",
                f"Seed: {request.seed}",
                f"Adapters: {', '.join(item['type'] for item in context.get('applied_adapters', [])) or 'none'}",
            ]
        )

        try:
            from PIL import Image, ImageDraw
        except ImportError:
            text_path = output_dir / f"{safe_name}.txt"
            text_path.write_text(description, encoding="utf-8")
            return GeneratedArtifact(path=text_path, kind="text")

        image_path = output_dir / f"{safe_name}.png"
        image = Image.new("RGB", self.image_size, self._background_color(request.prompt))
        draw = ImageDraw.Draw(image)
        draw.multiline_text((24, 24), description[:600], fill="white", spacing=8)
        image.save(image_path)
        return GeneratedArtifact(path=image_path)
