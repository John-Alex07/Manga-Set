from __future__ import annotations

import shutil
from pathlib import Path

from ..registry import REFINER_REGISTRY
from ..types import GeneratedArtifact, GenerationResult
from .base import BaseRefiner


@REFINER_REGISTRY.register("flux")
class FluxRefiner(BaseRefiner):
    """
    Integration point for a FLUX-style refinement stage.

    The current implementation keeps the interface stable and applies a light
    post-processing pass so the experiment runner remains executable without an
    external FLUX service.
    """

    def refine(self, result: GenerationResult, context: dict[str, object]) -> GenerationResult:
        refined_artifacts: list[GeneratedArtifact] = []
        save_copy = self.config.params.get("save_copy", True)

        for artifact in result.artifacts:
            destination = self._destination_path(artifact.path, save_copy=save_copy)
            self._refine_artifact(artifact.path, destination)
            refined_artifacts.append(
                GeneratedArtifact(
                    path=destination,
                    kind=artifact.kind,
                    metadata={**artifact.metadata, "refined_from": str(artifact.path)},
                )
            )

        result.artifacts = refined_artifacts
        result.metadata["refinement_backend"] = "placeholder_flux"
        return result

    @staticmethod
    def _destination_path(path: Path, save_copy: bool) -> Path:
        if not save_copy:
            return path
        return path.with_name(f"{path.stem}_refined{path.suffix}")

    @staticmethod
    def _refine_artifact(source_path: Path, destination_path: Path) -> None:
        try:
            from PIL import Image, ImageFilter, ImageOps
        except ImportError:
            shutil.copy2(source_path, destination_path)
            return

        if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            shutil.copy2(source_path, destination_path)
            return

        source = Image.open(source_path).convert("RGB")
        refined = ImageOps.autocontrast(source).filter(ImageFilter.DETAIL)
        refined.save(destination_path)
