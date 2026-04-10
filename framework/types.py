from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GenerationRequest:
    prompt_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    output_name: str | None = None
    reference_images: list[Path] = field(default_factory=list)
    controls: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedArtifact:
    path: Path
    kind: str = "image"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class GenerationResult:
    prompt_id: str
    artifacts: list[GeneratedArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": self.metadata,
            "timings": self.timings,
            "scores": self.scores,
        }
