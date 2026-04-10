from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..types import GenerationRequest, GenerationResult


class BaseGenerator(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        output_dir: Path,
        context: dict[str, Any],
    ) -> GenerationResult:
        raise NotImplementedError
