from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import RefinementConfig
from ..types import GenerationResult


class BaseRefiner(ABC):
    def __init__(self, config: RefinementConfig) -> None:
        self.config = config

    @abstractmethod
    def refine(self, result: GenerationResult, context: dict[str, Any]) -> GenerationResult:
        raise NotImplementedError
