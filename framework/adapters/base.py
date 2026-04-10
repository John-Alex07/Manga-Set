from __future__ import annotations

from abc import ABC
from dataclasses import replace
from typing import Any

from ..config import AdapterConfig
from ..types import GenerationRequest


class BaseAdapter(ABC):
    name = "adapter"

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    def before_generation(
        self,
        request: GenerationRequest,
        context: dict[str, Any],
    ) -> GenerationRequest:
        context.setdefault("applied_adapters", []).append(
            {
                "type": self.name,
                "weight": self.config.weight,
                "params": self.config.params,
            }
        )
        return request

    @staticmethod
    def append_to_prompt(request: GenerationRequest, text: str) -> GenerationRequest:
        if not text:
            return request
        prompt = f"{request.prompt}, {text}" if request.prompt else text
        return replace(request, prompt=prompt)
