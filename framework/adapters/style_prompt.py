from __future__ import annotations

from ..registry import ADAPTER_REGISTRY
from .base import BaseAdapter


@ADAPTER_REGISTRY.register("style_prompt")
class StylePromptAdapter(BaseAdapter):
    name = "style_prompt"

    def before_generation(self, request, context):
        request = super().before_generation(request, context)
        style_prompt = self.config.params.get("style_prompt", "")
        return self.append_to_prompt(request, style_prompt)
