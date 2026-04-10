from __future__ import annotations

from dataclasses import replace

from ..registry import ADAPTER_REGISTRY
from .base import BaseAdapter


@ADAPTER_REGISTRY.register("controlnet")
class ControlNetAdapter(BaseAdapter):
    name = "controlnet"

    def before_generation(self, request, context):
        request = super().before_generation(request, context)
        control_hints = dict(context.get("controls", {}))
        control_hints.update(self.config.params.get("controls", {}))
        context["controls"] = control_hints
        control_prompt = self.config.params.get("prompt_hint", "")
        if control_prompt:
            request = self.append_to_prompt(request, control_prompt)
        return replace(request, controls=control_hints)
