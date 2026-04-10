from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..registry import ADAPTER_REGISTRY
from .base import BaseAdapter


@ADAPTER_REGISTRY.register("ip_adapter")
class IPAdapter(BaseAdapter):
    name = "ip_adapter"

    def before_generation(self, request, context):
        request = super().before_generation(request, context)
        configured_refs = [Path(path) for path in self.config.params.get("reference_images", [])]
        reference_images = list(request.reference_images) + configured_refs
        identity_prompt = self.config.params.get("identity_prompt", "")
        if identity_prompt:
            request = self.append_to_prompt(request, identity_prompt)
        context["reference_images"] = reference_images
        return replace(request, reference_images=reference_images)
