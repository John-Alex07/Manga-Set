from __future__ import annotations

from ..registry import ADAPTER_REGISTRY
from .base import BaseAdapter


@ADAPTER_REGISTRY.register("lora")
class LoraAdapter(BaseAdapter):
    name = "lora"
