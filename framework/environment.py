from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DependencyStatus:
    name: str
    available: bool
    version: str
    required_for: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "required_for": self.required_for,
        }


CORE_DEPENDENCIES = [
    ("torch", "real local diffusion backends"),
    ("diffusers", "real local diffusion backends"),
    ("transformers", "text encoder and model loading"),
    ("PIL", "image save/load and qualitative outputs"),
    ("peft", "LoRA adapter loading"),
    ("safetensors", "safe model weight loading"),
    ("accelerate", "pipeline device management"),
    ("lpips", "perceptual similarity metric"),
    ("scipy", "statistical analysis"),
    ("huggingface_hub", "model artifact management"),
    ("torchvision", "image transforms for metrics"),
    ("streamlit", "interactive framework demo"),
]


def _get_module_version(name: str) -> str:
    """Return the installed version string for a module, or '' if unavailable."""
    import_name = name
    # PIL's version lives under the 'PIL' package but the dist is 'Pillow'
    if name == "PIL":
        import_name = "PIL"
    try:
        mod = __import__(import_name)
        for attr in ("__version__", "VERSION", "version"):
            v = getattr(mod, attr, None)
            if isinstance(v, str):
                return v
        # fallback: importlib.metadata
        dist_name = "Pillow" if name == "PIL" else name
        from importlib.metadata import version as dist_version
        return dist_version(dist_name)
    except Exception:
        return ""


def _inspect_gpu() -> dict[str, Any]:
    """Detect CUDA/GPU availability and basic device properties."""
    info: dict[str, Any] = {"cuda_available": False, "device_count": 0, "devices": []}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            count = torch.cuda.device_count()
            info["device_count"] = count
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_mb": round(props.total_mem / 1024 / 1024),
                    "major": props.major,
                    "minor": props.minor,
                })
            info["cuda_version"] = torch.version.cuda or ""
    except Exception:
        pass
    return info


def inspect_environment() -> dict[str, Any]:
    dependencies = [
        DependencyStatus(
            name=name,
            available=_has_module(name),
            version=_get_module_version(name) if _has_module(name) else "",
            required_for=required_for,
        )
        for name, required_for in CORE_DEPENDENCIES
    ]
    missing = [item.name for item in dependencies if not item.available]
    runtime_ready = not missing

    return {
        "runtime_ready": runtime_ready,
        "platform": {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
        },
        "gpu": _inspect_gpu(),
        "dependencies": [item.to_dict() for item in dependencies],
        "missing": missing,
        "summary": _build_summary(runtime_ready=runtime_ready, missing=missing),
    }


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _build_summary(runtime_ready: bool, missing: list[str]) -> str:
    if runtime_ready:
        return "Runtime is ready for local diffusion experiments."
    missing_text = ", ".join(missing)
    return f"Runtime is not ready for local diffusion experiments. Missing: {missing_text}."
