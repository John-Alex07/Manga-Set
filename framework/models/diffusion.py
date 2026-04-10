from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..registry import GENERATOR_REGISTRY
from ..types import GeneratedArtifact, GenerationRequest, GenerationResult
from .base import BaseGenerator

logger = logging.getLogger(__name__)

_FLUX_PIPELINE_NAMES = {"FluxPipeline", "FluxImg2ImgPipeline"}


@GENERATOR_REGISTRY.register("diffusers")
class DiffusersGenerator(BaseGenerator):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._pipeline = None
        self._pipeline_class_name: str = ""

    @property
    def is_flux(self) -> bool:
        return self._pipeline_class_name in _FLUX_PIPELINE_NAMES

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise RuntimeError("Diffusers backend requires torch and diffusers to be installed.") from exc

        dtype = getattr(torch, self.config.dtype, torch.float32)
        pipeline_params = dict(self.config.params)
        scheduler_name = pipeline_params.pop("scheduler", "")
        use_safetensors = pipeline_params.pop("use_safetensors", True)
        variant = pipeline_params.pop("variant", None)
        disable_safety_checker = pipeline_params.pop("disable_safety_checker", False)
        controlnet_id = pipeline_params.pop("controlnet_id", None)
        controlnet_revision = pipeline_params.pop("controlnet_revision", None)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": use_safetensors,
            **pipeline_params,
        }
        if variant is not None:
            load_kwargs["variant"] = variant
        if self.config.revision:
            load_kwargs["revision"] = self.config.revision

        if controlnet_id:
            pipeline = self._load_controlnet_pipeline(
                controlnet_id, controlnet_revision, dtype, load_kwargs,
            )
        else:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                self.config.model_id,
                **load_kwargs,
            )

        self._pipeline_class_name = pipeline.__class__.__name__
        logger.info("Loaded pipeline: %s", self._pipeline_class_name)

        if disable_safety_checker and hasattr(pipeline, "safety_checker"):
            pipeline.safety_checker = None
            if hasattr(pipeline, "requires_safety_checker"):
                pipeline.requires_safety_checker = False

        if scheduler_name:
            self._set_scheduler(pipeline, scheduler_name)

        pipeline = pipeline.to(self.config.device)
        self._pipeline = pipeline
        return pipeline

    def _load_controlnet_pipeline(
        self,
        controlnet_id: str,
        controlnet_revision: str | None,
        dtype: Any,
        load_kwargs: dict[str, Any],
    ) -> Any:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        cn_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if controlnet_revision:
            cn_kwargs["revision"] = controlnet_revision
        controlnet = ControlNetModel.from_pretrained(controlnet_id, **cn_kwargs)
        load_kwargs["controlnet"] = controlnet
        return StableDiffusionControlNetPipeline.from_pretrained(
            self.config.model_id, **load_kwargs,
        )

    @staticmethod
    def _set_scheduler(pipeline: Any, scheduler_name: str) -> None:
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
        )

        name = scheduler_name.lower()
        sched_config = pipeline.scheduler.config
        if name == "euler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(sched_config)
        elif name == "dpm":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sched_config)
        elif name == "euler_ancestral":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(sched_config)
        elif name in ("flow_match_euler", "default"):
            pass  # keep the pipeline's default scheduler
        else:
            raise ValueError(f"Unsupported scheduler {scheduler_name!r}.")

    @staticmethod
    def _enforce_determinism() -> None:
        """Set global torch flags that maximize reproducibility."""
        import torch

        torch.manual_seed(0)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def generate(
        self,
        request: GenerationRequest,
        output_dir: Path,
        context: dict[str, Any],
    ) -> GenerationResult:
        import torch

        self._enforce_determinism()
        pipeline = self._load_pipeline()
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = request.output_name or request.prompt_id
        image_path = output_dir / f"{safe_name}.png"

        generator_device = "cpu" if self.config.device == "mps" else self.config.device
        generator = torch.Generator(device=generator_device)
        generator = generator.manual_seed(request.seed)
        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "generator": generator,
        }

        if not self.is_flux and request.negative_prompt:
            call_kwargs["negative_prompt"] = request.negative_prompt

        control_image = self._resolve_control_image(request, context)
        if control_image is not None:
            call_kwargs["image"] = control_image
        elif request.reference_images and "image" in pipeline.__call__.__code__.co_varnames:
            call_kwargs["image"] = request.reference_images[0]

        self._apply_runtime_adapters(pipeline, context=context)
        image = pipeline(**call_kwargs).images[0]
        image.save(image_path)

        return GenerationResult(
            prompt_id=request.prompt_id,
            artifacts=[GeneratedArtifact(path=image_path)],
            metadata={
                "backend": "diffusers",
                "model_id": self.config.model_id,
                "pipeline_class": pipeline.__class__.__name__,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "device": self.config.device,
                "applied_adapters": context.get("applied_adapters", []),
            },
        )

    @staticmethod
    def _resolve_control_image(
        request: GenerationRequest, context: dict[str, Any],
    ) -> Any:
        """Build a ControlNet conditioning image from controls context.

        If a source image is provided, Canny edge detection is applied.
        Otherwise, a synthetic edge map is generated from Gaussian noise,
        which allows the ControlNet pipeline to run without external assets.
        """
        controls = context.get("controls", {})
        control_type = controls.get("type", "")
        if control_type != "canny":
            return None

        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            logger.warning("cv2 not available; skipping ControlNet canny preprocessing")
            return None

        low = int(controls.get("low_threshold", 100))
        high = int(controls.get("high_threshold", 200))

        source_path = controls.get("source_image")
        if source_path:
            img = np.array(Image.open(source_path).convert("RGB"))
        else:
            rng = np.random.RandomState(request.seed)
            img = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 0)

        edges = cv2.Canny(img, low, high)
        return Image.fromarray(np.stack([edges] * 3, axis=-1))

    def _apply_runtime_adapters(self, pipeline: Any, context: dict[str, Any]) -> None:
        adapters = context.get("applied_adapters", [])
        if not adapters:
            return

        if hasattr(pipeline, "unload_lora_weights"):
            try:
                pipeline.unload_lora_weights()
            except Exception:
                pass

        for index, adapter in enumerate(adapters):
            if adapter.get("type") != "lora":
                continue
            params = adapter.get("params", {})
            weights_path = params.get("weights_path") or params.get("lora_path")
            if not weights_path:
                continue

            adapter_name = params.get("adapter_name", f"lora_{index}")
            adapter_scale = float(params.get("adapter_scale", adapter.get("weight", 1.0)))
            revision = params.get("revision")
            if hasattr(pipeline, "load_lora_weights"):
                self._load_lora_with_fallback(
                    pipeline, weights_path, adapter_name, adapter_scale, revision=revision
                )

    def _load_lora_with_fallback(
        self,
        pipeline: Any,
        weights_path: str,
        adapter_name: str,
        adapter_scale: float,
        *,
        revision: str | None = None,
    ) -> None:
        """Load LoRA weights, handling legacy key formats via attn_procs fallback."""

        # Primary path: modern diffusers load_lora_weights API
        try:
            load_kwargs: dict[str, Any] = {"adapter_name": adapter_name}
            if revision:
                load_kwargs["revision"] = revision
            pipeline.load_lora_weights(weights_path, **load_kwargs)
            if hasattr(pipeline, "set_adapters"):
                pipeline.set_adapters([adapter_name], adapter_weights=[adapter_scale])
            logger.info("LoRA '%s' loaded via primary path (load_lora_weights)", adapter_name)
            return
        except Exception as exc:
            logger.warning(
                "Primary LoRA load failed for '%s': %s. Trying attn_procs fallback.",
                adapter_name,
                exc,
            )

        # Fallback path: resolve raw state dict and load via legacy attn_procs API
        state_dict = self._resolve_lora_state_dict(weights_path, revision=revision)
        if state_dict is None:
            logger.error("LoRA '%s': could not resolve state dict from '%s'", adapter_name, weights_path)
            return

        pipeline.unet.load_attn_procs(state_dict)

        # Apply adapter scale via manual weight scaling on the loaded attn processors
        if adapter_scale != 1.0:
            for key, proc in pipeline.unet.attn_processors.items():
                for param in getattr(proc, "parameters", lambda: [])():
                    param.data.mul_(adapter_scale)

        logger.info(
            "LoRA '%s' loaded via fallback path (load_attn_procs, scale=%.2f)",
            adapter_name,
            adapter_scale,
        )

    @staticmethod
    def _resolve_lora_state_dict(weights_path: str, revision: str | None = None) -> dict | None:
        """Load a LoRA state dict from a local file or HuggingFace repo."""
        import torch
        from pathlib import Path as _Path

        p = _Path(weights_path)
        if p.exists() and p.is_file():
            if p.suffix == ".safetensors":
                from safetensors.torch import load_file
                return load_file(str(p))
            return torch.load(str(p), map_location="cpu", weights_only=True)

        try:
            from huggingface_hub import hf_hub_download
            dl_kwargs: dict[str, Any] = {}
            if revision:
                dl_kwargs["revision"] = revision
            for fname in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"):
                try:
                    local = hf_hub_download(weights_path, filename=fname, **dl_kwargs)
                    if local.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        return load_file(local)
                    return torch.load(local, map_location="cpu", weights_only=True)
                except Exception:
                    continue
        except ImportError:
            pass
        return None
