from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    backend: str
    model_id: str
    device: str = "cpu"
    dtype: str = "float32"
    revision: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterConfig:
    type: str
    enabled: bool = True
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RefinementConfig:
    type: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationConfig:
    metrics: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptConfig:
    id: str
    prompt: str
    negative_prompt: str = ""
    output_name: str | None = None
    seed: int | None = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    reference_images: list[str] = field(default_factory=list)
    controls: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    description: str = ""
    seed: int = 42
    output_dir: str = "outputs"
    prompt_sources: list[str] = field(default_factory=list)
    prompt_filters: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(backend="mock", model_id="mock"))
    adapters: list[AdapterConfig] = field(default_factory=list)
    refinement: RefinementConfig | None = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompts: list[PromptConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_path: Path | None = None) -> "ExperimentConfig":
        model = ModelConfig(**data["model"])
        adapters = [_normalize_adapter_config(item) for item in data.get("adapters", [])]
        refinement_data = data.get("refinement")
        refinement = RefinementConfig(**refinement_data) if refinement_data else None
        evaluation = EvaluationConfig(**data.get("evaluation", {}))
        prompt_sources = data.get("prompt_sources", [])
        prompts = [PromptConfig(**item) for item in _load_prompt_entries(data, base_path=base_path)]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            seed=data.get("seed", 42),
            output_dir=data.get("output_dir", "outputs"),
            prompt_sources=prompt_sources,
            prompt_filters=data.get("prompt_filters", {}),
            model=model,
            adapters=adapters,
            refinement=refinement,
            evaluation=evaluation,
            prompts=prompts,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ExperimentConfig.from_dict(data, base_path=config_path.parent)


def _load_prompt_entries(data: dict[str, Any], base_path: Path | None) -> list[dict[str, Any]]:
    prompt_entries = list(data.get("prompts", []))
    for prompt_source in data.get("prompt_sources", []):
        source_path = Path(prompt_source)
        if not source_path.is_absolute() and base_path is not None:
            source_path = (base_path / source_path).resolve()
        with source_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(f"Prompt source {source_path} must contain a JSON list.")
        prompt_entries.extend(loaded)
    return _apply_prompt_filters(prompt_entries, data.get("prompt_filters", {}))


def _apply_prompt_filters(prompt_entries: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    if not filters:
        return prompt_entries

    allowed_ids = set(filters.get("ids", []))
    allowed_suites = set(filters.get("suites", []))
    limit = filters.get("limit")

    filtered: list[dict[str, Any]] = []
    for entry in prompt_entries:
        entry_id = entry.get("id", "")
        entry_suite = entry.get("metadata", {}).get("suite", "")

        if allowed_ids and entry_id not in allowed_ids:
            continue
        if allowed_suites and entry_suite not in allowed_suites:
            continue
        filtered.append(entry)

    if isinstance(limit, int) and limit >= 0:
        return filtered[:limit]
    return filtered


def _normalize_adapter_config(item: dict[str, Any]) -> AdapterConfig:
    adapter_type = item.get("type", "")
    params = dict(item.get("params", {}))

    if adapter_type == "lora" and "style_prompt" in params and not any(
        key in params for key in ("weights_path", "lora_path")
    ):
        import warnings
        warnings.warn(
            "Config uses type='lora' with a style_prompt and no weights_path. "
            "This is deprecated and will be removed. Use type='style_prompt' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        item = {
            **item,
            "type": "style_prompt",
            "params": {
                **params,
                "_legacy_alias": "lora",
            },
        }

    return AdapterConfig(**item)
