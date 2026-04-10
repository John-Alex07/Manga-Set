from __future__ import annotations

from pathlib import Path
from typing import Any


def prefetch_model_artifacts(
    model_id: str,
    *,
    local_dir: str | Path | None = None,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for model prefetch.") from exc

    kwargs: dict[str, Any] = {"repo_id": model_id}
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if token:
        kwargs["token"] = token
    if revision:
        kwargs["revision"] = revision

    destination = snapshot_download(**kwargs)
    return {
        "model_id": model_id,
        "revision": revision,
        "local_dir": str(destination),
        "allow_patterns": allow_patterns or [],
    }
