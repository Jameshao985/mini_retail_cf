"""Strict offline discovery of a usable Hugging Face snapshot."""

from __future__ import annotations

import os
from pathlib import Path

from config import DEFAULT_MODEL_CACHE


class ModelPathError(RuntimeError):
    pass


def _missing_components(path: Path) -> list[str]:
    missing: list[str] = []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
        missing.append("tokenizer files")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        missing.append("model weights")
    return missing


def validate_model_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise ModelPathError(f"Model directory does not exist: {candidate}")
    missing = _missing_components(candidate)
    if missing:
        raise ModelPathError(f"Incomplete model snapshot {candidate}; missing: {', '.join(missing)}")
    return candidate


def discover_model_path(explicit_path: str | Path | None = None) -> Path:
    override = explicit_path or os.environ.get("MODEL_PATH")
    if override:
        return validate_model_path(override)

    base = DEFAULT_MODEL_CACHE
    direct_missing = _missing_components(base) if base.is_dir() else ["snapshot directory"]
    if not direct_missing:
        return base.resolve()
    snapshots = base / "snapshots"
    if not snapshots.is_dir():
        raise ModelPathError(
            f"Offline model discovery failed: {snapshots} does not exist. "
            "Set MODEL_PATH or pass --model-path. Online download is disabled."
        )
    candidates = [path for path in snapshots.iterdir() if path.is_dir() and not _missing_components(path)]
    if not candidates:
        raise ModelPathError(
            f"No complete snapshot under {snapshots}. Expected config.json, tokenizer files, "
            "and model weights. Online download is disabled."
        )
    candidates.sort(key=lambda p: (len(list(p.iterdir())), p.stat().st_mtime), reverse=True)
    return candidates[0].resolve()

