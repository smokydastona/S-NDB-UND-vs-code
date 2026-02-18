from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelVersion:
    """Reference to a fine-tuned model variant (typically a LoRA).

    This is intentionally lightweight: it just resolves stable, named references
    into the low-level engine parameters (LoRA path/scale/trigger/etc).
    """

    key: str

    # Currently, the only supported target is inference-time LoRA loading for Stable Audio Open.
    engine: str = "stable_audio_open"

    # Optional: allow pinning a specific base model id.
    base_model: str | None = None

    lora_path: str | None = None
    trigger: str | None = None
    scale: float = 1.0
    negative_prompt: str | None = None

    notes: str | None = None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def load_model_versions(path: Path = Path("library") / "model_versions.json") -> dict[str, ModelVersion]:
    """Load model version definitions.

    Search order (first existing wins):
    - configs/model_versions.json (project config; can be committed)
    - library/model_versions.json (local override; gitignored in this repo)
    - configs/model_versions.example.json (in-repo example)

    File format (JSON):
      {
        "ghoul_v1": {
          "engine": "stable_audio_open",
          "base_model": "stabilityai/stable-audio-open-1.0",
          "lora_path": "lora/ghoul_v1.safetensors",
          "trigger": "ghoul",
          "scale": 0.8,
          "negative_prompt": "music, singing, speech"
        }
      }

    The file is optional; if it doesn't exist, returns {}.
    """

    path = Path(path)
    if not path.exists():
        for candidate in (
            Path("configs") / "model_versions.json",
            Path("library") / "model_versions.json",
            Path("configs") / "model_versions.example.json",
        ):
            if candidate.exists():
                path = candidate
                break
        else:
            return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid model versions file (expected object): {path}")

    out: dict[str, ModelVersion] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key or not isinstance(v, dict):
            continue

        eng = str(v.get("engine") or "stable_audio_open").strip().lower() or "stable_audio_open"
        base_model = _as_str(v.get("base_model"))
        lora_path = _as_str(v.get("lora_path"))
        trigger = _as_str(v.get("trigger"))
        negative_prompt = _as_str(v.get("negative_prompt"))

        scale = float(v.get("scale", 1.0))
        notes = _as_str(v.get("notes"))

        out[key] = ModelVersion(
            key=key,
            engine=eng,
            base_model=base_model,
            lora_path=lora_path,
            trigger=trigger,
            scale=scale,
            negative_prompt=negative_prompt,
            notes=notes,
        )

    return out


def resolve_model_version(key: str, *, path: Path = Path("library") / "model_versions.json") -> ModelVersion:
    versions = load_model_versions(path)
    k = str(key or "").strip()
    if not k:
        raise ValueError("Empty model version key")

    hit = versions.get(k)
    if hit is not None:
        return hit

    k2 = k.lower()
    for kk, vv in versions.items():
        if kk.lower() == k2:
            return vv

    raise KeyError(
        "Unknown model version '" + k + "'. "
        "Add it to configs/model_versions.json (project config) or library/model_versions.json (local override)."
    )
