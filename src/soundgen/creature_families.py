from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .json_utils import JsonParseError, load_json_file_lenient


@dataclass(frozen=True)
class CreatureFamily:
    key: str
    lora_path: str
    trigger: str | None = None
    scale: float = 1.0
    negative_prompt: str | None = None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def load_creature_families(path: Path = Path("library") / "creature_families.json") -> dict[str, CreatureFamily]:
    """Load creature-family LoRA configs.

    File format (JSON):
      {
        "ghoul": {"lora_path": "lora/ghoul.safetensors", "trigger": "ghoul", "scale": 0.8}
      }

    The file is optional; if it doesn't exist, returns {}.
    """

    # If the default path doesn't exist, fall back to a repo-friendly config location.
    # - configs/creature_families.json: project config (can be committed)
    # - library/creature_families.json: local override (often gitignored in this repo)
    path = Path(path)
    if not path.exists():
        for candidate in (Path("configs") / "creature_families.json", Path("library") / "creature_families.json"):
            if candidate.exists():
                path = candidate
                break
        else:
            return {}

    try:
        raw = load_json_file_lenient(path, context=f"Creature families JSON file: {path}")
    except JsonParseError as e:
        raise ValueError(str(e)) from e
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid creature families file (expected object): {path}")

    out: dict[str, CreatureFamily] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        if not isinstance(v, dict):
            continue

        lora_path = _as_str(v.get("lora_path"))
        if not lora_path:
            continue

        out[key] = CreatureFamily(
            key=key,
            lora_path=lora_path,
            trigger=_as_str(v.get("trigger")),
            scale=float(v.get("scale", 1.0)),
            negative_prompt=_as_str(v.get("negative_prompt")),
        )

    return out


def resolve_creature_family(key: str, *, path: Path = Path("library") / "creature_families.json") -> CreatureFamily:
    families = load_creature_families(path)
    k = str(key or "").strip()
    if not k:
        raise ValueError("Empty creature family key")

    fam = families.get(k)
    if fam is not None:
        return fam

    # Allow case-insensitive lookup.
    k2 = k.lower()
    for kk, vv in families.items():
        if kk.lower() == k2:
            return vv

    raise KeyError(
        "Unknown creature family '" + k + "'. "
        "Add it to configs/creature_families.json (project config) or library/creature_families.json (local override)."
    )


def apply_trigger(prompt: str, trigger: str | None) -> str:
    p = str(prompt or "").strip()
    t = str(trigger or "").strip()
    if not t:
        return p
    if t.lower() in p.lower():
        return p
    if not p:
        return t
    return f"{p}, {t}".strip()
