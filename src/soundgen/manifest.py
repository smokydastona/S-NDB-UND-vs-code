from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ManifestItem:
    prompt: str
    engine: str = "rfxgen"  # diffusers | rfxgen

    namespace: str = "soundgen"
    event: str = "generated.sound"
    sound_path: Optional[str] = None

    seconds: float = 3.0
    seed: Optional[int] = None
    preset: Optional[str] = None  # rfxgen

    variants: int = 1
    weight: int = 1
    volume: float = 1.0
    pitch: float = 1.0

    subtitle: Optional[str] = None
    subtitle_key: Optional[str] = None

    post: bool = True

    tags: tuple[str, ...] = ()


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _coerce_int(v: Any, default: int) -> int:
    if v is None or v == "":
        return default
    return int(v)


def _coerce_float(v: Any, default: float) -> float:
    if v is None or v == "":
        return default
    return float(v)


def _coerce_tags(v: Any) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(str(x) for x in v if str(x).strip())
    s = str(v).strip()
    if not s:
        return ()
    # comma or semicolon separated
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return tuple(p for p in parts if p)


def _item_from_mapping(m: dict[str, Any]) -> ManifestItem:
    return ManifestItem(
        prompt=str(m.get("prompt") or "").strip(),
        engine=str(m.get("engine") or "rfxgen").strip(),
        namespace=str(m.get("namespace") or "soundgen").strip(),
        event=str(m.get("event") or "generated.sound").strip(),
        sound_path=(str(m["sound_path"]).strip() if m.get("sound_path") else None),
        seconds=_coerce_float(m.get("seconds"), 3.0),
        seed=(int(m["seed"]) if m.get("seed") not in (None, "") else None),
        preset=(str(m["preset"]).strip() if m.get("preset") else None),
        variants=max(1, _coerce_int(m.get("variants"), 1)),
        weight=max(1, _coerce_int(m.get("weight"), 1)),
        volume=_coerce_float(m.get("volume"), 1.0),
        pitch=_coerce_float(m.get("pitch"), 1.0),
        subtitle=(str(m["subtitle"]).strip() if m.get("subtitle") else None),
        subtitle_key=(str(m["subtitle_key"]).strip() if m.get("subtitle_key") else None),
        post=_coerce_bool(m.get("post"), default=True),
        tags=_coerce_tags(m.get("tags")),
    )


def load_manifest(path: Path) -> list[ManifestItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("JSON manifest must be a list or {items:[...]}.")

        out: list[ManifestItem] = []
        for raw in items:
            if not isinstance(raw, dict):
                raise ValueError("Each JSON manifest entry must be an object.")
            item = _item_from_mapping(raw)
            if not item.prompt:
                raise ValueError("Manifest item missing 'prompt'.")
            out.append(item)
        return out

    if path.suffix.lower() in {".csv", ".tsv"}:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            out: list[ManifestItem] = []
            for row in reader:
                item = _item_from_mapping(row)
                if not item.prompt:
                    raise ValueError("Manifest row missing 'prompt'.")
                out.append(item)
            return out

    raise ValueError("Unsupported manifest type. Use .json or .csv/.tsv")
