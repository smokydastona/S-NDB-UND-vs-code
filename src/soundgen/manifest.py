from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ManifestItem:
    # Either `prompt` or `sfx_preset` must be provided.
    prompt: str = ""
    sfx_preset: Optional[str] = None
    sfx_preset_file: Optional[str] = None

    engine: Optional[str] = None  # diffusers | stable_audio_open | rfxgen | replicate | samplelib | synth | layered

    namespace: str = "soundgen"
    event: str = "generated.sound"
    sound_path: Optional[str] = None

    seconds: Optional[float] = None
    candidates: Optional[int] = None
    seed: Optional[int] = None
    preset: Optional[str] = None  # rfxgen

    # FX chains (optional; primarily used by batch workflows)
    fx_chain: Optional[str] = None
    fx_chain_json: Optional[str] = None

    # stable_audio_open (optional)
    stable_audio_model: Optional[str] = None
    stable_audio_negative_prompt: Optional[str] = None
    stable_audio_steps: Optional[int] = None
    stable_audio_guidance_scale: Optional[float] = None
    stable_audio_sampler: Optional[str] = None
    stable_audio_hf_token: Optional[str] = None

    # Optional per-item overrides (useful for workflow-ready batch generation)
    pro_preset: Optional[str] = None
    polish_profile: Optional[str] = None
    emotion: Optional[str] = None
    intensity: Optional[float] = None
    variation: Optional[float] = None
    pitch_contour: Optional[str] = None
    loop: Optional[bool] = None
    loop_crossfade_ms: Optional[int] = None

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
        sfx_preset=(str(m["sfx_preset"]).strip() if m.get("sfx_preset") else None),
        sfx_preset_file=(str(m["sfx_preset_file"]).strip() if m.get("sfx_preset_file") else None),
        engine=(str(m["engine"]).strip() if m.get("engine") else None),
        namespace=str(m.get("namespace") or "soundgen").strip(),
        event=str(m.get("event") or "generated.sound").strip(),
        sound_path=(str(m["sound_path"]).strip() if m.get("sound_path") else None),
        seconds=(_coerce_float(m.get("seconds"), 3.0) if m.get("seconds") not in (None, "") else None),
        candidates=(max(1, _coerce_int(m.get("candidates"), 1)) if m.get("candidates") not in (None, "") else None),
        seed=(int(m["seed"]) if m.get("seed") not in (None, "") else None),
        preset=(str(m["preset"]).strip() if m.get("preset") else None),

        fx_chain=(str(m["fx_chain"]).strip() if m.get("fx_chain") else None),
        fx_chain_json=(str(m["fx_chain_json"]).strip() if m.get("fx_chain_json") else None),

        stable_audio_model=(str(m["stable_audio_model"]).strip() if m.get("stable_audio_model") else None),
        stable_audio_negative_prompt=(
            str(m["stable_audio_negative_prompt"]).strip() if m.get("stable_audio_negative_prompt") else None
        ),
        stable_audio_steps=(int(m["stable_audio_steps"]) if m.get("stable_audio_steps") not in (None, "") else None),
        stable_audio_guidance_scale=(
            float(m["stable_audio_guidance_scale"]) if m.get("stable_audio_guidance_scale") not in (None, "") else None
        ),
        stable_audio_sampler=(str(m["stable_audio_sampler"]).strip() if m.get("stable_audio_sampler") else None),
        stable_audio_hf_token=(str(m["stable_audio_hf_token"]).strip() if m.get("stable_audio_hf_token") else None),

        pro_preset=(str(m["pro_preset"]).strip() if m.get("pro_preset") else None),
        polish_profile=(str(m["polish_profile"]).strip() if m.get("polish_profile") else None),
        emotion=(str(m["emotion"]).strip() if m.get("emotion") else None),
        intensity=(float(m["intensity"]) if m.get("intensity") not in (None, "") else None),
        variation=(float(m["variation"]) if m.get("variation") not in (None, "") else None),
        pitch_contour=(str(m["pitch_contour"]).strip() if m.get("pitch_contour") else None),
        loop=(_coerce_bool(m.get("loop"), default=False) if "loop" in m else None),
        loop_crossfade_ms=(_coerce_int(m.get("loop_crossfade_ms"), 100) if "loop_crossfade_ms" in m else None),

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
            if not item.prompt and not item.sfx_preset:
                raise ValueError("Manifest item missing 'prompt' (or provide 'sfx_preset').")
            out.append(item)
        return out

    if path.suffix.lower() in {".csv", ".tsv"}:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            out: list[ManifestItem] = []
            for row in reader:
                item = _item_from_mapping(row)
                if not item.prompt and not item.sfx_preset:
                    raise ValueError("Manifest row missing 'prompt' (or provide 'sfx_preset').")
                out.append(item)
            return out

    raise ValueError("Unsupported manifest type. Use .json or .csv/.tsv")
