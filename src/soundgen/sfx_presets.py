from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_utils import JsonParseError, load_json_file_lenient


@dataclass(frozen=True)
class SfxPreset:
    """Concrete SFX preset: engine + prompt + params + FX chain.

    Unlike `ProPreset`, which is a high-level "paid-tool-like" nudge layer, an
    `SfxPreset` is meant to represent a specific sound family target (e.g.
    `creature_medium_roar`) with a prompt and engine defaults.

    Presets are applied conservatively: they only override values that are still
    at their argparse defaults. This keeps manual CLI tweaks working naturally.
    """

    name: str
    engine: str
    prompt: str
    negative_prompt: str | None = None
    seconds: float | None = None
    seed: int | None = None
    variation_strength: float | None = None
    fx_chain: str | None = None
    post: bool | None = None

    # Extra arg patches (argparse dest -> value)
    engine_params: dict[str, Any] | None = None
    post_params: dict[str, Any] | None = None

    # Smart preset v2: prompt template variables (optional).
    # Values may be scalars or lists (lists are randomly sampled when rendering).
    vars: dict[str, Any] | None = None


def default_sfx_preset_paths() -> list[Path]:
    """Search order for preset libraries.

    - `library/sfx_presets.json` is treated as a local/user override (gitignored).
    - `configs/sfx_presets_v1.example.json` is the in-repo example library.
    """

    return [
        Path("library") / "sfx_presets.json",
        Path("configs") / "sfx_presets_v2.example.json",
        Path("configs") / "sfx_presets_v1.example.json",
    ]


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(dict(out[k]), dict(v))
        else:
            out[k] = v
    return out


def render_prompt_template(
    template: str,
    *,
    vars_def: dict[str, Any] | None,
    seed: int | None,
) -> tuple[str, dict[str, Any]]:
    """Render a prompt template using vars.

    - Template uses Python-style `{var}` placeholders.
    - vars values can be scalars or lists; lists are randomly sampled.
    - Returns (rendered_prompt, chosen_vars)
    """

    t = str(template or "").strip()
    if not t:
        return "", {}

    vars_def = dict(vars_def or {})
    rng = random.Random(int(seed) if seed is not None else None)

    chosen: dict[str, Any] = {}
    for k, v in vars_def.items():
        key = str(k)
        if isinstance(v, list):
            if v:
                chosen[key] = rng.choice(list(v))
        else:
            chosen[key] = v

    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    rendered = t.format_map(_SafeDict(**{k: str(v) for k, v in chosen.items()}))
    return rendered, chosen


# Backwards-compat alias (internal name used in older code paths).
_render_prompt_template = render_prompt_template


def load_sfx_preset_library(path: str | Path) -> dict[str, SfxPreset]:
    path = Path(path)
    try:
        obj = load_json_file_lenient(path, context=f"SFX preset library JSON file: {path}")
    except JsonParseError as e:
        raise ValueError(str(e)) from e
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid sfx preset library (expected object): {path}")

    presets_raw = obj.get("presets")
    if not isinstance(presets_raw, list):
        raise ValueError(f"Invalid sfx preset library (missing presets list): {path}")

    families_raw = obj.get("families")
    families: dict[str, dict[str, Any]] = {}
    if isinstance(families_raw, dict):
        for k, v in families_raw.items():
            if isinstance(v, dict):
                families[str(k)] = dict(v)

    # v2: resolve inheritance/family references from raw preset dicts.
    raw_by_name: dict[str, dict[str, Any]] = {}
    for p in presets_raw:
        if isinstance(p, dict) and str(p.get("name") or "").strip():
            raw_by_name[str(p.get("name")).strip()] = dict(p)

    resolving: set[str] = set()

    def _resolve_dict(name: str) -> dict[str, Any]:
        if name in resolving:
            raise ValueError(f"SFX preset inheritance cycle detected at '{name}' in {path}")
        resolving.add(name)
        base = dict(raw_by_name.get(name) or {})

        merged: dict[str, Any] = {}

        fam = base.get("family")
        if fam is not None:
            fam_key = str(fam).strip()
            if fam_key:
                fam_obj = families.get(fam_key)
                if fam_obj is None:
                    raise ValueError(f"Unknown sfx preset family '{fam_key}' used by '{name}' in {path}")
                merged = _deep_merge(merged, fam_obj)

        parent = base.get("inherits")
        if parent is not None:
            parent_key = str(parent).strip()
            if parent_key:
                if parent_key not in raw_by_name:
                    raise ValueError(f"Unknown sfx preset inherits '{parent_key}' used by '{name}' in {path}")
                merged = _deep_merge(merged, _resolve_dict(parent_key))

        merged = _deep_merge(merged, base)
        resolving.remove(name)
        return merged

    out: dict[str, SfxPreset] = {}
    for name in raw_by_name.keys():
        p = _resolve_dict(name)
        preset = SfxPreset(
            name=str(name),
            engine=str(p.get("engine") or "").strip(),
            prompt=str(p.get("prompt") or "").strip(),
            negative_prompt=(str(p["negative_prompt"]).strip() if p.get("negative_prompt") is not None else None),
            seconds=(float(p["seconds"]) if p.get("seconds") is not None else None),
            seed=(int(p["seed"]) if p.get("seed") is not None else None),
            variation_strength=(float(p["variation_strength"]) if p.get("variation_strength") is not None else None),
            fx_chain=(str(p["fx_chain"]).strip() if p.get("fx_chain") is not None else None),
            post=(bool(p["post"]) if p.get("post") is not None else None),
            engine_params=(dict(p.get("engine_params") or {}) if isinstance(p.get("engine_params"), dict) else None),
            post_params=(dict(p.get("post_params") or {}) if isinstance(p.get("post_params"), dict) else None),
            vars=(dict(p.get("vars") or {}) if isinstance(p.get("vars"), dict) else None),
        )

        if not preset.engine:
            raise ValueError(f"SFX preset '{name}' missing engine in {path}")
        if not preset.prompt:
            raise ValueError(f"SFX preset '{name}' missing prompt in {path}")

        out[name] = preset

    return out


def get_sfx_preset(
    preset_key: str,
    *,
    preset_file: str | None = None,
    search_paths: list[Path] | None = None,
) -> tuple[SfxPreset | None, Path | None]:
    key = str(preset_key or "").strip()
    if not key or key.lower() == "off":
        return None, None

    if preset_file:
        lib_path = Path(preset_file)
        lib = load_sfx_preset_library(lib_path)
        return lib.get(key), lib_path

    for p in (search_paths or default_sfx_preset_paths()):
        if p.exists():
            lib = load_sfx_preset_library(p)
            hit = lib.get(key)
            if hit is not None:
                return hit, p

    return None, None


def _apply_args_patch(*, args: Any, parser: Any, patch: dict[str, Any]) -> None:
    for dest, value in patch.items():
        if not hasattr(args, dest):
            continue
        try:
            default = parser.get_default(dest)
        except Exception:
            continue
        if getattr(args, dest) == default:
            setattr(args, dest, value)


def apply_sfx_preset(*, preset_key: str, preset_file: str | None, args: Any, parser: Any) -> tuple[SfxPreset | None, Path | None]:
    """Apply an SFX preset by patching argparse defaults.

    Returns (preset_obj, preset_library_path).
    """

    key = str(preset_key or "").strip()
    if not key or key.lower() == "off":
        return None, None

    preset, lib_path = get_sfx_preset(key, preset_file=preset_file)
    if preset is None:
        where = str(preset_file) if preset_file else "default search paths"
        raise ValueError(f"Unknown sfx preset '{key}' (searched: {where}).")

    # Core fields
    if hasattr(args, "engine") and getattr(args, "engine") == parser.get_default("engine"):
        setattr(args, "engine", preset.engine)

    # Prompt: allow explicit --prompt to override preset prompt.
    if hasattr(args, "prompt") and getattr(args, "prompt", None) in {None, ""}:
        setattr(args, "prompt", preset.prompt)

    # Preserve template vars for later render (batch can re-render per-item seed).
    if preset.vars is not None:
        try:
            setattr(args, "sfx_preset_vars", dict(preset.vars))
        except Exception:
            pass

    # Alias for Stable Audio negative prompt
    if preset.negative_prompt is not None and hasattr(args, "stable_audio_negative_prompt"):
        if getattr(args, "stable_audio_negative_prompt") == parser.get_default("stable_audio_negative_prompt"):
            setattr(args, "stable_audio_negative_prompt", preset.negative_prompt)

    if preset.seconds is not None and hasattr(args, "seconds"):
        if getattr(args, "seconds") == parser.get_default("seconds"):
            setattr(args, "seconds", float(preset.seconds))

    if preset.seed is not None and hasattr(args, "seed"):
        if getattr(args, "seed") == parser.get_default("seed"):
            setattr(args, "seed", int(preset.seed))

    if preset.variation_strength is not None and hasattr(args, "variation"):
        if getattr(args, "variation") == parser.get_default("variation"):
            setattr(args, "variation", float(preset.variation_strength))

    if preset.fx_chain is not None and hasattr(args, "fx_chain"):
        if getattr(args, "fx_chain") == parser.get_default("fx_chain"):
            setattr(args, "fx_chain", str(preset.fx_chain))

    if preset.post is True and hasattr(args, "post"):
        if getattr(args, "post") == parser.get_default("post"):
            setattr(args, "post", True)

    # Engine-specific mapping conveniences
    engine_patch: dict[str, Any] = dict(preset.engine_params or {})
    if "rfxgen_preset" in engine_patch and "preset" not in engine_patch:
        engine_patch["preset"] = engine_patch.pop("rfxgen_preset")

    _apply_args_patch(args=args, parser=parser, patch=engine_patch)
    _apply_args_patch(args=args, parser=parser, patch=(preset.post_params or {}))

    return preset, lib_path


def render_sfx_prompt_from_args(*, args: Any, seed: int | None) -> dict[str, Any] | None:
    """If args has an sfx preset template vars dict, render args.prompt in-place.

    Returns chosen vars dict if rendering happened, else None.
    """

    vars_def = getattr(args, "sfx_preset_vars", None)
    if not isinstance(vars_def, dict) or not getattr(args, "prompt", None):
        return None
    rendered, chosen = render_prompt_template(str(getattr(args, "prompt")), vars_def=vars_def, seed=seed)
    if rendered:
        setattr(args, "prompt", rendered)
        try:
            setattr(args, "sfx_preset_vars_chosen", dict(chosen))
        except Exception:
            pass
        return chosen
    return None
