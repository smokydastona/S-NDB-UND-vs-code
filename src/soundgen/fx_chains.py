from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_utils import JsonParseError, load_json_file_lenient


@dataclass(frozen=True)
class FXChain:
    """Named post-processing chain.

    This is intentionally implemented as an argparse "defaults patch": it sets
    CLI arg values only when those args are still at their argparse defaults.

    That means:
    - Users can pick a chain and still override individual knobs explicitly.
    - Chains compose safely with existing pro presets / polish profiles.
    """

    title: str
    description: str
    enable_post: bool = True
    enable_polish: bool = False
    args: dict[str, Any] | None = None


FX_CHAINS: dict[str, FXChain] = {
    "tight_game_ready": FXChain(
        title="Tight Game-ready",
        description="Punchy, clean, consistent output for creature/UI/impacts.",
        enable_post=True,
        enable_polish=False,
        args={
            "highpass_hz": 80.0,
            "lowpass_hz": 16000.0,
            "normalize_rms_db": -18.0,
            "normalize_peak_db": -1.0,
            "compressor_threshold_db": -18.0,
            "compressor_ratio": 3.0,
            "compressor_makeup_db": 0.0,
            "limiter_ceiling_db": -1.0,
            "exciter_amount": 0.20,
            # keep the chain tight and predictable
            "post_stack": "trim,filters,exciter,compressor,normalize,limiter,final_clip",
        },
    ),
    "creature_grit": FXChain(
        title="Creature Grit",
        description="Aggressive presence for roars/growls without killing lows.",
        enable_post=True,
        enable_polish=False,
        args={
            "highpass_hz": 60.0,
            "lowpass_hz": 15500.0,
            # presence lift via multiband mid boost in ~2-4k
            "multiband": True,
            "mb_low_hz": 2000.0,
            "mb_high_hz": 4500.0,
            "mb_mid_gain_db": 2.5,
            "exciter_amount": 0.45,
            "transient_attack": 0.25,
            "compressor_threshold_db": -20.0,
            "compressor_ratio": 3.0,
            "limiter_ceiling_db": -1.0,
            "normalize_rms_db": -18.0,
            "normalize_peak_db": -1.0,
            "post_stack": "trim,filters,multiband,transient,exciter,compressor,normalize,limiter,final_clip",
        },
    ),
    "distant_ambient": FXChain(
        title="Distant Ambient",
        description="Dark + spacious ambience bed with subtle noise floor.",
        enable_post=True,
        enable_polish=False,
        args={
            "no_trim": True,
            "fade_ms": 40,
            "highpass_hz": 30.0,
            "lowpass_hz": 4000.0,
            "reverb": "cave",
            "reverb_mix": 0.45,
            "reverb_time": 2.6,
            "noise_bed_db": -40.0,
            "normalize_rms_db": -22.0,
            "normalize_peak_db": -1.5,
            "post_stack": "filters,noise_bed,reverb,fade,normalize,final_clip",
        },
    ),
    "ui_polish": FXChain(
        title="UI Polish",
        description="Crisp, consistent clicks/beeps with tight low-cut.",
        enable_post=True,
        enable_polish=False,
        args={
            "highpass_hz": 200.0,
            "lowpass_hz": 16000.0,
            "compressor_threshold_db": -22.0,
            "compressor_ratio": 3.0,
            "exciter_amount": 0.08,
            "limiter_ceiling_db": -1.0,
            "normalize_rms_db": -19.0,
            "normalize_peak_db": -1.0,
            "fade_ms": 6,
            "post_stack": "trim,filters,exciter,compressor,normalize,limiter,final_clip",
        },
    ),
    "clean_normalized": FXChain(
        title="Clean Normalized",
        description="Minimal processing: normalize + fades only.",
        enable_post=True,
        enable_polish=False,
        args={
            "highpass_hz": 0.0,
            "lowpass_hz": 0.0,
            "normalize_rms_db": -18.0,
            "normalize_peak_db": -1.0,
            "fade_ms": 8,
            "post_stack": "trim,fade,normalize,final_clip",
        },
    ),
}


def fx_chain_keys() -> list[str]:
    return sorted(FX_CHAINS.keys())


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


def apply_fx_chain(*, chain_key: str, chain_json: str | None, args: Any, parser: Any) -> None:
    """Apply named or JSON-defined FX chain by patching argparse defaults."""

    key = str(chain_key or "off").strip()
    if key and key.lower() != "off":
        chain = FX_CHAINS.get(key)
        if chain is not None:
            if getattr(args, "post", None) == parser.get_default("post") and bool(chain.enable_post):
                setattr(args, "post", True)
            if getattr(args, "polish", None) == parser.get_default("polish") and bool(chain.enable_polish):
                setattr(args, "polish", True)
            _apply_args_patch(args=args, parser=parser, patch=(chain.args or {}))

    if chain_json:
        patch, enable_post, enable_polish = load_fx_chain_json(chain_json)
        if getattr(args, "post", None) == parser.get_default("post") and bool(enable_post):
            setattr(args, "post", True)
        if getattr(args, "polish", None) == parser.get_default("polish") and bool(enable_polish):
            setattr(args, "polish", True)
        _apply_args_patch(args=args, parser=parser, patch=patch)


def load_fx_chain_json(path: str | Path) -> tuple[dict[str, Any], bool, bool]:
    """Load a JSON FX chain.

    Supported formats:
    1) Args-patch format (simple):
       {"enable_post": true, "enable_polish": false, "args": {"highpass_hz": 80, ...}}

    2) Effect list format (compatible with the example in the prompt):
       {"name": "tight_game_ready", "chain": [{"effect": "HighPassFilter", "cutoff": 80}, ...]}

    The effect-list format is intentionally minimal and maps onto existing CLI knobs.
    Unknown effects/fields are ignored.
    """

    p = Path(path)
    try:
        obj = load_json_file_lenient(p, context=f"FX chain JSON file: {p}")
    except JsonParseError as e:
        raise ValueError(str(e)) from e

    if isinstance(obj, dict) and "args" in obj:
        enable_post = bool(obj.get("enable_post", True))
        enable_polish = bool(obj.get("enable_polish", False))
        patch = obj.get("args")
        return (dict(patch) if isinstance(patch, dict) else {}), enable_post, enable_polish

    enable_post = True
    enable_polish = False
    patch: dict[str, Any] = {}

    chain = obj.get("chain") if isinstance(obj, dict) else None
    if not isinstance(chain, list):
        return patch, enable_post, enable_polish

    for step in chain:
        if not isinstance(step, dict):
            continue
        eff = str(step.get("effect", "")).strip().lower()

        if eff in {"highpassfilter", "hpf", "high_pass", "high-pass"}:
            if "cutoff" in step:
                patch["highpass_hz"] = float(step["cutoff"])
        elif eff in {"lowpassfilter", "lpf", "low_pass", "low-pass"}:
            if "cutoff" in step:
                patch["lowpass_hz"] = float(step["cutoff"])
        elif eff in {"compressor"}:
            if "threshold" in step:
                patch["compressor_threshold_db"] = float(step["threshold"])
            if "ratio" in step:
                patch["compressor_ratio"] = float(step["ratio"])
            if "makeup" in step:
                patch["compressor_makeup_db"] = float(step["makeup"])
        elif eff in {"limiter"}:
            if "ceiling" in step:
                patch["limiter_ceiling_db"] = float(step["ceiling"])
        elif eff in {"saturation", "softclip", "soft_clipping", "soft-clipping"}:
            if "amount" in step:
                patch["exciter_amount"] = float(step["amount"])
        elif eff in {"reverb"}:
            if "preset" in step:
                patch["reverb"] = str(step["preset"])
            if "mix" in step:
                patch["reverb_mix"] = float(step["mix"])
            if "time_s" in step:
                patch["reverb_time"] = float(step["time_s"])
            if "time" in step:
                patch["reverb_time"] = float(step["time"])
        elif eff in {"normalize", "normalization"}:
            if "rms_db" in step:
                patch["normalize_rms_db"] = float(step["rms_db"])
            if "peak_db" in step:
                patch["normalize_peak_db"] = float(step["peak_db"])
        elif eff in {"fade", "fadeinout", "fade_in_out"}:
            if "ms" in step:
                patch["fade_ms"] = int(step["ms"])
        elif eff in {"noisebed", "noise_bed", "noise"}:
            if "db" in step:
                patch["noise_bed_db"] = float(step["db"])

    return patch, enable_post, enable_polish
