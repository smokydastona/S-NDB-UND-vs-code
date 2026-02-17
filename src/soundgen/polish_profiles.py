from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolishProfile:
    title: str
    description: str
    enable_post: bool = True
    enable_polish: bool = False
    args: dict[str, Any] | None = None


POLISH_PROFILES: dict[str, PolishProfile] = {
    "ui_clean": PolishProfile(
        title="UI Clean",
        description="Clean, click-safe UI/menus: light post, gentle multiband.",
        enable_post=True,
        enable_polish=False,
        args={
            "multiband": True,
            "mb_comp_threshold_db": -28.0,
            "mb_comp_ratio": 2.0,
            "reverb": "off",
            "reverb_mix": 0.0,
            "texture_preset": "off",
            "texture_amount": 0.0,
            "normalize_rms_db": -20.0,
            "highpass_hz": 80.0,
            "lowpass_hz": 16000.0,
            "fade_ms": 6,
            "silence_threshold_db": -45.0,
        },
    ),
    "foley_punchy": PolishProfile(
        title="Foley Punchy",
        description="Punchier impacts/foley: polish DSP + multiband + slightly tighter EQ.",
        enable_post=True,
        enable_polish=True,
        args={
            "multiband": True,
            "mb_comp_threshold_db": -26.0,
            "mb_comp_ratio": 2.5,
            "mb_high_gain_db": 1.5,
            "normalize_rms_db": -17.5,
            "highpass_hz": 70.0,
            "lowpass_hz": 15000.0,
            "fade_ms": 8,
            "reverb": "off",
            "reverb_mix": 0.0,
        },
    ),
    "creature_gritty": PolishProfile(
        title="Creature Gritty",
        description="Creature/monster sweetening: polish DSP + texture + cave space.",
        enable_post=True,
        enable_polish=True,
        args={
            "multiband": True,
            "mb_comp_threshold_db": -24.0,
            "mb_comp_ratio": 2.0,
            "texture_preset": "auto",
            "texture_amount": 0.22,
            "creature_size": -0.25,
            "reverb": "cave",
            "reverb_mix": 0.10,
            "reverb_time": 1.0,
            "highpass_hz": 50.0,
            "lowpass_hz": 14000.0,
        },
    ),
    "ambience_smooth": PolishProfile(
        title="Ambience Smooth",
        description="Ambient beds: keep tails, darker EQ, gentle space.",
        enable_post=True,
        enable_polish=True,
        args={
            "no_trim": True,
            "fade_ms": 30,
            "normalize_rms_db": -22.0,
            "normalize_peak_db": -1.5,
            "highpass_hz": 30.0,
            "lowpass_hz": 12000.0,
            "reverb": "forest",
            "reverb_mix": 0.18,
            "reverb_time": 2.4,
            "multiband": True,
            "mb_comp_threshold_db": -30.0,
            "mb_comp_ratio": 2.0,
        },
    ),
    "ambience_loop_ready": PolishProfile(
        title="Ambience Loop-ready",
        description="Loop-friendly ambience: no trim + longer fades + loop-clean seam crossfade.",
        enable_post=True,
        enable_polish=True,
        args={
            "no_trim": True,
            "fade_ms": 90,
            "loop": True,
            "loop_crossfade_ms": 100,
            "normalize_rms_db": -23.0,
            "normalize_peak_db": -1.5,
            "highpass_hz": 30.0,
            "lowpass_hz": 14000.0,
            "reverb": "off",
            "reverb_mix": 0.0,
            "multiband": True,
            "mb_comp_threshold_db": -30.0,
            "mb_comp_ratio": 2.0,
        },
    ),
}


def polish_profile_keys() -> list[str]:
    return sorted(POLISH_PROFILES.keys())


def apply_polish_profile(*, profile_key: str, args: Any, parser: Any) -> None:
    key = str(profile_key or "off").strip()
    if not key or key.lower() == "off":
        return

    prof = POLISH_PROFILES.get(key)
    if prof is None:
        return

    # Selecting a profile implies post-processing; some profiles also imply polish DSP.
    if getattr(args, "post", None) == parser.get_default("post") and bool(prof.enable_post):
        setattr(args, "post", True)
    if getattr(args, "polish", None) == parser.get_default("polish") and bool(prof.enable_polish):
        setattr(args, "polish", True)

    for dest, value in (prof.args or {}).items():
        if not hasattr(args, dest):
            continue
        try:
            default = parser.get_default(dest)
        except Exception:
            continue
        if getattr(args, dest) == default:
            setattr(args, dest, value)
