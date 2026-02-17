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
            "denoise_amount": 0.18,
            "transient_attack": 0.35,
            "transient_sustain": -0.05,
            "exciter_amount": 0.10,
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
    "creature_low_end": PolishProfile(
        title="Creature Low-end",
        description="Low-end emphasis + formant down; softer attack; subtle saturation; slower compression.",
        enable_post=True,
        enable_polish=True,
        args={
            "multiband": True,
            "mb_low_gain_db": 2.0,
            "mb_mid_gain_db": 0.6,
            "mb_high_gain_db": -0.8,
            "mb_comp_threshold_db": -26.0,
            "mb_comp_ratio": 2.0,
            "formant_shift": 0.86,
            "creature_size": 0.55,
            "transient_attack": -0.15,
            "transient_sustain": 0.20,
            "exciter_amount": 0.10,
            "compressor_attack_ms": 14.0,
            "compressor_release_ms": 150.0,
            "highpass_hz": 25.0,
            "lowpass_hz": 14500.0,
            "reverb": "off",
            "reverb_mix": 0.0,
        },
    ),
    "creature_snappy": PolishProfile(
        title="Creature Snappy",
        description="High-mid presence + transient snap; light distortion; tighter compression timing.",
        enable_post=True,
        enable_polish=True,
        args={
            "multiband": True,
            "mb_low_gain_db": -0.4,
            "mb_mid_gain_db": 0.9,
            "mb_high_gain_db": 2.0,
            "mb_comp_threshold_db": -25.0,
            "mb_comp_ratio": 2.2,
            "highpass_hz": 60.0,
            "lowpass_hz": 15500.0,
            "denoise_amount": 0.10,
            "transient_attack": 0.70,
            "transient_sustain": -0.10,
            "exciter_amount": 0.22,
            "compressor_attack_ms": 3.0,
            "compressor_release_ms": 75.0,
            "reverb": "off",
            "reverb_mix": 0.0,
        },
    ),
    "creature_hushed": PolishProfile(
        title="Creature Hushed",
        description="Dark, controlled highs; stronger noise-floor control; soft compression with longer release.",
        enable_post=True,
        enable_polish=True,
        args={
            "multiband": True,
            "mb_low_gain_db": 0.6,
            "mb_mid_gain_db": 0.4,
            "mb_high_gain_db": -1.5,
            "mb_comp_threshold_db": -30.0,
            "mb_comp_ratio": 2.0,
            "lowpass_hz": 11500.0,
            "denoise_amount": 0.35,
            "transient_attack": -0.25,
            "transient_sustain": 0.15,
            "compressor_attack_ms": 8.0,
            "compressor_release_ms": 230.0,
            "reverb": "off",
            "reverb_mix": 0.0,
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
    "ambience_warm_mono": PolishProfile(
        title="Ambience Warm (Mono)",
        description="Warm low-mids + high-cut; subtle reverb; stable mono bed.",
        enable_post=True,
        enable_polish=True,
        args={
            "no_trim": True,
            "fade_ms": 60,
            "normalize_rms_db": -23.0,
            "normalize_peak_db": -1.5,
            "highpass_hz": 28.0,
            "lowpass_hz": 11000.0,
            "reverb": "forest",
            "reverb_mix": 0.10,
            "reverb_time": 2.0,
            "multiband": True,
            "mb_low_gain_db": 0.8,
            "mb_mid_gain_db": 1.0,
            "mb_high_gain_db": -0.8,
            "mb_comp_threshold_db": -30.0,
            "mb_comp_ratio": 2.0,
            "compressor_attack_ms": 18.0,
            "compressor_release_ms": 260.0,
        },
    ),
    "ambience_glue_open": PolishProfile(
        title="Ambience Glue (Open)",
        description="Gentle EQ smile + glue compression; slightly brighter presence via exciter.",
        enable_post=True,
        enable_polish=True,
        args={
            "no_trim": True,
            "fade_ms": 60,
            "normalize_rms_db": -22.0,
            "normalize_peak_db": -1.5,
            "highpass_hz": 30.0,
            "lowpass_hz": 14000.0,
            "multiband": True,
            "mb_low_gain_db": 0.6,
            "mb_mid_gain_db": -0.5,
            "mb_high_gain_db": 0.9,
            "mb_comp_threshold_db": -32.0,
            "mb_comp_ratio": 2.0,
            "exciter_amount": 0.10,
            "compressor_attack_ms": 20.0,
            "compressor_release_ms": 300.0,
            "reverb": "room",
            "reverb_mix": 0.08,
            "reverb_time": 1.6,
        },
    ),
    "ambience_vhs": PolishProfile(
        title="Ambience VHS",
        description="Band-limited + subtle texture; lo-fi bed (mono).",
        enable_post=True,
        enable_polish=True,
        args={
            "no_trim": True,
            "fade_ms": 50,
            "normalize_rms_db": -24.0,
            "normalize_peak_db": -1.8,
            "highpass_hz": 80.0,
            "lowpass_hz": 7000.0,
            "denoise_amount": 0.08,
            "texture_preset": "buzz",
            "texture_amount": 0.06,
            "reverb": "off",
            "reverb_mix": 0.0,
            "multiband": True,
            "mb_comp_threshold_db": -32.0,
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

    "impact_hard_punch": PolishProfile(
        title="Impact Hard Punch",
        description="Strong transient + low-end punch; limiter-ready; impact-forward.",
        enable_post=True,
        enable_polish=True,
        args={
            "fade_ms": 6,
            "normalize_rms_db": -16.5,
            "normalize_peak_db": -1.0,
            "highpass_hz": 45.0,
            "lowpass_hz": 16000.0,
            "multiband": True,
            "mb_low_gain_db": 2.0,
            "mb_mid_gain_db": -0.2,
            "mb_high_gain_db": 0.6,
            "mb_comp_threshold_db": -24.0,
            "mb_comp_ratio": 2.4,
            "transient_attack": 0.80,
            "transient_sustain": -0.10,
            "exciter_amount": 0.08,
            "reverb": "off",
            "reverb_mix": 0.0,
        },
    ),
    "impact_soft_mid": PolishProfile(
        title="Impact Soft Mid",
        description="Softer attack, mid-focused tone; lighter feel with less aggressive transients.",
        enable_post=True,
        enable_polish=True,
        args={
            "fade_ms": 8,
            "normalize_rms_db": -18.0,
            "normalize_peak_db": -1.2,
            "highpass_hz": 55.0,
            "lowpass_hz": 14500.0,
            "multiband": True,
            "mb_low_gain_db": 0.6,
            "mb_mid_gain_db": 1.4,
            "mb_high_gain_db": -0.6,
            "mb_comp_threshold_db": -28.0,
            "mb_comp_ratio": 2.0,
            "transient_attack": -0.25,
            "transient_sustain": 0.25,
            "compressor_attack_ms": 16.0,
            "compressor_release_ms": 180.0,
            "reverb": "off",
            "reverb_mix": 0.0,
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
