from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ProPreset:
    """High-level, 'paid-tool-like' preset.

    This is intentionally engine-agnostic and mostly expressed as CLI/web knobs
    that already exist: polish/post, conditioning channels, and DSP modules.

    Presets are applied *conservatively*: we only override values that are still
    at their argparse defaults (so users can still manually tweak knobs).
    """

    title: str
    description: str

    # Optional prompt augmentation. Applied for AI/samplelib-style engines only.
    prompt_suffix: str | None = None

    # Recommended named polish profile (does not auto-apply; used for UI/help/traceability).
    recommended_polish_profile: str | None = None

    # Args overrides (dest -> value). Applied only when the current value equals
    # parser default.
    args: dict[str, Any] | None = None


def infer_recommended_polish_profile(preset_key: str) -> str | None:
    """Best-effort recommendation mapping.

    Keeps the system "designed": every preset can suggest a profile, but users
    remain free to pick a different profile (or keep it off).
    """

    key = str(preset_key or "").strip().lower()
    if not key or key == "off":
        return None

    if key.startswith("creature."):
        if "ghost" in key or "wail" in key:
            return "creature_hushed"
        if "whisper" in key or "ethereal" in key:
            return "creature_hushed"
        if "roar" in key or "large" in key:
            return "creature_low_end"
        if "scream" in key or "screech" in key:
            return "creature_snappy"
        if "chitter" in key or "buzz" in key or "insect" in key:
            return "creature_snappy"
        return "creature_gritty"

    if key.startswith("env."):
        if "loop" in key:
            return "ambience_loop_ready"
        if "cave" in key or "lava" in key:
            return "ambience_warm_mono"
        if "magical" in key:
            return "ambience_glue_open"
        return "ambience_smooth"

    if key.startswith("ui."):
        return "ui_clean"

    if key.startswith("impact."):
        return "impact_hard_punch"

    if key.startswith("foley."):
        return "foley_punchy"

    return "ui_clean"


def get_pro_preset(preset_key: str) -> ProPreset | None:
    key = str(preset_key or "").strip()
    if not key or key.lower() == "off":
        return None

    obj = PRO_PRESETS.get(key)
    if obj is None:
        return None
    if obj.recommended_polish_profile:
        return obj
    rec = infer_recommended_polish_profile(key)
    return replace(obj, recommended_polish_profile=rec)


def pro_preset_recommended_profile(preset_key: str) -> str | None:
    obj = get_pro_preset(preset_key)
    return None if obj is None else obj.recommended_polish_profile


# Keep keys stable (used in CLI/web).
PRO_PRESETS: dict[str, ProPreset] = {
    "creature.small_chitter": ProPreset(
        title="Creature: Small chitter",
        description="Fast insectoid chatter; crisp attacks; light room tone.",
        prompt_suffix="small creature chitter, insectoid, fast clicks, sharp transients",
        args={
            "post": True,
            "polish": True,
            "emotion": "scared",
            "intensity": 0.55,
            "variation": 0.35,
            "pitch_contour": "updown",
            "seconds": 1.2,
            "texture_preset": "chitter",
            "texture_amount": 0.55,
            "reverb": "off",
            "creature_size": -0.25,
            "multiband": True,
        },
    ),
    "creature.medium_growl": ProPreset(
        title="Creature: Medium growl",
        description="Focused midrange; controlled lows; moderate aggression.",
        prompt_suffix="creature growl, gritty, textured, clear midrange",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.65,
            "variation": 0.25,
            "pitch_contour": "fall",
            "seconds": 2.6,
            "texture_preset": "rasp",
            "texture_amount": 0.30,
            "reverb": "off",
            "creature_size": 0.15,
            "multiband": True,
        },
    ),
    "creature.large_roar": ProPreset(
        title="Creature: Large roar",
        description="Big body + controlled sub; longer tail; subtle space.",
        prompt_suffix="massive creature roar, powerful low end, wide chest resonance",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.85,
            "variation": 0.20,
            "pitch_contour": "fall",
            "seconds": 4.0,
            "texture_preset": "rasp",
            "texture_amount": 0.22,
            "reverb": "cave",
            "reverb_mix": 0.12,
            "reverb_time": 1.6,
            "creature_size": 0.75,
            "multiband": True,
        },
    ),
    "creature.insectoid_buzz": ProPreset(
        title="Creature: Insectoid buzz",
        description="Noisy buzz + chirps; bright; minimal lows.",
        prompt_suffix="insect buzz, wing flutter, chittering, bright",
        args={
            "post": True,
            "polish": True,
            "emotion": "scared",
            "intensity": 0.60,
            "variation": 0.50,
            "pitch_contour": "rise",
            "seconds": 2.0,
            "highpass_hz": 120.0,
            "texture_preset": "buzz",
            "texture_amount": 0.55,
            "reverb": "off",
            "creature_size": -0.35,
            "multiband": True,
        },
    ),
    "creature.ethereal_whisper": ProPreset(
        title="Creature: Ethereal whisper",
        description="Soft, airy, tail-heavy; gentle reverb.",
        prompt_suffix="ethereal whisper, airy, breathy, spectral",
        args={
            "post": True,
            "polish": True,
            "emotion": "calm",
            "intensity": 0.35,
            "variation": 0.25,
            "pitch_contour": "flat",
            "seconds": 3.2,
            "lowpass_hz": 13000.0,
            "reverb": "forest",
            "reverb_mix": 0.18,
            "reverb_time": 1.9,
            "creature_size": 0.10,
            "multiband": True,
        },
    ),
    "creature.undead_rasp": ProPreset(
        title="Creature: Undead rasp",
        description="Dry rasp + noise grit; controlled highs; no space.",
        prompt_suffix="undead rasp, gritty, breath noise, harsh texture",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.70,
            "variation": 0.30,
            "pitch_contour": "downup",
            "seconds": 2.4,
            "texture_preset": "rasp",
            "texture_amount": 0.55,
            "lowpass_hz": 14500.0,
            "reverb": "off",
            "creature_size": 0.35,
            "multiband": True,
        },
    ),
    "creature.slime_gurgle": ProPreset(
        title="Creature: Slime gurgle",
        description="Wet gurgle feel via darker tone + mild room reverb.",
        prompt_suffix="slime gurgle, wet squish, bubbly",
        args={
            "post": True,
            "polish": True,
            "emotion": "neutral",
            "intensity": 0.55,
            "variation": 0.35,
            "pitch_contour": "updown",
            "seconds": 2.2,
            "lowpass_hz": 12000.0,
            "reverb": "room",
            "reverb_mix": 0.10,
            "reverb_time": 1.3,
            "creature_size": 0.55,
            "multiband": True,
        },
    ),
    "creature.serpent_hiss": ProPreset(
        title="Creature: Serpent hiss",
        description="Tight hiss with subtle breath noise; light tail; minimal lows.",
        prompt_suffix="serpent hiss, tight sibilant, breath noise, threatening",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.50,
            "variation": 0.20,
            "pitch_contour": "rise",
            "seconds": 1.6,
            "highpass_hz": 120.0,
            "lowpass_hz": 15000.0,
            "texture_preset": "off",
            "texture_amount": 0.0,
            "reverb": "off",
            "creature_size": -0.05,
            "formant_shift": 1.15,
            "multiband": True,
        },
    ),
    "creature.goblin_bark": ProPreset(
        title="Creature: Goblin bark",
        description="Short, nasal bark; snappy transient; dry and readable.",
        prompt_suffix="goblin bark, short vocal, nasal, snappy",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.70,
            "variation": 0.35,
            "pitch_contour": "updown",
            "seconds": 1.1,
            "texture_preset": "rasp",
            "texture_amount": 0.20,
            "reverb": "off",
            "creature_size": -0.15,
            "formant_shift": 1.10,
            "transient_attack": 0.20,
            "multiband": True,
        },
    ),
    "creature.orc_brute_bark": ProPreset(
        title="Creature: Orc brute bark",
        description="Mid-low bark; heavier body; restrained tail; small room feel.",
        prompt_suffix="orc bark, brute, heavy chest, gritty",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.78,
            "variation": 0.25,
            "pitch_contour": "fall",
            "seconds": 1.7,
            "texture_preset": "rasp",
            "texture_amount": 0.24,
            "reverb": "room",
            "reverb_mix": 0.08,
            "reverb_time": 1.3,
            "creature_size": 0.40,
            "multiband": True,
        },
    ),
    "creature.wolf_snarl": ProPreset(
        title="Creature: Wolf snarl",
        description="Tense snarl; controlled lows; dry; bite without mud.",
        prompt_suffix="wolf snarl, animal growl, tense, teeth",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.70,
            "variation": 0.30,
            "pitch_contour": "downup",
            "seconds": 2.1,
            "texture_preset": "rasp",
            "texture_amount": 0.18,
            "reverb": "off",
            "creature_size": 0.10,
            "multiband": True,
        },
    ),
    "creature.ghost_wail": ProPreset(
        title="Creature: Ghost wail",
        description="Airy, haunting wail; longer tail; space-forward without harshness.",
        prompt_suffix="ghost wail, airy, haunting, spectral",
        args={
            "post": True,
            "polish": True,
            "emotion": "scared",
            "intensity": 0.45,
            "variation": 0.25,
            "pitch_contour": "rise",
            "seconds": 4.2,
            "lowpass_hz": 14500.0,
            "reverb": "cave",
            "reverb_mix": 0.22,
            "reverb_time": 2.4,
            "creature_size": 0.10,
            "multiband": True,
        },
    ),
    "creature.demon_scream": ProPreset(
        title="Creature: Demon scream",
        description="Bright scream with grit; controlled low end; minimal tail.",
        prompt_suffix="demon scream, harsh, intense, bright, tearing",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.85,
            "variation": 0.35,
            "pitch_contour": "rise",
            "seconds": 2.2,
            "highpass_hz": 70.0,
            "texture_preset": "screech",
            "texture_amount": 0.48,
            "reverb": "nether",
            "reverb_mix": 0.08,
            "reverb_time": 1.6,
            "creature_size": 0.55,
            "multiband": True,
        },
    ),
    "creature.dragon_bellow": ProPreset(
        title="Creature: Dragon bellow",
        description="Massive bellow; big body; controlled sub; longer tail.",
        prompt_suffix="dragon bellow, massive roar, powerful chest resonance",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.92,
            "variation": 0.20,
            "pitch_contour": "fall",
            "seconds": 5.2,
            "texture_preset": "rasp",
            "texture_amount": 0.18,
            "reverb": "cave",
            "reverb_mix": 0.12,
            "reverb_time": 2.2,
            "creature_size": 1.00,
            "multiband": True,
        },
    ),
    "creature.avian_screech": ProPreset(
        title="Creature: Avian screech",
        description="Piercing screech; fast attack; very light lows; dry.",
        prompt_suffix="avian screech, bird shriek, sharp, piercing",
        args={
            "post": True,
            "polish": True,
            "emotion": "scared",
            "intensity": 0.75,
            "variation": 0.40,
            "pitch_contour": "updown",
            "seconds": 1.4,
            "highpass_hz": 140.0,
            "texture_preset": "screech",
            "texture_amount": 0.40,
            "reverb": "off",
            "creature_size": -0.20,
            "multiband": True,
        },
    ),
    "creature.frog_croak": ProPreset(
        title="Creature: Frog croak",
        description="Round croak; darker tone; small space; gentle dynamics.",
        prompt_suffix="frog croak, round, wet throat, short vocal",
        args={
            "post": True,
            "polish": True,
            "emotion": "neutral",
            "intensity": 0.55,
            "variation": 0.35,
            "pitch_contour": "updown",
            "seconds": 1.8,
            "lowpass_hz": 12000.0,
            "reverb": "forest",
            "reverb_mix": 0.06,
            "reverb_time": 1.4,
            "creature_size": 0.20,
            "multiband": True,
        },
    ),
    "creature.aquatic_gurgle": ProPreset(
        title="Creature: Aquatic gurgle",
        description="Bubbly gurgle; darker highs; short tail; subtle room.",
        prompt_suffix="aquatic gurgle, bubbles, wet, muffled",
        args={
            "post": True,
            "polish": True,
            "emotion": "neutral",
            "intensity": 0.55,
            "variation": 0.35,
            "pitch_contour": "updown",
            "seconds": 2.0,
            "lowpass_hz": 11500.0,
            "reverb": "room",
            "reverb_mix": 0.07,
            "reverb_time": 1.3,
            "creature_size": 0.35,
            "multiband": True,
        },
    ),
    "creature.tiny_squeak": ProPreset(
        title="Creature: Tiny squeak",
        description="Very short squeak; bright; almost no lows; super clean.",
        prompt_suffix="tiny creature squeak, small critter, quick chirp",
        args={
            "post": True,
            "polish": True,
            "emotion": "scared",
            "intensity": 0.50,
            "variation": 0.45,
            "pitch_contour": "rise",
            "seconds": 0.6,
            "highpass_hz": 180.0,
            "reverb": "off",
            "creature_size": -0.60,
            "formant_shift": 1.25,
            "multiband": True,
        },
    ),
    "creature.golem_grind": ProPreset(
        title="Creature: Golem grind",
        description="Stone/metal creature grind; gritty mids; controlled tail.",
        prompt_suffix="golem grind, stone, metal, mechanical creature, heavy",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.65,
            "variation": 0.35,
            "pitch_contour": "fall",
            "seconds": 2.6,
            "highpass_hz": 70.0,
            "texture_preset": "chitter",
            "texture_amount": 0.22,
            "reverb": "cave",
            "reverb_mix": 0.08,
            "reverb_time": 1.8,
            "creature_size": 0.55,
            "transient_attack": 0.25,
            "multiband": True,
        },
    ),
    "env.cave_drone": ProPreset(
        title="Environment: Cave drone",
        description="Long, dark ambience; smooth dynamics; larger space.",
        prompt_suffix="cave ambience drone, deep rumble, distant air",
        args={
            "post": True,
            "polish": True,
            "emotion": "calm",
            "intensity": 0.40,
            "variation": 0.20,
            "pitch_contour": "flat",
            "seconds": 6.0,
            "highpass_hz": 25.0,
            "reverb": "cave",
            "reverb_mix": 0.22,
            "reverb_time": 2.6,
            "creature_size": 0.50,
            "multiband": True,
        },
    ),
    "env.lava_rumble": ProPreset(
        title="Environment: Lava rumble",
        description="Sub-heavy rumble; controlled highs; slight space.",
        prompt_suffix="lava rumble, bubbling, subterranean, heavy low end",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.60,
            "variation": 0.25,
            "pitch_contour": "flat",
            "seconds": 5.0,
            "lowpass_hz": 12000.0,
            "reverb": "nether",
            "reverb_mix": 0.10,
            "reverb_time": 1.8,
            "creature_size": 0.65,
            "multiband": True,
        },
    ),
    "env.forest_wind": ProPreset(
        title="Environment: Forest wind",
        description="Airy whoosh; brighter; moderate reverb.",
        prompt_suffix="forest wind, leaves rustle, airy whoosh",
        args={
            "post": True,
            "polish": True,
            "emotion": "calm",
            "intensity": 0.35,
            "variation": 0.35,
            "pitch_contour": "rise",
            "seconds": 5.0,
            "highpass_hz": 60.0,
            "reverb": "forest",
            "reverb_mix": 0.14,
            "reverb_time": 2.2,
            "creature_size": -0.10,
            "multiband": True,
        },
    ),
    "env.magical_hum": ProPreset(
        title="Environment: Magical hum",
        description="Smooth harmonic feel; gentle polish; subtle room.",
        prompt_suffix="magical hum, tonal, shimmering, harmonic",
        args={
            "post": True,
            "polish": True,
            "emotion": "neutral",
            "intensity": 0.45,
            "variation": 0.30,
            "pitch_contour": "updown",
            "seconds": 4.0,
            "reverb": "room",
            "reverb_mix": 0.12,
            "reverb_time": 1.5,
            "creature_size": 0.00,
            "multiband": True,
        },
    ),
    "env.mechanical_ambience": ProPreset(
        title="Environment: Mechanical ambience",
        description="Bright chatter + grit; controlled tail; minimal reverb.",
        prompt_suffix="mechanical ambience, gears, servos, clicks, industrial",
        args={
            "post": True,
            "polish": True,
            "emotion": "neutral",
            "intensity": 0.60,
            "variation": 0.45,
            "pitch_contour": "flat",
            "seconds": 4.0,
            "texture_preset": "chitter",
            "texture_amount": 0.30,
            "reverb": "off",
            "creature_size": -0.05,
            "multiband": True,
        },
    ),
    "foley.bone_crack": ProPreset(
        title="Foley: Bone crack",
        description="Short, snappy transient; dry; mid-forward.",
        prompt_suffix="bone crack, snap, dry, sharp transient",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.80,
            "variation": 0.20,
            "pitch_contour": "downup",
            "seconds": 0.8,
            "reverb": "off",
            "creature_size": -0.10,
            "multiband": True,
        },
    ),
    "foley.flesh_impact": ProPreset(
        title="Foley: Flesh impact",
        description="Thumpy body; short tail; slight room.",
        prompt_suffix="flesh impact, thud, wet hit",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.75,
            "variation": 0.25,
            "pitch_contour": "fall",
            "seconds": 1.1,
            "reverb": "room",
            "reverb_mix": 0.07,
            "reverb_time": 1.2,
            "creature_size": 0.25,
            "multiband": True,
        },
    ),
    "foley.metal_scrape": ProPreset(
        title="Foley: Metal scrape",
        description="Bright scrape; controlled low end; dry.",
        prompt_suffix="metal scrape, harsh, bright, friction",
        args={
            "post": True,
            "polish": True,
            "emotion": "aggressive",
            "intensity": 0.60,
            "variation": 0.45,
            "pitch_contour": "rise",
            "seconds": 1.6,
            "highpass_hz": 80.0,
            "reverb": "off",
            "creature_size": -0.25,
            "multiband": True,
        },
    ),
    "foley.footstep_variations": ProPreset(
        title="Foley: Footstep variations",
        description="Short steps; encourages variation; minimal space.",
        prompt_suffix="footsteps, short impacts, subtle grit",
        args={
            "post": True,
            "polish": True,
            "emotion": "neutral",
            "intensity": 0.55,
            "variation": 0.65,
            "pitch_contour": "flat",
            "seconds": 0.7,
            "reverb": "off",
            "creature_size": 0.05,
            "multiband": True,
        },
    ),
}


def pro_preset_keys() -> list[str]:
    return sorted(PRO_PRESETS.keys())


def apply_pro_preset(
    *,
    preset_key: str | None,
    args: Any,
    parser: Any,
) -> None:
    """Apply a pro preset to an argparse.Namespace in-place.

    We only override a destination if its current value equals the parser's
    default for that destination. This keeps presets from clobbering explicit
    user tuning.
    """

    if not preset_key:
        return

    key = str(preset_key).strip()
    if not key or key.lower() in {"off", "none", "false", "0"}:
        return

    preset = PRO_PRESETS.get(key)
    if preset is None:
        raise ValueError(f"Unknown pro preset: {key}")

    engine = str(getattr(args, "engine", "")).strip().lower()

    # Apply prompt augmentation for AI/sample selection engines.
    if preset.prompt_suffix and engine in {"diffusers", "stable_audio_open", "replicate", "samplelib", "layered"}:
        base_prompt = str(getattr(args, "prompt"))
        suffix = str(preset.prompt_suffix).strip()
        if suffix and suffix.lower() not in base_prompt.lower():
            setattr(args, "prompt", base_prompt.rstrip() + ", " + suffix)

    overrides = preset.args or {}
    for dest, value in overrides.items():
        try:
            default_value = parser.get_default(dest)
        except Exception:
            default_value = None

        current = getattr(args, dest, None)

        # Only override when user hasn't deviated from default.
        if current == default_value:
            setattr(args, dest, value)
