from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .io_utils import read_wav_mono
from .samplelib_backend import SampleLibParams, generate_with_samplelib
from .synth_backend import SynthParams, generate_with_synth


def _soft_clip(x: np.ndarray, *, drive: float) -> np.ndarray:
    d = float(np.clip(drive, 0.0, 1.0))
    if d <= 0.0:
        return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    k = 1.0 + 8.0 * d
    return np.tanh(k * x).astype(np.float32, copy=False)


def _fade_in(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size == 0:
        return x
    n = min(int(n), x.size)
    env = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = x.copy()
    y[:n] *= env
    return y


def _fade_out(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size == 0:
        return x
    n = min(int(n), x.size)
    env = np.linspace(1.0, 0.0, n, dtype=np.float32)
    y = x.copy()
    y[-n:] *= env
    return y


def _normalize_peak(x: np.ndarray, *, peak: float = 0.98) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    p = float(np.max(np.abs(x)))
    if p <= 0:
        return x.astype(np.float32, copy=False)
    g = float(peak) / p
    if g >= 1.0:
        return x.astype(np.float32, copy=False)
    return (x * g).astype(np.float32, copy=False)


def _ahd_env(
    n: int,
    *,
    attack_n: int,
    hold_n: int,
    decay_n: int,
    attack_curve: float = 1.0,
    decay_curve: float = 1.0,
) -> np.ndarray:
    """Simple attack-hold-decay envelope, length n, peak 1.

    Any remaining samples after A+H+D are 0.
    """

    n = max(0, int(n))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    a = max(0, int(attack_n))
    h = max(0, int(hold_n))
    d = max(0, int(decay_n))

    if a + h + d > n:
        # Prefer preserving attack/decay shape; trim hold first.
        overflow = (a + h + d) - n
        h = max(0, h - overflow)
        if a + h + d > n:
            # Still too long; trim decay.
            overflow = (a + h + d) - n
            d = max(0, d - overflow)
        if a + h + d > n:
            # Still too long; trim attack.
            overflow = (a + h + d) - n
            a = max(0, a - overflow)

    env = np.zeros(n, dtype=np.float32)
    idx = 0

    if a > 0:
        t = np.linspace(0.0, 1.0, a, dtype=np.float32)
        ac = float(max(0.01, attack_curve))
        env[idx : idx + a] = np.power(t, ac)
        idx += a

    if h > 0:
        env[idx : idx + h] = 1.0
        idx += h

    if d > 0:
        t = np.linspace(1.0, 0.0, d, dtype=np.float32)
        dc = float(max(0.01, decay_curve))
        env[idx : idx + d] = np.power(t, dc)

    return env


def _ms_to_n(ms: float, sr: int) -> int:
    return max(0, int(round(float(ms) * 0.001 * int(sr))))


def _sidechain_duck_gain(
    transient_full: np.ndarray,
    *,
    sr: int,
    amount: float,
    release_ms: float,
    window_ms: float = 4.0,
) -> np.ndarray:
    """Compute a body gain curve (0..1) that ducks based on transient energy.

    Fast attack (follows transient), slow-ish release (envelope follower).
    """

    amt = float(np.clip(amount, 0.0, 1.0))
    if amt <= 0.0 or transient_full.size == 0:
        return np.ones_like(transient_full, dtype=np.float32)

    x = transient_full.astype(np.float32, copy=False)
    # Envelope follower input: short-window RMS-ish via moving average of abs.
    w = max(1, _ms_to_n(window_ms, sr))
    kernel = np.ones(w, dtype=np.float32) / float(w)
    env = np.convolve(np.abs(x), kernel, mode="same")
    m = float(np.max(env))
    if m > 0:
        env = env / m
    env = np.clip(env, 0.0, 1.0)

    # Gain reduction curve with release smoothing.
    target_reduction = amt * env
    release_s = max(0.001, float(release_ms) * 0.001)
    release_coeff = float(np.exp(-1.0 / (float(sr) * release_s)))
    reduction = np.zeros_like(target_reduction, dtype=np.float32)
    r = 0.0
    for i in range(target_reduction.size):
        t = float(target_reduction[i])
        r = max(t, r * release_coeff)
        reduction[i] = r

    gain = 1.0 - reduction
    return np.clip(gain, 0.0, 1.0).astype(np.float32, copy=False)


def _one_pole_lowpass(x: np.ndarray, *, sr: int, cutoff_hz: float) -> np.ndarray:
    """Very small, stable lowpass for 'tilt' splitting (no scipy needed)."""

    if x.size == 0:
        return x.astype(np.float32, copy=False)
    fc = float(max(10.0, min(float(cutoff_hz), 0.45 * float(sr))))
    # Bilinear transform one-pole: alpha in (0,1)
    dt = 1.0 / float(sr)
    rc = 1.0 / (2.0 * np.pi * fc)
    alpha = float(dt / (rc + dt))
    y = np.empty_like(x, dtype=np.float32)
    acc = float(x[0])
    y[0] = acc
    for i in range(1, x.size):
        acc += alpha * (float(x[i]) - acc)
        y[i] = acc
    return y


def _apply_spectral_tilt(
    x: np.ndarray,
    *,
    sr: int,
    tilt: float,
    cutoff_hz: float,
    max_db: float = 6.0,
) -> np.ndarray:
    """Apply a simple spectral tilt by splitting low/high and rebalancing.

    tilt in [-1, 1]. Positive => brighter (more highs), negative => darker.
    """

    t = float(np.clip(tilt, -1.0, 1.0))
    if abs(t) < 1e-6 or x.size == 0:
        return x.astype(np.float32, copy=False)

    low = _one_pole_lowpass(x.astype(np.float32, copy=False), sr=sr, cutoff_hz=float(cutoff_hz))
    high = (x.astype(np.float32, copy=False) - low).astype(np.float32, copy=False)

    tilt_db = float(max_db) * t
    gain_high = float(10.0 ** (tilt_db / 20.0))
    gain_low = float(10.0 ** (-tilt_db / 20.0))
    y = (gain_low * low + gain_high * high).astype(np.float32, copy=False)
    return y


def _seed_offset(seed: int | None, offset: int) -> int:
    base = int(seed) if seed is not None else 1337
    return int((base * 1000003 + offset) & 0x7FFFFFFF)


@dataclass(frozen=True)
class LayeredParams:
    prompt: str
    seconds: float = 3.0
    seed: Optional[int] = None
    sample_rate: int = 44100

    preset: str = "auto"  # auto|ui|impact|whoosh|creature

    # samplelib
    library_zips: tuple[Path, ...] = ()
    library_pitch_min: float = 0.85
    library_pitch_max: float = 1.20
    library_mix_count: int = 1
    library_index_path: Optional[Path] = None

    # layer timing
    transient_ms: int = 110
    tail_ms: int = 350

    # layer gains
    transient_gain: float = 0.90
    body_gain: float = 0.55
    tail_gain: float = 0.65

    # per-layer envelopes (A/H/D) - designers can sculpt without changing sources
    transient_attack_ms: float = 1.0
    transient_hold_ms: float = 10.0
    transient_decay_ms: float = 90.0
    transient_attack_curve: float = 1.0
    transient_decay_curve: float = 1.0

    body_attack_ms: float = 5.0
    body_hold_ms: float = 0.0  # 0 => computed as remainder
    body_decay_ms: float = 80.0
    body_attack_curve: float = 1.0
    body_decay_curve: float = 1.0

    tail_attack_ms: float = 15.0
    tail_hold_ms: float = 0.0  # 0 => computed as remainder
    tail_decay_ms: float = 320.0
    tail_attack_curve: float = 1.0
    tail_decay_curve: float = 1.0

    # Optional envelope curve shape convenience (applies to all layers when set)
    # linear => (1.0, 1.0). exponential => snappier attack + faster initial decay.
    env_curve_shape: str = "linear"  # linear|exponential

    # Per-layer spectral tilt (-1 darker .. +1 brighter)
    transient_tilt: float = 0.0
    body_tilt: float = 0.0
    tail_tilt: float = 0.0

    # layer interaction: transient ducks body
    duck_amount: float = 0.35
    duck_release_ms: float = 90.0

    # Family generation: lock character (preset + core params), vary subtly per variant_index.
    preset_lock: bool = True
    variant_index: int = 0
    micro_variation: float = 0.0  # 0..1

    # Source lock: keep transient/tail samplelib member selection stable even when
    # body seed / variant index changes.
    source_lock: bool = False
    source_seed: Optional[int] = None

    # synth body
    synth_waveform: str = "sine"
    synth_freq_hz: float = 440.0
    synth_pitch_min: float = 0.90
    synth_pitch_max: float = 1.10
    synth_attack_ms: float = 5.0
    synth_decay_ms: float = 80.0
    synth_sustain_level: float = 0.35
    synth_release_ms: float = 120.0
    synth_noise_mix: float = 0.05
    synth_lowpass_hz: float = 16000.0
    synth_highpass_hz: float = 30.0
    synth_drive: float = 0.0


@dataclass(frozen=True)
class LayeredResult:
    audio: np.ndarray
    sample_rate: int
    sources: tuple[dict[str, Any], ...]
    credits_extra: dict[str, Any]


def _choose_body_waveform(prompt: str, rng: random.Random, default: str) -> str:
    p = (prompt or "").lower()
    if any(k in p for k in ("laser", "zap", "sci", "blaster")):
        return "saw"
    if any(k in p for k in ("coin", "pickup", "chime", "blip")):
        return "square"
    if any(k in p for k in ("whoosh", "wind", "swell")):
        return "triangle"
    if any(k in p for k in ("noise", "static", "hiss")):
        return "noise"
    # deterministic small variety
    return default or rng.choice(["sine", "square", "triangle"])  # type: ignore[return-value]


def _auto_preset(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(k in p for k in ("ui", "click", "button", "coin", "pickup", "blip", "menu")):
        return "ui"
    if any(k in p for k in ("whoosh", "swoosh", "wind", "swish", "swell")):
        return "whoosh"
    if any(k in p for k in ("creature", "monster", "growl", "roar")):
        return "creature"
    if any(k in p for k in ("hit", "impact", "thud", "smash", "slam", "punch")):
        return "impact"
    return "impact"


def _apply_preset(p: LayeredParams) -> LayeredParams:
    preset = (p.preset or "auto").strip().lower()
    if preset == "auto":
        preset = _auto_preset(p.prompt)

    if preset == "ui":
        return replace(
            p,
            preset="ui",
            seconds=min(float(p.seconds), 1.2),
            transient_ms=80,
            tail_ms=160,
            transient_gain=0.95,
            body_gain=0.35,
            tail_gain=0.35,
            transient_attack_ms=0.5,
            transient_hold_ms=6.0,
            transient_decay_ms=55.0,
            duck_amount=0.45,
            synth_noise_mix=max(float(p.synth_noise_mix), 0.03),
            synth_drive=min(float(p.synth_drive), 0.25),
        )

    if preset == "whoosh":
        return replace(
            p,
            preset="whoosh",
            transient_ms=70,
            tail_ms=900,
            transient_gain=0.55,
            body_gain=0.65,
            tail_gain=0.90,
            transient_attack_ms=1.0,
            transient_decay_ms=60.0,
            tail_attack_ms=25.0,
            tail_decay_ms=800.0,
            duck_amount=0.15,
            synth_noise_mix=max(float(p.synth_noise_mix), 0.20),
            synth_lowpass_hz=min(float(p.synth_lowpass_hz), 12000.0),
        )

    if preset == "creature":
        return replace(
            p,
            preset="creature",
            transient_ms=120,
            tail_ms=650,
            transient_gain=0.70,
            body_gain=0.75,
            tail_gain=0.70,
            body_attack_ms=10.0,
            body_decay_ms=140.0,
            duck_amount=0.20,
            synth_waveform=("saw" if p.synth_waveform == "sine" else p.synth_waveform),
            synth_noise_mix=max(float(p.synth_noise_mix), 0.10),
            synth_lowpass_hz=min(float(p.synth_lowpass_hz), 9000.0),
            synth_drive=max(float(p.synth_drive), 0.15),
        )

    # default: impact
    return replace(
        p,
        preset="impact",
        transient_ms=110,
        tail_ms=300,
        transient_gain=0.95,
        body_gain=0.55,
        tail_gain=0.55,
        transient_attack_ms=0.8,
        transient_hold_ms=12.0,
        transient_decay_ms=85.0,
        duck_amount=0.40,
    )


def _apply_curve_shape(p: LayeredParams) -> LayeredParams:
    shape = (p.env_curve_shape or "linear").strip().lower()
    if shape == "exponential":
        # Attack: quicker onset (curve < 1). Decay: stronger initial fall (curve > 1).
        return replace(
            p,
            transient_attack_curve=0.55,
            transient_decay_curve=2.4,
            body_attack_curve=0.70,
            body_decay_curve=2.1,
            tail_attack_curve=0.85,
            tail_decay_curve=1.9,
        )
    return replace(
        p,
        transient_attack_curve=1.0,
        transient_decay_curve=1.0,
        body_attack_curve=1.0,
        body_decay_curve=1.0,
        tail_attack_curve=1.0,
        tail_decay_curve=1.0,
    )


def _jitter(rng: random.Random, *, base: float, rel: float) -> float:
    # Multiplicative jitter around base: base * (1 + u*rel)
    u = (rng.random() * 2.0 - 1.0)  # [-1,1]
    return float(base) * float(1.0 + u * float(rel))


def _apply_micro_variation(p: LayeredParams, rng: random.Random) -> LayeredParams:
    mv = float(np.clip(float(p.micro_variation), 0.0, 1.0))
    if mv <= 0.0:
        return p

    # Small, designer-safe jitters. Keep within sane bounds.
    gain_rel = 0.06 * mv
    time_rel = 0.12 * mv
    tilt_rel = 0.25 * mv

    transient_ms = int(np.clip(round(_jitter(rng, base=float(p.transient_ms), rel=time_rel)), 40, 220))
    tail_ms = int(np.clip(round(_jitter(rng, base=float(p.tail_ms), rel=time_rel * 1.5)), 80, 2000))

    return replace(
        p,
        variant_index=int(p.variant_index),
        transient_ms=transient_ms,
        tail_ms=tail_ms,
        transient_gain=float(np.clip(_jitter(rng, base=float(p.transient_gain), rel=gain_rel), 0.05, 2.0)),
        body_gain=float(np.clip(_jitter(rng, base=float(p.body_gain), rel=gain_rel), 0.05, 2.0)),
        tail_gain=float(np.clip(_jitter(rng, base=float(p.tail_gain), rel=gain_rel), 0.05, 2.0)),
        synth_freq_hz=float(np.clip(_jitter(rng, base=float(p.synth_freq_hz), rel=0.05 * mv), 40.0, 8000.0)),
        synth_noise_mix=float(np.clip(_jitter(rng, base=float(p.synth_noise_mix), rel=0.30 * mv), 0.0, 0.8)),
        synth_drive=float(np.clip(_jitter(rng, base=float(p.synth_drive), rel=0.25 * mv), 0.0, 1.0)),
        transient_tilt=float(np.clip(_jitter(rng, base=float(p.transient_tilt), rel=tilt_rel), -1.0, 1.0)),
        body_tilt=float(np.clip(_jitter(rng, base=float(p.body_tilt), rel=tilt_rel), -1.0, 1.0)),
        tail_tilt=float(np.clip(_jitter(rng, base=float(p.tail_tilt), rel=tilt_rel), -1.0, 1.0)),
    )


def generate_with_layered(params: LayeredParams) -> LayeredResult:
    params = _apply_preset(params)
    params = _apply_curve_shape(params)

    sr = int(params.sample_rate)
    n = max(1, int(round(float(params.seconds) * sr)))

    # Family RNG: seed controls character. Variant index controls micro-variation.
    family_seed = _seed_offset(params.seed, 1)
    variant_seed = _seed_offset(params.seed, 1000 + int(params.variant_index))
    rng_family = random.Random(family_seed)
    rng_variant = random.Random(variant_seed)

    prompt = str(params.prompt)
    x = np.zeros(n, dtype=np.float32)
    sources: list[dict[str, Any]] = []

    transient_n = min(n, max(1, int(sr * (int(params.transient_ms) / 1000.0))))
    tail_n = min(n, max(1, int(sr * (int(params.tail_ms) / 1000.0))))

    transient_full = np.zeros(n, dtype=np.float32)

    # Transient + tail from samplelib (if any zips provided)
    # Apply micro-variation after preset/curve choices.
    params = _apply_micro_variation(params, rng_variant)

    # Source RNG base: can be pinned independently from body/variant.
    source_seed_base = params.source_seed if params.source_seed is not None else params.seed

    if params.library_zips:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # Transient (impact/click)
            t_seed = _seed_offset(source_seed_base if bool(params.source_lock) else params.seed, 101)
            t_wav = tmp_dir / "transient.wav"
            t_prompt = f"{prompt} impact click transient"
            t_params = SampleLibParams(
                prompt=t_prompt,
                out_path=t_wav,
                seconds=float(transient_n / sr),
                seed=t_seed,
                library_zips=tuple(params.library_zips),
                pitch_min=float(params.library_pitch_min),
                pitch_max=float(params.library_pitch_max),
                mix_count=int(params.library_mix_count),
                index_path=params.library_index_path,
                sample_rate=sr,
            )
            t_res = generate_with_samplelib(t_params)
            t_audio, _ = read_wav_mono(t_res.out_path)
            t_audio = t_audio[:transient_n]

            # Per-layer envelope (A/H/D)
            ta = _ms_to_n(params.transient_attack_ms, sr)
            th = _ms_to_n(params.transient_hold_ms, sr)
            td = _ms_to_n(params.transient_decay_ms, sr)
            tenv = _ahd_env(
                t_audio.size,
                attack_n=ta,
                hold_n=th,
                decay_n=td,
                attack_curve=float(params.transient_attack_curve),
                decay_curve=float(params.transient_decay_curve),
            )
            t_audio = (t_audio * tenv).astype(np.float32, copy=False)
            # Tiny safety fade to prevent decode clicks
            t_audio = _fade_out(t_audio, max(1, int(0.004 * sr)))

            # Tilt transient identity
            t_audio = _apply_spectral_tilt(t_audio, sr=sr, tilt=float(params.transient_tilt), cutoff_hz=1800.0)

            transient_full[: t_audio.size] = float(params.transient_gain) * t_audio
            x[: t_audio.size] += transient_full[: t_audio.size]
            for s in t_res.sources:
                sources.append(
                    {
                        "layer": "transient",
                        "zip": s.zip_path,
                        "member": s.member,
                        "repo": s.repo,
                        "attribution_files": list(s.attribution_files),
                    }
                )

            # Tail (air/noise/decay) placed at the end
            tail_seed = _seed_offset(source_seed_base if bool(params.source_lock) else params.seed, 202)
            tail_wav = tmp_dir / "tail.wav"
            tail_prompt = f"{prompt} tail decay whoosh"
            tail_params = SampleLibParams(
                prompt=tail_prompt,
                out_path=tail_wav,
                seconds=float(tail_n / sr),
                seed=tail_seed,
                library_zips=tuple(params.library_zips),
                pitch_min=float(params.library_pitch_min),
                pitch_max=float(params.library_pitch_max),
                mix_count=1,
                index_path=params.library_index_path,
                sample_rate=sr,
            )
            tail_res = generate_with_samplelib(tail_params)
            tail_audio, _ = read_wav_mono(tail_res.out_path)
            tail_audio = tail_audio[:tail_n]

            # Per-layer envelope (A/H/D)
            ta = _ms_to_n(params.tail_attack_ms, sr)
            th = _ms_to_n(params.tail_hold_ms, sr)
            td = _ms_to_n(params.tail_decay_ms, sr)
            tenv = _ahd_env(
                tail_audio.size,
                attack_n=ta,
                hold_n=th,
                decay_n=td,
                attack_curve=float(params.tail_attack_curve),
                decay_curve=float(params.tail_decay_curve),
            )
            tail_audio = (tail_audio * tenv).astype(np.float32, copy=False)
            tail_audio = _fade_in(tail_audio, max(1, int(0.004 * sr)))

            # Tilt tail emotion/space
            tail_audio = _apply_spectral_tilt(tail_audio, sr=sr, tilt=float(params.tail_tilt), cutoff_hz=2400.0)

            start = max(0, n - tail_audio.size)
            x[start : start + tail_audio.size] += float(params.tail_gain) * tail_audio
            for s in tail_res.sources:
                sources.append(
                    {
                        "layer": "tail",
                        "zip": s.zip_path,
                        "member": s.member,
                        "repo": s.repo,
                        "attribution_files": list(s.attribution_files),
                    }
                )

    # Body from synth across full duration
    body_seed = _seed_offset(params.seed, 303)
    if bool(params.preset_lock):
        # Lock waveform choice to the family, not prompt heuristics.
        waveform = str(params.synth_waveform) or rng_family.choice(["sine", "square", "triangle"])
        if waveform == "sine":
            waveform = rng_family.choice(["sine", "square", "triangle"])
    else:
        waveform = _choose_body_waveform(prompt, rng_family, str(params.synth_waveform))
    synth = SynthParams(
        prompt=prompt,
        seconds=float(params.seconds),
        seed=body_seed,
        sample_rate=sr,
        waveform=waveform,
        freq_hz=float(params.synth_freq_hz),
        pitch_min=float(params.synth_pitch_min),
        pitch_max=float(params.synth_pitch_max),
        attack_ms=float(params.synth_attack_ms),
        decay_ms=float(params.synth_decay_ms),
        sustain_level=float(params.synth_sustain_level),
        release_ms=float(params.synth_release_ms),
        noise_mix=float(params.synth_noise_mix),
        lowpass_hz=float(params.synth_lowpass_hz),
        highpass_hz=float(params.synth_highpass_hz),
        drive=float(params.synth_drive),
    )
    body, _ = generate_with_synth(synth)
    body = body[:n]

    # Layer-level body envelope (A/H/D)
    ba = _ms_to_n(params.body_attack_ms, sr)
    bd = _ms_to_n(params.body_decay_ms, sr)
    if float(params.body_hold_ms) <= 0.0:
        bh = max(0, body.size - ba - bd)
    else:
        bh = _ms_to_n(params.body_hold_ms, sr)
    benv = _ahd_env(
        body.size,
        attack_n=ba,
        hold_n=bh,
        decay_n=bd,
        attack_curve=float(params.body_attack_curve),
        decay_curve=float(params.body_decay_curve),
    )
    body = (body * benv).astype(np.float32, copy=False)

    # Tilt body character
    body = _apply_spectral_tilt(body, sr=sr, tilt=float(params.body_tilt), cutoff_hz=1600.0)

    # Interaction rule: transient ducks body.
    duck_gain = _sidechain_duck_gain(
        transient_full,
        sr=sr,
        amount=float(params.duck_amount),
        release_ms=float(params.duck_release_ms),
    )
    body = (body * duck_gain[: body.size]).astype(np.float32, copy=False)

    x[: body.size] += float(params.body_gain) * body

    # Slight glue: soft clip then normalize peak
    x = _soft_clip(x, drive=0.10)
    x = _normalize_peak(x, peak=0.98)
    x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)

    credits_extra = {
        "layered": {
            "preset": str(params.preset),
            "preset_lock": bool(params.preset_lock),
            "variant_index": int(params.variant_index),
            "micro_variation": float(params.micro_variation),
            "source_lock": bool(params.source_lock),
            "source_seed": (int(source_seed_base) if source_seed_base is not None else None),
            "transient_ms": int(params.transient_ms),
            "tail_ms": int(params.tail_ms),
            "gains": {
                "transient": float(params.transient_gain),
                "body": float(params.body_gain),
                "tail": float(params.tail_gain),
            },
            "envelopes_ms": {
                "transient": {
                    "attack": float(params.transient_attack_ms),
                    "hold": float(params.transient_hold_ms),
                    "decay": float(params.transient_decay_ms),
                },
                "body": {
                    "attack": float(params.body_attack_ms),
                    "hold": float(params.body_hold_ms),
                    "decay": float(params.body_decay_ms),
                },
                "tail": {
                    "attack": float(params.tail_attack_ms),
                    "hold": float(params.tail_hold_ms),
                    "decay": float(params.tail_decay_ms),
                },
            },
            "env_curve_shape": str(params.env_curve_shape),
            "tilt": {
                "transient": float(params.transient_tilt),
                "body": float(params.body_tilt),
                "tail": float(params.tail_tilt),
            },
            "interaction": {
                "duck_amount": float(params.duck_amount),
                "duck_release_ms": float(params.duck_release_ms),
            },
            "body_waveform": waveform,
            "body_freq_hz": float(params.synth_freq_hz),
        }
    }

    return LayeredResult(
        audio=x,
        sample_rate=sr,
        sources=tuple(sources),
        credits_extra=credits_extra,
    )
