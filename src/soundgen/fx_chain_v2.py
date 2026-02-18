from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import signal

from .io_utils import read_wav_mono


class FxChainV2FormatError(ValueError):
    pass


@dataclass(frozen=True)
class FxStepV2:
    module_id: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FxChainV2:
    name: str
    steps: tuple[FxStepV2, ...]
    description: str | None = None


@dataclass(frozen=True)
class FxModuleV2:
    module_id: str
    title: str
    description: str
    apply: Callable[[np.ndarray, int, dict[str, Any]], np.ndarray]


def _clip_mono(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        return bool(int(v) != 0)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(v)


def _ensure_mono_f32(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("FX chain expects mono 1D audio")
    return y


def _resample_poly(x: np.ndarray, *, up: int, down: int) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    y = signal.resample_poly(x, up, down)
    return y.astype(np.float32, copy=False)


def _resample_to_sr(x: np.ndarray, *, sr_in: int, sr_out: int) -> np.ndarray:
    if int(sr_in) == int(sr_out):
        return x.astype(np.float32, copy=False)
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    g = math.gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // g
    down = int(sr_in) // g
    return _resample_poly(x, up=up, down=down)


def _phase_vocoder_time_stretch(
    x: np.ndarray,
    sr: int,
    *,
    rate: float,
    n_fft: int = 2048,
    hop_length: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """Time-stretch using a phase vocoder.

    rate > 1.0 makes audio shorter (faster). rate < 1.0 makes it longer.
    """

    rr = float(rate)
    if x.size == 0 or abs(rr - 1.0) < 1e-6:
        return x.astype(np.float32, copy=False)
    if rr <= 0.0:
        raise ValueError("time_stretch rate must be > 0")

    n_fft = int(max(256, min(int(n_fft), 8192)))
    hop = int(hop_length) if hop_length is not None else int(n_fft // 4)
    hop = int(max(32, min(hop, n_fft - 1)))

    # STFT
    f, t, Z = signal.stft(
        x.astype(np.float32, copy=False),
        fs=float(sr),
        window=window,
        nperseg=n_fft,
        noverlap=(n_fft - hop),
        boundary=None,
        padded=False,
    )
    if Z.size == 0:
        return x.astype(np.float32, copy=False)

    n_bins, n_frames = Z.shape
    time_steps = np.arange(0.0, float(n_frames - 1), float(rr), dtype=np.float64)
    if time_steps.size < 2:
        return x.astype(np.float32, copy=False)

    omega = (2.0 * np.pi * hop) * (np.arange(n_bins, dtype=np.float64) / float(n_fft))

    phase_acc = np.angle(Z[:, 0]).astype(np.float64)
    last_phase = np.angle(Z[:, 0]).astype(np.float64)

    out = np.zeros((n_bins, int(time_steps.size)), dtype=np.complex64)

    for out_idx, step in enumerate(time_steps):
        i = int(np.floor(step))
        frac = float(step - float(i))
        if i + 1 >= n_frames:
            break

        a = Z[:, i]
        b = Z[:, i + 1]

        mag = (1.0 - frac) * np.abs(a) + frac * np.abs(b)

        phase = np.angle(a)
        phase_next = np.angle(b)

        # Phase advance
        delta = (phase_next - phase) - omega
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        true_advance = omega + delta

        phase_acc += true_advance
        out[:, out_idx] = (mag * np.exp(1.0j * phase_acc)).astype(np.complex64, copy=False)

        last_phase = phase_next

    # ISTFT
    _, y = signal.istft(
        out,
        fs=float(sr),
        window=window,
        nperseg=n_fft,
        noverlap=(n_fft - hop),
        input_onesided=True,
        boundary=None,
    )
    return y.astype(np.float32, copy=False)


def _fx_time_stretch(x: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    rate = float(params.get("rate", 1.0))
    n_fft = int(params.get("n_fft", 2048))
    hop = params.get("hop_length", None)
    hop_i = (int(hop) if hop is not None else None)
    return _phase_vocoder_time_stretch(x, sr, rate=rate, n_fft=n_fft, hop_length=hop_i)


def _fx_pitch_shift(x: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    semitones = float(params.get("semitones", 0.0))
    if abs(semitones) < 1e-6:
        return x.astype(np.float32, copy=False)

    factor = float(2.0 ** (semitones / 12.0))
    # Pitch shift without changing duration:
    # 1) time-stretch by 1/factor
    # 2) resample back by factor
    stretched = _phase_vocoder_time_stretch(x, sr, rate=(1.0 / factor), n_fft=int(params.get("n_fft", 2048)))

    # Resample to original duration by changing sample rate and then trunc/pad.
    y = _resample_to_sr(stretched, sr_in=int(sr), sr_out=int(round(float(sr) * factor)))

    # Now bring back to original SR.
    y = _resample_to_sr(y, sr_in=int(round(float(sr) * factor)), sr_out=int(sr))

    # Preserve original length (most SFX workflows expect this).
    target = int(x.size)
    if y.size > target:
        y = y[:target]
    elif y.size < target:
        y2 = np.zeros(target, dtype=np.float32)
        y2[: y.size] = y
        y = y2
    return y.astype(np.float32, copy=False)


def _fx_distortion(x: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    _ = sr
    drive = float(np.clip(float(params.get("drive", 0.0)), 0.0, 1.0))
    mode = str(params.get("mode", "tanh")).strip().lower()
    mix = float(np.clip(float(params.get("mix", 1.0)), 0.0, 1.0))

    if x.size == 0 or drive <= 1e-6 or mix <= 1e-6:
        return x.astype(np.float32, copy=False)

    pre = float(1.0 + 12.0 * drive)
    y = x.astype(np.float32, copy=False) * pre

    if mode in {"hard", "hardclip", "clip"}:
        y2 = np.clip(y, -1.0, 1.0)
    elif mode in {"soft", "softclip", "soft_clip"}:
        # Polynomial soft clip
        y2 = y - (y**3) / 3.0
        y2 = np.clip(y2, -1.0, 1.0)
    else:
        # tanh by default
        y2 = np.tanh(y)

    out = (1.0 - mix) * x + mix * y2
    return _clip_mono(out)


def _load_ir_mono(path: str | Path, *, sr: int) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ir, ir_sr = read_wav_mono(p)
    if ir.size == 0:
        return ir.astype(np.float32, copy=False)
    ir = ir.astype(np.float32, copy=False)
    if int(ir_sr) != int(sr):
        ir = _resample_to_sr(ir, sr_in=int(ir_sr), sr_out=int(sr))
    return ir


def _fx_convolution_reverb(x: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    path = params.get("ir_path") or params.get("path") or params.get("ir")
    if not path:
        return x.astype(np.float32, copy=False)

    mix = float(np.clip(float(params.get("mix", 0.25)), 0.0, 1.0))
    if x.size == 0 or mix <= 1e-6:
        return x.astype(np.float32, copy=False)

    normalize_ir = _parse_bool(params.get("normalize_ir", True))
    tail_s = params.get("tail_s", 1.0)
    pre_delay_ms = float(params.get("pre_delay_ms", 0.0))

    ir = _load_ir_mono(str(path), sr=int(sr))
    if ir.size == 0:
        return x.astype(np.float32, copy=False)

    if normalize_ir:
        peak = float(np.max(np.abs(ir))) if ir.size else 0.0
        if peak > 1e-6:
            ir = (ir / peak).astype(np.float32, copy=False)

    if pre_delay_ms > 0.0:
        pad = int(round(float(sr) * (pre_delay_ms / 1000.0)))
        if pad > 0:
            ir = np.pad(ir, (pad, 0), mode="constant")

    wet = signal.fftconvolve(x.astype(np.float32, copy=False), ir, mode="full").astype(np.float32, copy=False)

    if tail_s is not None:
        keep = int(x.size) + int(round(float(sr) * float(tail_s)))
        wet = wet[: max(1, min(int(wet.size), int(keep)))]

    # Mix dry into the front, then pad dry if needed.
    dry = x.astype(np.float32, copy=False)
    if wet.size > dry.size:
        dry2 = np.zeros(int(wet.size), dtype=np.float32)
        dry2[: dry.size] = dry
        dry = dry2

    out = (1.0 - mix) * dry + mix * wet
    return _clip_mono(out)


def _butter(x: np.ndarray, sr: int, *, kind: str, f1: float, f2: float | None = None, order: int = 2) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    nyq = 0.5 * float(sr)
    if kind in {"lowpass", "highpass"}:
        wn = float(f1) / nyq
        if wn <= 0.0 or wn >= 1.0:
            return x.astype(np.float32, copy=False)
        b, a = signal.butter(int(order), wn, btype=kind)
        return signal.filtfilt(b, a, x).astype(np.float32, copy=False)

    # bandpass
    assert f2 is not None
    lo = float(f1) / nyq
    hi = float(f2) / nyq
    lo = float(np.clip(lo, 1e-4, 0.999))
    hi = float(np.clip(hi, lo + 1e-4, 0.999))
    b, a = signal.butter(int(order), [lo, hi], btype="bandpass")
    return signal.filtfilt(b, a, x).astype(np.float32, copy=False)


def _fx_multiband_eq(x: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    low_hz = float(params.get("low_hz", 250.0))
    high_hz = float(params.get("high_hz", 3000.0))
    low_gain_db = float(params.get("low_gain_db", 0.0))
    mid_gain_db = float(params.get("mid_gain_db", 0.0))
    high_gain_db = float(params.get("high_gain_db", 0.0))

    if x.size == 0:
        return x.astype(np.float32, copy=False)

    low_hz = float(np.clip(low_hz, 40.0, 0.45 * float(sr)))
    high_hz = float(np.clip(high_hz, low_hz + 20.0, 0.49 * float(sr)))

    low = _butter(x, sr, kind="lowpass", f1=low_hz)
    band = _butter(x, sr, kind="bandpass", f1=low_hz, f2=high_hz)
    high = (x - low - band).astype(np.float32, copy=False)

    y = (
        low * _db_to_amp(low_gain_db)
        + band * _db_to_amp(mid_gain_db)
        + high * _db_to_amp(high_gain_db)
    )
    return _clip_mono(y)


def _fx_transient_shaper(x: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    # Lightweight transient shaper inspired by postprocess._transient_shaper_attack_sustain.
    attack = float(np.clip(float(params.get("attack", 0.0)), -1.0, 1.0))
    sustain = float(np.clip(float(params.get("sustain", 0.0)), -1.0, 1.0))
    split_hz = float(params.get("split_hz", 1200.0))

    if x.size == 0 or (abs(attack) < 1e-6 and abs(sustain) < 1e-6):
        return x.astype(np.float32, copy=False)

    split = float(np.clip(split_hz, 150.0, 0.45 * float(sr)))
    low = _butter(x, sr, kind="lowpass", f1=split)
    high = (x - low).astype(np.float32, copy=False)

    env = np.abs(high).astype(np.float32, copy=False)
    # Fast vs slow envelope.
    fast_tau_s = 0.006
    slow_tau_s = 0.080
    a_fast = float(np.exp(-1.0 / max(1.0, float(sr) * fast_tau_s)))
    a_slow = float(np.exp(-1.0 / max(1.0, float(sr) * slow_tau_s)))
    env_fast = signal.lfilter([1.0 - a_fast], [1.0, -a_fast], env)
    env_slow = signal.lfilter([1.0 - a_slow], [1.0, -a_slow], env)

    transient = np.maximum(env_fast - env_slow, 0.0).astype(np.float32, copy=False)
    sustain_env = env_slow

    # Normalize envelopes.
    t_norm = transient / (np.max(transient) + 1e-6)
    s_norm = sustain_env / (np.max(sustain_env) + 1e-6)

    # Map -1..1 to a gain curve.
    t_gain = 1.0 + 1.8 * attack * t_norm
    s_gain = 1.0 + 1.2 * sustain * s_norm

    shaped = high * (t_gain * s_gain).astype(np.float32, copy=False)
    y = low + shaped
    return _clip_mono(y)


def _fx_spectral_denoise(x: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    strength = float(np.clip(float(params.get("strength", 0.0)), 0.0, 1.0))
    if x.size == 0 or strength <= 1e-6:
        return x.astype(np.float32, copy=False)

    n_fft = int(params.get("n_fft", 2048))
    hop = int(params.get("hop_length", n_fft // 4))
    n_fft = int(max(256, min(n_fft, 8192)))
    hop = int(max(32, min(hop, n_fft - 1)))

    f, t, Z = signal.stft(
        x.astype(np.float32, copy=False),
        fs=float(sr),
        window="hann",
        nperseg=n_fft,
        noverlap=(n_fft - hop),
        boundary=None,
        padded=False,
    )
    if Z.size == 0:
        return x.astype(np.float32, copy=False)

    mag = np.abs(Z).astype(np.float32, copy=False)
    phase = np.angle(Z).astype(np.float32, copy=False)

    # Noise estimate: 10th percentile magnitude per-bin.
    noise = np.quantile(mag, 0.10, axis=1).astype(np.float32, copy=False)
    noise = noise.reshape(-1, 1)

    # Spectral subtraction (gentle).
    sub = mag - (strength * 1.6) * noise
    sub = np.maximum(sub, (1.0 - strength) * noise)

    Z2 = (sub * np.exp(1.0j * phase)).astype(np.complex64, copy=False)
    _, y = signal.istft(
        Z2,
        fs=float(sr),
        window="hann",
        nperseg=n_fft,
        noverlap=(n_fft - hop),
        input_onesided=True,
        boundary=None,
    )
    return _clip_mono(y.astype(np.float32, copy=False))


_FX_MODULES_V2: dict[str, FxModuleV2] = {
    "pitch_shift": FxModuleV2(
        module_id="pitch_shift",
        title="Pitch Shift",
        description="Phase-vocoder pitch shift by semitones (preserves duration).",
        apply=_fx_pitch_shift,
    ),
    "time_stretch": FxModuleV2(
        module_id="time_stretch",
        title="Time Stretch",
        description="Phase-vocoder time stretch (rate>1 faster, rate<1 slower).",
        apply=_fx_time_stretch,
    ),
    "distortion": FxModuleV2(
        module_id="distortion",
        title="Distortion",
        description="Simple waveshaper distortion with drive/mode/mix.",
        apply=_fx_distortion,
    ),
    "convolution_reverb": FxModuleV2(
        module_id="convolution_reverb",
        title="Convolution Reverb",
        description="FFT convolution with an external impulse response WAV.",
        apply=_fx_convolution_reverb,
    ),
    "multiband_eq": FxModuleV2(
        module_id="multiband_eq",
        title="Multi-band EQ",
        description="3-band split with per-band gain.",
        apply=_fx_multiband_eq,
    ),
    "transient_shaper": FxModuleV2(
        module_id="transient_shaper",
        title="Transient Shaper",
        description="Attack/sustain shaping focused on high band.",
        apply=_fx_transient_shaper,
    ),
    "spectral_denoise": FxModuleV2(
        module_id="spectral_denoise",
        title="Spectral Noise Reduction",
        description="Gentle spectral subtraction denoise.",
        apply=_fx_spectral_denoise,
    ),
}


def fx_module_v2_ids() -> list[str]:
    return sorted(_FX_MODULES_V2.keys())


def get_fx_module_v2(module_id: str) -> FxModuleV2 | None:
    return _FX_MODULES_V2.get(str(module_id).strip().lower())


def load_fx_chain_v2_json(path: str | Path) -> FxChainV2:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise FxChainV2FormatError("fx chain v2 json must be an object")

    version = obj.get("version", None)
    fmt = str(obj.get("format", obj.get("type", "")) or "").strip().lower()
    if int(version) != 2 or fmt not in {"sndbund_fx_chain", "sndbund_fx_chain_v2", "soundgen_fx_chain"}:
        raise FxChainV2FormatError("not a v2 fx chain json (expected format + version=2)")

    name = str(obj.get("name", p.stem)).strip() or p.stem
    description = (str(obj.get("description")).strip() if obj.get("description") is not None else None)

    steps_obj = obj.get("steps")
    if not isinstance(steps_obj, list):
        raise FxChainV2FormatError("v2 fx chain requires 'steps' list")

    steps: list[FxStepV2] = []
    for it in steps_obj:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("id") or it.get("module") or "").strip().lower()
        if not mid:
            continue
        params = it.get("params")
        params2 = dict(params) if isinstance(params, dict) else {}
        steps.append(FxStepV2(module_id=mid, params=params2))

    return FxChainV2(name=name, steps=tuple(steps), description=description)


def dump_fx_chain_v2_json(chain: FxChainV2) -> dict[str, Any]:
    return {
        "format": "sndbund_fx_chain",
        "version": 2,
        "name": str(chain.name),
        "description": (str(chain.description) if chain.description else None),
        "steps": [
            {"id": str(s.module_id), "params": dict(s.params or {})}
            for s in chain.steps
        ],
    }


def is_fx_chain_v2_json(path: str | Path) -> bool:
    try:
        _ = load_fx_chain_v2_json(path)
        return True
    except Exception:
        return False


def apply_fx_chain_v2(
    x: np.ndarray,
    sr: int,
    chain: FxChainV2,
) -> np.ndarray:
    y = _ensure_mono_f32(x)
    out = y

    for step in chain.steps:
        mod = get_fx_module_v2(step.module_id)
        if mod is None:
            # Unknown module: ignore for forward compatibility.
            continue
        out = _ensure_mono_f32(mod.apply(out, int(sr), dict(step.params or {})))
        out = out.astype(np.float32, copy=False)
        # keep in bounds between stages
        out = _clip_mono(out)

    return out.astype(np.float32, copy=False)
