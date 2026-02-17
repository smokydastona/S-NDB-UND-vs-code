from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class PostProcessParams:
    trim_silence: bool = True
    silence_threshold_db: float = -40.0
    silence_padding_ms: int = 30

    fade_ms: int = 8

    normalize_rms_db: Optional[float] = -18.0
    normalize_peak_db: float = -1.0

    highpass_hz: Optional[float] = 40.0
    lowpass_hz: Optional[float] = 16000.0

    # "Polish Mode" DSP (off by default)
    denoise_strength: float = 0.0  # 0..1 (gentle spectral subtraction)
    transient_amount: float = 0.0  # -1..+1 (attack emphasis)
    transient_sustain: float = 0.0  # -1..+1 (sustain emphasis)
    transient_split_hz: float = 1200.0

    # Harmonic enhancer / exciter (off by default)
    exciter_amount: float = 0.0  # 0..1
    exciter_cutoff_hz: float = 2500.0

    # Multi-band cleanup / dynamics (off by default)
    multiband: bool = False
    multiband_low_hz: float = 250.0
    multiband_high_hz: float = 3000.0
    multiband_low_gain_db: float = 0.0
    multiband_mid_gain_db: float = 0.0
    multiband_high_gain_db: float = 0.0
    multiband_comp_threshold_db: Optional[float] = None
    multiband_comp_ratio: float = 2.0
    multiband_comp_attack_ms: float = 4.0
    multiband_comp_release_ms: float = 120.0
    multiband_comp_makeup_db: float = 0.0

    # Formant-ish spectral warp (1.0 = none). Use creature_size as a friendlier knob.
    formant_shift: float = 1.0
    creature_size: float = 0.0  # -1 small/bright .. +1 large/dark

    # Procedural texture overlay (hybrid granular-ish layer)
    texture_preset: str = "off"  # off|auto|chitter|rasp|buzz|screech
    texture_amount: float = 0.0  # 0..1
    texture_grain_ms: float = 22.0
    texture_spray: float = 0.55

    # Synthetic convolution reverb (off by default)
    reverb_preset: str = "off"  # off|room|cave|forest|nether
    reverb_mix: float = 0.0  # 0..1
    reverb_time_s: float = 1.2

    # Determinism for stochastic DSP (texture/reverb). If None, effects are still deterministic
    # per run but not tied to engine seeds.
    random_seed: Optional[int] = None

    # Optional prompt hint for auto-selecting texture preset. Stored here because post_process_audio
    # operates on audio-only.
    prompt_hint: Optional[str] = None

    compressor_threshold_db: Optional[float] = None  # e.g. -18
    compressor_ratio: float = 4.0
    compressor_attack_ms: float = 5.0
    compressor_release_ms: float = 80.0
    compressor_makeup_db: float = 0.0
    limiter_ceiling_db: Optional[float] = None  # e.g. -1.0

    # Loop cleaning (for ambience): blend the end into the start to reduce seam clicks.
    loop_clean: bool = False
    loop_crossfade_ms: int = 100


@dataclass(frozen=True)
class PostProcessReport:
    trimmed: bool
    start_sample: int
    end_sample: int
    peak_before: float
    peak_after: float
    rms_before: float
    rms_after: float
    clipped_before: bool
    clipped_after: bool


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def _peak(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _is_clipped(x: np.ndarray, *, threshold: float = 0.999) -> bool:
    return bool(np.any(np.abs(x) >= threshold))


def _trim_silence_bounds(x: np.ndarray, sr: int, *, threshold_db: float, padding_ms: int) -> tuple[int, int]:
    if x.size == 0:
        return 0, 0

    thr = _db_to_amp(threshold_db)
    idx = np.flatnonzero(np.abs(x) > thr)
    if idx.size == 0:
        # keep a tiny blip so downstream doesn't crash
        keep = min(x.size, max(1, int(sr * 0.05)))
        return 0, keep

    start = int(idx[0])
    end = int(idx[-1]) + 1

    pad = int(sr * (padding_ms / 1000.0))
    start = max(0, start - pad)
    end = min(x.size, end + pad)
    if end <= start:
        return 0, min(x.size, max(1, int(sr * 0.05)))
    return start, end


def _apply_fade(x: np.ndarray, sr: int, *, fade_ms: int) -> np.ndarray:
    if x.size == 0 or fade_ms <= 0:
        return x
    fade_len = int(sr * (fade_ms / 1000.0))
    if fade_len <= 1:
        return x
    fade_len = min(fade_len, x.size // 2)
    if fade_len <= 1:
        return x

    y = x.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = fade_in[::-1]
    y[:fade_len] *= fade_in
    y[-fade_len:] *= fade_out
    return y


def _butter_filter(x: np.ndarray, sr: int, *, kind: str, cutoff_hz: float, order: int = 2) -> np.ndarray:
    if x.size == 0:
        return x
    nyq = 0.5 * sr
    if cutoff_hz <= 0:
        return x
    wn = cutoff_hz / nyq
    if wn <= 0 or wn >= 1:
        return x

    b, a = signal.butter(order, wn, btype=kind)
    # filtfilt avoids phase shift
    return signal.filtfilt(b, a, x).astype(np.float32, copy=False)


def _soft_limit(x: np.ndarray, *, ceiling_amp: float) -> np.ndarray:
    c = float(ceiling_amp)
    if x.size == 0 or c <= 0:
        return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    # tanh soft clip around ceiling
    y = np.tanh(x / c) * c
    return y.astype(np.float32, copy=False)


def _transient_shaper(x: np.ndarray, sr: int, *, amount: float, split_hz: float = 1200.0) -> np.ndarray:
    """Attack/sustain transient shaping focused on the high band.

    amount controls attack emphasis. Sustain is controlled separately via
    PostProcessParams.transient_sustain.
    """

    return _transient_shaper_attack_sustain(x, sr, attack=float(amount), sustain=0.0, split_hz=float(split_hz))


def _transient_shaper_attack_sustain(
    x: np.ndarray,
    sr: int,
    *,
    attack: float,
    sustain: float,
    split_hz: float = 1200.0,
) -> np.ndarray:
    a = float(np.clip(float(attack), -1.0, 1.0))
    s = float(np.clip(float(sustain), -1.0, 1.0))
    if x.size == 0 or (abs(a) < 1e-6 and abs(s) < 1e-6):
        return x.astype(np.float32, copy=False)

    split = float(np.clip(float(split_hz), 150.0, 0.45 * float(sr)))
    low = _butter_filter(x, sr, kind="lowpass", cutoff_hz=split, order=2)
    high = (x - low).astype(np.float32, copy=False)

    env = np.abs(high).astype(np.float32, copy=False)
    # Fast vs slow envelope to estimate transient vs sustain.
    fast_tau_s = 0.006
    slow_tau_s = 0.080
    a_fast = float(np.exp(-1.0 / max(1.0, float(sr) * fast_tau_s)))
    a_slow = float(np.exp(-1.0 / max(1.0, float(sr) * slow_tau_s)))
    env_fast = signal.lfilter([1.0 - a_fast], [1.0, -a_fast], env)
    env_slow = signal.lfilter([1.0 - a_slow], [1.0, -a_slow], env)

    transient = np.maximum(env_fast - env_slow, 0.0).astype(np.float32, copy=False)
    tmax = float(np.max(transient)) if transient.size else 0.0
    if tmax <= 1e-8:
        t = np.zeros_like(transient, dtype=np.float32)
    else:
        t = (transient / tmax).astype(np.float32, copy=False)

    # Gain curve: emphasize attack on transient regions, sustain elsewhere.
    attack_gain = 1.0 + 1.6 * a
    sustain_gain = 1.0 + 0.9 * s
    g = (sustain_gain + (attack_gain - sustain_gain) * t).astype(np.float32, copy=False)
    g = np.clip(g, 0.1, 3.0, out=g)

    y = (low + (high * g)).astype(np.float32, copy=False)
    return y


def _apply_exciter(x: np.ndarray, sr: int, *, amount: float, cutoff_hz: float) -> np.ndarray:
    """Subtle harmonic exciter by saturating an upper band and mixing back."""

    amt = float(np.clip(float(amount), 0.0, 1.0))
    if x.size == 0 or amt <= 1e-6:
        return x.astype(np.float32, copy=False)

    cutoff = float(np.clip(float(cutoff_hz), 600.0, 0.45 * float(sr)))
    band = _butter_filter(x, sr, kind="highpass", cutoff_hz=cutoff, order=2)

    drive = 1.5 + 6.0 * amt
    sat = np.tanh(band * drive).astype(np.float32, copy=False)
    # Normalize roughly so amt feels consistent.
    sat = (sat / float(np.tanh(drive))).astype(np.float32, copy=False)

    mix = 0.55 * amt
    y = (x + (sat * mix)).astype(np.float32, copy=False)
    return y


def _multiband_split(
    x: np.ndarray,
    sr: int,
    *,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into (low, mid, high) using zero-phase Butterworth filters.

    Recombining low+mid+high approximately reconstructs x.
    """

    if x.size == 0:
        z = x.astype(np.float32, copy=False)
        return z, z, z

    lo = float(np.clip(float(low_hz), 40.0, 0.35 * float(sr)))
    hi = float(np.clip(float(high_hz), lo * 1.2, 0.45 * float(sr)))

    low = _butter_filter(x, sr, kind="lowpass", cutoff_hz=lo, order=4)
    high = _butter_filter(x, sr, kind="highpass", cutoff_hz=hi, order=4)
    mid = (x - low - high).astype(np.float32, copy=False)
    return low, mid, high


def _apply_multiband(x: np.ndarray, sr: int, params: PostProcessParams) -> np.ndarray:
    if not bool(params.multiband) or x.size == 0:
        return x.astype(np.float32, copy=False)

    low, mid, high = _multiband_split(x, sr, low_hz=float(params.multiband_low_hz), high_hz=float(params.multiband_high_hz))

    gl = _db_to_amp(float(params.multiband_low_gain_db))
    gm = _db_to_amp(float(params.multiband_mid_gain_db))
    gh = _db_to_amp(float(params.multiband_high_gain_db))

    low = (low * gl).astype(np.float32, copy=False)
    mid = (mid * gm).astype(np.float32, copy=False)
    high = (high * gh).astype(np.float32, copy=False)

    if params.multiband_comp_threshold_db is not None:
        thr = float(params.multiband_comp_threshold_db)
        ratio = float(params.multiband_comp_ratio)
        atk = float(params.multiband_comp_attack_ms)
        rel = float(params.multiband_comp_release_ms)
        makeup = float(params.multiband_comp_makeup_db)
        low = _compressor(low, sr, threshold_db=thr, ratio=ratio, attack_ms=atk, release_ms=rel, makeup_db=makeup)
        mid = _compressor(mid, sr, threshold_db=thr, ratio=ratio, attack_ms=atk, release_ms=rel, makeup_db=makeup)
        high = _compressor(high, sr, threshold_db=thr, ratio=ratio, attack_ms=atk, release_ms=rel, makeup_db=makeup)

    y = (low + mid + high).astype(np.float32, copy=False)
    return y


def _effective_formant_shift(params: PostProcessParams) -> float:
    f = float(params.formant_shift)
    if abs(f - 1.0) > 1e-6:
        return float(np.clip(f, 0.5, 2.0))
    size = float(np.clip(float(params.creature_size), -1.0, 1.0))
    if abs(size) < 1e-6:
        return 1.0
    # Positive size => larger creature => lower formants (shift down)
    factor = float(2.0 ** (-0.35 * size))  # ~0.78..1.28
    return float(np.clip(factor, 0.5, 2.0))


def _formant_warp_stft(x: np.ndarray, sr: int, *, factor: float) -> np.ndarray:
    """Approximate formant shift by warping the STFT spectrum along frequency.

    Note: This is an approximation; it can subtly affect perceived pitch.
    """

    if x.size == 0:
        return x.astype(np.float32, copy=False)
    f = float(np.clip(float(factor), 0.5, 2.0))
    if abs(f - 1.0) < 1e-6:
        return x.astype(np.float32, copy=False)

    nper = 2048
    nover = 1536
    if x.size < nper:
        return x.astype(np.float32, copy=False)

    freqs, times, Z = signal.stft(x.astype(np.float32, copy=False), fs=sr, nperseg=nper, noverlap=nover, window="hann")
    # Warp in frequency domain: target bin i samples from source frequency / factor.
    src_freqs = freqs
    tgt_freqs = freqs
    sample_freqs = tgt_freqs / f

    real = np.empty_like(Z.real, dtype=np.float32)
    imag = np.empty_like(Z.imag, dtype=np.float32)

    # Vectorized per-frame interpolation
    for ti in range(Z.shape[1]):
        zr = Z.real[:, ti]
        zi = Z.imag[:, ti]
        real[:, ti] = np.interp(sample_freqs, src_freqs, zr, left=0.0, right=0.0).astype(np.float32, copy=False)
        imag[:, ti] = np.interp(sample_freqs, src_freqs, zi, left=0.0, right=0.0).astype(np.float32, copy=False)

    Z2 = real.astype(np.float32, copy=False) + 1j * imag.astype(np.float32, copy=False)
    _, y = signal.istft(Z2, fs=sr, nperseg=nper, noverlap=nover, window="hann")
    y = y[: x.size]
    return y.astype(np.float32, copy=False)


def _auto_texture_preset(prompt_hint: str | None) -> str:
    p = (prompt_hint or "").lower()
    if any(k in p for k in ("buzz", "wasp", "bee", "fly", "insect", "mosquito")):
        return "buzz"
    if any(k in p for k in ("screech", "shriek", "scream", "metal", "scrape")):
        return "screech"
    if any(k in p for k in ("rasp", "undead", "zombie", "grit", "gravel")):
        return "rasp"
    if any(k in p for k in ("chitter", "click", "skitter")):
        return "chitter"
    return "off"


def _synthesize_texture(
    n: int,
    sr: int,
    *,
    preset: str,
    grain_ms: float,
    spray: float,
    seed: int,
) -> np.ndarray:
    """Procedural granular-ish texture: windowed noise grains + band shaping."""

    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    g_ms = float(np.clip(float(grain_ms), 4.0, 80.0))
    grain = max(16, int(round(float(sr) * (g_ms / 1000.0))))
    grain = min(grain, max(16, n))

    # Density from spray
    dens = 0.15 + 1.25 * _clamp01(float(spray))
    hop = max(1, int(round(grain / (2.0 * dens))))

    win = np.hanning(grain).astype(np.float32)
    out = np.zeros((n,), dtype=np.float32)

    for start in range(0, n, hop):
        if rng.random() > (0.55 + 0.45 * _clamp01(float(spray))):
            continue
        end = min(n, start + grain)
        g = end - start
        noise = rng.standard_normal(g).astype(np.float32)
        w = win[:g]
        amp = float(0.35 + 0.65 * rng.random())
        out[start:end] += amp * noise * w

    # Band shaping per preset
    preset_l = (preset or "off").strip().lower()
    if preset_l == "chitter":
        out = _butter_filter(out, sr, kind="highpass", cutoff_hz=1600.0, order=2)
        out = _butter_filter(out, sr, kind="lowpass", cutoff_hz=9000.0, order=2)
    elif preset_l == "rasp":
        out = _butter_filter(out, sr, kind="highpass", cutoff_hz=350.0, order=2)
        out = _butter_filter(out, sr, kind="lowpass", cutoff_hz=4200.0, order=2)
    elif preset_l == "buzz":
        out = _butter_filter(out, sr, kind="highpass", cutoff_hz=500.0, order=2)
        out = _butter_filter(out, sr, kind="lowpass", cutoff_hz=2200.0, order=2)
    elif preset_l == "screech":
        out = _butter_filter(out, sr, kind="highpass", cutoff_hz=2200.0, order=2)
        out = _butter_filter(out, sr, kind="lowpass", cutoff_hz=14000.0, order=2)
    else:
        # off or unknown
        return np.zeros((n,), dtype=np.float32)

    # Soft clip and normalize lightly
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-6:
        out = (out / peak).astype(np.float32, copy=False)
    out = np.tanh(1.8 * out).astype(np.float32, copy=False)
    return out.astype(np.float32, copy=False)


def _apply_texture_overlay(x: np.ndarray, sr: int, params: PostProcessParams, *, prompt_hint: str | None = None) -> np.ndarray:
    amt = float(np.clip(float(params.texture_amount), 0.0, 1.0))
    if x.size == 0 or amt <= 1e-6:
        return x.astype(np.float32, copy=False)

    preset = (params.texture_preset or "off").strip().lower()
    if preset == "auto":
        preset = _auto_texture_preset(prompt_hint)
    if preset in {"off", "none", "0"}:
        return x.astype(np.float32, copy=False)

    seed = int(params.random_seed) if params.random_seed is not None else 0
    tex = _synthesize_texture(
        int(x.size),
        int(sr),
        preset=str(preset),
        grain_ms=float(params.texture_grain_ms),
        spray=float(params.texture_spray),
        seed=seed ^ 0xA53C_19D1,
    )

    # Mix in; keep conservative energy.
    y = (x + (amt * 0.35) * tex).astype(np.float32, copy=False)
    return y


def _synthetic_ir(
    sr: int,
    *,
    preset: str,
    time_s: float,
    seed: int,
) -> np.ndarray:
    t = float(np.clip(float(time_s), 0.1, 6.0))
    n = max(1, int(round(float(sr) * t)))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    p = (preset or "room").strip().lower()
    if p == "room":
        early_ms = 60.0
        tail_lp = 9000.0
        decay = 1.6
    elif p == "cave":
        early_ms = 110.0
        tail_lp = 6000.0
        decay = 2.4
    elif p == "forest":
        early_ms = 75.0
        tail_lp = 5000.0
        decay = 1.8
    elif p == "nether":
        early_ms = 90.0
        tail_lp = 4200.0
        decay = 2.1
    else:
        early_ms = 60.0
        tail_lp = 9000.0
        decay = 1.6

    ir = np.zeros((n,), dtype=np.float32)
    # Direct path
    ir[0] = 1.0

    # Early reflections
    early_n = int(round(float(sr) * (early_ms / 1000.0)))
    k = max(8, min(40, int(6 + 18 * t)))
    for _ in range(k):
        idx = int(rng.integers(1, max(2, early_n)))
        amp = float((0.25 + 0.75 * rng.random()) * np.exp(-3.0 * (idx / max(1, early_n))))
        ir[idx] += amp * float(rng.choice([1.0, -1.0]))

    # Late tail: exponentially decaying filtered noise
    noise = rng.standard_normal(n).astype(np.float32)
    env = np.exp(-np.linspace(0.0, decay, n, dtype=np.float32))
    tail = (noise * env).astype(np.float32, copy=False)
    tail = _butter_filter(tail, sr, kind="lowpass", cutoff_hz=float(tail_lp), order=2)
    ir = (ir + 0.22 * tail).astype(np.float32, copy=False)

    # Normalize IR energy
    peak = float(np.max(np.abs(ir))) if ir.size else 0.0
    if peak > 1e-6:
        ir = (ir / peak).astype(np.float32, copy=False)
    return ir.astype(np.float32, copy=False)


def _apply_reverb(x: np.ndarray, sr: int, params: PostProcessParams) -> np.ndarray:
    mix = float(np.clip(float(params.reverb_mix), 0.0, 1.0))
    preset = (params.reverb_preset or "off").strip().lower()
    if x.size == 0 or mix <= 1e-6 or preset in {"off", "none", "0"}:
        return x.astype(np.float32, copy=False)

    seed = int(params.random_seed) if params.random_seed is not None else 0
    ir = _synthetic_ir(int(sr), preset=str(preset), time_s=float(params.reverb_time_s), seed=seed ^ 0x6C2B_1E55)
    wet = signal.fftconvolve(x.astype(np.float32, copy=False), ir.astype(np.float32, copy=False), mode="full")
    wet = wet[: x.size].astype(np.float32, copy=False)

    # Damping to keep Minecraft-ish tails under control.
    wet = _butter_filter(wet, sr, kind="lowpass", cutoff_hz=12000.0, order=2)
    y = ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)
    return y


def _compressor(
    x: np.ndarray,
    sr: int,
    *,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float,
) -> np.ndarray:
    """Feed-forward mono compressor with envelope follower."""

    if x.size == 0:
        return x.astype(np.float32, copy=False)

    thr = float(threshold_db)
    r = float(max(1.0, ratio))
    atk = max(0.5, float(attack_ms)) * 0.001
    rel = max(1.0, float(release_ms)) * 0.001
    atk_c = float(np.exp(-1.0 / (float(sr) * atk)))
    rel_c = float(np.exp(-1.0 / (float(sr) * rel)))

    eps = 1e-9
    env = 0.0
    y = np.empty_like(x, dtype=np.float32)

    for i in range(x.size):
        s = float(abs(x[i]))
        # envelope follower (peak-ish)
        if s > env:
            env = atk_c * env + (1.0 - atk_c) * s
        else:
            env = rel_c * env + (1.0 - rel_c) * s

        lvl_db = 20.0 * float(np.log10(env + eps))
        if lvl_db <= thr:
            gain_db = 0.0
        else:
            # compress above threshold
            compressed_db = thr + (lvl_db - thr) / r
            gain_db = compressed_db - lvl_db

        gain = float(10.0 ** ((gain_db + float(makeup_db)) / 20.0))
        y[i] = float(x[i]) * gain

    return y.astype(np.float32, copy=False)


def _spectral_denoise(x: np.ndarray, sr: int, *, strength: float) -> np.ndarray:
    """Gentle spectral subtraction using STFT.

    This is intentionally conservative; it's aimed at reducing steady hiss,
    not removing strong textures.
    """

    s = float(np.clip(strength, 0.0, 1.0))
    if x.size == 0 or s <= 0.0:
        return x.astype(np.float32, copy=False)

    nper = 1024
    nover = 768
    if x.size < nper:
        return x.astype(np.float32, copy=False)

    f, t, Z = signal.stft(x.astype(np.float32, copy=False), fs=sr, nperseg=nper, noverlap=nover, window="hann")
    mag = np.abs(Z)
    phase = np.exp(1j * np.angle(Z))

    # Choose low-energy frames as noise reference.
    frame_energy = np.mean(mag * mag, axis=0)
    if frame_energy.size < 8:
        noise_idx = slice(0, frame_energy.size)
        noise_mag = np.median(mag[:, noise_idx], axis=1)
    else:
        k = max(4, int(round(0.15 * frame_energy.size)))
        idx = np.argsort(frame_energy)[:k]
        noise_mag = np.median(mag[:, idx], axis=1)

    # Subtract a fraction of the noise profile.
    sub = (s * noise_mag)[:, None]
    mag2 = np.maximum(mag - sub, 0.0)
    Z2 = mag2 * phase
    _, y = signal.istft(Z2, fs=sr, nperseg=nper, noverlap=nover, window="hann")
    y = y[: x.size]
    return y.astype(np.float32, copy=False)


def _apply_loop_clean(y: np.ndarray, sr: int, *, crossfade_ms: int) -> np.ndarray:
    """Reduce seam clicks when looping by blending the last window into the first.

    This is intentionally simple and robust for ambience: it keeps length the same and
    forces y[-1] == y[0] after processing.
    """

    if y.size == 0:
        return y.astype(np.float32, copy=False)

    ms = int(max(1, int(crossfade_ms)))
    n = int(round(float(sr) * (float(ms) / 1000.0)))
    if n < 2:
        y2 = y.astype(np.float32, copy=False)
        if y2.size >= 2:
            y2[-1] = y2[0]
        return y2

    # Keep the window reasonable.
    n = min(n, max(2, y.size // 4))
    if y.size < (n + 2):
        # Too short to do anything meaningful.
        y2 = y.astype(np.float32, copy=False)
        if y2.size >= 2:
            y2[-1] = y2[0]
        return y2

    head = y[:n].astype(np.float32, copy=False)
    tail = y[-n:].astype(np.float32, copy=False)
    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)

    # Blend tail into a time-reversed view of the head so that the very last sample
    # fades toward the very first sample (y[0]), making the loop boundary continuous.
    head_rev = head[::-1]
    tail2 = (tail * (1.0 - fade) + head_rev * fade).astype(np.float32, copy=False)

    y2 = y.copy()
    y2[-n:] = tail2
    # Enforce exact continuity at the boundary.
    y2[-1] = y2[0]
    return y2.astype(np.float32, copy=False)


def post_process_audio(x: np.ndarray, sr: int, params: PostProcessParams) -> tuple[np.ndarray, PostProcessReport]:
    """Apply a Minecraft-friendly post chain.

    Input/Output: mono float32 in [-1, 1].
    """

    if x.ndim != 1:
        raise ValueError("post_process_audio expects mono 1D audio")

    x = x.astype(np.float32, copy=False)

    peak_before = _peak(x)
    rms_before = _rms(x)
    clipped_before = _is_clipped(x)

    start, end = 0, x.size
    trimmed = False
    y = x

    if params.trim_silence:
        start, end = _trim_silence_bounds(
            y,
            sr,
            threshold_db=params.silence_threshold_db,
            padding_ms=params.silence_padding_ms,
        )
        y = y[start:end]
        trimmed = (start != 0) or (end != x.size)

    # Optional spectral denoise
    if float(params.denoise_strength) > 0.0:
        y = _spectral_denoise(y, sr, strength=float(params.denoise_strength))

    # Optional gentle EQ
    if params.highpass_hz is not None:
        y = _butter_filter(y, sr, kind="highpass", cutoff_hz=float(params.highpass_hz))
    if params.lowpass_hz is not None:
        y = _butter_filter(y, sr, kind="lowpass", cutoff_hz=float(params.lowpass_hz))

    # Optional multi-band cleanup
    y = _apply_multiband(y, sr, params)

    # Optional formant-ish spectral warp
    ff = _effective_formant_shift(params)
    if abs(ff - 1.0) > 1e-6:
        y = _formant_warp_stft(y, sr, factor=ff)

    # Optional texture overlay (hybrid granular-ish)
    y = _apply_texture_overlay(y, sr, params, prompt_hint=(params.prompt_hint or None))

    # Optional transient shaping
    if float(params.transient_amount) != 0.0 or float(getattr(params, "transient_sustain", 0.0)) != 0.0:
        y = _transient_shaper_attack_sustain(
            y,
            sr,
            attack=float(params.transient_amount),
            sustain=float(getattr(params, "transient_sustain", 0.0)),
            split_hz=float(params.transient_split_hz),
        )

    # Optional exciter (before fade / dynamics)
    if float(getattr(params, "exciter_amount", 0.0)) > 0.0:
        y = _apply_exciter(
            y,
            sr,
            amount=float(getattr(params, "exciter_amount", 0.0)),
            cutoff_hz=float(getattr(params, "exciter_cutoff_hz", 2500.0)),
        )

    # Fade to avoid clicks
    y = _apply_fade(y, sr, fade_ms=int(params.fade_ms))

    # Optional compression (before normalization)
    if params.compressor_threshold_db is not None:
        y = _compressor(
            y,
            sr,
            threshold_db=float(params.compressor_threshold_db),
            ratio=float(params.compressor_ratio),
            attack_ms=float(params.compressor_attack_ms),
            release_ms=float(params.compressor_release_ms),
            makeup_db=float(params.compressor_makeup_db),
        )

    # Optional reverb (before normalization so overall loudness stays consistent)
    y = _apply_reverb(y, sr, params)

    # Optional loop cleaning (before normalization so output loudness stays consistent)
    if bool(getattr(params, "loop_clean", False)):
        y = _apply_loop_clean(y, sr, crossfade_ms=int(getattr(params, "loop_crossfade_ms", 100)))

    # Loudness-ish normalization (RMS target) + safety peak cap.
    if params.normalize_rms_db is not None:
        target_rms = _db_to_amp(float(params.normalize_rms_db))
        cur_rms = _rms(y)
        if cur_rms > 0:
            y = (y * (target_rms / cur_rms)).astype(np.float32, copy=False)

    # Peak cap to avoid clipping.
    peak_target = _db_to_amp(float(params.normalize_peak_db))
    cur_peak = _peak(y)
    if cur_peak > 0:
        scale = min(1.0, peak_target / cur_peak)
        if scale != 1.0:
            y = (y * scale).astype(np.float32, copy=False)

    # Optional limiter at the end of the chain.
    if params.limiter_ceiling_db is not None:
        y = _soft_limit(y, ceiling_amp=_db_to_amp(float(params.limiter_ceiling_db)))

    # Hard clip as last resort (should be rare after scaling)
    y = np.clip(y, -1.0, 1.0, out=y)

    report = PostProcessReport(
        trimmed=trimmed,
        start_sample=int(start),
        end_sample=int(end),
        peak_before=float(peak_before),
        peak_after=_peak(y),
        rms_before=float(rms_before),
        rms_after=_rms(y),
        clipped_before=bool(clipped_before),
        clipped_after=_is_clipped(y),
    )
    return y.astype(np.float32, copy=False), report
