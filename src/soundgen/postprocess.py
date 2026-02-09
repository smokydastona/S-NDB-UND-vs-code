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

    # Optional gentle EQ
    if params.highpass_hz is not None:
        y = _butter_filter(y, sr, kind="highpass", cutoff_hz=float(params.highpass_hz))
    if params.lowpass_hz is not None:
        y = _butter_filter(y, sr, kind="lowpass", cutoff_hz=float(params.lowpass_hz))

    # Fade to avoid clicks
    y = _apply_fade(y, sr, fade_ms=int(params.fade_ms))

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
