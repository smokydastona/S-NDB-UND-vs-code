from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class QAMetrics:
    seconds: float
    sample_rate: int
    peak: float
    rms: float
    clipped: bool


def compute_metrics(x: np.ndarray, sr: int) -> QAMetrics:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("compute_metrics expects mono 1D audio")

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) if x.size else 0.0
    clipped = bool(np.any(np.abs(x) >= 0.999))
    seconds = float(x.size / sr) if sr > 0 else 0.0
    return QAMetrics(seconds=seconds, sample_rate=int(sr), peak=peak, rms=rms, clipped=clipped)


def detect_long_tail(
    x: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -45.0,
    tail_seconds: float = 0.35,
) -> bool:
    """Returns True if the last N seconds are above a silence threshold.

    Useful to catch "too-long tails" that feel laggy in Minecraft.
    """

    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or sr <= 0:
        return False

    tail_len = int(sr * tail_seconds)
    if tail_len <= 0:
        return False

    tail = x[-min(tail_len, x.size) :]
    thr = float(10.0 ** (threshold_db / 20.0))
    return bool(np.any(np.abs(tail) > thr))
