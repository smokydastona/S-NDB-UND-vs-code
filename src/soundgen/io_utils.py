from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a mono WAV file.

    audio: float32 array in [-1, 1]
    """
    ensure_parent_dir(path)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read audio as mono float32 in [-1, 1]."""

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1, dtype=np.float32)
    return mono.astype(np.float32, copy=False), int(sr)
