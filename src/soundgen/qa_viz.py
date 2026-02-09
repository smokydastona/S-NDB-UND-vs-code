from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from scipy import signal


def _to_image(fig) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def waveform_image(audio: np.ndarray, sr: int) -> Image.Image:
    import matplotlib.pyplot as plt

    audio = np.asarray(audio, dtype=np.float32)
    t = np.arange(audio.size, dtype=np.float32) / float(sr)

    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
    ax.plot(t, audio, linewidth=0.6)
    ax.set_title("Waveform")
    ax.set_xlabel("s")
    ax.set_ylabel("amp")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.2)

    img = _to_image(fig)
    plt.close(fig)
    return img


def spectrogram_image(audio: np.ndarray, sr: int) -> Image.Image:
    import matplotlib.pyplot as plt

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return Image.new("RGBA", (10, 10), (0, 0, 0, 0))

    f, t, sxx = signal.spectrogram(audio, fs=sr, nperseg=512, noverlap=384, scaling="spectrum")
    sxx_db = 10.0 * np.log10(np.maximum(sxx, 1e-12))

    fig = plt.figure(figsize=(8, 2.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        sxx_db,
        origin="lower",
        aspect="auto",
        extent=[float(t.min()), float(t.max()), float(f.min()), float(f.max())],
        cmap="magma",
    )
    ax.set_title("Spectrogram (dB)")
    ax.set_xlabel("s")
    ax.set_ylabel("Hz")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    img = _to_image(fig)
    plt.close(fig)
    return img
