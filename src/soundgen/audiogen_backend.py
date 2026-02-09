from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    seconds: float = 3.0
    seed: Optional[int] = None
    device: str = "cpu"  # "cpu" or "cuda"
    model: str = "cvssp/audioldm2"  # diffusers AudioLDM2 checkpoint
    sample_rate: int = 16000


def generate_audio(params: GenerationParams) -> tuple[np.ndarray, int]:
    """Generate mono audio from a text prompt using Diffusers.

    Returns:
        (audio, sample_rate) where audio is float32 in [-1, 1] shaped (num_samples,).
    """

    # Lazy import keeps `--help` fast even if deps are missing.
    from diffusers import AudioLDM2Pipeline
    from transformers import GPT2LMHeadModel

    device = params.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    torch_device = torch.device(device)

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(params.model, torch_dtype=dtype)
    pipe = pipe.to(torch_device)

    # Some environments end up with a plain GPT2Model loaded, but diffusers' AudioLDM2
    # generation path requires GPT2LMHeadModel generation helpers.
    if pipe.language_model.__class__.__name__ != "GPT2LMHeadModel":
        language_model = GPT2LMHeadModel.from_pretrained(params.model, subfolder="language_model", torch_dtype=dtype)
        pipe.language_model = language_model.to(torch_device)

    generator = None
    if params.seed is not None:
        generator = torch.Generator(device=torch_device)
        generator.manual_seed(int(params.seed))

    # Diffusers AudioLDM2 expects length in seconds via `audio_length_in_s`.
    result = pipe(
        prompt=params.prompt,
        audio_length_in_s=float(params.seconds),
        generator=generator,
    )

    audio = result.audios[0]
    # `audios[0]` can be (samples,) or (samples, channels); normalize to mono.
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sample_rate = getattr(pipe, "sample_rate", params.sample_rate)
    return audio, int(sample_rate)
