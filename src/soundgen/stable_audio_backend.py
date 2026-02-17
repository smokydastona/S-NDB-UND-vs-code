from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sys

import numpy as np


@dataclass(frozen=True)
class StableAudioOpenParams:
    prompt: str
    seconds: float
    seed: int | None
    device: str
    model: str
    negative_prompt: str | None = None
    num_inference_steps: int = 100
    guidance_scale: float = 7.0
    sampler: str | None = None
    hf_token: str | None = None
    lora_path: str | None = None
    lora_scale: float = 1.0
    lora_trigger: str | None = None


def _apply_sampler(pipe, sampler: str) -> None:
    """Best-effort scheduler selection for Stable Audio Open.

    diffusers supports multiple schedulers; we expose a small stable set here.
    If the requested scheduler isn't available, raise a clear error.
    """

    key = str(sampler or "").strip().lower()
    if not key or key in {"default", "auto"}:
        return

    try:
        from diffusers import (
            DDIMScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Unable to import diffusers schedulers. "
            "Ensure diffusers is installed (pip install -r requirements.txt)."
        ) from e

    sched_map = {
        "ddim": DDIMScheduler,
        "deis": DEISMultistepScheduler,
        "dpmpp": DPMSolverMultistepScheduler,
        "dpmpp_2m": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "euler_ancestral": EulerAncestralDiscreteScheduler,
    }

    cls = sched_map.get(key)
    if cls is None:
        opts = ", ".join(sorted(sched_map.keys()))
        raise ValueError(f"Unknown stable-audio sampler '{sampler}'. Options: {opts}")

    try:
        pipe.scheduler = cls.from_config(pipe.scheduler.config)
    except Exception as e:
        raise RuntimeError(f"Failed to apply sampler '{sampler}' to Stable Audio pipeline") from e


@lru_cache(maxsize=4)
def _load_pipeline(model: str, device: str, use_fp16: bool):
    try:
        import torch
        from diffusers import StableAudioPipeline
    except Exception as e:  # pragma: no cover
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Stable Audio Open requires the AI dependencies (torch + diffusers). "
            "Install with `pip install -r requirements.txt`. "
            "If installs fail, try Python 3.12 (the CI/EXE build uses 3.12). "
            f"Current Python: {py}."
        ) from e

    dtype = torch.float16 if use_fp16 else torch.float32

    try:
        pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype)
    except Exception as e:
        msg = str(e)
        if (
            "gated" in msg.lower()
            or "unauthorized" in msg.lower()
            or "401" in msg
            or e.__class__.__name__ in {"GatedRepoError", "RepositoryNotFoundError"}
        ):
            raise RuntimeError(
                "Stable Audio Open model is gated on Hugging Face. "
                "Accept the model terms on the model page and ensure your Hugging Face token is available "
                "(set HUGGINGFACE_HUB_TOKEN, run `huggingface-cli login`, or pass --hf-token / HF token in the UI)."
            ) from e
        raise
    pipe = pipe.to(device)

    # Small memory win; safe for inference.
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe


def _load_pipeline_with_token(model: str, device: str, use_fp16: bool, token: str):
    """Load a pipeline using an explicit HF token.

    Not cached (avoid keeping tokens in cache keys).
    """
    try:
        import torch
        from diffusers import StableAudioPipeline
    except Exception as e:  # pragma: no cover
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Stable Audio Open requires the AI dependencies (torch + diffusers). "
            "Install with `pip install -r requirements.txt`. "
            "If installs fail, try Python 3.12 (the CI/EXE build uses 3.12). "
            f"Current Python: {py}."
        ) from e

    dtype = torch.float16 if use_fp16 else torch.float32

    # diffusers/huggingface_hub moved from use_auth_token -> token.
    try:
        try:
            pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype, token=str(token))
        except TypeError:
            pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype, use_auth_token=str(token))
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "unauthorized" in msg.lower() or "401" in msg:
            raise RuntimeError(
                "Stable Audio Open model is gated on Hugging Face. "
                "Accept the model terms on the model page and ensure the provided token has access."
            ) from e
        raise
    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def _load_pipeline_uncached(model: str, device: str, use_fp16: bool, token: str | None):
    """Load a pipeline without caching.

    This is important when applying LoRA weights: we don't want to mutate a cached
    pipeline instance and have LoRA "stick" for subsequent generations.
    """

    if token:
        return _load_pipeline_with_token(model, device, use_fp16, str(token))

    # Inline of _load_pipeline without lru_cache.
    try:
        import torch
        from diffusers import StableAudioPipeline
    except Exception as e:  # pragma: no cover
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Stable Audio Open requires the AI dependencies (torch + diffusers). "
            "Install with `pip install -r requirements.txt`. "
            "If installs fail, try Python 3.12 (the CI/EXE build uses 3.12). "
            f"Current Python: {py}."
        ) from e

    dtype = torch.float16 if use_fp16 else torch.float32
    pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def _apply_lora(pipe, *, lora_path: str, lora_scale: float) -> dict[str, str | float]:
    """Best-effort LoRA load + scale.

    diffusers' LoRA APIs vary a bit by version; we support the common patterns.
    """

    p = str(lora_path).strip()
    if not p:
        return {"lora_path": "", "lora_scale": float(lora_scale)}

    # Prefer an explicit adapter name to avoid collisions.
    adapter_name = "creature"

    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(
            "This diffusers pipeline does not support LoRA loading (missing load_lora_weights). "
            "Upgrade diffusers, or use a different engine."
        )

    try:
        try:
            pipe.load_lora_weights(p, adapter_name=adapter_name)
        except TypeError:
            # Older signature
            pipe.load_lora_weights(p)
            adapter_name = "default"
    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA weights from: {p}") from e

    s = float(lora_scale)
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(adapter_name, adapter_weights=s)
        except TypeError:
            pipe.set_adapters([adapter_name], adapter_weights=[s])
        except Exception:
            # If adapter API is present but fails, fall back to fuse_lora below.
            pass

    if hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(lora_scale=s)
        except TypeError:
            # Some versions accept no args or different kw.
            pipe.fuse_lora()
        except Exception:
            pass

    return {"lora_path": p, "lora_scale": s, "lora_adapter": adapter_name}


def generate_audio(p: StableAudioOpenParams) -> tuple[np.ndarray, int]:
    """Generate audio using Stability AI Stable Audio Open 1.0 via diffusers.

    Returns:
      (audio_mono_float32, sample_rate)

    Notes:
    - Model is typically gated on Hugging Face; users must accept terms and have auth configured.
    - The pipeline can return stereo @ 44.1kHz; this function downmixes to mono.
    """

    prompt = str(p.prompt)
    trigger = str(p.lora_trigger or "").strip()
    if trigger and trigger.lower() not in prompt.lower():
        prompt = (prompt.rstrip() + ", " + trigger).strip().strip(",")
    seconds = float(p.seconds)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be > 0")

    # Stable Audio Open 1.0 supports up to ~47s.
    seconds = min(seconds, 47.0)

    device = str(p.device or "cpu")
    model = str(p.model)
    if not model:
        raise ValueError("Stable Audio model id must be provided")

    use_fp16 = device == "cuda"

    import torch

    hf_token = str(p.hf_token).strip() if p.hf_token else ""
    lora_path = str(p.lora_path).strip() if p.lora_path else ""
    if lora_path:
        pipe = _load_pipeline_uncached(model, device, use_fp16, hf_token or None)
        _apply_lora(pipe, lora_path=lora_path, lora_scale=float(p.lora_scale))
    else:
        if hf_token:
            pipe = _load_pipeline_with_token(model, device, use_fp16, hf_token)
        else:
            pipe = _load_pipeline(model, device, use_fp16)

    if p.sampler:
        _apply_sampler(pipe, str(p.sampler))

    generator = None
    if p.seed is not None:
        # Some pipelines want CPU generator even if on GPU; be permissive.
        try:
            generator = torch.Generator(device=device).manual_seed(int(p.seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(p.seed))

    kwargs = {
        "prompt": prompt,
        "audio_end_in_s": float(seconds),
        "num_inference_steps": int(p.num_inference_steps),
        "guidance_scale": float(p.guidance_scale),
    }
    if p.negative_prompt:
        kwargs["negative_prompt"] = str(p.negative_prompt)
    if generator is not None:
        kwargs["generator"] = generator

    try:
        out = pipe(**kwargs)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "accept" in msg.lower() or "401" in msg or "unauthorized" in msg.lower():
            raise RuntimeError(
                "Stable Audio Open model is gated on Hugging Face. "
                "Accept the model terms on the model page and ensure your Hugging Face token is available "
                "(e.g., set HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`)."
            ) from e
        raise

    # diffusers returns a list/np array; be robust.
    audio = getattr(out, "audios", None)
    if audio is None:
        raise RuntimeError("StableAudioPipeline returned no audio")

    audio_np = np.asarray(audio)
    # Common shapes:
    # - (batch, channels, samples)
    # - (batch, samples)
    # - (channels, samples)
    # - (samples,)
    if audio_np.ndim == 3:
        audio_np = audio_np[0]
    if audio_np.ndim == 2:
        # (channels, samples) or (samples, channels)
        if audio_np.shape[0] in {1, 2}:
            audio_np = audio_np.mean(axis=0)
        else:
            audio_np = audio_np.mean(axis=1)

    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    audio_np = np.clip(audio_np, -1.0, 1.0)

    sr = getattr(pipe, "sample_rate", None)
    if sr is None:
        # Stable Audio Open is 44.1kHz.
        sr = 44100

    return audio_np, int(sr)
