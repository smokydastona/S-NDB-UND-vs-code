from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .io_utils import read_wav_mono, write_wav


PostprocessFn = Callable[[np.ndarray, int], tuple[np.ndarray, str]]


_DEFAULT_SAMPLELIB_INDEX_PATH = Path("library") / "samplelib_index.json"


@dataclass(frozen=True)
class GeneratedWav:
    wav_path: Path
    post_info: str | None = None
    sources: tuple[dict[str, Any], ...] = ()
    credits_extra: dict[str, Any] = field(default_factory=dict)


def generate_wav(
    engine: str,
    *,
    prompt: str,
    seconds: float,
    seed: int | None,
    out_wav: Path,
    postprocess_fn: PostprocessFn | None = None,
    # diffusers
    device: str | None = None,
    model: str | None = None,
    # rfxgen
    preset: str | None = None,
    rfxgen_path: Path | None = None,
    # replicate
    replicate_model: str | None = None,
    replicate_token: str | None = None,
    replicate_input_json: str | None = None,
    # samplelib
    library_zips: tuple[Path, ...] = (),
    library_pitch_min: float = 0.85,
    library_pitch_max: float = 1.20,
    library_mix_count: int = 1,
    library_index_path: Optional[Path] = _DEFAULT_SAMPLELIB_INDEX_PATH,
    # synth
    synth_waveform: str = "sine",
    synth_freq_hz: float = 440.0,
    synth_attack_ms: float = 5.0,
    synth_decay_ms: float = 80.0,
    synth_sustain_level: float = 0.35,
    synth_release_ms: float = 120.0,
    synth_noise_mix: float = 0.05,
    synth_drive: float = 0.0,
    synth_pitch_min: float = 0.90,
    synth_pitch_max: float = 1.10,
    synth_lowpass_hz: float = 16000.0,
    synth_highpass_hz: float = 30.0,
    # layered
    layered_preset: str = "auto",
    layered_preset_lock: bool = True,
    layered_variant_index: int = 0,
    layered_micro_variation: float = 0.0,
    layered_env_curve_shape: str = "linear",
    layered_transient_tilt: float = 0.0,
    layered_body_tilt: float = 0.0,
    layered_tail_tilt: float = 0.0,
    layered_source_lock: bool = False,
    layered_source_seed: int | None = None,
    layered_transient_ms: int = 110,
    layered_tail_ms: int = 350,
    layered_transient_attack_ms: float = 1.0,
    layered_transient_hold_ms: float = 10.0,
    layered_transient_decay_ms: float = 90.0,
    layered_body_attack_ms: float = 5.0,
    layered_body_hold_ms: float = 0.0,
    layered_body_decay_ms: float = 80.0,
    layered_tail_attack_ms: float = 15.0,
    layered_tail_hold_ms: float = 0.0,
    layered_tail_decay_ms: float = 320.0,
    layered_duck_amount: float = 0.35,
    layered_duck_release_ms: float = 90.0,
    sample_rate: int = 44100,
) -> GeneratedWav:
    """Generate a mono WAV file for any engine.

    Returns a standardized result with engine-specific credits fields and optional sources.
    """

    engine = str(engine).strip().lower()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    post_info: str | None = None
    sources: tuple[dict[str, Any], ...] = ()
    credits_extra: dict[str, Any] = {}

    if engine == "diffusers":
        from .audiogen_backend import GenerationParams, generate_audio

        gp = GenerationParams(
            prompt=prompt,
            seconds=float(seconds),
            seed=seed,
            device=str(device or "cpu"),
            model=str(model or "cvssp/audioldm2"),
        )
        audio, sr = generate_audio(gp)
        if postprocess_fn is not None:
            audio, post_info = postprocess_fn(audio, sr)
        write_wav(out_wav, audio, sr)
        credits_extra = {"model": gp.model, "device": gp.device, "seed": seed}
        return GeneratedWav(wav_path=out_wav, post_info=post_info, sources=sources, credits_extra=credits_extra)

    if engine == "rfxgen":
        from .rfxgen_backend import RfxGenParams, generate_with_rfxgen

        rp = RfxGenParams(prompt=prompt, out_path=out_wav, preset=preset, rfxgen_path=rfxgen_path)
        written = generate_with_rfxgen(rp)
        if postprocess_fn is not None:
            a, sr = read_wav_mono(written)
            a, post_info = postprocess_fn(a, sr)
            write_wav(written, a, sr)
        credits_extra = {"preset": preset, "rfxgen_path": str(rfxgen_path) if rfxgen_path else None}
        return GeneratedWav(wav_path=written, post_info=post_info, sources=sources, credits_extra=credits_extra)

    if engine == "samplelib":
        from .samplelib_backend import SampleLibParams, generate_with_samplelib

        sp = SampleLibParams(
            prompt=prompt,
            out_path=out_wav,
            seconds=float(seconds),
            seed=seed,
            library_zips=tuple(Path(p) for p in library_zips),
            pitch_min=float(library_pitch_min),
            pitch_max=float(library_pitch_max),
            mix_count=int(library_mix_count),
            index_path=library_index_path,
            sample_rate=int(sample_rate),
        )
        result = generate_with_samplelib(sp)
        written = result.out_path
        if postprocess_fn is not None:
            a, sr = read_wav_mono(written)
            a, post_info = postprocess_fn(a, sr)
            write_wav(written, a, sr)

        sources = tuple(
            {
                "zip": s.zip_path,
                "member": s.member,
                "repo": s.repo,
                "attribution_files": list(s.attribution_files),
            }
            for s in result.sources
        )
        credits_extra = {"mix_count": int(sp.mix_count)}
        return GeneratedWav(wav_path=written, post_info=post_info, sources=sources, credits_extra=credits_extra)

    if engine == "replicate":
        from .replicate_backend import ReplicateParams, generate_with_replicate

        rp = ReplicateParams(
            prompt=prompt,
            seconds=float(seconds),
            out_path=out_wav,
            model=str(replicate_model or "").strip(),
            api_token=(str(replicate_token).strip() if replicate_token else None),
            extra_input_json=(str(replicate_input_json) if replicate_input_json else None),
        )
        written = generate_with_replicate(rp)
        if postprocess_fn is not None and written.suffix.lower() == ".wav":
            a, sr = read_wav_mono(written)
            a, post_info = postprocess_fn(a, sr)
            write_wav(written, a, sr)
        credits_extra = {"replicate_model": replicate_model}
        return GeneratedWav(wav_path=written, post_info=post_info, sources=sources, credits_extra=credits_extra)

    if engine == "synth":
        from .synth_backend import SynthParams, generate_with_synth

        sp = SynthParams(
            prompt=prompt,
            seconds=float(seconds),
            seed=seed,
            sample_rate=int(sample_rate),
            waveform=str(synth_waveform),
            freq_hz=float(synth_freq_hz),
            pitch_min=float(synth_pitch_min),
            pitch_max=float(synth_pitch_max),
            attack_ms=float(synth_attack_ms),
            decay_ms=float(synth_decay_ms),
            sustain_level=float(synth_sustain_level),
            release_ms=float(synth_release_ms),
            noise_mix=float(synth_noise_mix),
            lowpass_hz=float(synth_lowpass_hz),
            highpass_hz=float(synth_highpass_hz),
            drive=float(synth_drive),
        )
        audio, sr = generate_with_synth(sp)
        if postprocess_fn is not None:
            audio, post_info = postprocess_fn(audio, sr)
        write_wav(out_wav, audio, sr)
        credits_extra = {"waveform": sp.waveform, "freq_hz": sp.freq_hz}
        return GeneratedWav(wav_path=out_wav, post_info=post_info, sources=sources, credits_extra=credits_extra)

    if engine == "layered":
        from .layered_backend import LayeredParams, generate_with_layered

        lp = LayeredParams(
            prompt=prompt,
            seconds=float(seconds),
            seed=seed,
            sample_rate=int(sample_rate),

            preset=str(layered_preset),
            preset_lock=bool(layered_preset_lock),
            variant_index=int(layered_variant_index),
            micro_variation=float(layered_micro_variation),
            env_curve_shape=str(layered_env_curve_shape),
            transient_tilt=float(layered_transient_tilt),
            body_tilt=float(layered_body_tilt),
            tail_tilt=float(layered_tail_tilt),
            source_lock=bool(layered_source_lock),
            source_seed=(int(layered_source_seed) if layered_source_seed is not None else None),

            library_zips=tuple(Path(p) for p in library_zips),
            library_pitch_min=float(library_pitch_min),
            library_pitch_max=float(library_pitch_max),
            library_mix_count=int(library_mix_count),
            library_index_path=library_index_path,

            transient_ms=int(layered_transient_ms),
            tail_ms=int(layered_tail_ms),

            transient_attack_ms=float(layered_transient_attack_ms),
            transient_hold_ms=float(layered_transient_hold_ms),
            transient_decay_ms=float(layered_transient_decay_ms),

            body_attack_ms=float(layered_body_attack_ms),
            body_hold_ms=float(layered_body_hold_ms),
            body_decay_ms=float(layered_body_decay_ms),

            tail_attack_ms=float(layered_tail_attack_ms),
            tail_hold_ms=float(layered_tail_hold_ms),
            tail_decay_ms=float(layered_tail_decay_ms),

            duck_amount=float(layered_duck_amount),
            duck_release_ms=float(layered_duck_release_ms),

            synth_waveform=str(synth_waveform),
            synth_freq_hz=float(synth_freq_hz),
            synth_pitch_min=float(synth_pitch_min),
            synth_pitch_max=float(synth_pitch_max),
            synth_attack_ms=float(synth_attack_ms),
            synth_decay_ms=float(synth_decay_ms),
            synth_sustain_level=float(synth_sustain_level),
            synth_release_ms=float(synth_release_ms),
            synth_noise_mix=float(synth_noise_mix),
            synth_lowpass_hz=float(synth_lowpass_hz),
            synth_highpass_hz=float(synth_highpass_hz),
            synth_drive=float(synth_drive),
        )

        res = generate_with_layered(lp)
        audio = res.audio
        sr = int(res.sample_rate)
        if postprocess_fn is not None:
            audio, post_info = postprocess_fn(audio, sr)
        write_wav(out_wav, audio, sr)
        sources = tuple(res.sources)
        credits_extra = dict(res.credits_extra)
        return GeneratedWav(wav_path=out_wav, post_info=post_info, sources=sources, credits_extra=credits_extra)

    raise ValueError(f"Unknown engine: {engine}")
