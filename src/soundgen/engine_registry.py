from __future__ import annotations

from dataclasses import dataclass, field
import os
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .io_utils import read_wav_mono, write_wav


PostprocessFn = Callable[[np.ndarray, int], tuple[np.ndarray, str]]


@dataclass(frozen=True)
class EnginePluginResult:
    audio: np.ndarray
    sample_rate: int
    credits_extra: dict[str, Any] = field(default_factory=dict)
    sources: tuple[dict[str, Any], ...] = ()


EnginePluginFn = Callable[[dict[str, Any]], EnginePluginResult]


BUILTIN_ENGINES: tuple[str, ...] = (
    "diffusers",
    "stable_audio_open",
    "rfxgen",
    "replicate",
    "samplelib",
    "synth",
    "layered",
)

_ENGINE_PLUGINS: dict[str, EnginePluginFn] = {}
_PLUGINS_LOADED: bool = False


def register_engine(*, engine_name: str, engine_fn: EnginePluginFn, overwrite: bool = False) -> None:
    """Register an engine plugin.

    Plugins are expected to be *audio-producing* engines: they return mono audio
    and a sample rate; `generate_wav()` handles postprocess + WAV writing.
    """

    key = str(engine_name or "").strip().lower()
    if not key:
        raise ValueError("engine_name is required")
    if not overwrite and key in _ENGINE_PLUGINS:
        raise ValueError(f"Engine already registered: {key}")
    _ENGINE_PLUGINS[key] = engine_fn


def _load_plugins_once() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True

    try:
        from .plugin_loader import load_engine_plugins

        load_engine_plugins(register_engine=lambda name, fn: register_engine(engine_name=name, engine_fn=fn))
    except Exception:
        # Plugin loading is best-effort; core engines must keep working.
        return


def available_engines() -> list[str]:
    _load_plugins_once()
    return sorted(set(BUILTIN_ENGINES).union(_ENGINE_PLUGINS.keys()))


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
    candidates: int = 1,
    postprocess_fn: PostprocessFn | None = None,
    # diffusers
    device: str | None = None,
    model: str | None = None,
    diffusers_multiband: bool = False,
    diffusers_multiband_mode: str = "auto",  # auto|2band|3band
    diffusers_multiband_low_hz: float = 250.0,
    diffusers_multiband_high_hz: float = 3000.0,
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
    layered_granular_preset: str = "off",
    layered_granular_amount: float = 0.0,
    layered_granular_grain_ms: float = 28.0,
    layered_granular_spray: float = 0.35,
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
    # stable audio open
    stable_audio_model: str | None = None,
    stable_audio_negative_prompt: str | None = None,
    stable_audio_steps: int = 100,
    stable_audio_guidance_scale: float = 7.0,
    stable_audio_sampler: str | None = None,
    stable_audio_hf_token: str | None = None,
    stable_audio_lora_path: str | None = None,
    stable_audio_lora_scale: float = 1.0,
    stable_audio_lora_trigger: str | None = None,
    sample_rate: int = 44100,
) -> GeneratedWav:
    """Generate a mono WAV file for any engine.

    Returns a standardized result with engine-specific credits fields and optional sources.
    """

    engine = str(engine).strip().lower()

    # Best-effort plugin discovery (no-op if no plugins are present).
    _load_plugins_once()

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cand_n = int(candidates or 1)
    if cand_n < 1:
        cand_n = 1
    if cand_n > 1:
        # Best-of-N wrapper: generate multiple candidate WAVs and pick the best
        # using QA metrics (clip + peak/rms + long-tail detection).
        base_seed = seed
        if base_seed is None:
            base_seed = int.from_bytes(os.urandom(4), "big")
        cand_paths: list[Path] = []
        cand_results: list[GeneratedWav] = []
        for i in range(cand_n):
            cand_out = out_wav.with_name(f"{out_wav.stem}__cand{i + 1}{out_wav.suffix}")
            res = generate_wav(
                engine,
                prompt=prompt,
                seconds=seconds,
                seed=int(base_seed) + i,
                out_wav=cand_out,
                candidates=1,
                postprocess_fn=postprocess_fn,

                device=device,
                model=model,
                diffusers_multiband=diffusers_multiband,
                diffusers_multiband_mode=diffusers_multiband_mode,
                diffusers_multiband_low_hz=diffusers_multiband_low_hz,
                diffusers_multiband_high_hz=diffusers_multiband_high_hz,

                preset=preset,
                rfxgen_path=rfxgen_path,

                replicate_model=replicate_model,
                replicate_token=replicate_token,
                replicate_input_json=replicate_input_json,

                library_zips=library_zips,
                library_pitch_min=library_pitch_min,
                library_pitch_max=library_pitch_max,
                library_mix_count=library_mix_count,
                library_index_path=library_index_path,

                synth_waveform=synth_waveform,
                synth_freq_hz=synth_freq_hz,
                synth_attack_ms=synth_attack_ms,
                synth_decay_ms=synth_decay_ms,
                synth_sustain_level=synth_sustain_level,
                synth_release_ms=synth_release_ms,
                synth_noise_mix=synth_noise_mix,
                synth_drive=synth_drive,
                synth_pitch_min=synth_pitch_min,
                synth_pitch_max=synth_pitch_max,
                synth_lowpass_hz=synth_lowpass_hz,
                synth_highpass_hz=synth_highpass_hz,

                layered_preset=layered_preset,
                layered_preset_lock=layered_preset_lock,
                layered_variant_index=layered_variant_index,
                layered_micro_variation=layered_micro_variation,
                layered_env_curve_shape=layered_env_curve_shape,
                layered_transient_tilt=layered_transient_tilt,
                layered_body_tilt=layered_body_tilt,
                layered_tail_tilt=layered_tail_tilt,
                layered_source_lock=layered_source_lock,
                layered_source_seed=layered_source_seed,
                layered_granular_preset=layered_granular_preset,
                layered_granular_amount=layered_granular_amount,
                layered_granular_grain_ms=layered_granular_grain_ms,
                layered_granular_spray=layered_granular_spray,
                layered_transient_ms=layered_transient_ms,
                layered_tail_ms=layered_tail_ms,
                layered_transient_attack_ms=layered_transient_attack_ms,
                layered_transient_hold_ms=layered_transient_hold_ms,
                layered_transient_decay_ms=layered_transient_decay_ms,
                layered_body_attack_ms=layered_body_attack_ms,
                layered_body_hold_ms=layered_body_hold_ms,
                layered_body_decay_ms=layered_body_decay_ms,
                layered_tail_attack_ms=layered_tail_attack_ms,
                layered_tail_hold_ms=layered_tail_hold_ms,
                layered_tail_decay_ms=layered_tail_decay_ms,
                layered_duck_amount=layered_duck_amount,
                layered_duck_release_ms=layered_duck_release_ms,

                stable_audio_model=stable_audio_model,
                stable_audio_negative_prompt=stable_audio_negative_prompt,
                stable_audio_steps=stable_audio_steps,
                stable_audio_guidance_scale=stable_audio_guidance_scale,
                stable_audio_sampler=stable_audio_sampler,

                sample_rate=sample_rate,
            )
            cand_paths.append(res.wav_path)
            cand_results.append(res)

        def _rms_db(v: float) -> float:
            return 20.0 * math.log10(max(float(v), 1e-9))

        def _score(wav_path: Path) -> tuple[float, dict[str, Any] | None]:
            if wav_path.suffix.lower() != ".wav" or not wav_path.exists():
                return -1e9, None
            from .qa import compute_metrics, detect_long_tail

            x, sr = read_wav_mono(wav_path)
            m = compute_metrics(x, sr)
            long_tail = detect_long_tail(x, sr)

            # Heuristic score: hard-avoid clipping, prefer healthy loudness and peaks near full-scale,
            # and lightly penalize long tails.
            score = 0.0
            if bool(m.clipped):
                score -= 1000.0
            # Peak: penalize both low (weak) and too close to 1.0 (risk).
            peak = float(m.peak)
            score -= abs(peak - 0.98) * 12.0
            if peak > 0.995:
                score -= 10.0
            # RMS: aim for a moderate range (-24 .. -12 dBFS).
            rms_db = _rms_db(float(m.rms))
            if rms_db < -24.0:
                score -= (-24.0 - rms_db) * 2.0
            elif rms_db > -12.0:
                score -= (rms_db + 12.0) * 2.0
            if bool(long_tail):
                score -= 3.0

            info = {
                "peak": float(m.peak),
                "rms": float(m.rms),
                "clipped": bool(m.clipped),
                "long_tail": bool(long_tail),
                "seconds": float(m.seconds),
                "sample_rate": int(m.sample_rate),
            }
            return float(score), info

        scored: list[tuple[float, dict[str, Any] | None]] = [_score(p) for p in cand_paths]
        wav_indices = [i for i, pth in enumerate(cand_paths) if pth.exists() and pth.suffix.lower() == ".wav"]
        if not wav_indices:
            # Some engines may return non-wav outputs; in that case, don't attempt to
            # pick/rename candidates. Return the first result unchanged.
            first = cand_results[0]
            merged_credits = dict(first.credits_extra)
            merged_credits["best_of_n"] = {
                "candidates": int(cand_n),
                "seed_base": int(base_seed),
                "picked_index": 1,
                "picked_seed": int(base_seed),
                "candidate_scores": [float(s[0]) for s in scored],
                "candidate_metrics": [s[1] for s in scored],
                "note": "best-of-N selection skipped (no WAV candidates)",
                "candidate_suffixes": [str(p.suffix) for p in cand_paths],
            }
            return GeneratedWav(
                wav_path=first.wav_path,
                post_info=first.post_info,
                sources=first.sources,
                credits_extra=merged_credits,
            )

        best_idx = max(wav_indices, key=lambda i: scored[i][0])
        best_res = cand_results[best_idx]
        best_path = cand_paths[best_idx]

        # Move best candidate into requested out_wav path.
        if best_path.suffix.lower() == ".wav" and best_path != out_wav:
            if out_wav.exists():
                out_wav.unlink(missing_ok=True)
            best_path.replace(out_wav)
            best_res = GeneratedWav(
                wav_path=out_wav,
                post_info=best_res.post_info,
                sources=best_res.sources,
                credits_extra=dict(best_res.credits_extra),
            )

        # Cleanup non-selected candidates.
        for i, pth in enumerate(cand_paths):
            if i == best_idx:
                continue
            try:
                pth.unlink(missing_ok=True)
            except Exception:
                pass

        # Record selection metadata.
        pick_info = {
            "candidates": int(cand_n),
            "seed_base": int(base_seed),
            "picked_index": int(best_idx) + 1,
            "picked_seed": int(base_seed) + int(best_idx),
            "candidate_scores": [float(s[0]) for s in scored],
            "candidate_metrics": [s[1] for s in scored],
        }
        merged_credits = dict(best_res.credits_extra)
        merged_credits["best_of_n"] = pick_info
        return GeneratedWav(
            wav_path=best_res.wav_path,
            post_info=best_res.post_info,
            sources=best_res.sources,
            credits_extra=merged_credits,
        )

    post_info: str | None = None
    sources: tuple[dict[str, Any], ...] = ()
    credits_extra: dict[str, Any] = {}

    # Plugin engines (audio-producing).
    plugin_fn = _ENGINE_PLUGINS.get(engine)
    if plugin_fn is not None:
        req = {
            "engine": engine,
            "prompt": prompt,
            "seconds": float(seconds),
            "seed": seed,
            "device": (str(device) if device is not None else None),
            "sample_rate": int(sample_rate),
        }
        res = plugin_fn(req)
        audio = res.audio
        sr = int(res.sample_rate)
        if postprocess_fn is not None:
            audio, post_info = postprocess_fn(audio, sr)
        write_wav(out_wav, audio, sr)
        return GeneratedWav(
            wav_path=out_wav,
            post_info=post_info,
            sources=tuple(res.sources),
            credits_extra=dict(res.credits_extra),
        )

    if engine == "diffusers":
        from .audiogen_backend import GenerationParams, generate_audio

        gp = GenerationParams(
            prompt=prompt,
            seconds=float(seconds),
            seed=seed,
            device=str(device or "cpu"),
            model=str(model or "cvssp/audioldm2"),
            multiband=bool(diffusers_multiband),
            multiband_mode=str(diffusers_multiband_mode or "auto"),
            multiband_low_hz=float(diffusers_multiband_low_hz),
            multiband_high_hz=float(diffusers_multiband_high_hz),
        )
        audio, sr = generate_audio(gp)
        if postprocess_fn is not None:
            audio, post_info = postprocess_fn(audio, sr)
        write_wav(out_wav, audio, sr)
        credits_extra = {
            "model": gp.model,
            "device": gp.device,
            "seed": seed,
            "diffusers_multiband": bool(gp.multiband),
            "diffusers_multiband_mode": (str(gp.multiband_mode) if bool(gp.multiband) else None),
            "diffusers_multiband_low_hz": (float(gp.multiband_low_hz) if bool(gp.multiband) else None),
            "diffusers_multiband_high_hz": (float(gp.multiband_high_hz) if bool(gp.multiband) else None),
        }
        return GeneratedWav(wav_path=out_wav, post_info=post_info, sources=sources, credits_extra=credits_extra)

    if engine == "stable_audio_open":
        from .stable_audio_backend import StableAudioOpenParams, generate_audio

        sp = StableAudioOpenParams(
            prompt=prompt,
            seconds=float(seconds),
            seed=seed,
            device=str(device or "cpu"),
            model=str(stable_audio_model or "stabilityai/stable-audio-open-1.0"),
            negative_prompt=(str(stable_audio_negative_prompt) if stable_audio_negative_prompt else None),
            num_inference_steps=int(stable_audio_steps),
            guidance_scale=float(stable_audio_guidance_scale),
            sampler=(str(stable_audio_sampler) if stable_audio_sampler else None),
            hf_token=(str(stable_audio_hf_token).strip() if stable_audio_hf_token else None),
            lora_path=(str(stable_audio_lora_path).strip() if stable_audio_lora_path else None),
            lora_scale=float(stable_audio_lora_scale),
            lora_trigger=(str(stable_audio_lora_trigger).strip() if stable_audio_lora_trigger else None),
        )
        audio, sr = generate_audio(sp)
        if postprocess_fn is not None:
            audio, post_info = postprocess_fn(audio, sr)
        write_wav(out_wav, audio, sr)
        credits_extra = {
            "model": sp.model,
            "device": sp.device,
            "seed": seed,
            "stable_audio_steps": int(sp.num_inference_steps),
            "stable_audio_guidance_scale": float(sp.guidance_scale),
            "stable_audio_sampler": (str(sp.sampler) if sp.sampler else None),
            "stable_audio_negative_prompt": (str(sp.negative_prompt) if sp.negative_prompt else None),
            "stable_audio_lora_path": (str(sp.lora_path) if sp.lora_path else None),
            "stable_audio_lora_scale": (float(sp.lora_scale) if sp.lora_path else None),
            "stable_audio_lora_trigger": (str(sp.lora_trigger) if sp.lora_trigger else None),
        }
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

            granular_preset=str(layered_granular_preset),
            granular_amount=float(layered_granular_amount),
            granular_grain_ms=float(layered_granular_grain_ms),
            granular_spray=float(layered_granular_spray),

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
