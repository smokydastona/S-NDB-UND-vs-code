from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from .engine_registry import available_engines, generate_wav
from .io_utils import read_wav_mono
from .postprocess import PostProcessParams, post_process_audio
from .qa import compute_metrics, detect_long_tail


@dataclass(frozen=True)
class RunResult:
    engine: str
    prompt_id: str
    prompt: str
    seed: int
    seconds: float
    candidates: int
    post: bool
    polish: bool
    wav_path: str
    elapsed_s: float
    peak: float
    rms: float
    clipped: bool
    long_tail: bool
    sample_rate: int
    error: str | None = None


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _suite_default() -> list[tuple[str, str, float]]:
    """A small, mixed prompt suite to compare engines.

    Format: (prompt_id, prompt_text, seconds)
    """

    return [
        ("ui_click", "ui click, short, clean, no reverb", 0.35),
        ("coin_pickup", "coin pickup, bright, tiny chime, short", 0.55),
        ("laser_zap", "laser zap, sci-fi blaster, tight transient, no tail", 0.70),
        ("impact_thud", "impact thud, punchy, mid-heavy, no music", 0.85),
        ("whoosh", "whoosh swish, fast pass-by, airy tail", 1.20),
        ("ambience_cave", "cave ambience drone, deep rumble, distant air", 6.00),
        ("creature_growl", "creature growl, gritty, textured, clear midrange", 2.20),
        ("creature_roar", "massive creature roar, powerful low end, chest resonance", 3.60),
        ("creature_chitter", "small creature chitter, insectoid, fast clicks, sharp transients", 1.20),
        ("creature_hiss", "creature hiss, breathy, wet air, short", 1.40),
    ]


def _suite_creature() -> list[tuple[str, str, float]]:
    """Creature vocalization benchmark suite.

    This suite aims to stress a range of creature timbres and envelopes.
    Prompts explicitly ask for a single, isolated vocalization (no music).

    Format: (prompt_id, prompt_text, seconds)
    """

    base = "single creature vocalization, isolated, no music, no melody, no speech, no words"

    return [
        ("growl_short", f"{base}, short guttural growl, gritty texture, tight tail", 1.60),
        ("growl_long", f"{base}, sustained growl, rumbling low end, subtle throat rasp", 2.60),
        ("roar_short", f"{base}, short roar, aggressive burst, chest resonance, no reverb", 2.20),
        ("roar_long", f"{base}, massive roar, powerful low end, wide chest resonance, long decay", 3.80),
        ("snarl", f"{base}, snarl, teeth grit, harsh breath, midrange bite", 1.40),
        ("hiss", f"{base}, hiss, wet breath, close-up air, short", 1.20),
        ("chitter", f"{base}, chitter, insectoid clicks, fast transients, bright", 1.20),
        ("screech", f"{base}, screech, piercing, rough distortion edge, brief", 1.60),
        ("moan", f"{base}, eerie moan, hollow resonance, breathy, low intensity", 2.60),
        ("bark", f"{base}, bark, sharp attack, animalistic, dry", 1.00),
        ("yelp", f"{base}, yelp, startled, high pitch, quick", 0.85),
        ("howl", f"{base}, howl, sustained, tonal but not musical, subtle vibrato", 3.40),
    ]


def _pp_params(*, seed: int | None, prompt_hint: str | None, polish: bool) -> PostProcessParams:
    # Keep this conservative and stable; benchmarking should avoid lots of auto-magic.
    # (Users can always run full generate.py for the full Pro Mode system.)
    return PostProcessParams(
        trim_silence=True,
        silence_threshold_db=-45.0,
        silence_padding_ms=18,
        fade_ms=10,
        normalize_rms_db=(-18.0 if polish else -20.0),
        normalize_peak_db=-1.0,
        highpass_hz=30.0,
        lowpass_hz=None,
        denoise_strength=(0.25 if polish else 0.0),
        transient_amount=(0.35 if polish else 0.0),
        transient_sustain=0.0,
        transient_split_hz=1200.0,
        exciter_amount=(0.10 if polish else 0.0),
        multiband=bool(polish),
        multiband_low_hz=250.0,
        multiband_high_hz=3000.0,
        multiband_low_gain_db=0.0,
        multiband_mid_gain_db=0.0,
        multiband_high_gain_db=0.0,
        multiband_comp_threshold_db=(-24.0 if polish else None),
        multiband_comp_ratio=2.0,
        formant_shift=1.0,
        creature_size=0.0,
        texture_preset="off",
        texture_amount=0.0,
        texture_grain_ms=28.0,
        texture_spray=0.35,
        reverb_preset="off",
        reverb_mix=0.0,
        reverb_time_s=1.2,
        random_seed=(int(seed) if seed is not None else None),
        prompt_hint=(str(prompt_hint) if prompt_hint else None),
        compressor_threshold_db=(-18.0 if polish else None),
        compressor_ratio=4.0,
        compressor_attack_ms=5.0,
        compressor_release_ms=90.0,
        compressor_makeup_db=(3.0 if polish else 0.0),
        limiter_ceiling_db=(-1.0 if polish else None),
        loop_clean=False,
        loop_crossfade_ms=100,
    )


def _postprocess_fn(*, seed: int | None, prompt_hint: str | None, post: bool, polish: bool):
    if not (post or polish):
        return None

    def _pp(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
        processed, rep = post_process_audio(
            audio,
            sr,
            _pp_params(seed=seed, prompt_hint=prompt_hint, polish=bool(polish)),
        )
        m = compute_metrics(processed, sr)
        flags: list[str] = []
        if m.clipped:
            flags.append("CLIPPING")
        if detect_long_tail(processed, sr):
            flags.append("LONG_TAIL")
        flag_s = (" " + " ".join(flags)) if flags else ""
        info = f"post: trimmed={rep.trimmed} qa: peak={m.peak:.3f} rms={m.rms:.3f}{flag_s}".strip()
        return processed, info

    return _pp


def _summarize(results: list[RunResult]) -> list[dict[str, Any]]:
    # Group by engine+prompt_id
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.engine, r.prompt_id), []).append(r)

    rows: list[dict[str, Any]] = []
    for (engine, prompt_id), rs in sorted(groups.items()):
        ok = [x for x in rs if not x.error]
        err = [x for x in rs if x.error]
        peaks = [x.peak for x in ok]
        rms = [x.rms for x in ok]
        times = [x.elapsed_s for x in ok]
        clip_rate = (sum(1 for x in ok if x.clipped) / len(ok)) if ok else 0.0
        tail_rate = (sum(1 for x in ok if x.long_tail) / len(ok)) if ok else 0.0

        row: dict[str, Any] = {
            "engine": engine,
            "prompt_id": prompt_id,
            "runs": len(rs),
            "ok": len(ok),
            "errors": len(err),
            "mean_elapsed_s": (mean(times) if times else None),
            "peak_mean": (mean(peaks) if peaks else None),
            "peak_std": (pstdev(peaks) if len(peaks) >= 2 else 0.0),
            "rms_mean": (mean(rms) if rms else None),
            "rms_std": (pstdev(rms) if len(rms) >= 2 else 0.0),
            "clip_rate": clip_rate,
            "long_tail_rate": tail_rate,
        }
        rows.append(row)

    return rows


def run_benchmark(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark quality/consistency across engines (QA proxy metrics).")
    p.add_argument(
        "--engines",
        nargs="+",
        choices=available_engines(),
        default=["diffusers", "stable_audio_open", "layered", "rfxgen"],
        help="Engines to run (built-ins + optional plugins).",
    )
    p.add_argument(
        "--engine-choices",
        action="store_true",
        help="Print available engine names (including plugins) and exit.",
    )
    p.add_argument("--repeats", type=int, default=4, help="Runs per prompt per engine.")
    p.add_argument("--seed", type=int, default=1000, help="Base seed; repeat i uses seed+i.")
    p.add_argument("--candidates", type=int, default=1, help="Best-of-N candidate selection.")
    p.add_argument("--post", action="store_true", help="Apply post-processing.")
    p.add_argument("--polish", action="store_true", help="Apply polish DSP (implies --post).")
    p.add_argument(
        "--suite",
        choices=["default", "creature"],
        default="default",
        help="Which prompt suite to run.",
    )
    p.add_argument("--out-dir", default=None, help="Output folder. Default outputs/bench/<timestamp>.")
    p.add_argument("--keep-wavs", action="store_true", help="Keep WAVs for listening (default: false).")

    # AI engine knobs (kept minimal)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--diffusers-model", default="cvssp/audioldm2")
    p.add_argument("--stable-audio-model", default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--stable-audio-steps", type=int, default=100)
    p.add_argument("--stable-audio-guidance-scale", type=float, default=7.0)
    p.add_argument("--stable-audio-negative-prompt", default="music, melody, singing, speech, words")
    p.add_argument("--hf-token", default=None, help="Optional HF token for stable_audio_open.")

    args = p.parse_args(argv)

    if bool(getattr(args, "engine_choices", False)):
        print("\n".join(available_engines()))
        return 0

    engines = [str(e).strip().lower() for e in (args.engines or []) if str(e).strip()]
    repeats = max(1, int(args.repeats))
    base_seed = int(args.seed)
    candidates = max(1, int(args.candidates))
    post = bool(args.post) or bool(args.polish)
    polish = bool(args.polish)

    out_dir = Path(args.out_dir) if args.out_dir else (Path("outputs") / "bench" / _now_slug())
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "summary.csv"

    suite_name = str(args.suite).strip().lower()
    suite = _suite_creature() if suite_name == "creature" else _suite_default()
    results: list[RunResult] = []

    print(f"Benchmark out: {out_dir}")
    print(f"Engines: {', '.join(engines)}")
    print(
        f"Suite: {suite_name}  prompts: {len(suite)}  repeats: {repeats}  candidates: {candidates}  post={post} polish={polish}"
    )

    with runs_path.open("w", encoding="utf-8") as f:
        for engine in engines:
            for prompt_id, prompt, seconds in suite:
                for i in range(repeats):
                    seed_i = base_seed + i
                    wav_path = out_dir / f"{engine}__{prompt_id}__r{i+1:02d}.wav"
                    t0 = time.perf_counter()
                    err: str | None = None
                    try:
                        gen = generate_wav(
                            engine,
                            prompt=str(prompt),
                            seconds=float(seconds),
                            seed=int(seed_i),
                            out_wav=wav_path,
                            candidates=int(candidates),
                            postprocess_fn=_postprocess_fn(
                                seed=int(seed_i),
                                prompt_hint=str(prompt),
                                post=bool(post),
                                polish=bool(polish),
                            ),
                            device=str(args.device),
                            model=str(args.diffusers_model),
                            stable_audio_model=str(args.stable_audio_model),
                            stable_audio_negative_prompt=str(args.stable_audio_negative_prompt or "") or None,
                            stable_audio_steps=int(args.stable_audio_steps),
                            stable_audio_guidance_scale=float(args.stable_audio_guidance_scale),
                            stable_audio_hf_token=(str(args.hf_token).strip() if args.hf_token else None),
                            sample_rate=44100,
                        )
                        written = Path(gen.wav_path)
                        audio, sr = read_wav_mono(written)
                        m = compute_metrics(audio, sr)
                        r = RunResult(
                            engine=str(engine),
                            prompt_id=str(prompt_id),
                            prompt=str(prompt),
                            seed=int(seed_i),
                            seconds=float(seconds),
                            candidates=int(candidates),
                            post=bool(post),
                            polish=bool(polish),
                            wav_path=str(written),
                            elapsed_s=float(time.perf_counter() - t0),
                            peak=float(m.peak),
                            rms=float(m.rms),
                            clipped=bool(m.clipped),
                            long_tail=bool(detect_long_tail(audio, sr)),
                            sample_rate=int(sr),
                            error=None,
                        )

                        if not args.keep_wavs:
                            try:
                                written.unlink(missing_ok=True)
                            except Exception:
                                pass
                    except Exception as e:
                        err = str(e).strip() or e.__class__.__name__
                        r = RunResult(
                            engine=str(engine),
                            prompt_id=str(prompt_id),
                            prompt=str(prompt),
                            seed=int(seed_i),
                            seconds=float(seconds),
                            candidates=int(candidates),
                            post=bool(post),
                            polish=bool(polish),
                            wav_path=str(wav_path),
                            elapsed_s=float(time.perf_counter() - t0),
                            peak=0.0,
                            rms=0.0,
                            clipped=False,
                            long_tail=False,
                            sample_rate=0,
                            error=err,
                        )

                    results.append(r)
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                    f.flush()

                    status = "OK" if not err else "FAIL"
                    print(
                        f"[{status}] {engine:16s} {prompt_id:16s} r={i+1:02d} seed={seed_i} time={r.elapsed_s:.2f}s"
                        + ("" if not err else f"  err={err}")
                    )

    summary = _summarize(results)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    print(f"\nWrote runs: {runs_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


def main() -> None:
    raise SystemExit(run_benchmark())


if __name__ == "__main__":
    main()
