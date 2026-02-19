from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .fx_chain_v2 import FxChainV2FormatError, apply_fx_chain_v2, load_fx_chain_v2_json
from .fx_chains import FX_CHAINS, load_fx_chain_json
from .io_utils import read_wav_mono, write_wav
from .json_utils import JsonParseError, load_json_file_lenient
from .postprocess import PostProcessParams, post_process_audio


def _clip_mono(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        y = np.reshape(y, (-1,)).astype(np.float32, copy=False)
    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


def _sanitize_for_filename(s: str) -> str:
    s2 = (s or "").strip().replace(" ", "_")
    out = []
    for ch in s2:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
    return "".join(out) or "region"


def _post_params_from_patch(patch: dict[str, Any]) -> PostProcessParams:
    # Neutral baseline; the chain's post_stack decides what actually runs.
    params = PostProcessParams(
        trim_silence=False,
        fade_ms=0,
        normalize_rms_db=None,
        normalize_peak_db=-1.0,
        highpass_hz=None,
        lowpass_hz=None,
        denoise_strength=0.0,
        transient_amount=0.0,
        transient_sustain=0.0,
        exciter_amount=0.0,
        multiband=False,
        reverb_preset="off",
        reverb_mix=0.0,
        reverb_time_s=1.2,
        compressor_threshold_db=None,
        limiter_ceiling_db=None,
        post_stack="final_clip",
        noise_bed_db=None,
    )

    def _get(k: str):
        return patch.get(k)

    # Map CLI-ish patch keys into PostProcessParams fields.
    if _get("no_trim") is not None:
        params = params.__class__(**{**params.__dict__, "trim_silence": (not bool(_get("no_trim")))} )
    if _get("silence_threshold_db") is not None:
        params = params.__class__(**{**params.__dict__, "silence_threshold_db": float(_get("silence_threshold_db"))})
    if _get("silence_padding_ms") is not None:
        params = params.__class__(**{**params.__dict__, "silence_padding_ms": int(_get("silence_padding_ms"))})
    if _get("fade_ms") is not None:
        params = params.__class__(**{**params.__dict__, "fade_ms": int(_get("fade_ms"))})
    if _get("normalize_rms_db") is not None:
        v = float(_get("normalize_rms_db"))
        params = params.__class__(**{**params.__dict__, "normalize_rms_db": (None if abs(v) < 1e-9 else v)})
    if _get("normalize_peak_db") is not None:
        params = params.__class__(**{**params.__dict__, "normalize_peak_db": float(_get("normalize_peak_db"))})
    if _get("highpass_hz") is not None:
        v = float(_get("highpass_hz"))
        params = params.__class__(**{**params.__dict__, "highpass_hz": (None if abs(v) < 1e-9 else v)})
    if _get("lowpass_hz") is not None:
        v = float(_get("lowpass_hz"))
        params = params.__class__(**{**params.__dict__, "lowpass_hz": (None if abs(v) < 1e-9 else v)})
    if _get("denoise_amount") is not None:
        params = params.__class__(**{**params.__dict__, "denoise_strength": float(_get("denoise_amount"))})
    if _get("transient_attack") is not None:
        params = params.__class__(**{**params.__dict__, "transient_amount": float(_get("transient_attack"))})
    if _get("transient_sustain") is not None:
        params = params.__class__(**{**params.__dict__, "transient_sustain": float(_get("transient_sustain"))})
    if _get("exciter_amount") is not None:
        params = params.__class__(**{**params.__dict__, "exciter_amount": float(_get("exciter_amount"))})
    if _get("multiband") is not None:
        params = params.__class__(**{**params.__dict__, "multiband": bool(_get("multiband"))})
    if _get("mb_low_hz") is not None:
        params = params.__class__(**{**params.__dict__, "multiband_low_hz": float(_get("mb_low_hz"))})
    if _get("mb_high_hz") is not None:
        params = params.__class__(**{**params.__dict__, "multiband_high_hz": float(_get("mb_high_hz"))})
    if _get("mb_low_gain_db") is not None:
        params = params.__class__(**{**params.__dict__, "multiband_low_gain_db": float(_get("mb_low_gain_db"))})
    if _get("mb_mid_gain_db") is not None:
        params = params.__class__(**{**params.__dict__, "multiband_mid_gain_db": float(_get("mb_mid_gain_db"))})
    if _get("mb_high_gain_db") is not None:
        params = params.__class__(**{**params.__dict__, "multiband_high_gain_db": float(_get("mb_high_gain_db"))})
    if _get("mb_comp_threshold_db") is not None:
        params = params.__class__(**{**params.__dict__, "multiband_comp_threshold_db": float(_get("mb_comp_threshold_db"))})
    if _get("mb_comp_ratio") is not None:
        params = params.__class__(**{**params.__dict__, "multiband_comp_ratio": float(_get("mb_comp_ratio"))})
    if _get("reverb") is not None:
        params = params.__class__(**{**params.__dict__, "reverb_preset": str(_get("reverb"))})
    if _get("reverb_mix") is not None:
        params = params.__class__(**{**params.__dict__, "reverb_mix": float(_get("reverb_mix"))})
    if _get("reverb_time") is not None:
        params = params.__class__(**{**params.__dict__, "reverb_time_s": float(_get("reverb_time"))})
    if _get("reverb_time_s") is not None:
        params = params.__class__(**{**params.__dict__, "reverb_time_s": float(_get("reverb_time_s"))})
    if _get("compressor_threshold_db") is not None:
        params = params.__class__(**{**params.__dict__, "compressor_threshold_db": float(_get("compressor_threshold_db"))})
    if _get("compressor_ratio") is not None:
        params = params.__class__(**{**params.__dict__, "compressor_ratio": float(_get("compressor_ratio"))})
    if _get("compressor_makeup_db") is not None:
        params = params.__class__(**{**params.__dict__, "compressor_makeup_db": float(_get("compressor_makeup_db"))})
    if _get("limiter_ceiling_db") is not None:
        params = params.__class__(**{**params.__dict__, "limiter_ceiling_db": float(_get("limiter_ceiling_db"))})
    if _get("noise_bed_db") is not None:
        params = params.__class__(**{**params.__dict__, "noise_bed_db": float(_get("noise_bed_db"))})
    if _get("post_stack") is not None:
        params = params.__class__(**{**params.__dict__, "post_stack": str(_get("post_stack"))})

    return params


def _resolve_chain_json_path(edits_path: Path, chain_json: str | None) -> Path | None:
    if chain_json is None:
        return None
    s = str(chain_json).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p
    return (edits_path.parent / p).resolve()


def _apply_region_fx(seg: np.ndarray, *, sr: int, fx_chain: str | None, fx_chain_json: str | None, edits_path: Path) -> np.ndarray:
    key = str(fx_chain or "off").strip()
    chain_json_path = _resolve_chain_json_path(edits_path, fx_chain_json)

    patch: dict[str, Any] = {}
    if key and key.lower() != "off":
        chain = FX_CHAINS.get(key)
        if chain is not None and chain.args:
            patch.update(dict(chain.args))

    if chain_json_path is not None:
        p2, _enable_post, _enable_polish = load_fx_chain_json(str(chain_json_path))
        patch.update(dict(p2))

    # If this looks like a v2 modular chain (effect-list), apply it directly.
    if chain_json_path is not None and not patch:
        try:
            chain2 = load_fx_chain_v2_json(str(chain_json_path))
            return apply_fx_chain_v2(seg.astype(np.float32, copy=False), int(sr), chain2).astype(np.float32, copy=False)
        except FxChainV2FormatError:
            pass
        except Exception:
            pass

    if not patch:
        return seg.astype(np.float32, copy=False)

    params = _post_params_from_patch(patch)
    y, _rep = post_process_audio(seg.astype(np.float32, copy=False), int(sr), params)
    return y.astype(np.float32, copy=False)


def _default_edits_path_for_wav(wav_path: Path) -> Path:
    sidecar = wav_path.with_suffix("")
    return sidecar.with_name(sidecar.name + ".edits.json")


def export_regions(
    *,
    wav_path: Path,
    edits_path: Path | None,
    out_dir: Path | None,
    apply_fx: bool,
) -> list[Path]:
    wav_path = Path(wav_path)
    edits_path = Path(edits_path) if edits_path is not None else _default_edits_path_for_wav(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(str(wav_path))
    if not edits_path.exists():
        raise FileNotFoundError(str(edits_path))

        try:
            obj = load_json_file_lenient(edits_path, context=f"Regions edits JSON file: {edits_path}")
        except JsonParseError as e:
            raise ValueError(str(e)) from e
    rr = obj.get("regions") if isinstance(obj, dict) else None
    if not isinstance(rr, list) or not rr:
        return []

    audio, sr = read_wav_mono(wav_path)
    audio = _clip_mono(audio)
    sr = int(sr)

    out_base_dir = Path(out_dir) if out_dir is not None else wav_path.parent
    out_base_dir.mkdir(parents=True, exist_ok=True)

    out_files: list[Path] = []
    for r in rr:
        if not isinstance(r, dict):
            continue

        try:
            start_s = float(r.get("start_s", 0.0))
            end_s = float(r.get("end_s", 0.0))
        except Exception:
            continue
        if end_s <= start_s:
            continue

        lo = int(round(start_s * sr))
        hi = int(round(end_s * sr))
        lo = max(0, min(lo, int(audio.size)))
        hi = max(0, min(hi, int(audio.size)))
        if hi <= lo:
            continue

        name = _sanitize_for_filename(str(r.get("name") or "region"))
        base = out_base_dir / f"{wav_path.stem}__{name}{wav_path.suffix}"
        out_path = base
        k = 2
        while out_path.exists():
            out_path = out_base_dir / f"{wav_path.stem}__{name}_{k}{wav_path.suffix}"
            k += 1

        seg = audio[lo:hi].copy()
        if apply_fx:
            seg = _apply_region_fx(
                seg,
                sr=sr,
                fx_chain=(str(r.get("fx_chain")) if r.get("fx_chain") is not None else None),
                fx_chain_json=(str(r.get("fx_chain_json")) if r.get("fx_chain_json") is not None else None),
                edits_path=edits_path,
            )
        seg = _clip_mono(seg)
        write_wav(out_path, seg, sr)

        # Sidecar for region exports (minimal but useful, matches editor behavior).
        sidecar = out_path.with_suffix("")
        sidecar = sidecar.with_name(sidecar.name + ".edits.json")
        rec: dict[str, Any] = {
            "source_wav": str(wav_path),
            "source_edits": str(edits_path),
            "exported_wav": str(out_path),
            "region": {
                "name": str(r.get("name") or "region"),
                "start_s": float(lo) / float(sr),
                "end_s": float(hi) / float(sr),
                "fx_chain": str(r.get("fx_chain") or "off"),
                "fx_chain_json": (str(r.get("fx_chain_json")) if r.get("fx_chain_json") is not None else None),
                "loop_s": r.get("loop_s"),
            },
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        sidecar.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        out_files.append(out_path)

    return out_files


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Editor region tools (v2.7 CLI v2)\n\n"
            "Exports regions non-interactively from a <wav>.edits.json sidecar created by the built-in editor."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export", help="Export region WAVs from <wav>.edits.json")
    exp.add_argument("--wav", required=True, help="Path to the source WAV")
    exp.add_argument(
        "--edits",
        default=None,
        help=("Path to .edits.json. Default: infer from --wav (e.g. foo.wav -> foo.edits.json)."),
    )
    exp.add_argument(
        "--out-dir",
        default=None,
        help="Output folder for exported regions (default: alongside the source wav)",
    )
    exp.add_argument(
        "--no-fx",
        action="store_true",
        help="Do not apply per-region FX chain on export (export raw slices only).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "export":
        out = export_regions(
            wav_path=Path(str(args.wav)),
            edits_path=(Path(str(args.edits)) if args.edits else None),
            out_dir=(Path(str(args.out_dir)) if args.out_dir else None),
            apply_fx=(not bool(args.no_fx)),
        )
        if not out:
            print("No regions found.")
            return 0
        for f in out:
            print(f"Exported region: {f}")
        return 0

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
