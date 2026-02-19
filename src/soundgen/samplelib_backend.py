from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy.signal import resample_poly

from .io_utils import find_ffmpeg, read_wav_mono, write_wav
from .json_utils import load_json_file_lenient


_AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Process-local cache to avoid re-scanning ZIP contents for every variant.
_ZIP_MEMBER_CACHE: dict[Path, list[str]] = {}

_DEFAULT_INDEX_PATH = Path("library") / "samplelib_index.json"
_INDEX_VERSION = 1


@dataclass(frozen=True)
class SampleLibSource:
    zip_path: str
    member: str
    repo: str
    attribution_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleLibResult:
    out_path: Path
    sources: tuple[SampleLibSource, ...]


@dataclass(frozen=True)
class SampleLibParams:
    prompt: str
    out_path: Path
    seconds: float = 3.0
    seed: Optional[int] = None

    # ZIP sound libraries (like .examples/sound libraies/*.zip)
    library_zips: tuple[Path, ...] = ()

    # Persistent index cache on disk (speeds selection a lot).
    # Set to None to disable.
    index_path: Optional[Path] = _DEFAULT_INDEX_PATH

    # Randomization
    pitch_min: float = 0.85
    pitch_max: float = 1.20

    # If 2, pick 2 samples and mix them (more chaotic output).
    mix_count: int = 1

    # Target output
    sample_rate: int = 44100


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _score(prompt_tokens: set[str], candidate_tokens: set[str]) -> int:
    if not prompt_tokens or not candidate_tokens:
        return 0
    return len(prompt_tokens & candidate_tokens)


def _iter_zip_audio_entries(zip_path: Path) -> Iterable[str]:
    zip_path = Path(zip_path)
    cached = _ZIP_MEMBER_CACHE.get(zip_path)
    if cached is None:
        members: list[str] = []
        with zipfile.ZipFile(zip_path, "r") as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                suffix = Path(info.filename).suffix.lower()
                if suffix in _AUDIO_EXTS:
                    members.append(info.filename)
        _ZIP_MEMBER_CACHE[zip_path] = members
        cached = members

    yield from cached


def _looks_like_attribution_file(name: str) -> bool:
    n = name.lower()
    base = Path(n).name
    if base in {"license", "license.txt", "license.md", "copying", "copying.txt"}:
        return True
    if base.startswith("license") or base.startswith("copying"):
        return True
    if base in {"readme", "readme.txt", "readme.md", "credits", "credits.txt", "credits.md"}:
        return True
    if base.startswith("readme") or base.startswith("credit"):
        return True
    return False


def _zip_metadata(zip_path: Path) -> dict[str, Any]:
    st = zip_path.stat()
    return {"mtime": int(st.st_mtime), "size": int(st.st_size)}


def _load_index(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        data = load_json_file_lenient(path, context=f"SampleLib index JSON file: {path}")
        if not isinstance(data, dict) or data.get("version") != _INDEX_VERSION:
            return None
        if not isinstance(data.get("zips"), dict):
            return None
        return data
    except Exception:
        return None


def _save_index(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_index(zip_paths: tuple[Path, ...]) -> dict[str, Any]:
    zips: dict[str, Any] = {}
    for zp in zip_paths:
        zp = Path(zp)
        zp_abs = str(zp.resolve())
        meta = _zip_metadata(zp)

        audio: list[dict[str, Any]] = []
        attribution_files: list[str] = []
        with zipfile.ZipFile(zp, "r") as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                suffix = Path(name).suffix.lower()
                if suffix in _AUDIO_EXTS:
                    audio.append(
                        {
                            "member": name,
                            "ext": suffix,
                            "tokens": sorted(_tokens(name) | _tokens(zp.stem)),
                        }
                    )
                elif _looks_like_attribution_file(name):
                    attribution_files.append(name)

        zips[zp_abs] = {
            **meta,
            "repo": zp.stem,
            "attribution_files": sorted(set(attribution_files)),
            "audio": audio,
        }

    return {"version": _INDEX_VERSION, "zips": zips}


def _ensure_index(zip_paths: tuple[Path, ...], index_path: Optional[Path]) -> dict[str, Any]:
    if index_path is None:
        return _build_index(zip_paths)

    index_path = Path(index_path)
    existing = _load_index(index_path)
    if existing is not None:
        # Validate freshness.
        zips = existing.get("zips", {})
        ok = True
        for zp in zip_paths:
            zp_abs = str(Path(zp).resolve())
            entry = zips.get(zp_abs)
            if not isinstance(entry, dict):
                ok = False
                break
            try:
                meta = _zip_metadata(Path(zp_abs))
            except FileNotFoundError:
                ok = False
                break
            if int(entry.get("mtime", -1)) != meta["mtime"] or int(entry.get("size", -1)) != meta["size"]:
                ok = False
                break

        if ok:
            return existing

    built = _build_index(zip_paths)
    _save_index(index_path, built)
    return built


def _pick_entries(
    prompt: str,
    index: dict[str, Any],
    rng: random.Random,
    *,
    count: int,
) -> list[tuple[str, str]]:
    prompt_tokens = _tokens(prompt)
    zips = index.get("zips", {})
    if not isinstance(zips, dict):
        raise ValueError("Invalid samplelib index")

    # Find best score across all candidates (pre-tokenized).
    top_score = -1
    top: list[tuple[str, str]] = []  # (zip_abs, member)

    for zip_abs, entry in zips.items():
        if not isinstance(entry, dict):
            continue
        audio = entry.get("audio")
        if not isinstance(audio, list):
            continue
        for a in audio:
            if not isinstance(a, dict):
                continue
            member = a.get("member")
            toks = a.get("tokens")
            if not isinstance(member, str) or not isinstance(toks, list):
                continue
            s = _score(prompt_tokens, set(str(t) for t in toks))
            if s > top_score:
                top_score = s
                top = [(zip_abs, member)]
            elif s == top_score:
                top.append((zip_abs, member))

    if not top:
        raise FileNotFoundError(
            "No audio files found in the provided library ZIPs. Supported: .wav .ogg .mp3 .flac"
        )

    count = max(1, int(count))
    if count == 1 or len(top) == 1:
        return [rng.choice(top)]

    # Mix mode: pick 2 distinct entries if possible.
    first = rng.choice(top)
    if len(top) == 1:
        return [first]
    second = rng.choice([t for t in top if t != first])
    return [first, second]


def _decode_to_wav(input_path: Path, output_wav: Path, *, sample_rate: int) -> None:
    cmd = [
        find_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "wav",
        str(output_wav),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed decoding sample library audio:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )


def _extract_member(zip_path: Path, member: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(member).name
    # Ensure a stable filename even if duplicates exist.
    if out.exists():
        out = dest_dir / f"{out.stem}_{abs(hash(member)) % 1_000_000}{out.suffix}"

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


def _ensure_length(audio: np.ndarray, sr: int, *, seconds: float) -> np.ndarray:
    target = max(1, int(round(float(seconds) * int(sr))))
    if audio.shape[0] == target:
        return audio
    if audio.shape[0] > target:
        return audio[:target]
    pad = target - audio.shape[0]
    return np.pad(audio, (0, pad), mode="constant")


def _apply_pitch(audio: np.ndarray, *, pitch: float) -> np.ndarray:
    # Resample by 1/pitch to raise pitch when pitch>1 (shorter waveform) and vice versa.
    if pitch <= 0:
        return audio
    inv = 1.0 / float(pitch)
    # Convert to rational approximation for polyphase resampling.
    up = int(round(1000))
    down = int(round(1000 * inv))
    down = max(1, down)
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def _safe_normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def generate_with_samplelib(params: SampleLibParams) -> SampleLibResult:
    out_path = Path(params.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zip_paths = tuple(Path(p) for p in params.library_zips)
    if not zip_paths:
        raise ValueError("No library ZIPs provided.")

    rng = random.Random(params.seed)

    index = _ensure_index(zip_paths, params.index_path)
    picks = _pick_entries(params.prompt, index, rng, count=int(params.mix_count))

    sources: list[SampleLibSource] = []
    decoded_audios: list[np.ndarray] = []
    decoded_sr: Optional[int] = None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for idx, (zip_abs, member) in enumerate(picks):
            zp = Path(zip_abs)
            extracted = _extract_member(zp, member, tmp_dir / "src")

            decoded_wav = tmp_dir / f"decoded_{idx}.wav"
            _decode_to_wav(extracted, decoded_wav, sample_rate=params.sample_rate)

            audio, sr = read_wav_mono(decoded_wav)
            decoded_audios.append(audio)
            decoded_sr = int(sr) if decoded_sr is None else decoded_sr

            entry = index.get("zips", {}).get(zip_abs, {})
            attribution_files: tuple[str, ...] = ()
            if isinstance(entry, dict) and isinstance(entry.get("attribution_files"), list):
                attribution_files = tuple(str(x) for x in entry.get("attribution_files", []) if str(x).strip())
            repo = str(entry.get("repo") or Path(zip_abs).stem) if isinstance(entry, dict) else Path(zip_abs).stem

            sources.append(
                SampleLibSource(
                    zip_path=str(zip_abs),
                    member=str(member),
                    repo=str(repo),
                    attribution_files=attribution_files,
                )
            )

    if not decoded_audios or decoded_sr is None:
        raise RuntimeError("Samplelib decode produced no audio")

    # Random pitch variation per source.
    pmin = float(params.pitch_min)
    pmax = float(params.pitch_max)
    if pmax < pmin:
        pmin, pmax = pmax, pmin

    pitched: list[np.ndarray] = []
    for a in decoded_audios:
        pitch = rng.uniform(pmin, pmax)
        pitched.append(_apply_pitch(a, pitch=pitch))

    # Mix if requested.
    if int(params.mix_count) >= 2 and len(pitched) >= 2:
        a0, a1 = pitched[0], pitched[1]
        # Random weights around 50/50.
        w0 = rng.uniform(0.40, 0.60)
        w1 = 1.0 - w0
        n = max(a0.shape[0], a1.shape[0])
        a0 = np.pad(a0, (0, max(0, n - a0.shape[0])), mode="constant")
        a1 = np.pad(a1, (0, max(0, n - a1.shape[0])), mode="constant")
        audio = (w0 * a0 + w1 * a1).astype(np.float32, copy=False)
    else:
        audio = pitched[0].astype(np.float32, copy=False)

    audio = _ensure_length(audio, decoded_sr, seconds=float(params.seconds))
    audio = _safe_normalize(audio)

    write_wav(out_path, audio, int(decoded_sr))
    return SampleLibResult(out_path=out_path, sources=tuple(sources))
