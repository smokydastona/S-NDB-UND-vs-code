from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_wav_mono, write_wav
from .runtime_config import configure_runtime


_MAX_HISTORY = 30


def _clip_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _normalize_peak(audio: np.ndarray, *, peak_db: float = -1.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-12:
        return audio.astype(np.float32, copy=False)
    target = _db_to_lin(float(peak_db))
    gain = target / peak
    return _clip_mono(audio * gain)


def _fade(audio: np.ndarray, *, sr: int, mode: str, ms: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(audio.size)
    if n == 0:
        return audio.astype(np.float32, copy=False)

    fade_n = int(round(float(ms) * float(sr) / 1000.0))
    fade_n = max(0, min(fade_n, n))
    if fade_n <= 0:
        return audio.astype(np.float32, copy=False)

    out = audio.copy()
    if mode == "in":
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        out[:fade_n] *= ramp
    elif mode == "out":
        ramp = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        out[-fade_n:] *= ramp
    else:
        raise ValueError("mode must be 'in' or 'out'")
    return _clip_mono(out)


def _safe_slice(audio: np.ndarray, start: int, end: int) -> tuple[int, int]:
    n = int(audio.size)
    s = max(0, min(int(start), n))
    e = max(0, min(int(end), n))
    if e < s:
        s, e = e, s
    return s, e


@dataclass
class SessionState:
    session_id: str
    session_dir: Path
    sample_rate: int
    cur: str
    clipboard: str | None
    undo: list[str]
    redo: list[str]
    history: list[dict[str, Any]]

    @property
    def cur_path(self) -> Path:
        return self.session_dir / self.cur


def _session_dir(base: Path, session_id: str) -> Path:
    return base / "editor_sessions" / session_id


def _state_path(sess_dir: Path) -> Path:
    return sess_dir / "session.json"


def _load_state(base: Path, session_id: str) -> SessionState:
    sess_dir = _session_dir(base, session_id)
    p = _state_path(sess_dir)
    if not p.exists():
        raise FileNotFoundError("Unknown editor session.")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return SessionState(
        session_id=str(obj["session_id"]),
        session_dir=sess_dir,
        sample_rate=int(obj["sample_rate"]),
        cur=str(obj["cur"]),
        clipboard=(str(obj["clipboard"]) if obj.get("clipboard") else None),
        undo=[str(x) for x in (obj.get("undo") or [])],
        redo=[str(x) for x in (obj.get("redo") or [])],
        history=[dict(x) for x in (obj.get("history") or [])],
    )


def _save_state(st: SessionState) -> None:
    obj = {
        "session_id": st.session_id,
        "sample_rate": int(st.sample_rate),
        "cur": st.cur,
        "clipboard": st.clipboard,
        "undo": list(st.undo),
        "redo": list(st.redo),
        "history": list(st.history),
    }
    _state_path(st.session_dir).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_state_filename(st: SessionState) -> str:
    # state_0000.wav, state_0001.wav, ...
    existing = sorted(st.session_dir.glob("state_*.wav"))
    nxt = len(existing)
    return f"state_{nxt:04d}.wav"


def _clipboard_path(st: SessionState) -> Path | None:
    if not st.clipboard:
        return None
    return st.session_dir / str(st.clipboard)


def _push_history(st: SessionState, op: str, params: dict[str, Any]) -> None:
    st.history.append({"op": str(op), "params": params})


def _commit_new_audio(st: SessionState, audio: np.ndarray, *, op: str, params: dict[str, Any]) -> None:
    audio = _clip_mono(audio)

    # Save previous pointer for undo.
    st.undo.append(st.cur)
    if len(st.undo) > _MAX_HISTORY:
        # Drop oldest; keep the wav around to avoid accidental deletion of active files.
        st.undo = st.undo[-_MAX_HISTORY:]

    st.redo = []

    fn = _new_state_filename(st)
    out_path = st.session_dir / fn
    write_wav(out_path, audio, int(st.sample_rate), subtype="PCM_16")
    st.cur = fn

    _push_history(st, op, params)
    _save_state(st)


def _op_copy(st: SessionState, *, start: int, end: int) -> dict[str, Any]:
    audio, sr = read_wav_mono(st.cur_path)
    if int(sr) != int(st.sample_rate):
        st.sample_rate = int(sr)

    s, e = _safe_slice(audio, start, end)
    seg = audio[s:e].copy()
    if seg.size == 0:
        raise ValueError("copy requires a non-empty selection")

    fn = "clipboard.wav"
    write_wav(st.session_dir / fn, _clip_mono(seg), int(st.sample_rate), subtype="PCM_16")
    st.clipboard = fn
    _push_history(st, "copy", {"start": int(s), "end": int(e), "samples": int(seg.size)})
    _save_state(st)
    return {}


def _op_cut(st: SessionState, *, start: int, end: int) -> dict[str, Any]:
    audio, sr = read_wav_mono(st.cur_path)
    if int(sr) != int(st.sample_rate):
        st.sample_rate = int(sr)

    s, e = _safe_slice(audio, start, end)
    seg = audio[s:e].copy()
    if seg.size == 0:
        raise ValueError("cut requires a non-empty selection")

    fn = "clipboard.wav"
    write_wav(st.session_dir / fn, _clip_mono(seg), int(st.sample_rate), subtype="PCM_16")
    st.clipboard = fn

    out = np.concatenate([audio[:s], audio[e:]], axis=0).astype(np.float32, copy=False)
    if out.size == 0:
        out = np.zeros(1, dtype=np.float32)
    _commit_new_audio(st, out, op="cut", params={"start": int(s), "end": int(e), "samples": int(seg.size)})
    return {}


def _op_paste(st: SessionState, *, cursor: int, start: int | None, end: int | None) -> dict[str, Any]:
    audio, sr = read_wav_mono(st.cur_path)
    if int(sr) != int(st.sample_rate):
        st.sample_rate = int(sr)

    cbp = _clipboard_path(st)
    if cbp is None or not cbp.exists():
        raise ValueError("Nothing to paste (clipboard empty)")
    clip, clip_sr = read_wav_mono(cbp)
    if int(clip_sr) != int(st.sample_rate):
        # Clipboard should match session SR; if not, treat as error to keep MVP simple.
        raise ValueError("Clipboard sample rate mismatch")

    n = int(audio.size)
    cur = max(0, min(int(cursor), n))

    if start is not None and end is not None:
        s, e = _safe_slice(audio, int(start), int(end))
        out = np.concatenate([audio[:s], clip, audio[e:]], axis=0).astype(np.float32, copy=False)
        _commit_new_audio(
            st,
            out,
            op="paste",
            params={"cursor": int(cur), "replace": True, "start": int(s), "end": int(e), "clip_samples": int(clip.size)},
        )
        return {}

    out = np.concatenate([audio[:cur], clip, audio[cur:]], axis=0).astype(np.float32, copy=False)
    _commit_new_audio(
        st,
        out,
        op="paste",
        params={"cursor": int(cur), "replace": False, "clip_samples": int(clip.size)},
    )
    return {}


def _op_insert_silence(st: SessionState, *, cursor: int, ms: float) -> dict[str, Any]:
    audio, sr = read_wav_mono(st.cur_path)
    if int(sr) != int(st.sample_rate):
        st.sample_rate = int(sr)

    n = int(audio.size)
    cur = max(0, min(int(cursor), n))
    sil_n = int(round(float(ms) * float(st.sample_rate) / 1000.0))
    sil_n = max(1, sil_n)
    zeros = np.zeros(sil_n, dtype=np.float32)
    out = np.concatenate([audio[:cur], zeros, audio[cur:]], axis=0).astype(np.float32, copy=False)
    _commit_new_audio(st, out, op="silence_insert", params={"cursor": int(cur), "ms": float(ms), "samples": int(sil_n)})
    return {}


def _op_trim(st: SessionState, *, start: int, end: int) -> dict[str, Any]:
    audio, sr = read_wav_mono(st.cur_path)
    if int(sr) != int(st.sample_rate):
        st.sample_rate = int(sr)

    s, e = _safe_slice(audio, start, end)
    out = audio[s:e].copy()
    if out.size == 0:
        out = np.zeros(1, dtype=np.float32)

    _commit_new_audio(st, out, op="trim", params={"start": int(s), "end": int(e)})
    return {}


def _op_reverse(st: SessionState, *, start: int | None, end: int | None) -> dict[str, Any]:
    audio, _sr = read_wav_mono(st.cur_path)
    if start is None or end is None:
        out = audio[::-1].copy()
        _commit_new_audio(st, out, op="reverse", params={"range": None})
        return {}

    s, e = _safe_slice(audio, int(start), int(end))
    out = audio.copy()
    out[s:e] = out[s:e][::-1]
    _commit_new_audio(st, out, op="reverse", params={"start": int(s), "end": int(e)})
    return {}


def _op_fade(st: SessionState, *, mode: str, ms: float, start: int | None, end: int | None) -> dict[str, Any]:
    audio, _sr = read_wav_mono(st.cur_path)
    if start is None or end is None:
        out = _fade(audio, sr=int(st.sample_rate), mode=str(mode), ms=float(ms))
        _commit_new_audio(st, out, op="fade", params={"mode": str(mode), "ms": float(ms), "range": None})
        return {}

    s, e = _safe_slice(audio, int(start), int(end))
    seg = audio[s:e]
    seg2 = _fade(seg, sr=int(st.sample_rate), mode=str(mode), ms=float(ms))
    out = audio.copy()
    out[s:e] = seg2
    _commit_new_audio(st, out, op="fade", params={"mode": str(mode), "ms": float(ms), "start": int(s), "end": int(e)})
    return {}


def _op_normalize(st: SessionState, *, peak_db: float, start: int | None, end: int | None) -> dict[str, Any]:
    audio, _sr = read_wav_mono(st.cur_path)
    if start is None or end is None:
        out = _normalize_peak(audio, peak_db=float(peak_db))
        _commit_new_audio(st, out, op="normalize", params={"peak_db": float(peak_db), "range": None})
        return {}

    s, e = _safe_slice(audio, int(start), int(end))
    seg = audio[s:e]
    seg2 = _normalize_peak(seg, peak_db=float(peak_db))
    out = audio.copy()
    out[s:e] = seg2
    _commit_new_audio(st, out, op="normalize", params={"peak_db": float(peak_db), "start": int(s), "end": int(e)})
    return {}


def _op_pitch(st: SessionState, *, semitones: float, start: int | None, end: int | None) -> dict[str, Any]:
    from .editor.launch import _fit_to_length, _pitch_shift_preserve_duration

    audio, _sr = read_wav_mono(st.cur_path)

    if start is None or end is None:
        y = _pitch_shift_preserve_duration(audio, sr=int(st.sample_rate), semitones=float(semitones))
        y = _fit_to_length(y, int(audio.size))
        _commit_new_audio(st, y, op="pitch", params={"semitones": float(semitones), "range": None})
        return {}

    s, e = _safe_slice(audio, int(start), int(end))
    seg = audio[s:e].copy()
    y = _pitch_shift_preserve_duration(seg, sr=int(st.sample_rate), semitones=float(semitones))
    y = _fit_to_length(y, int(seg.size))

    out = audio.copy()
    out[s:e] = _clip_mono(y)
    _commit_new_audio(st, out, op="pitch", params={"semitones": float(semitones), "start": int(s), "end": int(e)})
    return {}


def _op_eq3(
    st: SessionState,
    *,
    low_cut_hz: float,
    mid_freq_hz: float,
    mid_gain_db: float,
    mid_q: float,
    high_cut_hz: float,
    start: int | None,
    end: int | None,
) -> dict[str, Any]:
    from .editor.launch import _eq_three_band, _fit_to_length

    audio, _sr = read_wav_mono(st.cur_path)

    def apply(seg: np.ndarray) -> np.ndarray:
        y = _eq_three_band(
            seg,
            sr=int(st.sample_rate),
            low_cut_hz=float(low_cut_hz),
            mid_freq_hz=float(mid_freq_hz),
            mid_gain_db=float(mid_gain_db),
            mid_q=float(mid_q),
            high_cut_hz=float(high_cut_hz),
        )
        return _fit_to_length(_clip_mono(y), int(seg.size))

    params = {
        "low_cut_hz": float(low_cut_hz),
        "mid_freq_hz": float(mid_freq_hz),
        "mid_gain_db": float(mid_gain_db),
        "mid_q": float(mid_q),
        "high_cut_hz": float(high_cut_hz),
    }

    if start is None or end is None:
        out = apply(audio)
        _commit_new_audio(st, out, op="eq3", params={**params, "range": None})
        return {}

    s, e = _safe_slice(audio, int(start), int(end))
    seg = audio[s:e].copy()
    y = apply(seg)

    out = audio.copy()
    out[s:e] = y
    _commit_new_audio(st, out, op="eq3", params={**params, "start": int(s), "end": int(e)})
    return {}


def _op_undo(st: SessionState) -> dict[str, Any]:
    if not st.undo:
        return {}
    prev = st.undo.pop()
    st.redo.append(st.cur)
    st.cur = prev
    _push_history(st, "undo", {})
    _save_state(st)
    return {}


def _op_redo(st: SessionState) -> dict[str, Any]:
    if not st.redo:
        return {}
    nxt = st.redo.pop()
    st.undo.append(st.cur)
    st.cur = nxt
    _push_history(st, "redo", {})
    _save_state(st)
    return {}


def _op_export(st: SessionState, *, out_wav: Path) -> dict[str, Any]:
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(st.cur_path, out_wav)
    _push_history(st, "export", {"out": str(out_wav)})
    _save_state(st)
    return {"out": str(out_wav)}


def _op_close(st: SessionState) -> dict[str, Any]:
    try:
        shutil.rmtree(st.session_dir, ignore_errors=True)
    except Exception:
        pass
    return {}


def _resp(st: SessionState, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    audio, _sr = read_wav_mono(st.cur_path)
    cbp = _clipboard_path(st)
    out: dict[str, Any] = {
        "session_id": st.session_id,
        "sample_rate": int(st.sample_rate),
        "length_samples": int(audio.size),
        "duration_s": float(audio.size / float(st.sample_rate)) if st.sample_rate else 0.0,
        "current_wav": str(st.cur_path),
        "has_clipboard": bool(cbp and cbp.exists()),
        "can_undo": bool(st.undo),
        "can_redo": bool(st.redo),
        "history": list(st.history),
    }
    if extra:
        out.update(extra)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soundgen editops", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Start an edit session")
    p_init.add_argument("--in", dest="in_path", required=True)

    p_info = sub.add_parser("info", help="Get current session info")
    p_info.add_argument("--session", required=True)

    p_op = sub.add_parser("op", help="Apply an operation")
    p_op.add_argument("--session", required=True)
    p_op.add_argument(
        "--type",
        required=True,
        choices=["trim", "reverse", "fade", "normalize", "pitch", "eq3", "copy", "cut", "paste", "silence_insert"],
    )
    p_op.add_argument("--start", type=int)
    p_op.add_argument("--end", type=int)
    p_op.add_argument("--cursor", type=int)

    p_op.add_argument("--silence-ms", type=float, default=250.0)

    p_op.add_argument("--fade-mode", choices=["in", "out"], default="in")
    p_op.add_argument("--fade-ms", type=float, default=30.0)

    p_op.add_argument("--normalize-peak-db", type=float, default=-1.0)

    p_op.add_argument("--pitch-semitones", type=float, default=0.0)

    p_op.add_argument("--eq-low-cut-hz", type=float, default=80.0)
    p_op.add_argument("--eq-mid-freq-hz", type=float, default=1200.0)
    p_op.add_argument("--eq-mid-gain-db", type=float, default=0.0)
    p_op.add_argument("--eq-mid-q", type=float, default=1.0)
    p_op.add_argument("--eq-high-cut-hz", type=float, default=16000.0)

    p_undo = sub.add_parser("undo", help="Undo")
    p_undo.add_argument("--session", required=True)

    p_redo = sub.add_parser("redo", help="Redo")
    p_redo.add_argument("--session", required=True)

    p_export = sub.add_parser("export", help="Export WAV")
    p_export.add_argument("--session", required=True)
    p_export.add_argument("--out", required=True)

    p_close = sub.add_parser("close", help="Close session and delete temp files")
    p_close.add_argument("--session", required=True)

    args = parser.parse_args([] if argv is None else argv)

    base = configure_runtime()

    if args.cmd == "init":
        in_path = Path(args.in_path)
        if not in_path.exists():
            raise SystemExit(f"Input not found: {in_path}")

        session_id = uuid.uuid4().hex
        sess_dir = _session_dir(base, session_id)
        sess_dir.mkdir(parents=True, exist_ok=True)

        audio, sr = read_wav_mono(in_path)
        cur = "state_0000.wav"
        write_wav(sess_dir / cur, _clip_mono(audio), int(sr), subtype="PCM_16")

        st = SessionState(
            session_id=session_id,
            session_dir=sess_dir,
            sample_rate=int(sr),
            cur=cur,
            clipboard=None,
            undo=[],
            redo=[],
            history=[],
        )
        _push_history(st, "init", {"in": str(in_path)})
        _save_state(st)
        print(json.dumps(_resp(st), ensure_ascii=False))
        return 0

    if args.cmd == "info":
        st = _load_state(base, str(args.session))
        print(json.dumps(_resp(st), ensure_ascii=False))
        return 0

    if args.cmd == "op":
        st = _load_state(base, str(args.session))
        t = str(args.type)
        start = args.start
        end = args.end
        cursor = args.cursor

        if t == "trim":
            if start is None or end is None:
                raise SystemExit("trim requires --start and --end")
            _op_trim(st, start=int(start), end=int(end))
        elif t == "reverse":
            _op_reverse(st, start=start, end=end)
        elif t == "fade":
            _op_fade(st, mode=str(args.fade_mode), ms=float(args.fade_ms), start=start, end=end)
        elif t == "normalize":
            _op_normalize(st, peak_db=float(args.normalize_peak_db), start=start, end=end)
        elif t == "pitch":
            _op_pitch(st, semitones=float(args.pitch_semitones), start=start, end=end)
        elif t == "eq3":
            _op_eq3(
                st,
                low_cut_hz=float(args.eq_low_cut_hz),
                mid_freq_hz=float(args.eq_mid_freq_hz),
                mid_gain_db=float(args.eq_mid_gain_db),
                mid_q=float(args.eq_mid_q),
                high_cut_hz=float(args.eq_high_cut_hz),
                start=start,
                end=end,
            )
        elif t == "copy":
            if start is None or end is None:
                raise SystemExit("copy requires --start and --end")
            _op_copy(st, start=int(start), end=int(end))
        elif t == "cut":
            if start is None or end is None:
                raise SystemExit("cut requires --start and --end")
            _op_cut(st, start=int(start), end=int(end))
        elif t == "paste":
            if cursor is None:
                raise SystemExit("paste requires --cursor")
            _op_paste(st, cursor=int(cursor), start=start, end=end)
        elif t == "silence_insert":
            if cursor is None:
                raise SystemExit("silence_insert requires --cursor")
            _op_insert_silence(st, cursor=int(cursor), ms=float(args.silence_ms))
        else:
            raise SystemExit(f"Unknown op type: {t}")

        print(json.dumps(_resp(st), ensure_ascii=False))
        return 0

    if args.cmd == "undo":
        st = _load_state(base, str(args.session))
        _op_undo(st)
        print(json.dumps(_resp(st), ensure_ascii=False))
        return 0

    if args.cmd == "redo":
        st = _load_state(base, str(args.session))
        _op_redo(st)
        print(json.dumps(_resp(st), ensure_ascii=False))
        return 0

    if args.cmd == "export":
        st = _load_state(base, str(args.session))
        _op_export(st, out_wav=Path(args.out))
        print(json.dumps(_resp(st), ensure_ascii=False))
        return 0

    if args.cmd == "close":
        st = _load_state(base, str(args.session))
        _op_close(st)
        print(json.dumps({"ok": True, "session_id": st.session_id}, ensure_ascii=False))
        return 0

    raise SystemExit("unreachable")
