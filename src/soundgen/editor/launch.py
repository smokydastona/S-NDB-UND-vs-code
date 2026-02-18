from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


from ..io_utils import read_wav_mono, write_wav


def _clip_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _lin_to_db(x: float) -> float:
    x = max(1e-12, float(x))
    return float(20.0 * math.log10(x))


def _downsample_for_display(x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, y) for plotting with at most max_points samples.

    Uses min/max binning per chunk to preserve peaks.
    """

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    max_points = max(256, int(max_points))
    if n <= max_points:
        t = np.arange(n, dtype=np.float32)
        return t, x

    bins = max_points // 2
    step = int(math.ceil(n / float(bins)))
    mins: list[float] = []
    maxs: list[float] = []
    ts: list[int] = []
    for i in range(0, n, step):
        chunk = x[i : i + step]
        if chunk.size == 0:
            continue
        mins.append(float(np.min(chunk)))
        maxs.append(float(np.max(chunk)))
        ts.append(i)
    # Interleave min/max to draw vertical ranges.
    y = np.empty(len(mins) * 2, dtype=np.float32)
    y[0::2] = np.asarray(mins, dtype=np.float32)
    y[1::2] = np.asarray(maxs, dtype=np.float32)
    t = np.repeat(np.asarray(ts, dtype=np.float32), 2)
    return t, y


def _nearest_zero_crossing(x: np.ndarray, idx: int, *, radius: int) -> int:
    idx = int(idx)
    radius = max(0, int(radius))
    n = int(x.size)
    if n <= 1:
        return max(0, min(idx, n))

    lo = max(1, idx - radius)
    hi = min(n - 1, idx + radius)
    if lo >= hi:
        return max(0, min(idx, n - 1))

    best = idx
    best_abs = float("inf")
    for i in range(lo, hi + 1):
        a = float(x[i - 1])
        b = float(x[i])
        if (a <= 0.0 <= b) or (b <= 0.0 <= a):
            return i
        ab = abs(b)
        if ab < best_abs:
            best_abs = ab
            best = i
    return best


def _fade_in(x: np.ndarray) -> np.ndarray:
    n = int(x.size)
    if n <= 1:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return (x.astype(np.float32) * ramp).astype(np.float32)


def _fade_out(x: np.ndarray) -> np.ndarray:
    n = int(x.size)
    if n <= 1:
        return x
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    return (x.astype(np.float32) * ramp).astype(np.float32)


def _normalize_peak(x: np.ndarray, *, target_db: float = -1.0) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(x)) + 1e-12)
    peak_db = _lin_to_db(peak)
    target_lin = _db_to_lin(float(target_db))
    gain = float(target_lin / peak)
    return _clip_mono(x * gain), gain


@dataclass
class _State:
    audio: np.ndarray
    sample_rate: int
    cursor: int = 0
    selection: tuple[int, int] | None = None
    clipboard: np.ndarray | None = None
    loop: tuple[int, int] | None = None
    markers: list[dict[str, Any]] | None = None
    edits: list[dict[str, Any]] | None = None
    export_count: int = 0
    view_mode: str = "wave"  # wave|spec
    spec_fft: int = 1024
    spec_fmax_hz: float | None = None

    def clone(self) -> "_State":
        return _State(
            audio=self.audio.copy(),
            sample_rate=int(self.sample_rate),
            cursor=int(self.cursor),
            selection=None if self.selection is None else (int(self.selection[0]), int(self.selection[1])),
            clipboard=None if self.clipboard is None else self.clipboard.copy(),
            loop=None if self.loop is None else (int(self.loop[0]), int(self.loop[1])),
            markers=[] if not self.markers else [dict(m) for m in self.markers],
            edits=[] if not self.edits else [dict(e) for e in self.edits],
            export_count=int(self.export_count),
            view_mode=str(self.view_mode),
            spec_fft=int(self.spec_fft),
            spec_fmax_hz=(None if self.spec_fmax_hz is None else float(self.spec_fmax_hz)),
        )


class _Undo:
    def __init__(self, *, limit: int = 50):
        self._undo: list[_State] = []
        self._redo: list[_State] = []
        self._limit = int(limit)

    def push(self, st: _State) -> None:
        self._undo.append(st.clone())
        if len(self._undo) > self._limit:
            self._undo = self._undo[-self._limit :]
        self._redo.clear()

    def undo(self, cur: _State) -> _State | None:
        if not self._undo:
            return None
        self._redo.append(cur.clone())
        return self._undo.pop()

    def redo(self, cur: _State) -> _State | None:
        if not self._redo:
            return None
        self._undo.append(cur.clone())
        return self._redo.pop()


class _Player:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._playing = False
        self._started_at = 0.0
        self._started_cursor = 0
        self._loop = False

    @property
    def playing(self) -> bool:
        return bool(self._playing)

    def stop(self) -> None:
        self._stop.set()
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_ASYNC)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.25)
        self._thread = None
        self._playing = False

    def pause_cursor(self, *, sr: int) -> int | None:
        if not self._playing:
            return None
        elapsed = max(0.0, time.time() - float(self._started_at))
        return int(self._started_cursor + elapsed * float(sr))

    def play_segment(self, *, audio: np.ndarray, sr: int, start: int, end: int, loop: bool) -> None:
        self.stop()
        self._stop.clear()
        self._loop = bool(loop)

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        sr = int(sr)
        start = max(0, min(int(start), int(audio.size)))
        end = max(start, min(int(end), int(audio.size)))
        seg = audio[start:end].copy()
        if seg.size == 0:
            return

        self._playing = True
        self._started_at = time.time()
        self._started_cursor = int(start)

        def _run() -> None:
            try:
                if sys.platform == "win32":
                    import winsound

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        tmp_path = Path(f.name)
                    try:
                        write_wav(tmp_path, _clip_mono(seg), sr)
                        if self._loop:
                            while not self._stop.is_set():
                                winsound.PlaySound(str(tmp_path), winsound.SND_FILENAME)
                        else:
                            winsound.PlaySound(str(tmp_path), winsound.SND_FILENAME)
                    finally:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    # Non-Windows: playback is best-effort only.
                    return
            finally:
                self._playing = False

        self._thread = threading.Thread(target=_run, name="soundgen-editor-playback", daemon=True)
        self._thread.start()


def _write_edits_sidecar(wav_path: Path, *, state: _State, mode: str) -> None:
    sidecar = wav_path.with_suffix("")
    sidecar = sidecar.with_name(sidecar.name + ".edits.json")

    rec: dict[str, Any] = {
        "edited_wav": str(wav_path),
        "sample_rate": int(state.sample_rate),
        "mode": str(mode),
        "cursor_s": float(state.cursor) / float(state.sample_rate),
        "selection_s": None,
        "loop_s": None,
        "markers": [],
        "edits": list(state.edits or []),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if state.selection is not None:
        a, b = state.selection
        rec["selection_s"] = [float(a) / float(state.sample_rate), float(b) / float(state.sample_rate)]
    if state.loop is not None:
        a, b = state.loop
        rec["loop_s"] = [float(a) / float(state.sample_rate), float(b) / float(state.sample_rate)]
    if state.markers:
        rec["markers"] = [
            {
                "kind": str(m.get("kind") or "marker"),
                "time_s": float(m.get("sample", 0)) / float(state.sample_rate),
            }
            for m in state.markers
        ]

    sidecar.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _help_text() -> str:
    return (
        "Built-in editor keybinds:\n"
        "  mouse drag: select region\n"
        "  click: set cursor\n"
        "  space: play/pause from cursor\n"
        "  p: play selection\n"
        "  l: loop selection (play selection repeatedly)\n"
        "  s: stop\n"
        "\n"
        "  S: toggle spectrogram view\n"
        "  F: cycle spectrogram FFT size\n"
        "  Z: cycle spectrogram frequency zoom\n"
        "  T: auto-mark transients\n"
        "  J: snap selection to nearest transients\n"
        "  A: auto loop points + crossfaded loop segment\n"
        "\n"
        "  c: copy selection\n"
        "  x: cut selection\n"
        "  v: paste at cursor\n"
        "  delete/backspace: delete selection\n"
        "  t: trim to selection (leave only selection)\n"
        "\n"
        "  f: fade in (selection)\n"
        "  g: fade out (selection)\n"
        "  k: crossfade-delete selection (smooth splice)\n"
        "  n: normalize peak to -1 dBFS (whole file)\n"
        "  +/-: gain ±1 dB (selection if present else whole file)\n"
        "\n"
        "  m: add marker at cursor\n"
        "  M: add transient marker at cursor\n"
        "  G: add good-take marker at cursor\n"
        "  [: set loop start (selection start if present else cursor)\n"
        "  ]: set loop end (selection end if present else cursor)\n"
        "  0: snap selection bounds to zero crossings\n"
        "\n"
        "  u: undo\n"
        "  r: redo\n"
        "\n"
        "  e: export overwrite\n"
        "  E: export as variation\n"
        "  h: print this help\n"
    )


def launch_editor(wav_path: str | Path) -> None:
    """Launch the built-in destructive editor.

    Implementation notes:
    - UI: matplotlib (waveform + selection)
    - Playback: Windows winsound (best-effort)
    - Audio format: mono float32 in [-1, 1]
    """

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(str(wav_path))

    audio, sr = read_wav_mono(wav_path)
    st = _State(audio=_clip_mono(audio), sample_rate=int(sr), cursor=0, selection=None, markers=[], edits=[])
    undo = _Undo(limit=60)
    player = _Player()

    import matplotlib

    # Ensure we have an interactive backend; default is fine on Windows.
    matplotlib.use(matplotlib.get_backend())
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    fig, (ax_wav, ax_spec) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(11, 6),
        gridspec_kw={"height_ratios": [2.1, 1.2]},
    )
    fig.canvas.manager.set_window_title(f"S-NDB-UND Editor — {wav_path.name}")
    fig.tight_layout()

    def _clear_artist(artist: Any) -> None:
        try:
            if artist is None:
                return
            artist.remove()
        except Exception:
            return

    def _render() -> None:
        ax_wav.cla()
        ax_wav.set_title(wav_path.name)
        ax_wav.set_ylabel("Amplitude")
        ax_wav.set_ylim(-1.05, 1.05)

        ax_spec.cla()
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Hz")

        max_points = 4000
        t, y = _downsample_for_display(st.audio, max_points=max_points)
        tt = t / float(st.sample_rate)
        ax_wav.plot(tt, y, linewidth=0.8)

        cur_s = float(st.cursor) / float(st.sample_rate)
        ax_wav.axvline(cur_s, linewidth=1)
        ax_spec.axvline(cur_s, linewidth=1)

        # Selection
        if st.selection is not None:
            a, b = st.selection
            a_s = float(a) / float(st.sample_rate)
            b_s = float(b) / float(st.sample_rate)
            ax_wav.axvspan(min(a_s, b_s), max(a_s, b_s), alpha=0.18)
            ax_spec.axvspan(min(a_s, b_s), max(a_s, b_s), alpha=0.18)

        # Loop
        if st.loop is not None:
            a, b = st.loop
            a_s = float(a) / float(st.sample_rate)
            b_s = float(b) / float(st.sample_rate)
            ax_wav.axvline(a_s, linewidth=1)
            ax_wav.axvline(b_s, linewidth=1)
            ax_spec.axvline(a_s, linewidth=1)
            ax_spec.axvline(b_s, linewidth=1)

        # Markers
        if st.markers:
            for m in st.markers:
                ms = int(m.get("sample") or 0)
                kind = str(m.get("kind") or "marker").strip().lower()
                m_s = float(ms) / float(st.sample_rate)
                ax_wav.axvline(m_s, linewidth=0.9, alpha=0.9)
                ax_spec.axvline(m_s, linewidth=0.9, alpha=0.9)

        # Spectrogram
        show_spec = str(st.view_mode).strip().lower() == "spec"
        ax_spec.set_visible(show_spec)
        if show_spec:
            nfft = int(st.spec_fft)
            nfft = max(256, min(nfft, 8192))
            noverlap = int(nfft // 2)
            ax_spec.specgram(
                st.audio,
                NFFT=nfft,
                Fs=float(st.sample_rate),
                noverlap=noverlap,
                scale="dB",
                cmap=None,
            )
            fmax = st.spec_fmax_hz
            if fmax is not None:
                ax_spec.set_ylim(0.0, float(max(100.0, min(float(fmax), float(st.sample_rate) / 2.0))))
        fig.canvas.draw_idle()

    def _push_edit(kind: str, extra: dict[str, Any] | None = None) -> None:
        rec: dict[str, Any] = {
            "kind": str(kind),
            "cursor_s": float(st.cursor) / float(st.sample_rate),
            "selection_s": None,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if st.selection is not None:
            a, b = st.selection
            rec["selection_s"] = [float(a) / float(st.sample_rate), float(b) / float(st.sample_rate)]
        if extra:
            rec.update(extra)
        (st.edits or []).append(rec)

    def _sel_range() -> tuple[int, int] | None:
        if st.selection is None:
            return None
        a, b = st.selection
        lo = max(0, min(int(a), int(b)))
        hi = max(0, max(int(a), int(b)))
        if hi <= lo:
            return None
        return lo, min(hi, int(st.audio.size))

    def _set_cursor(sample_idx: int) -> None:
        st.cursor = max(0, min(int(sample_idx), int(st.audio.size)))

    def _on_select(xmin: float, xmax: float) -> None:
        a = int(round(float(xmin) * float(st.sample_rate)))
        b = int(round(float(xmax) * float(st.sample_rate)))
        a = max(0, min(a, int(st.audio.size)))
        b = max(0, min(b, int(st.audio.size)))
        st.selection = (a, b)
        _render()

    span = SpanSelector(
        ax_wav,
        _on_select,
        "horizontal",
        useblit=True,
        interactive=True,
        props=dict(alpha=0.2),
    )

    def _on_click(ev: Any) -> None:
        if ev.inaxes not in {ax_wav, ax_spec}:
            return
        if ev.xdata is None:
            return
        _set_cursor(int(round(float(ev.xdata) * float(st.sample_rate))))
        _render()

    def _apply_gain(db: float) -> None:
        rng = _sel_range()
        undo.push(st)
        g = _db_to_lin(float(db))
        if rng is None:
            st.audio = _clip_mono(st.audio * g)
        else:
            lo, hi = rng
            out = st.audio.copy()
            out[lo:hi] = _clip_mono(out[lo:hi] * g)
            st.audio = out
        _push_edit("gain", {"db": float(db)})
        _render()

    def _do_copy() -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        st.clipboard = st.audio[lo:hi].copy()
        _push_edit("copy", {"samples": int(hi - lo)})

    def _do_delete(cut: bool) -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        undo.push(st)
        if cut:
            st.clipboard = st.audio[lo:hi].copy()
        st.audio = np.concatenate([st.audio[:lo], st.audio[hi:]], axis=0)
        st.selection = None
        _set_cursor(lo)
        _push_edit("cut" if cut else "delete", {"samples": int(hi - lo)})
        _render()

    def _do_crossfade_delete() -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        left = st.audio[:lo]
        right = st.audio[hi:]
        if left.size == 0 or right.size == 0:
            _do_delete(cut=False)
            return

        undo.push(st)
        # Use a small fixed crossfade (40ms) to smooth the splice.
        cf = int(0.040 * float(st.sample_rate))
        cf = max(16, cf)
        cf = min(cf, int(left.size), int(right.size))

        if cf > 0:
            t = np.linspace(0.0, 1.0, cf, endpoint=False, dtype=np.float32)
            fade_out = np.cos(0.5 * math.pi * t).astype(np.float32)
            fade_in = np.sin(0.5 * math.pi * t).astype(np.float32)
            xfade = (left[-cf:] * fade_out + right[:cf] * fade_in).astype(np.float32)
            out = np.concatenate([left[:-cf], xfade, right[cf:]], axis=0)
        else:
            out = np.concatenate([left, right], axis=0)

        st.audio = _clip_mono(out)
        st.selection = None
        _set_cursor(max(0, min(lo, int(st.audio.size))))
        _push_edit("crossfade_delete", {"removed_samples": int(hi - lo), "crossfade_ms": 40.0})
        _render()

    def _do_paste() -> None:
        if st.clipboard is None or st.clipboard.size == 0:
            return
        undo.push(st)
        cur = int(st.cursor)
        st.audio = np.concatenate([st.audio[:cur], st.clipboard, st.audio[cur:]], axis=0)
        _set_cursor(cur + int(st.clipboard.size))
        st.selection = None
        _push_edit("paste", {"samples": int(st.clipboard.size)})
        _render()

    def _do_trim_to_selection() -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        undo.push(st)
        st.audio = st.audio[lo:hi].copy()
        st.selection = None
        _set_cursor(0)
        _push_edit("leave", {"samples": int(hi - lo)})
        _render()

    def _do_fade(kind: str) -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        undo.push(st)
        out = st.audio.copy()
        seg = out[lo:hi]
        out[lo:hi] = _fade_in(seg) if kind == "in" else _fade_out(seg)
        st.audio = _clip_mono(out)
        _push_edit("fade_in" if kind == "in" else "fade_out", {"samples": int(hi - lo)})
        _render()

    def _do_normalize() -> None:
        undo.push(st)
        y, gain = _normalize_peak(st.audio, target_db=-1.0)
        st.audio = y
        _push_edit("normalize_peak", {"target_db": -1.0, "gain": float(gain)})
        _render()

    def _do_marker(kind: str) -> None:
        kind2 = str(kind or "marker").strip().lower()
        (st.markers or []).append({"kind": kind2, "sample": int(st.cursor)})
        (st.markers or []).sort(key=lambda m: int(m.get("sample") or 0))
        _push_edit("marker", {"kind": kind2, "sample": int(st.cursor)})
        _render()

    def _snap_selection_to_zero() -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        rad = int(0.005 * float(st.sample_rate))
        lo2 = _nearest_zero_crossing(st.audio, lo, radius=rad)
        hi2 = _nearest_zero_crossing(st.audio, hi, radius=rad)
        st.selection = (int(lo2), int(hi2))
        _push_edit("snap_zero", {"radius_ms": 5.0})
        _render()

    def _set_loop(which: str) -> None:
        a = int(st.cursor)
        b = int(st.cursor)
        if st.selection is not None:
            lohi = _sel_range()
            if lohi is not None:
                a, b = lohi
        rad = int(0.005 * float(st.sample_rate))
        a = _nearest_zero_crossing(st.audio, a, radius=rad)
        b = _nearest_zero_crossing(st.audio, b, radius=rad)

        if st.loop is None:
            st.loop = (a, b)
        la, lb = st.loop
        if which == "start":
            la = a
        else:
            lb = b
        la = max(0, min(la, int(st.audio.size)))
        lb = max(0, min(lb, int(st.audio.size)))
        if lb < la:
            la, lb = lb, la
        st.loop = (la, lb)
        _push_edit("loop_point", {"loop": [float(la) / sr, float(lb) / sr], "snapped_zero": True})
        _render()

    def _export(overwrite: bool) -> None:
        undo.push(st)
        out_path = wav_path
        mode = "overwrite"
        if not overwrite:
            st.export_count += 1
            out_path = wav_path.with_name(f"{wav_path.stem}_edit{st.export_count}{wav_path.suffix}")
            mode = "variation"
        write_wav(out_path, _clip_mono(st.audio), int(st.sample_rate))
        _write_edits_sidecar(out_path, state=st, mode=mode)
        _push_edit("export", {"mode": mode, "path": str(out_path)})
        print(f"Exported: {out_path}")

    def _play_from_cursor(loop: bool) -> None:
        if sys.platform != "win32":
            print("Playback is only implemented on Windows in this editor.")
            return
        player.play_segment(audio=st.audio, sr=st.sample_rate, start=int(st.cursor), end=int(st.audio.size), loop=loop)

    def _play_selection(loop: bool) -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        if sys.platform != "win32":
            print("Playback is only implemented on Windows in this editor.")
            return
        player.play_segment(audio=st.audio, sr=st.sample_rate, start=lo, end=hi, loop=loop)

    def _toggle_spec() -> None:
        st.view_mode = "spec" if str(st.view_mode).strip().lower() != "spec" else "wave"
        _push_edit("view", {"mode": st.view_mode})
        _render()

    def _cycle_fft() -> None:
        sizes = [256, 512, 1024, 2048, 4096]
        cur = int(st.spec_fft)
        if cur not in sizes:
            st.spec_fft = 1024
        else:
            st.spec_fft = sizes[(sizes.index(cur) + 1) % len(sizes)]
        _push_edit("spec_fft", {"nfft": int(st.spec_fft)})
        _render()

    def _cycle_fzoom() -> None:
        nyq = float(st.sample_rate) / 2.0
        opts: list[float | None] = [2000.0, 8000.0, None]
        cur = st.spec_fmax_hz
        idx = 0
        for i, o in enumerate(opts):
            if (o is None and cur is None) or (o is not None and cur is not None and abs(float(o) - float(cur)) < 1e-6):
                idx = i
                break
        nxt = opts[(idx + 1) % len(opts)]
        if nxt is not None:
            nxt = float(min(float(nxt), nyq))
        st.spec_fmax_hz = nxt
        _push_edit("spec_fzoom", {"fmax_hz": None if nxt is None else float(nxt)})
        _render()

    def _detect_transients() -> None:
        # Simple transient detector: energy envelope derivative peaks.
        x = st.audio.astype(np.float32, copy=False)
        sr2 = int(st.sample_rate)
        win = max(8, int(0.005 * sr2))  # 5ms abs smoothing
        absx = np.abs(x)
        kernel = np.ones(win, dtype=np.float32) / float(win)
        env = np.convolve(absx, kernel, mode="same")
        d = np.diff(env, prepend=env[:1])

        thr = float(np.mean(d) + 3.5 * (np.std(d) + 1e-9))
        min_gap = max(1, int(0.060 * sr2))

        peaks: list[int] = []
        last = -10**9
        for i in range(1, int(d.size) - 1):
            if d[i] > thr and d[i] >= d[i - 1] and d[i] >= d[i + 1]:
                if i - last >= min_gap:
                    peaks.append(i)
                    last = i

        # Remove old transient markers and add new ones.
        kept: list[dict[str, Any]] = []
        for m in (st.markers or []):
            if str(m.get("kind") or "").strip().lower() != "transient":
                kept.append(m)
        for p in peaks:
            kept.append({"kind": "transient", "sample": int(p)})
        kept.sort(key=lambda m: int(m.get("sample") or 0))
        st.markers = kept
        _push_edit("detect_transients", {"count": int(len(peaks)), "threshold": float(thr)})
        _render()

    def _snap_selection_to_transients() -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        trans = [int(m.get("sample") or 0) for m in (st.markers or []) if str(m.get("kind") or "").strip().lower() == "transient"]
        if not trans:
            return

        def _nearest(v: int) -> int:
            best = trans[0]
            best_d = abs(best - v)
            for t in trans[1:]:
                dd = abs(int(t) - v)
                if dd < best_d:
                    best_d = dd
                    best = int(t)
            return int(best)

        st.selection = (_nearest(lo), _nearest(hi))
        _push_edit("snap_transients", {})
        _render()

    def _auto_loop_crossfade() -> None:
        from ..loop_suite import apply_loop_crossfade, find_auto_loop_points

        undo.push(st)
        res = find_auto_loop_points(st.audio, st.sample_rate, min_loop_s=1.5, max_loop_s=10.0)
        seg = apply_loop_crossfade(
            st.audio,
            loop_start=res.loop_start,
            loop_end=res.loop_end,
            crossfade_ms=40.0,
            sr=st.sample_rate,
        )
        st.audio = _clip_mono(seg)
        st.cursor = 0
        st.selection = None
        st.loop = (0, int(st.audio.size))
        _push_edit(
            "auto_loop",
            {
                "score": float(res.score),
                "loop_len_s": float((res.loop_end - res.loop_start) / float(st.sample_rate)),
                "crossfade_ms": 40.0,
            },
        )
        _render()

    def _on_key(ev: Any) -> None:
        key = str(getattr(ev, "key", "") or "")
        if not key:
            return

        if key in {"h"}:
            print(_help_text())
            return

        if key in {"S"}:
            _toggle_spec()
            return

        if key in {"F"}:
            _cycle_fft()
            return

        if key in {"Z"}:
            _cycle_fzoom()
            return

        if key in {"T"}:
            _detect_transients()
            return

        if key in {"J"}:
            _snap_selection_to_transients()
            return

        if key in {"A"}:
            _auto_loop_crossfade()
            return

        if key in {"s"}:
            player.stop()
            return

        if key == "space":
            if player.playing:
                cur = player.pause_cursor(sr=st.sample_rate)
                player.stop()
                if cur is not None:
                    _set_cursor(cur)
                _render()
            else:
                _play_from_cursor(loop=False)
            return

        if key in {"p"}:
            _play_selection(loop=False)
            return

        if key in {"l"}:
            _play_selection(loop=True)
            return

        if key in {"c"}:
            _do_copy()
            return

        if key in {"x"}:
            _do_delete(cut=True)
            return

        if key in {"delete", "backspace"}:
            _do_delete(cut=False)
            return

        if key in {"v"}:
            _do_paste()
            return

        if key in {"t"}:
            _do_trim_to_selection()
            return

        if key in {"f"}:
            _do_fade("in")
            return

        if key in {"g"}:
            _do_fade("out")
            return

        if key in {"k"}:
            _do_crossfade_delete()
            return

        if key in {"n"}:
            _do_normalize()
            return

        if key in {"+", "="}:
            _apply_gain(+1.0)
            return

        if key in {"-", "_"}:
            _apply_gain(-1.0)
            return

        if key in {"u", "ctrl+z"}:
            prev = undo.undo(st)
            if prev is not None:
                player.stop()
                st.audio = prev.audio
                st.sample_rate = prev.sample_rate
                st.cursor = prev.cursor
                st.selection = prev.selection
                st.clipboard = prev.clipboard
                st.loop = prev.loop
                st.markers = prev.markers
                st.edits = prev.edits
                st.export_count = prev.export_count
                _render()
            return

        if key in {"r", "ctrl+y"}:
            nxt = undo.redo(st)
            if nxt is not None:
                player.stop()
                st.audio = nxt.audio
                st.sample_rate = nxt.sample_rate
                st.cursor = nxt.cursor
                st.selection = nxt.selection
                st.clipboard = nxt.clipboard
                st.loop = nxt.loop
                st.markers = nxt.markers
                st.edits = nxt.edits
                st.export_count = nxt.export_count
                _render()
            return

        if key == "m":
            _do_marker("marker")
            return

        if key == "M":
            _do_marker("transient")
            return

        if key == "G":
            _do_marker("good")
            return

        if key == "[":
            _set_loop("start")
            return

        if key == "]":
            _set_loop("end")
            return

        if key == "0":
            _snap_selection_to_zero()
            return

        if key == "e":
            _export(overwrite=True)
            return

        if key == "E":
            _export(overwrite=False)
            return

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    print(_help_text())
    _render()
    plt.show()

    # Ensure playback stops on exit.
    player.stop()
