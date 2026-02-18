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


def _cyclic_crossfade(x: np.ndarray, *, crossfade: int) -> np.ndarray:
    """Apply a cyclic crossfade so end->start loops more smoothly.

    This is intended for *preview* (loop audition) and small repair operations.
    """

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    crossfade = int(crossfade)
    if n <= 2 or crossfade <= 0:
        return x
    cf = max(1, min(crossfade, n // 2))

    head = x[:cf].copy()
    tail = x[-cf:].copy()

    # Equal-power-ish blend across the loop boundary.
    t = np.linspace(0.0, 1.0, cf, endpoint=False, dtype=np.float32)
    w_in = np.sin(0.5 * math.pi * t).astype(np.float32)
    w_out = np.cos(0.5 * math.pi * t).astype(np.float32)

    y = x.copy()
    y[:cf] = (tail * w_out + head * w_in).astype(np.float32)
    y[-cf:] = (tail * w_in + head * w_out).astype(np.float32)
    return y


def _repair_click_linear(x: np.ndarray, *, lo: int, hi: int) -> np.ndarray:
    """Replace [lo, hi) with a smooth bridge between endpoints."""

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    lo = max(0, min(int(lo), n))
    hi = max(lo, min(int(hi), n))
    if hi - lo <= 1 or n <= 1:
        return x

    left = float(x[lo - 1]) if lo - 1 >= 0 else float(x[hi])
    right = float(x[hi]) if hi < n else float(x[lo - 1])
    fill = np.linspace(left, right, hi - lo, endpoint=False, dtype=np.float32)

    y = x.copy()
    y[lo:hi] = fill
    return y


def _resample_linear(x: np.ndarray, *, rate: float) -> np.ndarray:
    """Resample using linear interpolation.

    rate > 1.0 makes the audio shorter and higher-pitched (playback-rate style).
    """

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    rate = float(rate)
    if n <= 1 or abs(rate - 1.0) < 1e-9:
        return x
    rate = max(0.05, min(rate, 20.0))

    new_n = int(max(1, round(float(n) / rate)))
    src_idx = np.arange(new_n, dtype=np.float32) * float(rate)
    src_idx = np.clip(src_idx, 0.0, float(n - 1))
    xp = np.arange(n, dtype=np.float32)
    y = np.interp(src_idx, xp, x).astype(np.float32)
    return y


def _apply_edge_fades(x: np.ndarray, *, fade_in: int, fade_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n <= 1:
        return x
    fi = max(0, min(int(fade_in), n))
    fo = max(0, min(int(fade_out), n))
    y = x.copy()
    if fi > 0:
        y[:fi] = _fade_in(y[:fi])
    if fo > 0:
        y[-fo:] = _fade_out(y[-fo:])
    return y


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
    regions: list[dict[str, Any]] | None = None
    active_region: int | None = None
    layers: list[dict[str, Any]] | None = None
    active_layer: int | None = None
    edits: list[dict[str, Any]] | None = None
    export_count: int = 0
    view_mode: str = "wave"  # wave|spec
    spec_fft: int = 1024
    spec_fmax_hz: float | None = None

    def clone(self) -> "_State":
        layers: list[dict[str, Any]] = []
        if self.layers:
            for l in self.layers:
                ll = dict(l)
                a = ll.get("audio")
                if isinstance(a, np.ndarray):
                    ll["audio"] = a.astype(np.float32, copy=True)
                layers.append(ll)
        return _State(
            audio=self.audio.copy(),
            sample_rate=int(self.sample_rate),
            cursor=int(self.cursor),
            selection=None if self.selection is None else (int(self.selection[0]), int(self.selection[1])),
            clipboard=None if self.clipboard is None else self.clipboard.copy(),
            loop=None if self.loop is None else (int(self.loop[0]), int(self.loop[1])),
            markers=[] if not self.markers else [dict(m) for m in self.markers],
            regions=[] if not self.regions else [dict(r) for r in self.regions],
            active_region=(None if self.active_region is None else int(self.active_region)),
            layers=layers,
            active_layer=(None if self.active_layer is None else int(self.active_layer)),
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
        "regions": [],
        "layers": [],
        "active_region": (int(state.active_region) if state.active_region is not None else None),
        "active_layer": (int(state.active_layer) if state.active_layer is not None else None),
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

    if state.regions:
        out: list[dict[str, Any]] = []
        for r in state.regions:
            try:
                start = int(r.get("start") or 0)
                end = int(r.get("end") or 0)
            except Exception:
                continue
            out.append(
                {
                    "name": str(r.get("name") or "region"),
                    "start_s": float(start) / float(state.sample_rate),
                    "end_s": float(end) / float(state.sample_rate),
                    "fx_chain": (str(r.get("fx_chain")) if r.get("fx_chain") is not None else None),
                    "fx_chain_json": (str(r.get("fx_chain_json")) if r.get("fx_chain_json") is not None else None),
                    "loop_s": (
                        [
                            float(int(r.get("loop_start") or 0)) / float(state.sample_rate),
                            float(int(r.get("loop_end") or 0)) / float(state.sample_rate),
                        ]
                        if (r.get("loop_start") is not None and r.get("loop_end") is not None)
                        else None
                    ),
                }
            )
        rec["regions"] = out

    if state.layers:
        out_layers: list[dict[str, Any]] = []
        for l in state.layers:
            ll = dict(l)
            a = ll.pop("audio", None)
            if isinstance(a, np.ndarray):
                ll["samples"] = int(a.size)
            out_layers.append(ll)
        rec["layers"] = out_layers

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
        "  R: add region from selection\n"
        "  , / .: previous / next region\n"
        "  X: select active region\n"
        "  N: rename active region\n"
        "  C: set active region FX chain (export hook)\n"
        "  {: set active region loop start\n"
        "  }: set active region loop end\n"
        "  D: delete active region\n"
        "  O: export all regions\n"
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
        "  K: de-click (repair selection, or small window at cursor)\n"
        "  n: normalize peak to -1 dBFS (whole file)\n"
        "  +/-: gain ±1 dB (selection if present else whole file)\n"
        "\n"
        "  i: import layer WAV at cursor (max 4 layers total)\n"
        "  y: cycle active layer\n"
        "  L: edit active layer params (gain/pan/fades/pitch)\n"
        "  Y: delete active layer\n"
        "  b: play mixed (layers) from cursor\n"
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
    st = _State(
        audio=_clip_mono(audio),
        sample_rate=int(sr),
        cursor=0,
        selection=None,
        markers=[],
        regions=[],
        active_region=None,
        layers=[],
        active_layer=None,
        edits=[],
    )
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

        # Regions
        if st.regions:
            for idx, r in enumerate(st.regions):
                try:
                    a = int(r.get("start") or 0)
                    b = int(r.get("end") or 0)
                except Exception:
                    continue
                a = max(0, min(a, int(st.audio.size)))
                b = max(0, min(b, int(st.audio.size)))
                if b <= a:
                    continue
                a_s = float(a) / float(st.sample_rate)
                b_s = float(b) / float(st.sample_rate)
                is_active = (st.active_region is not None) and (int(st.active_region) == int(idx))
                alpha = 0.16 if is_active else 0.08
                ax_wav.axvspan(a_s, b_s, alpha=alpha)
                ax_spec.axvspan(a_s, b_s, alpha=alpha)

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
        sample = int(round(float(ev.xdata) * float(st.sample_rate)))
        _set_cursor(sample)
        # Right-click (esp. in spectrogram view) targets a small de-click selection.
        # This gives quick "click removal targeting" without needing modifier keys.
        try:
            btn = int(getattr(ev, "button", 1) or 1)
        except Exception:
            btn = 1
        if btn == 3:
            half = int(0.006 * float(st.sample_rate))  # ~12ms window
            a = max(0, min(sample - half, int(st.audio.size)))
            b = max(0, min(sample + half, int(st.audio.size)))
            if b > a:
                st.selection = (a, b)
        _render()

    def _region_list() -> list[dict[str, Any]]:
        if st.regions is None:
            st.regions = []
        return st.regions

    def _layer_list() -> list[dict[str, Any]]:
        if st.layers is None:
            st.layers = []
        return st.layers

    def _set_active_layer(idx: int | None) -> None:
        if idx is None:
            st.active_layer = None
            return
        ll = _layer_list()
        if not ll:
            st.active_layer = None
            return
        st.active_layer = int(max(0, min(int(idx), len(ll) - 1)))

    def _active_layer_obj() -> dict[str, Any] | None:
        if st.active_layer is None:
            return None
        ll = _layer_list()
        idx = int(st.active_layer)
        if idx < 0 or idx >= len(ll):
            return None
        return ll[idx]

    def _layer_prev_next(step: int) -> None:
        ll = _layer_list()
        if not ll:
            st.active_layer = None
            return
        if st.active_layer is None:
            _set_active_layer(0)
        else:
            _set_active_layer((int(st.active_layer) + int(step)) % len(ll))
        _push_edit("layer_select", {"active": int(st.active_layer) if st.active_layer is not None else None})
        _render()

    def _import_layer() -> None:
        ll = _layer_list()
        # Max 4 layers total: base + up to 3 overlays.
        if len(ll) >= 3:
            print("Max overlay layers reached (3).")
            return
        try:
            p = input("Layer WAV path: ").strip().strip('"')
        except Exception:
            return
        if not p:
            return
        path = Path(p)
        if not path.exists():
            print(f"Not found: {path}")
            return
        x2, sr2 = read_wav_mono(path)
        if int(sr2) != int(st.sample_rate):
            print(f"Layer sample rate {sr2} != project {st.sample_rate}; import uses original samples (no resample).")
        undo.push(st)
        rec = {
            "name": str(path.stem),
            "path": str(path),
            "offset": int(st.cursor),
            "gain_db": 0.0,
            "pan": 0.0,
            "fade_in_ms": 0.0,
            "fade_out_ms": 0.0,
            "pitch_semitones": 0.0,
            "audio": x2.astype(np.float32, copy=True),
        }
        ll.append(rec)
        _set_active_layer(len(ll) - 1)
        _push_edit("layer_import", {"name": rec["name"], "offset": int(rec["offset"]), "samples": int(rec["audio"].size)})
        _render()

    def _delete_active_layer() -> None:
        ll = _layer_list()
        if not ll or st.active_layer is None:
            return
        idx = int(st.active_layer)
        if idx < 0 or idx >= len(ll):
            return
        undo.push(st)
        name = str(ll[idx].get("name") or "layer")
        del ll[idx]
        if not ll:
            st.active_layer = None
        else:
            st.active_layer = int(max(0, min(idx, len(ll) - 1)))
        _push_edit("layer_delete", {"name": name})
        _render()

    def _edit_active_layer_params() -> None:
        layer = _active_layer_obj()
        if layer is None:
            print("No active layer.")
            return
        try:
            print("Layer params (press enter to keep current):")
            print(f"  name={layer.get('name')}  offset={layer.get('offset')} samples")
            g = input(f"  gain_db [{layer.get('gain_db', 0.0)}]: ").strip()
            p = input(f"  pan [-1..1] [{layer.get('pan', 0.0)}]: ").strip()
            fi = input(f"  fade_in_ms [{layer.get('fade_in_ms', 0.0)}]: ").strip()
            fo = input(f"  fade_out_ms [{layer.get('fade_out_ms', 0.0)}]: ").strip()
            ps = input(f"  pitch_semitones [{layer.get('pitch_semitones', 0.0)}]: ").strip()
        except Exception:
            return

        undo.push(st)
        if g:
            layer["gain_db"] = float(g)
        if p:
            layer["pan"] = float(max(-1.0, min(1.0, float(p))))
        if fi:
            layer["fade_in_ms"] = float(max(0.0, float(fi)))
        if fo:
            layer["fade_out_ms"] = float(max(0.0, float(fo)))
        if ps:
            layer["pitch_semitones"] = float(ps)
        _push_edit(
            "layer_params",
            {
                "name": str(layer.get("name") or "layer"),
                "gain_db": float(layer.get("gain_db") or 0.0),
                "pan": float(layer.get("pan") or 0.0),
                "fade_in_ms": float(layer.get("fade_in_ms") or 0.0),
                "fade_out_ms": float(layer.get("fade_out_ms") or 0.0),
                "pitch_semitones": float(layer.get("pitch_semitones") or 0.0),
            },
        )
        _render()

    def _mix_audio(*, stereo: bool) -> np.ndarray:
        base = st.audio.astype(np.float32, copy=False)
        layers = _layer_list()
        if not layers:
            if stereo:
                return np.stack([base, base], axis=1).astype(np.float32, copy=False)
            return base

        # Pre-compute output length.
        out_len = int(base.size)
        for layer in layers:
            x = np.asarray(layer.get("audio"), dtype=np.float32).reshape(-1)
            off = int(layer.get("offset") or 0)
            out_len = max(out_len, int(off) + int(x.size))
        out_len = max(out_len, 1)

        if stereo:
            out = np.zeros((out_len, 2), dtype=np.float32)
            out[: int(base.size), 0] += base
            out[: int(base.size), 1] += base
        else:
            out = np.zeros(out_len, dtype=np.float32)
            out[: int(base.size)] += base

        for layer in layers:
            x = np.asarray(layer.get("audio"), dtype=np.float32).reshape(-1)
            if x.size == 0:
                continue

            semis = float(layer.get("pitch_semitones") or 0.0)
            rate = float(2.0 ** (semis / 12.0))
            x2 = _resample_linear(x, rate=rate)

            fi = int(round(float(layer.get("fade_in_ms") or 0.0) / 1000.0 * float(st.sample_rate)))
            fo = int(round(float(layer.get("fade_out_ms") or 0.0) / 1000.0 * float(st.sample_rate)))
            x2 = _apply_edge_fades(x2, fade_in=fi, fade_out=fo)

            gain = _db_to_lin(float(layer.get("gain_db") or 0.0))
            x2 = (x2 * float(gain)).astype(np.float32)

            off = int(layer.get("offset") or 0)
            off = max(0, min(off, out_len))
            end = min(out_len, off + int(x2.size))
            if end <= off:
                continue
            x2 = x2[: end - off]

            if stereo:
                pan = float(layer.get("pan") or 0.0)
                pan = float(max(-1.0, min(1.0, pan)))
                ang = float((pan + 1.0) * (math.pi / 4.0))
                gl = float(math.cos(ang))
                gr = float(math.sin(ang))
                out[off:end, 0] += (x2 * gl).astype(np.float32)
                out[off:end, 1] += (x2 * gr).astype(np.float32)
            else:
                out[off:end] += x2

        return out

    def _active_region_obj() -> dict[str, Any] | None:
        if st.active_region is None:
            return None
        rr = _region_list()
        idx = int(st.active_region)
        if idx < 0 or idx >= len(rr):
            return None
        return rr[idx]

    def _set_active_region(idx: int | None) -> None:
        if idx is None:
            st.active_region = None
            return
        rr = _region_list()
        if not rr:
            st.active_region = None
            return
        st.active_region = int(max(0, min(int(idx), len(rr) - 1)))

    def _default_region_name() -> str:
        rr = _region_list()
        return f"region_{len(rr) + 1:02d}"

    def _add_region_from_selection() -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        undo.push(st)
        rr = _region_list()
        rr.append(
            {
                "name": _default_region_name(),
                "start": int(lo),
                "end": int(hi),
                # export hook (off by default)
                "fx_chain": "off",
                "fx_chain_json": None,
                # optional loop markers (absolute sample positions)
                "loop_start": None,
                "loop_end": None,
            }
        )
        _set_active_region(len(rr) - 1)
        _push_edit("region_add", {"name": str(rr[-1]["name"]), "start": int(lo), "end": int(hi)})
        _render()

    def _region_prev_next(step: int) -> None:
        rr = _region_list()
        if not rr:
            return
        if st.active_region is None:
            _set_active_region(0)
        else:
            _set_active_region((int(st.active_region) + int(step)) % len(rr))
        _push_edit("region_select", {"active": int(st.active_region) if st.active_region is not None else None})
        _render()

    def _select_active_region() -> None:
        r = _active_region_obj()
        if r is None:
            return
        try:
            lo = int(r.get("start") or 0)
            hi = int(r.get("end") or 0)
        except Exception:
            return
        lo = max(0, min(lo, int(st.audio.size)))
        hi = max(0, min(hi, int(st.audio.size)))
        if hi <= lo:
            return
        st.selection = (lo, hi)
        _set_cursor(lo)
        _push_edit("region_to_selection", {"active": int(st.active_region) if st.active_region is not None else None})
        _render()

    def _rename_active_region() -> None:
        r = _active_region_obj()
        if r is None:
            return
        try:
            new_name = input("Region name: ").strip()
        except Exception:
            return
        if not new_name:
            return
        undo.push(st)
        old = str(r.get("name") or "region")
        r["name"] = str(new_name)
        _push_edit("region_rename", {"from": old, "to": str(new_name)})
        _render()

    def _set_active_region_fx_chain() -> None:
        r = _active_region_obj()
        if r is None:
            return
        from ..fx_chains import fx_chain_keys

        opts = ["off", *fx_chain_keys()]
        cur = str(r.get("fx_chain") or "off")
        try:
            print("FX chains:")
            for o in opts:
                mark = "*" if o == cur else " "
                print(f"  {mark} {o}")
            v = input("Set fx_chain (or 'off'): ").strip()
        except Exception:
            return
        if not v:
            return
        if v not in opts:
            print(f"Unknown fx_chain '{v}'.")
            return
        undo.push(st)
        r["fx_chain"] = str(v)
        _push_edit("region_fx_chain", {"name": str(r.get("name") or "region"), "fx_chain": str(v)})
        _render()

    def _set_active_region_loop(which: str) -> None:
        r = _active_region_obj()
        if r is None:
            return

        # Use selection bounds if present; else cursor.
        a = int(st.cursor)
        b = int(st.cursor)
        if st.selection is not None:
            lohi = _sel_range()
            if lohi is not None:
                a, b = lohi

        # Snap to zero crossing for clean boundaries.
        rad = int(0.005 * float(st.sample_rate))
        a = _nearest_zero_crossing(st.audio, a, radius=rad)
        b = _nearest_zero_crossing(st.audio, b, radius=rad)

        try:
            rlo = int(r.get("start") or 0)
            rhi = int(r.get("end") or 0)
        except Exception:
            return
        rlo = max(0, min(rlo, int(st.audio.size)))
        rhi = max(0, min(rhi, int(st.audio.size)))
        if rhi <= rlo:
            return

        a = max(rlo, min(a, rhi))
        b = max(rlo, min(b, rhi))
        if b < a:
            a, b = b, a

        undo.push(st)
        if which == "start":
            r["loop_start"] = int(a)
            if r.get("loop_end") is None:
                r["loop_end"] = int(rhi)
        else:
            r["loop_end"] = int(b)
            if r.get("loop_start") is None:
                r["loop_start"] = int(rlo)

        try:
            ls = int(r.get("loop_start") or rlo)
            le = int(r.get("loop_end") or rhi)
        except Exception:
            ls, le = rlo, rhi

        ls = max(rlo, min(ls, rhi))
        le = max(rlo, min(le, rhi))
        if le < ls:
            ls, le = le, ls
        r["loop_start"] = int(ls)
        r["loop_end"] = int(le)

        _push_edit(
            "region_loop_point",
            {
                "region": str(r.get("name") or "region"),
                "loop_s": [float(ls) / float(st.sample_rate), float(le) / float(st.sample_rate)],
                "snapped_zero": True,
            },
        )
        _render()

    def _delete_active_region() -> None:
        rr = _region_list()
        if not rr or st.active_region is None:
            return
        idx = int(st.active_region)
        if idx < 0 or idx >= len(rr):
            return
        undo.push(st)
        name = str(rr[idx].get("name") or "region")
        del rr[idx]
        if not rr:
            st.active_region = None
        else:
            st.active_region = int(max(0, min(idx, len(rr) - 1)))
        _push_edit("region_delete", {"name": name})
        _render()

    def _sanitize_for_filename(s: str) -> str:
        s2 = (s or "").strip().replace(" ", "_")
        out = []
        for ch in s2:
            if ch.isalnum() or ch in {"_", "-", "."}:
                out.append(ch)
        return "".join(out) or "region"

    def _post_params_from_patch(patch: dict[str, Any]) -> "PostProcessParams":
        from ..postprocess import PostProcessParams

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
            params = params.__class__(**{**params.__dict__, "trim_silence": (not bool(_get("no_trim")))})
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

    def _apply_region_fx(x: np.ndarray, *, sr: int, region: dict[str, Any]) -> np.ndarray:
        key = str(region.get("fx_chain") or "off").strip()
        chain_json = region.get("fx_chain_json")
        patch: dict[str, Any] = {}

        if key and key.lower() != "off":
            from ..fx_chains import FX_CHAINS

            chain = FX_CHAINS.get(key)
            if chain is not None and chain.args:
                patch.update(dict(chain.args))

        if chain_json:
            from ..fx_chains import load_fx_chain_json

            p2, _, _ = load_fx_chain_json(str(chain_json))
            patch.update(dict(p2))

        if not patch:
            return x.astype(np.float32, copy=False)

        from ..postprocess import post_process_audio

        params = _post_params_from_patch(patch)
        y, _rep = post_process_audio(x.astype(np.float32, copy=False), int(sr), params)
        return y.astype(np.float32, copy=False)

    def _export_all_regions() -> None:
        rr = _region_list()
        if not rr:
            return
        for r in rr:
            try:
                lo = int(r.get("start") or 0)
                hi = int(r.get("end") or 0)
            except Exception:
                continue
            lo = max(0, min(lo, int(st.audio.size)))
            hi = max(0, min(hi, int(st.audio.size)))
            if hi <= lo:
                continue

            name = _sanitize_for_filename(str(r.get("name") or "region"))
            base = wav_path.with_name(f"{wav_path.stem}__{name}{wav_path.suffix}")
            out_path = base
            k = 2
            while out_path.exists():
                out_path = wav_path.with_name(f"{wav_path.stem}__{name}_{k}{wav_path.suffix}")
                k += 1

            seg = st.audio[lo:hi].copy()
            seg2 = _clip_mono(_apply_region_fx(seg, sr=st.sample_rate, region=r))
            write_wav(out_path, seg2, int(st.sample_rate))

            # Sidecar for region exports (minimal but useful).
            sidecar = out_path.with_suffix("")
            sidecar = sidecar.with_name(sidecar.name + ".edits.json")
            rec = {
                "source_wav": str(wav_path),
                "exported_wav": str(out_path),
                "region": {
                    "name": str(r.get("name") or "region"),
                    "start_s": float(lo) / float(st.sample_rate),
                    "end_s": float(hi) / float(st.sample_rate),
                    "fx_chain": str(r.get("fx_chain") or "off"),
                    "fx_chain_json": (str(r.get("fx_chain_json")) if r.get("fx_chain_json") is not None else None),
                    "loop_s": None,
                },
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            loop_s = None
            if r.get("loop_start") is not None and r.get("loop_end") is not None:
                try:
                    ls = int(r.get("loop_start") or 0)
                    le = int(r.get("loop_end") or 0)
                    if le > ls and ls >= lo and le <= hi:
                        loop_s = [float(ls - lo) / float(st.sample_rate), float(le - lo) / float(st.sample_rate)]
                except Exception:
                    loop_s = None
            elif st.loop is not None:
                try:
                    gls, gle = st.loop
                    gls = int(gls)
                    gle = int(gle)
                    if gle > gls and gls >= lo and gle <= hi:
                        loop_s = [float(gls - lo) / float(st.sample_rate), float(gle - lo) / float(st.sample_rate)]
                except Exception:
                    loop_s = None

            rec["region"]["loop_s"] = loop_s
            sidecar.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            print(f"Exported region: {out_path}")
        _push_edit("export_regions", {"count": int(len(rr))})

    def _apply_state(src: _State) -> None:
        player.stop()
        st.audio = src.audio
        st.sample_rate = src.sample_rate
        st.cursor = src.cursor
        st.selection = src.selection
        st.clipboard = src.clipboard
        st.loop = src.loop
        st.markers = src.markers
        st.regions = src.regions
        st.active_region = src.active_region
        st.layers = src.layers
        st.active_layer = src.active_layer
        st.edits = src.edits
        st.export_count = src.export_count
        st.view_mode = src.view_mode
        st.spec_fft = src.spec_fft
        st.spec_fmax_hz = src.spec_fmax_hz

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

    def _do_declick() -> None:
        rng = _sel_range()
        if rng is None:
            # Default to a small window at cursor.
            half = int(0.006 * float(st.sample_rate))
            lo = max(0, min(int(st.cursor) - half, int(st.audio.size)))
            hi = max(0, min(int(st.cursor) + half, int(st.audio.size)))
            if hi <= lo:
                return
            rng = (lo, hi)

        lo, hi = rng
        undo.push(st)
        y = _repair_click_linear(st.audio, lo=lo, hi=hi)
        # Light cyclic smoothing over the boundary neighborhood.
        cf = int(0.003 * float(st.sample_rate))  # 3ms
        lo2 = max(0, lo - cf)
        hi2 = min(int(y.size), hi + cf)
        seg = _cyclic_crossfade(y[lo2:hi2], crossfade=min(cf, max(1, (hi2 - lo2) // 4)))
        y2 = y.copy()
        y2[lo2:hi2] = seg
        st.audio = _clip_mono(y2)
        st.selection = None
        _set_cursor(lo)
        _push_edit(
            "declick",
            {
                "start_s": float(lo) / float(st.sample_rate),
                "end_s": float(hi) / float(st.sample_rate),
                "method": "linear_bridge",
            },
        )
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
        # If layers use pan, export as stereo so pan is meaningful.
        use_stereo = any(abs(float(l.get("pan") or 0.0)) > 1e-6 for l in (_layer_list() or []))
        rendered = _mix_audio(stereo=use_stereo)
        if use_stereo:
            import soundfile as sf

            sf.write(str(out_path), np.clip(rendered, -1.0, 1.0), int(st.sample_rate), subtype="PCM_16")
        else:
            write_wav(out_path, _clip_mono(rendered), int(st.sample_rate))
        _write_edits_sidecar(out_path, state=st, mode=mode)
        _push_edit("export", {"mode": mode, "path": str(out_path)})
        print(f"Exported: {out_path}")

    def _play_from_cursor(loop: bool) -> None:
        if sys.platform != "win32":
            print("Playback is only implemented on Windows in this editor.")
            return
        # Base audio playback (edits apply here). Layered preview is on 'b'.
        player.play_segment(audio=st.audio, sr=st.sample_rate, start=int(st.cursor), end=int(st.audio.size), loop=loop)

    def _play_mix_from_cursor() -> None:
        if sys.platform != "win32":
            print("Playback is only implemented on Windows in this editor.")
            return
        use_stereo = any(abs(float(l.get("pan") or 0.0)) > 1e-6 for l in (_layer_list() or []))
        mix = _mix_audio(stereo=use_stereo)
        start = max(0, min(int(st.cursor), int(mix.shape[0] if mix.ndim == 2 else mix.size)))
        end = int(mix.shape[0] if mix.ndim == 2 else mix.size)
        player.play_segment(audio=mix, sr=st.sample_rate, start=start, end=end, loop=False)

    def _play_selection(loop: bool) -> None:
        rng = _sel_range()
        if rng is None:
            return
        lo, hi = rng
        if sys.platform != "win32":
            print("Playback is only implemented on Windows in this editor.")
            return
        seg = st.audio[lo:hi].copy()
        if loop and seg.size > 8:
            # Seamless loop preview: apply a cyclic crossfade so the loop boundary is smoother.
            cf = int(0.040 * float(st.sample_rate))
            seg = _cyclic_crossfade(seg, crossfade=cf)
        player.play_segment(audio=seg, sr=st.sample_rate, start=0, end=int(seg.size), loop=loop)

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

        if key in {"R"}:
            _add_region_from_selection()
            return

        if key in {","}:
            _region_prev_next(-1)
            return

        if key in {"."}:
            _region_prev_next(+1)
            return

        if key in {"X"}:
            _select_active_region()
            return

        if key in {"N"}:
            _rename_active_region()
            return

        if key in {"C"}:
            _set_active_region_fx_chain()
            return

        if key in {"{"}:
            _set_active_region_loop("start")
            return

        if key in {"}"}:
            _set_active_region_loop("end")
            return

        if key in {"D"}:
            _delete_active_region()
            return

        if key in {"O"}:
            _export_all_regions()
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

        if key in {"b"}:
            _play_mix_from_cursor()
            return

        if key in {"i"}:
            _import_layer()
            return

        if key in {"y"}:
            _layer_prev_next(+1)
            return

        if key in {"L"}:
            _edit_active_layer_params()
            return

        if key in {"Y"}:
            _delete_active_layer()
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

        if key in {"K"}:
            _do_declick()
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
                _apply_state(prev)
                _render()
            return

        if key in {"r", "ctrl+y"}:
            nxt = undo.redo(st)
            if nxt is not None:
                _apply_state(nxt)
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
