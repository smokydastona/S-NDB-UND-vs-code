# Built-in editor (v1) — UI layout + interaction spec

Goal: a **single-file, destructive** SFX editor that covers the “quick fix” workflow (trim/fade/normalize/loop audition) without becoming a DAW.

## Window layout (v1)

Single main window:

1) **Top bar (file + export)**
- File: Open WAV, Open last generated, Save (overwrite), Save as variation
- Export: Export WAV, (future) Export OGG/Minecraft hook
- Metadata: show engine/prompt/seed/fx chain (read-only in v1)

2) **Transport bar (playback)**
- Play/Pause
- Stop
- Loop selection toggle
- Play selection button
- Time readout (cursor position + selection length)

3) **Waveform panel (main)**
- Waveform view (mono)
- Marker lane above waveform:
  - transient markers
  - loop start/end markers
  - “good take” markers
- Cursor (single vertical line)
- Selection (click-drag region)
- Scroll bar / mini overview (optional v1; can be plain pan)

4) **Right sidebar (tools)**
- **Selection tools**
  - Trim to selection
  - Cut selection
  - Delete selection
  - Insert silence (ms)
- **Fades**
  - Fade in (ms)
  - Fade out (ms)
  - Crossfade (ms) (applies over selection; symmetric)
- **Gain / Normalize**
  - Gain ±dB (apply)
  - Normalize to peak (default −1.0 dBFS)
- **Markers**
  - Add marker at cursor
  - Add loop start / loop end
  - Delete marker

5) **Bottom bar (history)**
- Undo / Redo buttons
- Small history list (last N operations with names)

## Interaction model

### Zoom + pan
- Mouse wheel: zoom horizontally around cursor (or mouse position)
- Shift+wheel: horizontal scroll
- Ctrl+wheel: zoom faster (optional)
- Shortcuts:
  - `+` / `-` zoom in/out
  - `0` zoom to fit

### Cursor + selection
- Click: set cursor
- Click-drag: create selection
- Drag selection edges: resize selection
- Double-click: select “meaningful region” (optional; can be omitted v1)

### Markers
- `M`: marker at cursor
- `[` / `]`: set loop start / loop end at cursor
- Markers are stored in **sample indices** (sr-aware) so edits shift markers correctly.

### Playback
- Space: play/pause from cursor
- Enter: play selection
- `L`: toggle loop selection

### Editing operations
All operations are **destructive on a working buffer** but must be undoable.

- Trim: keep selection, discard everything else
- Cut: remove selection and copy to clipboard
- Delete: remove selection (no clipboard)
- Insert silence: insert at cursor (or replace selection)
- Fade in/out: apply to selection or to file head/tail (v1 can be selection-only)
- Normalize: peak normalize to target dBFS
- Gain: constant gain for selection or whole file (v1 can default to selection if exists else whole)

## File outputs + naming

- Overwrite: writes the current working buffer to the original WAV path.
- Save as variation: writes alongside the original with a suffix:
  - `_edit1`, `_trim`, `_fade`, `_norm`, `_loopfix` (implementation chooses the next available)

## Data model (for implementation)

- Audio buffer: `float32` mono in `[-1, 1]`
- Sample rate: int
- Cursor: sample index
- Selection: `(start_idx, end_idx)` sample indices (or None)
- Markers: list of `{kind, index, label}`
- History: list of operations storing enough info to undo/redo efficiently

## Non-goals (explicitly not v1)

- Spectrogram
- Multi-region clips per file
- Multitrack / complex layering
- Advanced EQ graphing
- Automation / envelopes
