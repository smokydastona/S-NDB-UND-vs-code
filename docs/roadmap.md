# S‑NDB‑UND roadmap (status + next milestones)

This doc maps the aspirational roadmap to what exists in the repo today, and what’s realistically next.

For a tight definition of “one‑stop shop” (and a buildable MVP scope), see `docs/one_stop_shop.md`.

For model ecosystem notes (what’s worth copying, what’s risky), see `docs/audio_generation_landscape.md`.

Legend: **Done** / **In repo (basic)** / **Next** / **Later**

## Phase 1 — Core foundations (toward 1.0)

- **1.1 Multi‑engine architecture** — **Done**
  - Central dispatcher: `soundgen.engine_registry.generate_wav`
  - Engines: `diffusers`, `stable_audio_open`, `rfxgen`, `replicate`, `samplelib`, `synth`, `layered`

- **1.2 Stable Audio Open engine** — **Done**
  - Wrapper: `soundgen.stable_audio_backend`
  - Exposed params: prompt, negative prompt, seed, duration, CFG, sampler, model
  - Error handling for gated HF models: explicit message paths

- **1.3 Preset system (usability)** — **Done (two layers)**
  - **Pro presets**: high-level “one click defaults” that apply conservatively
  - Prompt suffix + engine-aware conditioning supported

- **1.4 Named post‑processing chains** — **Done**
  - Implemented as **polish profiles** (named tuned chains)

- **1.5 Batch generation v1** — **Done**
  - JSON/CSV manifests, per-item overrides, naming, seeds, metadata/catalog output

## Phase 2 — Modder‑first features (1.0 → 2.0)

- **2.1 Minecraft mob soundset generator** — **Done**
  - Command: `S‑NDB‑UND.exe mobset …` / `python -m soundgen.app mobset …`
  - Generates hurt/death/ambient/step, variants, updates `sounds.json`, optional snippet

- **2.2 Preset library v1 (content pack)** — **Done (demo pack)**
  - Curated demo pack exists: `soundpack_v1/` (WAV + Minecraft-ready OGG + credits).

- **2.3 Engine registry (plugin discovery)** — **Done**
  - Engine plugin discovery supports `./soundgen_plugins/` + Python entry points.
  - Docs: `docs/plugins.md`.

- **2.4 Unified metadata system** — **Done**
  - Sidecar credits JSON + pack credits JSON; includes timestamps and reproducibility fields.

## Phase 3 — Professional workflow features (2.0 → 3.0)

- **3.1 GUI frontend** — **Done (practical) / Later (Electron/Qt)**
  - Gradio Web UI + desktop wrapper exist and cover controls.
  - If you still want Electron/Qt, that’s a separate “productization” milestone.

- **3.2 Fine‑tuning support** — **Done (practical), Next (training recipes)**
  - Dataset export tool exists: `python -m soundgen.creature_finetune prepare …`
  - Inference-time LoRA loading exists for `stable_audio_open` (creature families).
  - Next step: provide training recipes for one or two popular stacks (diffusers+accelerate recommended).
  - Also: add a license-aware “engine plugin” guide (many state-of-the-art weights are non-commercial).

- **3.3 Multi‑engine blending** — **Done (layered v1)**
  - `layered` mixes samplelib transient/tail with synth body.
  - Added: controlled crossfades (transient→body, body→tail) and optional per-layer FX (HP/LP/drive/gain).

- **3.4 Looping + ambience tools** — **In repo (basic), Next (loop suite)**
  - Loop-clean + crossfade exists.
  - Next: “loop suite” commands (auto-loop points, tail trimming heuristics, noise bed helpers).

- **3.5 Editor-grade DSP + analysis (Audacity/Ardour inspired)** — **Done**
  - Export analysis report (stored in credits / best-of-N selection).
  - Fade/crossfade curve shapes for loop-clean + layered blending.
  - Envelope follower utilities shared by transient/ducking-style tools.
  - Ducking / sidechain-style polish (offline “duck bed under transients”).
  - Offline effect stack abstraction (configurable post stack + presets).
  - Best-of-N is consistent across workflows: `--candidates` (single + web) and `candidates` in batch/docs manifests; implemented centrally in `soundgen.engine_registry.generate_wav`.

- **3.6 Built-in SFX editor (laser-focused, not a DAW)** — **Next (v1)**
  - Goal: a small, destructive single-file editor for generated SFX and WAVs from disk.
  - Fits pipeline: **Engine → Post-FX → Editor → Export**
    - Generated sounds open directly in editor after creation (optional toggle).
    - Editor can open any WAV from disk.
    - Edits apply to a working copy; full multi-step undo/redo.
    - Final export writes processed WAV plus updated metadata (trim points, gain, edits applied).
  - **Definition of “complete” for S‑NDB‑UND v1 editor** (creature SFX + ambience-ready):
    - **Waveform view**: zoom (wheel + shortcuts), scroll/pan, time cursor, drag selection.
    - **Basic edits**: cut/copy/paste, delete selection, trim to selection.
    - **Fades**: fade in/out, crossfade over selection.
    - **Gain**: ±dB, normalize to peak target (default −1 dBFS).
    - **Loop tools**: snap selection to loop boundaries, audition loop (seam check).
    - **Markers**: mark transients, loop points, “good takes”.
    - **Playback**: play from cursor, play selection, loop selection.
    - **Export**: overwrite, “save as new variation” (auto suffix: `_edit1`, `_trim`, `_loopfix`, …).

  **Editor repo takeaways (applied, scope-safe)**
  - **Mode split (Zoom vs Selection)**: copy the explicit mode toggle idea (zoom-only interactions vs selection/edit interactions) to reduce accidental edits and simplify gestures (from Web-Audio-Editor).
  - **“Leave” / isolate selection**: keep a single destructive operation that retains only the selection and discards the rest (from Web-Audio-Editor).
  - **Undo/redo as a first-class requirement**: treat multi-step undo/redo as non-negotiable for destructive workflows (from Treble).
  - **Reverse / repeat / append**: useful v2 destructive ops for SFX iteration (reverse for whooshes; repeat for rattles; append with optional fixed gap for compound stingers) (from Treble).
  - **Pitch change**: add pitch shift as a v2 offline transform (distinct from playback-rate preview) for “variants without regen” workflows (from Sound-Editor).
  - **Simple export configuration**: keep export as “one-click” with a small set of settings surfaced (format, sample rate, loudness target) rather than a deep dialog (inspired by AudioEditorKit).
  - **Kira-inspired realtime preview layer (editor playback)**
    - Inspiration: https://github.com/tesselode/kira (tweens, mixer tracks + FX, clocks, spatial audio).
    - **Backend-agnostic playback engine**: abstract a small `AudioManager`-like layer so editor playback isn’t tied to one library.
    - **Tweens / ramps for click-free control**: smooth parameter changes (gain, pan, playback rate) to avoid zipper noise and clicks.
    - **Mixer-style tracks**: master track + optional preview sub-track(s) for audition-only FX.
    - **Preview FX chain**: lightweight effects for monitoring (e.g., low-pass “muffle”, simple filter) separate from offline post chain.
    - **Clock/scheduling (minimal v1)**: schedule “play selection” / “loop audition” precisely and reproducibly (no UI metronome requirement).
  - **Resound-inspired DSP bus model (editor playback)**
    - Inspiration: https://github.com/SolarLune/resound (DSP channels / buses + ordered effect stacks).
    - **DSP channels (buses)**: route preview playback through named buses (e.g., `preview_master`, `preview_fx`) for shared volume/pan/filter.
    - **Ordered effects stack**: keep effect order explicit and user-visible for preview (Delay → Distort → Volume etc).
    - **Effects wrap a stream**: model preview FX as wrappers around the playback stream so it stays composable.
    - **Pitfalls to avoid** (called out by Resound):
      - Tail-extending FX (reverb/delay) must not be hard-cut when the source ends (editor should optionally render/drain tails).
      - Realtime parameter changes need synchronization to avoid racey reads/writes (thread-safe params or audio-thread ownership).
  - Tech stack (Python): PySide6/PyQt6 + (pyqtgraph **or** downsampled QPainter) + sounddevice for playback; processing remains numpy/scipy.
  - **v2 nice-to-haves (Later)**:
    - Spectrogram view (click/noise hunting).
    - Multi-region per file (slice one WAV into multiple exports).
    - Simple layering (overlay 2–3 sounds; not full multitrack).
    - Spatial preview (game-ish): stereo panning + distance attenuation; later: Doppler-style pitch shift.
    - AI-assisted repair tools (optional): inpaint/replace selection, extend tail, audio-to-audio “style transfer” for seam fixing.

## Phase 4 — Ecosystem & expansion (3.0 → 4.0)

- **4.1 Plugin ecosystem** — **Later** (depends on 2.3 plugin discovery)
- **4.2 Game engine integrations (Unity/Unreal/Godot)** — **Later**
- **4.3 Cloud mode / inference server** — **Later**

## Phase 5 — Best-in-world milestones

- **5.x Specialized creature engine + hybrid AI/procedural** — **Later**
  - Most of the building blocks exist (layering, polish profiles, creature conditioning);
    the gap is curated datasets + training recipes + stronger blending controls.

## The best next step (highest leverage)

If your goal is "creature families", the most leverage comes from:

1) **Training recipe + tooling doc** (diffusers+accelerate LoRA), tied to the dataset exporter
2) **Curated creature family preset pack** (prompt templates + recommended polish profiles)
3) **Plugin discovery** (so community can ship new engines/presets cleanly)
