# S‑NDB‑UND roadmap (status + next milestones)

This doc maps the aspirational roadmap to what exists in the repo today, and what’s realistically next.

For a tight definition of “one‑stop shop” (and a buildable MVP scope), see `docs/one_stop_shop.md`.

For model ecosystem notes (what’s worth copying, what’s risky), see `docs/audio_generation_landscape.md`.

Legend: **Done** / **In repo (basic)** / **Next** / **Later (optional)**

---

# v2 — Professional Sound‑Design Environment (Spec)

- Spec doc: [docs/v2_spec.md](v2_spec.md)
- Status: **Next** (multi-iteration)

## v2 Editor (early)

- **In repo (basic)**: spectrogram view toggle + adjustable FFT + frequency zoom (`S`/`F`/`Z`)
- **In repo (basic)**: transient detection auto-mark + snap selection to transient (`T`/`J`)
- **In repo (basic)**: auto loop points + auto-crossfade loop creation (`A`)
- **In repo (basic)**: multi-region slicing + export-all (`R`, `O`) with per-region FX hook (`C`)
- **In repo (basic)**: per-region loop point metadata export (`{`, `}`)

## v2 Completion Checklist (Definition A)

v2 is considered **complete** when every section below is **Done** end-to-end (code + docs + wired into CLI/GUI where applicable).

- **v2.1 Editor v2** — **Next**
  - **Done gate**: spectrogram view w/ adjustable FFT + frequency zoom + click-removal targeting
  - **Done gate**: transient detection + snap-to-transient + usable UX for selection snapping
  - **Done gate**: multi-region workflow (name regions, slice/export, delete, navigate)
  - **Done gate**: region-based FX (per-region FX assignment + applied on export)
  - **Done gate**: layering (2–4 layers max) with per-layer gain/pan/fade/pitch shift
  - **Done gate**: loop tools (auto-crossfade loop creation + seamless loop preview)
  - **Done gate**: loop point metadata export (for region exports + main export)
  - **In repo (basic)**: spectrogram/transients/regions/loop metadata shipped (see above)

- **v2.2 Presets v2 (Smart Presets)** — **Next**
  - **Done gate**: parameter variables + randomized prompt fragments + families
  - **Done gate**: preset inheritance (base → variants)
  - **Done gate**: schema documented + example preset(s) included
  - **Done gate**: wired through CLI + batch + docs-driven generation

- **v2.3 FX Chains v2 (Modular + Visual)** — **Next**
  - **Done gate**: new FX modules implemented (pitch shift, time stretch, distortion, convolution reverb IR, multi-band EQ, transient shaper, spectral noise reduction)
  - **Done gate**: FX chain editor supports reorder + parameter editing + audition + save
  - **Done gate**: chain format stable + documented

- **v2.4 Project System (Sound Packs & Mobs)** — **Next**
  - **Done gate**: project create/load + track generated/edited versions + metadata
  - **Done gate**: pack export (Minecraft-ready where applicable) + versioning per sound
  - **Done gate**: Minecraft project mode (generate → edit → export) workflow documented

- **v2.5 Hybrid Engine Mode (AI + Procedural Layering)** — **Next**
  - **Done gate**: AI base + procedural transient/noise texture layers + automix
  - **Done gate**: apply FX chain to final composite + credits capture

- **v2.6 Fine‑Tuning Support (Optional)** — **Next**
  - **Done gate**: training script + validation preview + model versioning
  - **Done gate**: preset linking to fine-tuned models

- **v2.7 CLI v2** — **Next**
  - **Done gate**: project creation + export commands
  - **Done gate**: hybrid mode controls + preset variables + FX override + editor region export controls
  - **Done gate**: README usage examples

- **v2.8 GUI v1 (First Real Interface)** — **Next**
  - **Done gate**: preset browser + generate panel + waveform editor + FX chain editor + project browser + export panel
  - **Done gate**: install/run docs

---

# v3 — Sound‑Design Platform (Spec)

- Spec doc: [docs/v3_spec.md](v3_spec.md)
- Status: **Later (optional)** (ambitious long-horizon)

## v3 Completion Checklist (Spec)

v3 is considered **complete** when every section below is **Done** end-to-end (code + docs + GUI/CLI wiring + export profiles).

- **v3.1 Multi‑track sound design (mini‑DAW)** — **Later (optional)**
  - **Done gate**: timeline with clips (drag/move/trim/split/crossfade)
  - **Done gate**: tracks with mute/solo/gain/pan + per-track FX chain
  - **Done gate**: snap modes (grid/transient/region boundaries)
  - **Done gate**: per-clip gain/fade/pitch/time-stretch

- **v3.2 AI‑assisted editing** — **Later (optional)**
  - **Done gate**: auto-clean tools (denoise/de-click/hum removal)
  - **Done gate**: auto-loop suggestions + auto crossfade
  - **Done gate**: auto-FX suggestions + auto-layer suggestions
  - **Done gate**: auto naming + tagging into metadata

- **v3.3 Plugin system (platform)** — **Later (optional)**
  - **Done gate**: plugin manifest format (`sndbund_plugin.json`) + loader from `plugins/`
  - **Done gate**: plugin types: engines, FX modules, FX chains, preset packs, exporters, editor tools
  - **Done gate**: versioning + dependency validation + safe failure modes

- **v3.4 Game engine export (Unity/Unreal/Godot)** — **Later (optional)**
  - **Done gate**: Unity export profile (WAV + optional metadata templates)
  - **Done gate**: Unreal export profile (WAV + cue definitions + optional MetaSound templates)
  - **Done gate**: Godot export profile (WAV + import hints)

- **v3.5 Hybrid engine v3 (AI + procedural + sample)** — **Later (optional)**
  - **Done gate**: pipeline supports AI base + procedural transients + sample layer + automix
  - **Done gate**: configurable via presets (layers active/weights/layer FX)

- **v3.6 GUI v2 (full workstation UI)** — **Later (optional)**
  - **Done gate**: transport + timeline + track list + FX rack + preset browser + engine panel
  - **Done gate**: project tree + metadata inspector + export panel
  - **Done gate**: modes (Generate/Edit/Mix/Export) in one window

- **v3.7 Sound pack builder v2** — **Later (optional)**
  - **Done gate**: pack categories + status/tags/source tracking
  - **Done gate**: pack export profiles (Minecraft + Unity/Unreal/Godot)
  - **Done gate**: loudness normalization across pack

- **v3.8 CLI v3 extensions** — **Later (optional)**
  - **Done gate**: project create/build/export commands
  - **Done gate**: hybrid engine CLI + pack normalize command

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

- **3.2 Fine‑tuning support** — **Done**
  - Dataset export tool exists: `python -m soundgen.creature_finetune prepare …`
  - Inference-time LoRA loading exists for `stable_audio_open` (creature families).
  - Training recipe docs: `docs/creature_family_training_windows.md`.
  - License-aware plugin guidance: `docs/plugins.md` + optional loader gate via `SOUNDGEN_ACCEPT_PLUGIN_LICENSES`.

- **3.3 Multi‑engine blending** — **Done (layered v1)**
  - `layered` mixes samplelib transient/tail with synth body.
  - Added: controlled crossfades (transient→body, body→tail) and optional per-layer FX (HP/LP/drive/gain).

- **3.4 Looping + ambience tools** — **Done**
  - Loop-clean + crossfade exists.
  - Loop suite commands implemented: `python -m soundgen.loop_suite` (auto loop points, tail trim heuristics, noise bed helpers) and `S-NDB-UND.exe loop ...`.

- **3.5 Editor-grade DSP + analysis (Audacity/Ardour inspired)** — **Done**
  - Export analysis report (stored in credits / best-of-N selection).
  - Fade/crossfade curve shapes for loop-clean + layered blending.
  - Envelope follower utilities shared by transient/ducking-style tools.
  - Ducking / sidechain-style polish (offline “duck bed under transients”).
  - Offline effect stack abstraction (configurable post stack + presets).
  - Best-of-N is consistent across workflows: `--candidates` (single + web) and `candidates` in batch/docs manifests; implemented centrally in `soundgen.engine_registry.generate_wav`.

- **3.6 Built-in SFX editor (laser-focused, not a DAW)** — **Done (v1)**
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
  - Implemented in `soundgen.editor.launch` (matplotlib waveform UI + Windows playback), wired via `python -m soundgen.generate --edit` and `S-NDB-UND.exe edit <wav>`.

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
  - Tech stack (Python): matplotlib (interactive waveform UI) + stdlib playback on Windows; processing remains numpy/scipy.
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
