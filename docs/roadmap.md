# S‑NDB‑UND roadmap (status + next milestones)

This doc maps the aspirational roadmap to what exists in the repo today, and what’s realistically next.

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

- **2.2 Preset library v1 (content pack)** — **Next**
  - The *system* exists; what’s missing is shipping a curated preset pack (20+ entries) with names, descriptions, and “when to use” guidance.

- **2.3 Engine registry (plugin discovery)** — **In repo (basic), Next (plugins)**
  - Registry exists as a dispatcher, but not yet a true plugin folder / entry-points discovery mechanism.
  - Next step: define a plugin API and load engines from `soundgen_plugins/` or Python entry points.
  - Update: plugin discovery is now implemented; see `docs/plugins.md`.

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

- **3.3 Multi‑engine blending** — **In repo (layered), Next (more blending)**
  - `layered` already mixes samplelib transient/tail with synth body.
  - Next: add optional parallel FX chains per layer + controlled crossfades.

- **3.4 Looping + ambience tools** — **In repo (basic), Next (loop suite)**
  - Loop-clean + crossfade exists.
  - Next: “loop suite” commands (auto-loop points, tail trimming heuristics, noise bed helpers).

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
