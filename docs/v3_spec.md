# S‑NDB‑UND v3 — Sound‑Design Platform (Final Spec)

Theme: from “workstation” to a full sound‑design platform for game/mod audio.

## 1) Multi‑track sound design (mini‑DAW)

Core:

- Unlimited tracks (practical limit = system resources)
- Per‑track controls: mute, solo, gain, pan
- Per‑track FX chain (using the FX system)
- Clips on timeline: drag, move, trim, split, crossfade
- Snap modes: grid, transient, region boundaries
- Per‑clip: gain, fade in/out, pitch shift, time‑stretch

Goal: build full creature roars, ambiences, UI suites inside S‑NDB‑UND.

## 2) AI‑assisted editing

Tools:

- Auto‑clean: denoise, de‑click, hum removal
- Auto‑loop: suggest seamless loop points + auto crossfade
- Auto‑FX: suggest FX chain based on sound type (creature, UI, ambience)
- Auto‑layer: propose extra layers (low rumble, breath, texture)
- Auto‑naming + tagging: semantic names + tags in metadata

Goal: speed up polish and reduce manual cleanup.

## 3) Plugin system

Plugin types:

- Engines (AI, procedural, hybrid)
- FX modules
- FX chains
- Preset packs
- Exporters (Unity, Unreal, Godot, custom)
- Editor tools (special selection ops)

Format:

- Python module + manifest (`sndbund_plugin.json`)
- Declares type, version, dependencies, entrypoints
- Loaded from `plugins/` directory

Goal: let others extend S‑NDB‑UND without touching core.

## 4) Game engine export (Unity / Unreal / Godot)

Unity:

- Export WAVs + optional `.meta` presets
- Optional ScriptableObject sound bank (JSON or C# template)

Unreal:

- Export WAVs + cue definition JSON/INI
- Optional MetaSound graph templates (text assets)

Godot:

- Export WAVs + `.import` config hints (looping, compression)

Goal: “Export to engine” is a first‑class action.

## 5) Hybrid engine v3 (AI + procedural + sample)

Pipeline:

- AI base layer (Stable Audio Open or fine‑tuned variant)
- Procedural transient layer (clicks, impacts, consonants)
- Sample layer (user library: breaths, cloth, metal)
- Auto‑mix: level balance, EQ match, stereo placement
- FX chain: final polish (limiter, saturation)

Configurable via preset:

- which layers are active
- layer weights
- layer FX

Goal: sounds that feel “designed”, not just “generated”.

## 6) GUI v2 — full workstation UI

Main layout:

- Top: transport (play/stop/loop; record later if needed), time ruler
- Center: multi‑track timeline (waveform + optional spectrogram per track)
- Left: track list (name, mute/solo, FX, meter)
- Right: FX rack + preset browser + engine panel
- Bottom: project tree + metadata inspector + export panel

Modes (same window): Generate / Edit / Mix / Export.

Goal: live in this UI for entire sound packs.

## 7) Sound pack builder v2

Features:

- Project marked as a “sound pack”
- Categories: creature, UI, ambience, footsteps, magic, etc.
- Each sound has status (draft/approved/final), tags, source (engine/preset/hybrid)
- Pack export: folder structure, zip, mod pack (Minecraft), engine export profiles
- Loudness normalization across pack

Goal: ship a whole pack from one project.

## 8) CLI v3 extensions

Examples:

- `sndbund project create <name>`
- `sndbund project build` (render all pending sounds)
- `sndbund project export --target unity`
- `sndbund engine hybrid --preset creature_large_bellow --count 8`
- `sndbund pack normalize --target -14`

Goal: full automation for power users and CI.

## One‑sentence summary

A multi‑track, AI‑assisted, plugin‑extensible, game‑engine‑aware sound‑design platform where you can generate, design, organize, and export entire sound packs without leaving S‑NDB‑UND.
