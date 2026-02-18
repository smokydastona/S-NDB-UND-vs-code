# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on *Keep a Changelog*.

## Unreleased

### Added

- Web UI: “Advanced post (3.5)” controls (crossfade curve shapes, duck-bed under transients, post-stack override, compressor follower mode, layered xfade curve).
- FX chains (v1): `--fx-chain` presets and `--fx-chain-json` loader for shareable post recipes.
- Post-processing: `noise_bed` module + `--noise-bed-db` / `--noise-bed-seed` controls.

### Changed

- Web UI: FX chains can be selected via an **FX chain (v1)** accordion.
- Web UI (Control Panel): audio iteration is now **in-memory** (variants are cached; files are export artifacts).
- Web UI (Control Panel): temporary WAVs are deleted by default after ingest; set `SOUNDGEN_CONTROL_PANEL_KEEP_WAVS=1` to keep them for debugging.

### Fixed

- Web UI: Gradio demo build no longer crashes due to `gr.Update` type hints.
