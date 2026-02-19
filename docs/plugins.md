# Plugins (engine discovery)

S‑NDB‑UND supports **engine plugins** so you (or the community) can add new generation backends without editing core files.

If you’re looking for **audio effect plugins** (EQ/compressor/reverb style processing) via **CLAP/LV2**, see:

- [Audio FX plugins (CLAP + LV2)](audio_fx_plugins.md)

## What plugins can do

- Register **new engines** that output mono audio (NumPy) + sample rate.
- Appear automatically in:
  - `python -m soundgen.generate --engine ...`
  - Gradio Web UI engine selector
  - `python -m soundgen.mob_soundset --engine ...`
  - `python -m soundgen.from_docs --engine ...`
  - `python -m soundgen.benchmark --engines ...`

## Plugin API (engine plugins)

An engine plugin registers a callable:

- Input: a request dict with keys like `prompt`, `seconds`, `seed`, `device`, `sample_rate`
- Output: `soundgen.engine_registry.EnginePluginResult` with:
  - `audio`: mono `np.ndarray` float32-ish
  - `sample_rate`: int
  - optional `credits_extra` and `sources`

Core handles post-processing (if enabled) and writes the final WAV.

## Option A: Local folder plugin (easiest)

1) Create a folder at the repo root:

- `soundgen_plugins/`

2) Add a module, e.g. `soundgen_plugins/my_engine.py`:

```python
import numpy as np

from soundgen.engine_registry import EnginePluginResult


def register_soundgen(register_engine):
    def my_engine(req: dict) -> EnginePluginResult:
        sr = int(req.get("sample_rate") or 44100)
        seconds = float(req.get("seconds") or 1.0)
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        return EnginePluginResult(audio=audio, sample_rate=sr, credits_extra={"engine": "my_engine"})

    register_engine("my_engine", my_engine)
```

3) Run:

- `python -m soundgen.generate --engine my_engine --prompt "test" --seconds 1.0 --out outputs\\test.wav`

## Option B: Python package entry point (shareable)

If you want a pip-installable plugin, expose an entry point in your package:

- Entry point group: `soundgen.engines`
- Entry point object: a callable or module that provides `register_soundgen(register_engine)`

This repo’s loader will `ep.load()` and call the register hook.

## Notes

- Plugin discovery is best-effort. If a plugin fails to import, core engines still work.
- Keep plugin imports light if you care about `--help` startup time.

## License-aware plugins (non-commercial weights, gated models)

Many state-of-the-art audio model weights are **non-commercial**, **gated**, or require accepting terms.
This repo can’t enforce legal compliance for you, but it can help make the licensing surface area explicit.

### How to declare a license notice in a plugin

If your plugin wraps model weights with special terms, define a module-level dict:

```python
SOUNDGEN_PLUGIN_LICENSE = {
  "id": "myengine-nc-2026",
  "name": "MyEngine Model Weights (Non-Commercial)",
  "url": "https://example.com/license",
  "notice": "Non-commercial use only. See license URL for terms.",
  "requires_acceptance": True,
}
```

If `requires_acceptance` is `True`, the plugin will only load when the user explicitly opts in by setting:

- `SOUNDGEN_ACCEPT_PLUGIN_LICENSES=myengine-nc-2026`

You can list multiple accepted license ids separated by commas.

### Recommendation

- Treat weight licenses separately from code licenses.
- Always include a clear `notice` and `url`.
- For gated models (Hugging Face), mention the exact model id and the “accept terms” step in the plugin README.
