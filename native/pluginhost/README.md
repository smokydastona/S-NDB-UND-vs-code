# soundgen_pluginhost (native helper)

This is a **native helper executable** intended to let SÖNDBÖUND host external audio plugins.

## Current scope (MVP)

- **Scan** for CLAP (`.clap`) and LV2 bundles (`*.lv2`) on disk.
- **Render through a CLAP plugin offline** (destructive render) for preview.

> Note: True real-time hosting (low-latency streaming, automation, UI hosting) is planned, but not yet implemented here.

## Build (Windows)

Prereqs:
- Visual Studio Build Tools (MSVC)
- CMake 3.21+
- Git

From repo root:

```powershell
cmake -S native/pluginhost -B native/pluginhost/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/pluginhost/build --config Release
```

The executable should land at:
- `native/pluginhost/build/Release/soundgen_pluginhost.exe`

## Commands

- `soundgen_pluginhost scan`
  - Prints JSON with discovered CLAP/LV2 paths.

- `soundgen_pluginhost clap-render --plugin <path> --in <wav> --out <wav>`
  - Renders mono WAV through the **first** plugin in the CLAP library.
  - For now, it only supports plugins with **one stereo or mono audio in/out** pair and no params.

## Environment variables

- `SOUNDGEN_CLAP_PATHS` – `;` separated extra search paths
- `LV2_PATH` – `;` separated LV2 search paths (bundle directories)
