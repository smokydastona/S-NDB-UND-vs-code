from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .engine_registry import GeneratedWav, available_engines, generate_wav
from .io_utils import convert_audio_with_ffmpeg, read_wav_mono, write_wav
from .postprocess import PostProcessParams, post_process_audio
from .qa import compute_metrics, detect_long_tail
from .minecraft import export_wav_to_minecraft_pack
from .credits import upsert_pack_credits, write_sidecar_credits
from .controls import map_prompt_to_controls
from .pro_presets import apply_pro_preset, pro_preset_keys
from .polish_profiles import apply_polish_profile, polish_profile_keys
from .fx_chains import apply_fx_chain, fx_chain_keys
from .sfx_presets import apply_sfx_preset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a sound effect WAV from a text prompt.")
    p.add_argument(
        "--engine",
        choices=available_engines(),
        default="diffusers",
        help="Generation engine (built-ins + optional plugins). Built-ins: diffusers, stable_audio_open, rfxgen, replicate, samplelib, synth, layered.",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Text prompt describing the sound. If omitted, you must pass --sfx-preset.",
    )
    p.add_argument("--seconds", type=float, default=3.0, help="Duration in seconds.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for repeatability.")
    p.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Generate N candidates and pick the best using QA metrics (clip/peak/rms/long-tail). Default 1.",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated models (e.g. Stable Audio Open). If omitted, uses your HF login / env vars.",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Compute device.")
    p.add_argument(
        "--model",
        default="cvssp/audioldm2",
        help="Diffusers pretrained model id (e.g. cvssp/audioldm2).",
    )

    # Stable Audio Open 1.0 (diffusers StableAudioPipeline)
    p.add_argument(
        "--stable-audio-model",
        default="stabilityai/stable-audio-open-1.0",
        help="For engine=stable_audio_open: Hugging Face model id (typically gated; accept terms + login).",
    )
    p.add_argument(
        "--stable-audio-negative-prompt",
        default=None,
        help="For engine=stable_audio_open: optional negative prompt.",
    )
    p.add_argument(
        "--stable-audio-steps",
        type=int,
        default=100,
        help="For engine=stable_audio_open: diffusion steps (default 100).",
    )
    p.add_argument(
        "--stable-audio-guidance-scale",
        type=float,
        default=7.0,
        help="For engine=stable_audio_open: guidance/CFG scale (default 7.0).",
    )
    p.add_argument(
        "--stable-audio-sampler",
        default="auto",
        choices=["auto", "ddim", "deis", "dpmpp", "dpmpp_2m", "euler", "euler_a"],
        help="For engine=stable_audio_open: sampler/scheduler (default auto).",
    )

    # Creature-family fine-tuning (LoRA) for Stable Audio Open (inference-time loading).
    p.add_argument(
        "--creature-family",
        default=None,
        help=(
            "Optional creature family key. Loads LoRA settings from library/creature_families.json and "
            "applies them to engine=stable_audio_open."
        ),
    )
    p.add_argument(
        "--stable-audio-lora-path",
        default=None,
        help="For engine=stable_audio_open: path to LoRA weights (.safetensors / diffusers format).",
    )
    p.add_argument(
        "--stable-audio-lora-scale",
        type=float,
        default=None,
        help="For engine=stable_audio_open: LoRA scale/strength (default 1.0, or family default if set).",
    )
    p.add_argument(
        "--stable-audio-lora-trigger",
        default=None,
        help="For engine=stable_audio_open: optional trigger token/phrase appended to the prompt.",
    )

    # Diffusers multi-band mode (runs multiple generations and recombines bands).
    p.add_argument(
        "--diffusers-multiband",
        action="store_true",
        help="For engine=diffusers: run a multi-band strategy (2-3 model runs) and recombine for cleaner lows/mids/highs. Slower.",
    )
    p.add_argument(
        "--diffusers-mb-mode",
        choices=["auto", "2band", "3band"],
        default="auto",
        help="For engine=diffusers: multi-band mode. 'auto' uses 2band for short sounds and 3band for longer.",
    )
    p.add_argument(
        "--diffusers-mb-low-hz",
        type=float,
        default=250.0,
        help="For engine=diffusers multi-band: low crossover Hz (default 250).",
    )
    p.add_argument(
        "--diffusers-mb-high-hz",
        type=float,
        default=3000.0,
        help="For engine=diffusers multi-band: high crossover Hz (default 3000).",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="rfxgen preset override (coin, laser, explosion, powerup, hit, jump, blip). If omitted, inferred from prompt.",
    )
    p.add_argument(
        "--rfxgen-path",
        default=None,
        help="Path to rfxgen executable (e.g. tools/rfxgen/rfxgen.exe). If omitted, uses PATH.",
    )

    # Sample library engine (ZIP repos)
    p.add_argument(
        "--library-zip",
        action="append",
        default=None,
        help="Path to a ZIP sound library. Can be specified multiple times. Default: .examples/sound libraies/*.zip",
    )
    p.add_argument(
        "--library-pitch-min",
        type=float,
        default=0.85,
        help="For samplelib: min random pitch factor (default 0.85).",
    )
    p.add_argument(
        "--library-pitch-max",
        type=float,
        default=1.20,
        help="For samplelib: max random pitch factor (default 1.20).",
    )
    p.add_argument(
        "--library-mix-count",
        type=int,
        default=1,
        help="For samplelib: number of samples to mix (1 or 2). Default 1.",
    )
    p.add_argument(
        "--library-index",
        default="library/samplelib_index.json",
        help="For samplelib: persistent index cache path. Set to empty string to disable.",
    )

    # Replicate (optional paid API backend)
    p.add_argument("--replicate-model", default=None, help="Replicate model id (e.g. owner/model)")
    p.add_argument("--replicate-token", default=None, help="Replicate API token (or set REPLICATE_API_TOKEN)")
    p.add_argument(
        "--replicate-input-json",
        default=None,
        help="Extra JSON object merged into Replicate input (model-specific).",
    )

    # Minecraft resource pack export
    p.add_argument(
        "--minecraft",
        action="store_true",
        help="Export as a Minecraft resource pack sound (.ogg) + update sounds.json.",
    )
    p.add_argument(
        "--pack-root",
        default="resourcepack",
        help="Root folder for output. For resource packs: the pack root. For Forge: usually <mod>/src/main/resources.",
    )
    p.add_argument(
        "--mc-target",
        choices=["resourcepack", "forge"],
        default="resourcepack",
        help="Export target layout. 'resourcepack' creates pack.mcmeta. 'forge' writes under a mod resources folder.",
    )
    p.add_argument(
        "--namespace",
        default="soundgen",
        help="Minecraft namespace (modid) for assets/<namespace>/... (e.g. yourmodid).",
    )
    p.add_argument(
        "--event",
        default=None,
        help="sounds.json event key (e.g. laser.fire). Full id becomes <namespace>:<event>. Default: generated.<slug>",
    )
    p.add_argument(
        "--sound-path",
        default=None,
        help="Path under assets/<namespace>/sounds/ WITHOUT extension (e.g. sfx/laser_01). Default: generated/<slug>",
    )
    p.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of sound variants to generate and register under the same event (default 1).",
    )
    p.add_argument(
        "--weight",
        type=int,
        default=1,
        help="Weight for each variant entry in sounds.json (default 1).",
    )
    p.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Volume for sounds.json entry (default 1.0).",
    )
    p.add_argument(
        "--pitch",
        type=float,
        default=1.0,
        help="Pitch for sounds.json entry (default 1.0).",
    )
    p.add_argument(
        "--subtitle",
        default=None,
        help="Subtitle text to add to sounds.json and assets/<ns>/lang/en_us.json (optional).",
    )
    p.add_argument(
        "--subtitle-key",
        default=None,
        help="Translation key to use for subtitle (optional). Default: subtitles.<namespace>.<event>",
    )
    p.add_argument(
        "--ogg-quality",
        type=int,
        default=5,
        help="Vorbis VBR quality 0-10 for Minecraft .ogg export (default 5).",
    )
    p.add_argument(
        "--mc-sample-rate",
        type=int,
        default=44100,
        help="Sample rate for Minecraft .ogg export (default 44100).",
    )
    p.add_argument(
        "--mc-channels",
        type=int,
        default=1,
        help="Channels for Minecraft .ogg export: 1 mono or 2 stereo (default 1).",
    )

    # Post-processing / QA
    p.add_argument("--post", action="store_true", help="Enable post-processing (trim/fade/normalize/EQ).")
    p.add_argument(
        "--polish",
        action="store_true",
        help="Enable 'Polish Mode' DSP (denoise + transient shaping + compression + limiter). Implies --post.",
    )
    p.add_argument(
        "--polish-profile",
        choices=["off", *polish_profile_keys()],
        default="off",
        help="Named post/polish profile (AAA-style chain). Only overrides values still at their defaults.",
    )

    # FX chains (v1): named chains + optional JSON definition.
    p.add_argument(
        "--fx-chain",
        choices=["off", *fx_chain_keys()],
        default="off",
        help="Named FX chain preset (shareable post chain). Only overrides values still at their defaults.",
    )
    p.add_argument(
        "--fx-chain-json",
        default=None,
        help="Load an FX chain JSON file. Supports either an args-patch JSON or an effect-list JSON.",
    )

    # Concrete SFX presets (v1): engine + prompt + defaults + FX chain.
    p.add_argument(
        "--sfx-preset",
        default="off",
        help=(
            "Apply a concrete SFX preset from a preset library (engine + prompt + defaults). "
            "Searches library/sfx_presets.json then configs/sfx_presets_v1.example.json. "
            "Use --sfx-preset-file to load a specific JSON file."
        ),
    )
    p.add_argument(
        "--sfx-preset-file",
        default=None,
        help="Path to a JSON preset library file containing a 'presets' list.",
    )
    p.add_argument(
        "--map-controls",
        action="store_true",
        help="Map common prompt keywords to control hints (loud/soft/bright/muffled/clicky/etc).",
    )
    p.add_argument("--no-trim", action="store_true", help="Disable trimming silence (post-processing).")
    p.add_argument("--silence-threshold-db", type=float, default=-40.0, help="Trim threshold in dBFS.")
    p.add_argument("--silence-padding-ms", type=int, default=30, help="Padding kept around trimmed audio.")
    p.add_argument("--fade-ms", type=int, default=8, help="Fade in/out duration to prevent clicks.")
    p.add_argument(
        "--loop",
        action="store_true",
        help="Loop-clean output (ambience): blend the end into the start to reduce seam clicks.",
    )
    p.add_argument(
        "--loop-crossfade-ms",
        type=int,
        default=100,
        help="Loop-clean crossfade window in milliseconds (default 100).",
    )
    p.add_argument(
        "--loop-crossfade-curve",
        choices=["linear", "equal_power", "exponential"],
        default="linear",
        help="Loop-clean crossfade curve shape (linear/equal_power/exponential).",
    )
    p.add_argument(
        "--normalize-rms-db",
        type=float,
        default=-18.0,
        help="Approx loudness target (RMS). Use 0 to disable RMS normalization.",
    )
    p.add_argument("--normalize-peak-db", type=float, default=-1.0, help="Peak cap target in dBFS.")
    p.add_argument("--highpass-hz", type=float, default=40.0, help="Highpass cutoff. Use 0 to disable.")
    p.add_argument("--lowpass-hz", type=float, default=16000.0, help="Lowpass cutoff. Use 0 to disable.")

    # Pro controls (conditioning channels)
    p.add_argument(
        "--pro-preset",
        choices=["off", *pro_preset_keys()],
        default="off",
        help="High-level preset that sets sensible defaults (polish/conditioning/DSP). Only overrides values still at their defaults.",
    )
    p.add_argument(
        "--emotion",
        choices=["neutral", "aggressive", "calm", "scared"],
        default="neutral",
        help="Optional conditioning channel: emotion (nudges DSP + engine params).",
    )
    p.add_argument(
        "--intensity",
        type=float,
        default=0.0,
        help="Optional conditioning channel: intensity 0..1 (0 disables conditioning).",
    )
    p.add_argument(
        "--variation",
        type=float,
        default=0.0,
        help="Optional conditioning channel: variation 0..1 (nudges pitch/micro-variation).",
    )
    p.add_argument(
        "--pitch-contour",
        choices=["flat", "rise", "fall", "updown", "downup"],
        default="flat",
        help="Optional conditioning channel: pitch contour (best-effort; strongest on synth/layered body).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Alias for --seconds (optional conditioning channel: duration).",
    )

    # Pro DSP modules (apply via post-process across all engines)
    p.add_argument("--multiband", action="store_true", help="Enable multi-band cleanup/dynamics stage (post).")
    p.add_argument("--mb-low-hz", type=float, default=250.0, help="Multiband crossover low Hz (default 250).")
    p.add_argument("--mb-high-hz", type=float, default=3000.0, help="Multiband crossover high Hz (default 3000).")
    p.add_argument("--mb-low-gain-db", type=float, default=0.0, help="Multiband low band gain (dB).")
    p.add_argument("--mb-mid-gain-db", type=float, default=0.0, help="Multiband mid band gain (dB).")
    p.add_argument("--mb-high-gain-db", type=float, default=0.0, help="Multiband high band gain (dB).")
    p.add_argument(
        "--mb-comp-threshold-db",
        type=float,
        default=None,
        help="Multiband per-band compressor threshold (dBFS). If omitted, no multiband compression.",
    )
    p.add_argument("--mb-comp-ratio", type=float, default=2.0, help="Multiband compressor ratio (default 2.0).")

    p.add_argument(
        "--creature-size",
        type=float,
        default=0.0,
        help="Creature size control -1..+1 (formant-ish spectral warp). +1 = larger/darker.",
    )
    p.add_argument(
        "--formant-shift",
        type=float,
        default=1.0,
        help="Formant shift factor (0.5..2.0). 1.0 disables. Overrides --creature-size if not 1.0.",
    )

    p.add_argument(
        "--texture-preset",
        choices=["off", "auto", "chitter", "rasp", "buzz", "screech"],
        default="off",
        help="Procedural texture overlay (hybrid granular-ish). Applies in post across engines.",
    )
    p.add_argument("--texture-amount", type=float, default=0.0, help="Texture overlay amount (0..1).")
    p.add_argument("--texture-grain-ms", type=float, default=22.0, help="Texture grain size (ms).")
    p.add_argument("--texture-spray", type=float, default=0.55, help="Texture randomness/density (0..1).")

    p.add_argument(
        "--reverb",
        choices=["off", "room", "cave", "forest", "nether"],
        default="off",
        help="Synthetic convolution reverb preset (post).",
    )
    p.add_argument("--reverb-mix", type=float, default=0.0, help="Reverb wet mix (0..1).")
    p.add_argument("--reverb-time", type=float, default=1.2, help="Reverb tail time in seconds.")

    # Pro Mode overrides (optional). If omitted, polish/conditioning selects conservative values.
    p.add_argument(
        "--denoise-amount",
        type=float,
        default=None,
        help="Override spectral denoise amount 0..1 (omit to use profile/polish defaults).",
    )
    p.add_argument(
        "--transient-attack",
        type=float,
        default=None,
        help="Override transient shaper attack emphasis -1..+1 (omit to use profile/polish defaults).",
    )
    p.add_argument(
        "--transient-sustain",
        type=float,
        default=None,
        help="Override transient shaper sustain emphasis -1..+1 (omit to use profile/polish defaults).",
    )
    p.add_argument(
        "--exciter-amount",
        type=float,
        default=None,
        help="Override harmonic exciter amount 0..1 (omit to use profile defaults).",
    )
    p.add_argument(
        "--compressor-attack-ms",
        type=float,
        default=None,
        help="Override compressor attack (ms) for polish mode (omit to use defaults).",
    )
    p.add_argument(
        "--compressor-release-ms",
        type=float,
        default=None,
        help="Override compressor release (ms) for polish mode (omit to use defaults).",
    )

    p.add_argument(
        "--compressor-threshold-db",
        type=float,
        default=None,
        help="Override compressor threshold (dBFS). If omitted, uses polish/conditioning defaults.",
    )
    p.add_argument(
        "--compressor-ratio",
        type=float,
        default=None,
        help="Override compressor ratio (e.g. 3.0). If omitted, uses default.",
    )
    p.add_argument(
        "--compressor-makeup-db",
        type=float,
        default=None,
        help="Override compressor makeup gain (dB). If omitted, uses polish/conditioning defaults.",
    )
    p.add_argument(
        "--limiter-ceiling-db",
        type=float,
        default=None,
        help="Override limiter ceiling (dBFS). If omitted, uses polish defaults (when enabled).",
    )
    p.add_argument(
        "--compressor-follower-mode",
        choices=["peak", "rms"],
        default="peak",
        help="Advanced: compressor detector mode (peak or RMS-ish).",
    )

    # Advanced polish (optional)
    p.add_argument(
        "--duck-bed",
        action="store_true",
        help="Advanced: duck low/mid bed under transients (offline sidechain-style; post).",
    )
    p.add_argument("--duck-bed-split-hz", type=float, default=1800.0, help="Duck split frequency (Hz).")
    p.add_argument("--duck-bed-amount", type=float, default=0.35, help="Duck amount (0..1).")
    p.add_argument("--duck-bed-attack-ms", type=float, default=2.0, help="Duck attack (ms).")
    p.add_argument("--duck-bed-release-ms", type=float, default=120.0, help="Duck release (ms).")
    p.add_argument(
        "--post-stack",
        default=None,
        help="Advanced: override post chain ordering (comma-separated block keys).",
    )

    p.add_argument(
        "--noise-bed-db",
        type=float,
        default=None,
        help="Add a noise bed at this level (dBFS), e.g. -40. Omit to disable.",
    )
    p.add_argument(
        "--noise-bed-seed",
        type=int,
        default=None,
        help="Noise bed RNG seed (optional). Defaults to post random seed if set, else 0.",
    )

    # Synth engine (DSP) controls
    p.add_argument("--synth-waveform", default="sine", help="synth waveform: sine|square|saw|triangle|noise")
    p.add_argument("--synth-freq", type=float, default=440.0, help="synth base frequency (Hz)")
    p.add_argument("--synth-attack-ms", type=float, default=5.0)
    p.add_argument("--synth-decay-ms", type=float, default=80.0)
    p.add_argument("--synth-sustain", type=float, default=0.35)
    p.add_argument("--synth-release-ms", type=float, default=120.0)
    p.add_argument("--synth-noise-mix", type=float, default=0.05)
    p.add_argument("--synth-drive", type=float, default=0.0)

    # Layered engine controls
    p.add_argument(
        "--layered-preset",
        choices=["auto", "ui", "impact", "whoosh", "creature"],
        default="auto",
        help="layered preset (auto/ui/impact/whoosh/creature)",
    )
    p.add_argument("--layered-transient-ms", type=int, default=110, help="layered transient window length (ms)")
    p.add_argument("--layered-tail-ms", type=int, default=350, help="layered tail window length (ms)")

    p.add_argument("--layered-transient-attack-ms", type=float, default=1.0)
    p.add_argument("--layered-transient-hold-ms", type=float, default=10.0)
    p.add_argument("--layered-transient-decay-ms", type=float, default=90.0)

    p.add_argument("--layered-body-attack-ms", type=float, default=5.0)
    p.add_argument("--layered-body-hold-ms", type=float, default=0.0, help="0 => auto remainder")
    p.add_argument("--layered-body-decay-ms", type=float, default=80.0)

    p.add_argument("--layered-tail-attack-ms", type=float, default=15.0)
    p.add_argument("--layered-tail-hold-ms", type=float, default=0.0, help="0 => auto remainder")
    p.add_argument("--layered-tail-decay-ms", type=float, default=320.0)

    p.add_argument("--layered-duck", type=float, default=0.35, help="Transient ducks body amount (0..1)")
    p.add_argument("--layered-duck-release-ms", type=float, default=90.0)

    p.add_argument(
        "--layered-curve",
        choices=["linear", "exponential"],
        default="linear",
        help="Layered envelope curve shape (applies to all layers).",
    )
    p.add_argument("--layered-transient-tilt", type=float, default=0.0, help="Transient spectral tilt (-1..+1)")
    p.add_argument("--layered-body-tilt", type=float, default=0.0, help="Body spectral tilt (-1..+1)")
    p.add_argument("--layered-tail-tilt", type=float, default=0.0, help="Tail spectral tilt (-1..+1)")

    p.add_argument(
        "--layered-xfade-transient-to-body-ms",
        type=float,
        default=0.0,
        help="Layered: crossfade transient-to-body (ms). 0 disables (legacy add-layers).",
    )
    p.add_argument(
        "--layered-xfade-body-to-tail-ms",
        type=float,
        default=0.0,
        help="Layered: crossfade body-to-tail (ms). 0 disables (legacy add-layers).",
    )
    p.add_argument(
        "--layered-xfade-curve",
        choices=["linear", "equal_power", "exponential"],
        default="linear",
        help="Layered: crossfade curve shape (linear/equal_power/exponential).",
    )

    # Layered per-layer FX (optional). 0 disables.
    p.add_argument("--layered-transient-hp-hz", type=float, default=0.0, help="Layered transient HPF (Hz)")
    p.add_argument("--layered-transient-lp-hz", type=float, default=0.0, help="Layered transient LPF (Hz)")
    p.add_argument("--layered-transient-drive", type=float, default=0.0, help="Layered transient drive (0..1)")
    p.add_argument("--layered-transient-gain-db", type=float, default=0.0, help="Layered transient gain (dB)")

    p.add_argument("--layered-body-hp-hz", type=float, default=0.0, help="Layered body HPF (Hz)")
    p.add_argument("--layered-body-lp-hz", type=float, default=0.0, help="Layered body LPF (Hz)")
    p.add_argument("--layered-body-drive", type=float, default=0.0, help="Layered body drive (0..1)")
    p.add_argument("--layered-body-gain-db", type=float, default=0.0, help="Layered body gain (dB)")

    p.add_argument("--layered-tail-hp-hz", type=float, default=0.0, help="Layered tail HPF (Hz)")
    p.add_argument("--layered-tail-lp-hz", type=float, default=0.0, help="Layered tail LPF (Hz)")
    p.add_argument("--layered-tail-drive", type=float, default=0.0, help="Layered tail drive (0..1)")
    p.add_argument("--layered-tail-gain-db", type=float, default=0.0, help="Layered tail gain (dB)")

    p.add_argument(
        "--layered-family",
        action="store_true",
        help="Family mode: keep --seed as the family seed across variants and apply micro-variation per variant.",
    )
    p.add_argument(
        "--layered-micro-variation",
        type=float,
        default=0.0,
        help="Family mode: subtle deterministic variation amount (0..1).",
    )

    p.add_argument(
        "--layered-source-lock",
        action="store_true",
        help="Pin samplelib transient/tail source selection across variants (sources won't re-pick when body varies).",
    )
    p.add_argument(
        "--layered-source-seed",
        type=int,
        default=None,
        help="Optional override seed used for source lock (transient/tail). If omitted, uses the base --seed.",
    )

    p.add_argument(
        "--layered-granular-preset",
        choices=["off", "auto", "chitter", "rasp", "buzz", "screech"],
        default="off",
        help="Layered: add a procedural granular texture layer mixed into body (off/auto/chitter/rasp/buzz/screech).",
    )
    p.add_argument(
        "--layered-granular-amount",
        type=float,
        default=0.0,
        help="Layered: granular texture mix amount (0..1).",
    )
    p.add_argument(
        "--layered-granular-grain-ms",
        type=float,
        default=28.0,
        help="Layered: granular texture grain size (ms).",
    )
    p.add_argument(
        "--layered-granular-spray",
        type=float,
        default=0.35,
        help="Layered: granular texture randomness/density (0..1).",
    )

    p.add_argument("--out", default="outputs/out.wav", help="Output WAV path.")

    # Export format options (non-Minecraft)
    p.add_argument(
        "--out-format",
        choices=["wav", "flac", "mp3", "ogg"],
        default=None,
        help="Non-Minecraft: output format. If omitted, inferred from --out extension (default wav).",
    )
    p.add_argument(
        "--out-sample-rate",
        type=int,
        default=None,
        help="Non-Minecraft: resample output to this sample rate (e.g. 44100).",
    )
    p.add_argument(
        "--wav-subtype",
        choices=["PCM_16", "PCM_24", "FLOAT"],
        default="PCM_16",
        help="Non-Minecraft WAV encoding subtype (default PCM_16).",
    )
    p.add_argument(
        "--mp3-bitrate",
        default="192k",
        help="Non-Minecraft MP3 bitrate for --out-format mp3 (default 192k).",
    )
    return p


def _slug_from_prompt(prompt: str) -> str:
    import re

    s = prompt.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:48] or "sound"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Apply concrete SFX preset first so it can set engine/prompt/fx-chain,
    # then let the FX chain patch remaining defaults, then pro presets.
    try:
        sfx_preset_obj, sfx_preset_lib = apply_sfx_preset(
            preset_key=str(getattr(args, "sfx_preset", "off")),
            preset_file=getattr(args, "sfx_preset_file", None),
            args=args,
            parser=parser,
        )
    except ValueError as e:
        parser.error(str(e))
        raise

    if not getattr(args, "prompt", None):
        parser.error("Either --prompt or --sfx-preset is required.")

    # Apply FX chain first so later presets only fill remaining defaults,
    # and explicit user flags are never clobbered.
    apply_fx_chain(chain_key=str(getattr(args, "fx_chain", "off")), chain_json=getattr(args, "fx_chain_json", None), args=args, parser=parser)

    # Apply pro preset after parsing so we can compare against argparse defaults.
    apply_pro_preset(preset_key=str(args.pro_preset), args=args, parser=parser)
    apply_polish_profile(profile_key=str(getattr(args, "polish_profile", "off")), args=args, parser=parser)

    if args.duration is not None:
        args.seconds = float(args.duration)

    out_path = Path(args.out)
    slug = _slug_from_prompt(args.prompt)

    def _infer_out_format() -> str:
        if args.out_format:
            return str(args.out_format)
        suf = out_path.suffix.lower().lstrip(".")
        if suf in {"wav", "flac", "mp3", "ogg"}:
            return suf
        return "wav"

    def _normalize_out_path(fmt: str) -> Path:
        fmt = fmt.lower()
        if out_path.suffix.lower().lstrip(".") == fmt:
            return out_path
        if out_path.suffix:
            return out_path.with_suffix("." + fmt)
        return Path(str(out_path) + "." + fmt)

    hints = map_prompt_to_controls(args.prompt) if args.map_controls else None

    def _pp_params(*, post_seed: int | None, prompt_hint: str | None) -> PostProcessParams:
        rms = float(args.normalize_rms_db)
        hp = float(args.highpass_hz)
        lp = float(args.lowpass_hz)

        if hints is not None:
            if hints.loudness_rms_db is not None:
                rms = float(hints.loudness_rms_db)
            if hints.highpass_hz is not None:
                hp = float(hints.highpass_hz)
            if hints.lowpass_hz is not None:
                lp = float(hints.lowpass_hz)

        # Conditioning
        intensity = float(np.clip(float(args.intensity), 0.0, 1.0))
        variation = float(np.clip(float(args.variation), 0.0, 1.0))
        emotion = str(args.emotion or "neutral").strip().lower()

        # Emotion/intensity nudges for polish defaults (only if user opted in via intensity > 0)
        denoise = (0.25 if args.polish else 0.0)
        transient = (0.25 if args.polish else 0.0)
        transient_sustain = 0.0
        comp_thr = (-18.0 if args.polish else None)
        comp_makeup = (3.0 if args.polish else 0.0)
        limiter = (-1.0 if args.polish else None)

        if intensity > 0.0:
            transient = float(np.clip(transient + 0.35 * intensity, -1.0, 1.0))
            # More intensity => slightly more compression
            comp_thr = float(-18.0 - 8.0 * intensity)
            comp_makeup = float(comp_makeup + 2.0 * intensity)
            # A touch of denoise for aggressive textures
            denoise = float(np.clip(denoise + (0.10 * intensity), 0.0, 1.0))

            if emotion == "aggressive":
                transient = float(np.clip(transient + 0.20, -1.0, 1.0))
            elif emotion == "calm":
                transient = float(np.clip(transient - 0.15, -1.0, 1.0))
            elif emotion == "scared":
                transient = float(np.clip(transient + 0.10, -1.0, 1.0))

        # Optional Pro Mode overrides (win over conditioning/polish defaults).
        denoise_ovr = getattr(args, "denoise_amount", None)
        if denoise_ovr is not None:
            denoise = float(np.clip(float(denoise_ovr), 0.0, 1.0))

        trans_attack_ovr = getattr(args, "transient_attack", None)
        if trans_attack_ovr is not None:
            transient = float(np.clip(float(trans_attack_ovr), -1.0, 1.0))

        trans_sustain_ovr = getattr(args, "transient_sustain", None)
        if trans_sustain_ovr is not None:
            transient_sustain = float(np.clip(float(trans_sustain_ovr), -1.0, 1.0))

        exciter = 0.0
        exciter_ovr = getattr(args, "exciter_amount", None)
        if exciter_ovr is not None:
            exciter = float(np.clip(float(exciter_ovr), 0.0, 1.0))

        # Compressor/Limiter explicit overrides (win over polish defaults).
        comp_thr_ovr = getattr(args, "compressor_threshold_db", None)
        if comp_thr_ovr is not None:
            comp_thr = float(comp_thr_ovr)
        comp_ratio_ovr = getattr(args, "compressor_ratio", None)
        comp_makeup_ovr = getattr(args, "compressor_makeup_db", None)
        limiter_ovr = getattr(args, "limiter_ceiling_db", None)
        if limiter_ovr is not None:
            limiter = float(limiter_ovr)

        # Texture: if user didn't set explicit texture controls, let conditioning gently enable it.
        texture_preset = str(args.texture_preset)
        texture_amount = float(args.texture_amount)
        if intensity > 0.0 and (texture_preset == "off") and texture_amount <= 0.0:
            if emotion in {"scared", "aggressive"}:
                texture_preset = "auto"
                texture_amount = float(np.clip(0.18 + 0.25 * intensity + 0.10 * variation, 0.0, 1.0))

        # Reverb: keep off unless explicitly requested or calm emotion.
        reverb_preset = str(args.reverb)
        reverb_mix = float(args.reverb_mix)
        reverb_time = float(args.reverb_time)
        if intensity > 0.0 and reverb_preset == "off" and reverb_mix <= 0.0 and emotion == "calm":
            reverb_preset = "room"
            reverb_mix = float(np.clip(0.06 + 0.10 * intensity, 0.0, 0.35))

        return PostProcessParams(
            trim_silence=(not args.no_trim),
            silence_threshold_db=float(args.silence_threshold_db),
            silence_padding_ms=int(args.silence_padding_ms),
            fade_ms=int(args.fade_ms),
            normalize_rms_db=(None if float(rms) == 0.0 else float(rms)),
            normalize_peak_db=float(args.normalize_peak_db),
            highpass_hz=(None if float(hp) == 0.0 else float(hp)),
            lowpass_hz=(None if float(lp) == 0.0 else float(lp)),
            # Polish mode DSP (conservative defaults + conditioning)
            denoise_strength=float(denoise),
            transient_amount=float(transient),
            transient_sustain=float(transient_sustain),
            transient_split_hz=1200.0,
            exciter_amount=float(exciter),
            multiband=bool(args.multiband or (args.polish and intensity > 0.0)),
            multiband_low_hz=float(args.mb_low_hz),
            multiband_high_hz=float(args.mb_high_hz),
            multiband_low_gain_db=float(args.mb_low_gain_db),
            multiband_mid_gain_db=float(args.mb_mid_gain_db),
            multiband_high_gain_db=float(args.mb_high_gain_db),
            multiband_comp_threshold_db=(
                float(args.mb_comp_threshold_db)
                if args.mb_comp_threshold_db is not None
                else (-24.0 if (args.polish and intensity > 0.0) else None)
            ),
            multiband_comp_ratio=float(args.mb_comp_ratio),
            formant_shift=float(args.formant_shift),
            creature_size=float(args.creature_size),
            texture_preset=str(texture_preset),
            texture_amount=float(texture_amount),
            texture_grain_ms=float(args.texture_grain_ms),
            texture_spray=float(args.texture_spray),
            reverb_preset=str(reverb_preset),
            reverb_mix=float(reverb_mix),
            reverb_time_s=float(reverb_time),
            random_seed=(int(post_seed) if post_seed is not None else None),
            prompt_hint=(str(prompt_hint) if prompt_hint else None),
            compressor_threshold_db=(float(comp_thr) if comp_thr is not None else None),
            compressor_ratio=float(comp_ratio_ovr) if comp_ratio_ovr is not None else (4.0 if args.polish else 4.0),
            compressor_attack_ms=float(getattr(args, "compressor_attack_ms", 5.0) if getattr(args, "compressor_attack_ms", None) is not None else (5.0 if args.polish else 5.0)),
            compressor_release_ms=float(getattr(args, "compressor_release_ms", 90.0) if getattr(args, "compressor_release_ms", None) is not None else (90.0 if args.polish else 80.0)),
            compressor_makeup_db=float(comp_makeup_ovr) if comp_makeup_ovr is not None else float(comp_makeup),
            limiter_ceiling_db=(float(limiter) if limiter is not None else None),
            loop_clean=bool(getattr(args, "loop", False)),
            loop_crossfade_ms=int(getattr(args, "loop_crossfade_ms", 100)),
            loop_crossfade_curve=str(getattr(args, "loop_crossfade_curve", "linear")),

            duck_bed=bool(getattr(args, "duck_bed", False)),
            duck_bed_split_hz=float(getattr(args, "duck_bed_split_hz", 1800.0)),
            duck_bed_amount=float(getattr(args, "duck_bed_amount", 0.35)),
            duck_bed_attack_ms=float(getattr(args, "duck_bed_attack_ms", 2.0)),
            duck_bed_release_ms=float(getattr(args, "duck_bed_release_ms", 120.0)),
            post_stack=(str(args.post_stack) if getattr(args, "post_stack", None) else None),
            compressor_follower_mode=str(getattr(args, "compressor_follower_mode", "peak")),

            noise_bed_db=(float(args.noise_bed_db) if getattr(args, "noise_bed_db", None) is not None else None),
            noise_bed_seed=(int(args.noise_bed_seed) if getattr(args, "noise_bed_seed", None) is not None else None),
        )

    def _qa_info(audio: np.ndarray, sr: int) -> str:
        m = compute_metrics(audio, sr)
        flags: list[str] = []
        if m.clipped:
            flags.append("CLIPPING")
        if detect_long_tail(audio, sr):
            flags.append("LONG_TAIL")
        flag_s = (" " + " ".join(flags)) if flags else ""
        return f"qa: peak={m.peak:.3f} rms={m.rms:.3f}{flag_s}".strip()

    def _postprocess_fn_for_engine(engine: str, *, post_seed: int | None, prompt_hint: str | None):
        if args.post or args.polish:
            def _pp_apply(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
                processed, rep = post_process_audio(audio, sr, _pp_params(post_seed=post_seed, prompt_hint=prompt_hint))
                info = _qa_info(processed, sr)
                return processed, f"post: trimmed={rep.trimmed} {info}".strip()

            return _pp_apply

        if engine in {"diffusers", "synth", "layered"}:
            def _qa_only(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
                return audio, _qa_info(audio, sr)

            return _qa_only

        return None

    def export_if_minecraft(wav_path: Path, *, sound_path: str) -> Path | None:
        if not args.minecraft:
            return None

        pack_root = Path(args.pack_root)
        namespace = args.namespace
        effective_sound_path = args.sound_path or sound_path
        event = args.event or f"generated.{slug}"
        write_pack_mcmeta = args.mc_target == "resourcepack"

        ogg_path = export_wav_to_minecraft_pack(
            wav_path,
            pack_root=pack_root,
            namespace=namespace,
            event=event,
            sound_path=effective_sound_path,
            weight=int(args.weight),
            volume=float(args.volume),
            pitch=float(args.pitch),
            subtitle=args.subtitle,
            subtitle_key=args.subtitle_key,
            ogg_quality=args.ogg_quality,
            sample_rate=args.mc_sample_rate,
            channels=args.mc_channels,
            description="S-NDB-UND pack",
            write_pack_mcmeta=write_pack_mcmeta,
        )
        print(f"Minecraft export: {ogg_path}")
        print(f"Minecraft playsound id: {namespace}:{event}")
        return ogg_path

    def _write_credits(*, sound_path: str, wav_path: Path | None, ogg_path: Path | None, generated: GeneratedWav) -> None:
        data: dict[str, object] = {
            "engine": str(args.engine),
            "prompt": str(args.prompt),
            "emotion": str(args.emotion),
            "intensity": float(args.intensity),
            "variation": float(args.variation),
            "pitch_contour": str(args.pitch_contour),
            "sound_path": str(sound_path),
            **{k: v for k, v in generated.credits_extra.items() if v is not None},
        }
        if sfx_preset_obj is not None:
            data["sfx_preset"] = str(sfx_preset_obj.name)
        if sfx_preset_lib is not None:
            data["sfx_preset_library"] = str(sfx_preset_lib)
        data["pro_preset"] = str(getattr(args, "pro_preset", "off"))
        data["polish_profile"] = str(getattr(args, "polish_profile", "off"))
        data["loop_clean"] = bool(getattr(args, "loop", False))
        data["loop_crossfade_ms"] = int(getattr(args, "loop_crossfade_ms", 100))
        if generated.sources:
            data["sources"] = list(generated.sources)

        if wav_path is not None and not args.minecraft:
            write_sidecar_credits(Path(wav_path), data)
        if ogg_path is not None and args.minecraft:
            pack_root = Path(args.pack_root)
            namespace = args.namespace
            event = args.event or f"generated.{slug}"
            upsert_pack_credits(
                pack_root=pack_root,
                namespace=namespace,
                event=event,
                sound_path=sound_path,
                credits=data,
            )
    # samplelib defaults
    default_zips = sorted(Path(".examples").joinpath("sound libraies").glob("*.zip"))
    zip_args = args.library_zip if args.library_zip else [str(p) for p in default_zips]
    if args.engine == "samplelib" and not zip_args:
        raise FileNotFoundError(
            "No --library-zip provided and no default zips found at .examples/sound libraies/*.zip"
        )

    index_path = None if str(args.library_index).strip() == "" else Path(str(args.library_index))

    def _gen_one(*, out_wav: Path, seed: int | None, variant_index: int = 0, source_seed: int | None = None) -> GeneratedWav:
        # Apply hint overrides for synth controls.
        synth_attack = float(hints.attack_ms) if hints and hints.attack_ms is not None else float(args.synth_attack_ms)
        synth_release = float(hints.release_ms) if hints and hints.release_ms is not None else float(args.synth_release_ms)
        synth_pitch_min = float(hints.pitch_min) if hints and hints.pitch_min is not None else 0.90
        synth_pitch_max = float(hints.pitch_max) if hints and hints.pitch_max is not None else 1.10
        synth_lp = float(hints.lowpass_hz) if hints and hints.lowpass_hz is not None else 16000.0
        synth_hp = float(hints.highpass_hz) if hints and hints.highpass_hz is not None else 30.0
        synth_drive = float(hints.drive) if hints and hints.drive is not None else float(args.synth_drive)

        # Conditioning nudges across engines.
        intensity = float(np.clip(float(args.intensity), 0.0, 1.0))
        variation = float(np.clip(float(args.variation), 0.0, 1.0))
        emotion = str(args.emotion or "neutral").strip().lower()

        if intensity > 0.0:
            synth_drive = float(np.clip(synth_drive + 0.55 * intensity, 0.0, 1.0))
            if emotion == "aggressive":
                synth_drive = float(np.clip(synth_drive + 0.20, 0.0, 1.0))
            if emotion == "calm":
                synth_lp = float(max(2000.0, synth_lp - 3500.0 * intensity))

        # Variation widens pitch randomization for synth/samplelib and increases layered micro variation.
        v_pitch = 0.08 * variation
        synth_pitch_min = float(np.clip(synth_pitch_min - v_pitch, 0.60, 1.20))
        synth_pitch_max = float(np.clip(synth_pitch_max + v_pitch, 0.80, 1.60))

        # Pitch contour (best-effort): we encode it into the prompt so AI engines can respond,
        # and we also nudge synth pitch range.
        contour = str(args.pitch_contour or "flat").strip().lower()
        prompt2 = str(args.prompt)
        if intensity > 0.0 and contour != "flat":
            prompt2 = f"{prompt2} pitch {contour}".strip()
            if contour in {"rise", "updown"}:
                synth_pitch_max = float(np.clip(synth_pitch_max + 0.10, 0.80, 1.80))
            if contour in {"fall", "downup"}:
                synth_pitch_min = float(np.clip(synth_pitch_min - 0.10, 0.50, 1.20))

        layered_micro = float(args.layered_micro_variation)
        if variation > 0.0:
            layered_micro = float(np.clip(layered_micro + 0.60 * variation, 0.0, 1.0))

        stable_audio_lora_path = args.stable_audio_lora_path
        stable_audio_lora_scale: float = float(args.stable_audio_lora_scale) if args.stable_audio_lora_scale is not None else 1.0
        stable_audio_lora_trigger = args.stable_audio_lora_trigger

        if args.creature_family:
            from .creature_families import apply_trigger, resolve_creature_family

            fam = resolve_creature_family(str(args.creature_family))
            # Only apply to stable_audio_open (for now). We still allow other engines to run unchanged.
            stable_audio_lora_path = stable_audio_lora_path or fam.lora_path
            if args.stable_audio_lora_scale is None:
                stable_audio_lora_scale = float(fam.scale)
            stable_audio_lora_trigger = stable_audio_lora_trigger or fam.trigger
            if args.engine == "stable_audio_open" and fam.negative_prompt and not args.stable_audio_negative_prompt:
                args.stable_audio_negative_prompt = fam.negative_prompt
            if args.engine == "stable_audio_open" and stable_audio_lora_trigger:
                prompt2 = apply_trigger(prompt2, stable_audio_lora_trigger)

        return generate_wav(
            args.engine,
            prompt=prompt2,
            seconds=float(args.seconds),
            seed=seed,
            out_wav=out_wav,
            candidates=max(1, int(args.candidates or 1)),
            postprocess_fn=_postprocess_fn_for_engine(
                args.engine,
                post_seed=(int(seed) if seed is not None else None),
                prompt_hint=prompt2,
            ),
            device=str(args.device),
            model=str(args.model),
            stable_audio_model=str(args.stable_audio_model or "stabilityai/stable-audio-open-1.0"),
            stable_audio_negative_prompt=(args.stable_audio_negative_prompt or None),
            stable_audio_steps=int(args.stable_audio_steps),
            stable_audio_guidance_scale=float(args.stable_audio_guidance_scale),
            stable_audio_sampler=(None if str(getattr(args, "stable_audio_sampler", "auto")).strip().lower() in {"", "auto", "default"} else str(getattr(args, "stable_audio_sampler"))),
            stable_audio_hf_token=(args.hf_token or None),
            stable_audio_lora_path=(stable_audio_lora_path or None),
            stable_audio_lora_scale=float(stable_audio_lora_scale),
            stable_audio_lora_trigger=(stable_audio_lora_trigger or None),
            diffusers_multiband=bool(args.diffusers_multiband),
            diffusers_multiband_mode=str(args.diffusers_mb_mode or "auto"),
            diffusers_multiband_low_hz=float(args.diffusers_mb_low_hz),
            diffusers_multiband_high_hz=float(args.diffusers_mb_high_hz),
            preset=args.preset,
            rfxgen_path=(Path(args.rfxgen_path) if args.rfxgen_path else None),
            replicate_model=args.replicate_model,
            replicate_token=args.replicate_token,
            replicate_input_json=args.replicate_input_json,
            library_zips=tuple(Path(p) for p in zip_args),
            library_pitch_min=float(np.clip(float(args.library_pitch_min) - 0.10 * variation, 0.50, 1.20)),
            library_pitch_max=float(np.clip(float(args.library_pitch_max) + 0.10 * variation, 0.80, 2.00)),
            library_mix_count=int(args.library_mix_count),
            library_index_path=index_path,
            synth_waveform=str(args.synth_waveform),
            synth_freq_hz=float(args.synth_freq),
            synth_attack_ms=synth_attack,
            synth_decay_ms=float(args.synth_decay_ms),
            synth_sustain_level=float(args.synth_sustain),
            synth_release_ms=synth_release,
            synth_noise_mix=float(args.synth_noise_mix),
            synth_drive=synth_drive,
            synth_pitch_min=synth_pitch_min,
            synth_pitch_max=synth_pitch_max,
            synth_lowpass_hz=synth_lp,
            synth_highpass_hz=synth_hp,
            sample_rate=44100,

            layered_preset=str(args.layered_preset),
            layered_preset_lock=True,
            layered_variant_index=int(variant_index),
            layered_micro_variation=float(layered_micro),
            layered_env_curve_shape=str(args.layered_curve),
            layered_transient_tilt=float(args.layered_transient_tilt),
            layered_body_tilt=float(args.layered_body_tilt),
            layered_tail_tilt=float(args.layered_tail_tilt),
            layered_xfade_transient_to_body_ms=float(args.layered_xfade_transient_to_body_ms),
            layered_xfade_body_to_tail_ms=float(args.layered_xfade_body_to_tail_ms),
            layered_xfade_curve_shape=str(getattr(args, "layered_xfade_curve", "linear")),
            layered_transient_hp_hz=float(args.layered_transient_hp_hz),
            layered_transient_lp_hz=float(args.layered_transient_lp_hz),
            layered_transient_drive=float(args.layered_transient_drive),
            layered_transient_gain_db=float(args.layered_transient_gain_db),
            layered_body_hp_hz=float(args.layered_body_hp_hz),
            layered_body_lp_hz=float(args.layered_body_lp_hz),
            layered_body_drive=float(args.layered_body_drive),
            layered_body_gain_db=float(args.layered_body_gain_db),
            layered_tail_hp_hz=float(args.layered_tail_hp_hz),
            layered_tail_lp_hz=float(args.layered_tail_lp_hz),
            layered_tail_drive=float(args.layered_tail_drive),
            layered_tail_gain_db=float(args.layered_tail_gain_db),
            layered_source_lock=bool(args.layered_source_lock),
            layered_source_seed=(int(source_seed) if source_seed is not None else None),
            layered_granular_preset=str(args.layered_granular_preset),
            layered_granular_amount=float(args.layered_granular_amount),
            layered_granular_grain_ms=float(args.layered_granular_grain_ms),
            layered_granular_spray=float(args.layered_granular_spray),
            layered_transient_ms=int(args.layered_transient_ms),
            layered_tail_ms=int(args.layered_tail_ms),
            layered_transient_attack_ms=float(args.layered_transient_attack_ms),
            layered_transient_hold_ms=float(args.layered_transient_hold_ms),
            layered_transient_decay_ms=float(args.layered_transient_decay_ms),
            layered_body_attack_ms=float(args.layered_body_attack_ms),
            layered_body_hold_ms=float(args.layered_body_hold_ms),
            layered_body_decay_ms=float(args.layered_body_decay_ms),
            layered_tail_attack_ms=float(args.layered_tail_attack_ms),
            layered_tail_hold_ms=float(args.layered_tail_hold_ms),
            layered_tail_decay_ms=float(args.layered_tail_decay_ms),
            layered_duck_amount=float(args.layered_duck),
            layered_duck_release_ms=float(args.layered_duck_release_ms),
        )

    if args.minecraft:
        with tempfile.TemporaryDirectory() as tmp:
            base_sound_path = args.sound_path or f"generated/{slug}"
            variants = max(1, int(args.variants))
            base_seed = args.seed if args.seed is not None else 1337

            for i in range(variants):
                suffix = f"_{i+1:02d}" if variants > 1 else ""
                tmp_wav = Path(tmp) / f"{args.engine}{suffix}.wav"
                if args.engine in {"diffusers", "stable_audio_open", "samplelib", "synth"}:
                    seed_i = int(base_seed) + i
                elif args.engine == "layered":
                    seed_i = int(base_seed) if args.layered_family else (int(base_seed) + i)
                else:
                    seed_i = None

                source_seed_i: int | None = None
                if args.engine == "layered" and args.layered_source_lock:
                    source_seed_i = int(args.layered_source_seed) if args.layered_source_seed is not None else int(base_seed)
                # Preserve previous behavior: replicate writes to --out even when --minecraft.
                if args.engine == "replicate":
                    tmp_wav = out_path
                generated = _gen_one(out_wav=tmp_wav, seed=seed_i, variant_index=i, source_seed=source_seed_i)
                if generated.post_info:
                    print(generated.post_info)

                sp = f"{base_sound_path}{suffix}"
                ogg = export_if_minecraft(generated.wav_path, sound_path=sp)
                _write_credits(sound_path=sp, wav_path=None, ogg_path=ogg, generated=generated)
        return 0

    # Non-minecraft: generate WAV then optionally resample/convert.
    fmt = _infer_out_format()
    final_path = _normalize_out_path(fmt)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tmp_wav = tmp_dir / "generated.wav"
        source_seed_single: int | None = None
        if args.engine == "layered" and args.layered_source_lock:
            if args.layered_source_seed is not None:
                source_seed_single = int(args.layered_source_seed)
            elif args.seed is not None:
                source_seed_single = int(args.seed)
            else:
                source_seed_single = 1337

        generated = _gen_one(out_wav=tmp_wav, seed=args.seed, variant_index=0, source_seed=source_seed_single)
        if generated.post_info:
            print(generated.post_info)

        src_wav = Path(generated.wav_path)
        out_sr = int(args.out_sample_rate) if args.out_sample_rate else None

        # If final is wav, rewrite with optional resample + subtype.
        if fmt == "wav":
            audio, sr = read_wav_mono(src_wav)
            if out_sr is not None and out_sr > 0 and out_sr != sr:
                audio = resample_poly(audio, out_sr, sr).astype(np.float32, copy=False)
                sr = out_sr
            write_wav(final_path, audio, sr, subtype=str(args.wav_subtype))
        else:
            # Convert via ffmpeg (optionally resampling).
            convert_audio_with_ffmpeg(
                src_wav,
                final_path,
                sample_rate=out_sr,
                channels=1,
                out_format=fmt,
                ogg_quality=int(args.ogg_quality),
                mp3_bitrate=str(args.mp3_bitrate),
            )

        print(f"Wrote {final_path}")

        sp = str(args.sound_path or f"generated/{slug}")
        _write_credits(sound_path=sp, wav_path=final_path, ogg_path=None, generated=generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
