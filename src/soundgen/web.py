from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from .audiogen_backend import GenerationParams, generate_audio
from .io_utils import read_wav_mono, write_wav
from .rfxgen_backend import SUPPORTED_PRESETS, RfxGenParams, generate_with_rfxgen
from .minecraft import export_wav_to_minecraft_pack
from .postprocess import PostProcessParams, post_process_audio
from .qa import compute_metrics, detect_long_tail
from .qa_viz import spectrogram_image, waveform_image


def _generate(
    engine: str,
    prompt: str,
    seconds: float,
    seed: int | None,
    device: str,
    model: str,
    preset: str,
    rfxgen_path: str,
    export_minecraft: bool,
    mc_target: str,
    pack_root: str,
    namespace: str,
    event: str,
    sound_path: str,
    subtitle: str,
    variants: int,
    weight: int,
    volume: float,
    pitch: float,
    ogg_quality: int,

    post: bool,
) -> tuple[str, str, str, object, object]:
    out_path = Path("outputs") / "web.wav"

    def _pp_params() -> PostProcessParams:
        return PostProcessParams()

    def _qa_info(audio: np.ndarray, sr: int) -> str:
        m = compute_metrics(audio, sr)
        flags: list[str] = []
        if m.clipped:
            flags.append("CLIPPING")
        if detect_long_tail(audio, sr):
            flags.append("LONG_TAIL")
        flag_s = (" " + " ".join(flags)) if flags else ""
        return f"qa: {m.seconds:.2f}s @ {m.sample_rate}Hz peak={m.peak:.3f} rms={m.rms:.3f}{flag_s}".strip()

    def _maybe_postprocess_array(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
        if not post:
            return audio, _qa_info(audio, sr)
        processed, rep = post_process_audio(audio, sr, _pp_params())
        return processed, f"post: trimmed={rep.trimmed} {_qa_info(processed, sr)}".strip()

    def _minecraft_export(wav_path: Path, sp: str) -> tuple[str, str]:
        ogg_path = export_wav_to_minecraft_pack(
            wav_path,
            pack_root=Path(pack_root or "resourcepack"),
            namespace=(namespace or "soundgen"),
            event=(event or "generated.web"),
            sound_path=(sound_path or sp),
            subtitle=(subtitle or None),
            ogg_quality=int(ogg_quality),
            weight=max(1, int(weight)),
            volume=float(volume),
            pitch=float(pitch),
            write_pack_mcmeta=(mc_target == "resourcepack"),
        )
        playsound = f"/playsound {(namespace or 'soundgen')}:{(event or 'generated.web')} master @s"
        return str(ogg_path), playsound

    if engine == "rfxgen":
        # For Minecraft export we generate variants by calling rfxgen multiple times.
        v = max(1, int(variants))
        last_file: Path | None = None
        info = ""
        wav_img = None
        spec_img = None
        playsound = ""

        for i in range(v):
            suffix = f"_{i+1:02d}" if v > 1 else ""
            wav = Path("outputs") / f"web_rfxgen{suffix}.wav"
            rfx_params = RfxGenParams(
                prompt=prompt,
                out_path=wav,
                preset=(preset or None),
                rfxgen_path=Path(rfxgen_path) if rfxgen_path else None,
            )
            last_file = generate_with_rfxgen(rfx_params)

            a, sr = read_wav_mono(last_file)
            a, info = _maybe_postprocess_array(a, sr)
            if post:
                write_wav(last_file, a, sr)
            wav_img = waveform_image(a, sr)
            spec_img = spectrogram_image(a, sr)

            if export_minecraft:
                sp = (sound_path or f"generated/web{suffix}")
                ogg, playsound = _minecraft_export(last_file, sp)
        if export_minecraft:
            # Return last ogg for download
            return ogg, playsound, info, wav_img, spec_img
        return str(last_file), playsound, info, wav_img, spec_img

    params = GenerationParams(
        prompt=prompt,
        seconds=float(seconds),
        seed=seed,
        device=device,
        model=model,
    )
    audio, sr = generate_audio(params)
    audio, info = _maybe_postprocess_array(audio, sr)
    write_wav(out_path, audio, sr)
    wav_img = waveform_image(audio, sr)
    spec_img = spectrogram_image(audio, sr)
    if export_minecraft:
        sp = (sound_path or "generated/web")
        ogg, playsound = _minecraft_export(out_path, sp)
        return ogg, playsound, info, wav_img, spec_img
    return str(out_path), "", info, wav_img, spec_img


def main() -> None:
    with gr.Blocks(title="Sound Generator") as demo:
        gr.Markdown("# Prompt → Sound Effect (WAV)")
        engine = gr.Radio(["diffusers", "rfxgen"], value="diffusers", label="Engine")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. laser zap, sci-fi blaster, short")
        with gr.Row():
            seconds = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Seconds")
            seed = gr.Number(value=None, precision=0, label="Seed (optional)")
        with gr.Row():
            device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")
            model = gr.Dropdown(
                ["cvssp/audioldm2"],
                value="cvssp/audioldm2",
                label="Model",
            )
        with gr.Row():
            preset = gr.Dropdown(list(SUPPORTED_PRESETS), value="blip", label="rfxgen preset")
            rfxgen_path = gr.Textbox(value="", label="rfxgen path (optional)", placeholder="e.g. tools/rfxgen/rfxgen.exe")

        post = gr.Checkbox(value=True, label="Post-process (trim/fade/normalize/EQ)")

        gr.Markdown("## Minecraft export (1.20.1)")
        export_minecraft = gr.Checkbox(value=False, label="Export to Minecraft (.ogg + sounds.json)")
        with gr.Row():
            mc_target = gr.Dropdown(["resourcepack", "forge"], value="resourcepack", label="Target")
            pack_root = gr.Textbox(value="resourcepack", label="Pack/Resources root")
        with gr.Row():
            namespace = gr.Textbox(value="soundgen", label="Namespace (modid)")
            event = gr.Textbox(value="generated.web", label="Event (sounds.json key)")
        with gr.Row():
            sound_path = gr.Textbox(value="generated/web", label="Sound path (under sounds/, no extension)")
            subtitle = gr.Textbox(value="", label="Subtitle (optional)")
        with gr.Row():
            variants = gr.Slider(1, 10, value=1, step=1, label="Variants")
            weight = gr.Slider(1, 20, value=1, step=1, label="Weight")
            volume = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Volume")
            pitch = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Pitch")
            ogg_quality = gr.Slider(0, 10, value=5, step=1, label="OGG quality")

        btn = gr.Button("Generate")
        out_file = gr.File(label="Generated file")
        playsound_cmd = gr.Textbox(label="Minecraft playsound", interactive=False)
        info = gr.Textbox(label="QA / post-process", interactive=False)
        with gr.Row():
            wave = gr.Image(label="Waveform", type="pil")
            spec = gr.Image(label="Spectrogram", type="pil")

        btn.click(
            fn=_generate,
            inputs=[
                engine,
                prompt,
                seconds,
                seed,
                device,
                model,
                preset,
                rfxgen_path,
                export_minecraft,
                mc_target,
                pack_root,
                namespace,
                event,
                sound_path,
                subtitle,
                variants,
                weight,
                volume,
                pitch,
                ogg_quality,
                post,
            ],
            outputs=[out_file, playsound_cmd, info, wave, spec],
        )

    demo.launch()


if __name__ == "__main__":
    main()
