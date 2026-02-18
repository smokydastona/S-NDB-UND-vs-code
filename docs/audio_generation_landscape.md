# Audio generation landscape — takeaways for S‑NDB‑UND

This note is a practical “what can we learn + apply” summary from a few popular audio-gen repos.

Scope: S‑NDB‑UND is a modder-first SFX tool; we care about **short SFX quality**, **repeatability**, **batch throughput**, **license clarity**, and **editor-friendly repair workflows**.

## Stable Audio Open (SaladTechnologies/stable-audio-open)

What it shows:
- A minimal **Gradio + Docker** path for serving Stable Audio Open with GPU.
- Operational detail: download weights at runtime via `HF_TOKEN`, cache mount for cold-start speed.

What to apply here:
- **Optional container path** for web UI deployments (especially for friends/teams) without forcing it as a default.
- Treat model weights as an external artifact: user-supplied token + cache location config.

Notes:
- This repo is primarily deployment glue; the deeper “model knobs” live in the underlying Stable Audio Open tooling/model card.

## AudioCraft (facebookresearch/audiocraft)

What it provides:
- A full stack for generation research and production-ish inference:
  - **AudioGen** (text-to-sound) and **MusicGen** (text-to-music)
  - **EnCodec** (neural audio codec / tokenizer)
  - Training pipeline docs (configs, recipes)
- Clear license split:
  - Code is permissive (MIT)
  - **Model weights are CC-BY-NC 4.0** (non-commercial)

What to apply here:
- Architecture: treat “codec/tokenizer” as a first-class building block if we ever add LM-based engines.
- Product reality: keep any AudioCraft-based engine behind a **plugin / optional dependency** boundary.
- Packaging pattern: explicit model cache directory (`AUDIOCRAFT_CACHE_DIR`) style configuration.

## AudioLDM (haoheliu/AudioLDM)

What it provides:
- Text-to-audio plus repair-adjacent workflows:
  - **Audio-to-audio generation**
  - **Text-guided style transfer** (with a `transfer_strength` knob)
  - References to super-resolution/inpainting pipelines
- Practical quality tips:
  - Better prompts (more specific adjectives)
  - Try multiple seeds
- A built-in concept of **best-of-N candidate generation** (`n_candidate_gen_per_text`) for quality control.

What to apply here:
- Add a first-class “**best-of-N**” notion in S‑NDB‑UND engines where feasible:
  - generate N candidates → score them → keep best → record the decision in credits.
- For an editor v2: “**repair tools**” powered by audio-to-audio / inpainting style operations:
  - remove a click/pop, re-synthesize a small region, extend a tail, make a loop seam less obvious.

## ThinkSound (FunAudioLLM/ThinkSound)

What it provides:
- A modern “generation + editing” framing:
  - Multi-stage pipeline (generate base → refine → targeted edit)
  - Strong focus on **interactive editing** and compositional control
- Clear warning: repo states Apache-2.0 for code, but also “research/educational only” and includes third-party components with separate licenses.

What to apply here:
- UX idea (not necessarily the model): split editor workflow into:
  1) base render
  2) targeted refinement
  3) small-region edits
- Keep licensing front-and-center: if we integrate anything like this, it should be an opt-in engine with explicit license disclosure.

## Amphion (open-mmlab/Amphion)

What it provides:
- A broad toolkit of **recipes** for many tasks (TTS/VC/TTA/etc).
- A big emphasis on **evaluation metrics** and **dataset preprocessing**.

What to apply here:
- Evaluation: expand our QA beyond “basic waveform stats” to include a small set of objective metrics that matter for our use case.
  - SFX-focused candidates: loudness/crest factor, bandwidth/rolloff, transient salience, optionally CLAP-based similarity.
- Recipe culture: keep “how to train/finetune” as reproducible recipes (config + commands + expected outputs).

## TTS-WebUI (rsxdalv/TTS-WebUI)

What it provides:
- A battle-tested pattern for hosting a zoo of models:
  - **extension marketplace**
  - installer scripts
  - explicit docs around dependency/license collisions

What to apply here:
- For S‑NDB‑UND: we already have plugin discovery; the next step is “**curated plugins + clear license gates**”, not a huge model zoo in-core.
- Adopt the attitude: code can be MIT, but *weights* and *deps* may not be—surface that to users.

## GitHub topic: audio-generation

Use it as:
- A discovery pool for future engine plugins.

Apply a rule:
- Don’t ship model weights in-repo; treat weights as user-supplied downloads with explicit license disclosure.

## Concrete S‑NDB‑UND action items (high leverage)

1) **License-aware engine plugins**
   - For any engine with non-commercial weights: allow integration, but require users to fetch weights and acknowledge license.

2) **Best-of-N generation with scoring**
   - Generalize “generate N candidates” into a standard cross-engine option.
   - Score with existing analysis + (optional) embedding similarity (if we add CLAP-like scoring later).

3) **Editor v2: repair tools**
   - Add “inpaint/replace selection” and “extend tail” operations as optional AI-assisted tools.

4) **Training recipe docs**
   - Document 1–2 blessed paths for finetuning (hardware expectations, cache mgmt, reproducibility fields).
