# Creature preset ↔ polish profile matrix

Goal: make creature SFX feel “designed” by pairing a **Pro preset** (generation defaults) with a **Polish profile** (finishing chain), plus 1–2 safe knob suggestions.

These keys are the real, stable keys from the repo registries:
- Pro presets: `src/soundgen/pro_presets.py` (`--pro-preset`)
- Polish profiles: `src/soundgen/polish_profiles.py` (`--polish-profile`)

Conventions:
- Pro preset: `--pro-preset <key>` (or the Web “Pro preset” dropdown)
- Polish profile: `--polish-profile <key>` (or the Web “Polish profile” dropdown)
- Optional knobs are additive tweaks when you need more push.

## Matrix (12 creature archetypes)

1) **Small critter (mouse, gecko, sprite)**
- Pro preset: `creature.small_chitter`
- Polish profile: `ui_clean`
- Optional knobs: `--creature-size -0.6`, `--texture-amount 0.35`

2) **Insect / skitter (spider, beetle swarm)**
- Pro preset: `creature.insectoid_buzz`
- Polish profile: `creature_snappy`
- Optional knobs: `--texture-preset buzz --texture-amount 0.65`, `--highpass-hz 140`

3) **Serpent / hiss**
- Pro preset: `creature.medium_growl`
- Polish profile: `creature_hushed`
- Optional knobs: `--formant-shift 1.20`, `--texture-amount 0.10 --lowpass-hz 13000`

4) **Goblin / imp voice-bark**
- Pro preset: `creature.medium_growl`
- Polish profile: `creature_snappy`
- Optional knobs: `--creature-size -0.15 --formant-shift 1.10`, `--seconds 1.2`

5) **Orc / brute bark**
- Pro preset: `creature.medium_growl`
- Polish profile: `creature_low_end`
- Optional knobs: `--creature-size 0.35`, `--transient-attack 0.15`

6) **Wolf / beast growl**
- Pro preset: `creature.medium_growl`
- Polish profile: `creature_gritty`
- Optional knobs: `--texture-preset rasp --texture-amount 0.30`, `--exciter-amount 0.12`

7) **Undead / hollow moan**
- Pro preset: `creature.undead_rasp`
- Polish profile: `creature_hushed`
- Optional knobs: `--formant-shift 0.86`, `--reverb cave --reverb-mix 0.10 --reverb-time 1.6`

8) **Ghost / whisper**
- Pro preset: `creature.ethereal_whisper`
- Polish profile: `ambience_smooth`
- Optional knobs: `--denoise-amount 0.35`, `--seconds 4.5`

9) **Demon / infernal roar**
- Pro preset: `creature.large_roar`
- Polish profile: `creature_gritty`
- Optional knobs: `--reverb nether --reverb-mix 0.10 --reverb-time 1.8`, `--exciter-amount 0.14`

10) **Dragon / massive roar**
- Pro preset: `creature.large_roar`
- Polish profile: `creature_low_end`
- Optional knobs: `--creature-size 1.0`, `--mb-comp-threshold-db -24 --mb-comp-ratio 2.0`

11) **Slime / squelch**
- Pro preset: `creature.slime_gurgle`
- Polish profile: `impact_soft_mid`
- Optional knobs: `--transient-sustain 0.25`, `--reverb room --reverb-mix 0.08`

12) **Construct / golem (mechanical creature, stone/metal)**
- Pro preset: `env.mechanical_ambience`
- Polish profile: `foley_punchy`
- Optional knobs: `--multiband --mb-mid-gain-db 1.0`, `--texture-preset chitter --texture-amount 0.20`

## One-liner CLI template

Example:
`python -m soundgen.generate --engine rfxgen --prompt "imp bark" --seconds 1.2 --post --polish --pro-preset creature.medium_growl --polish-profile creature_snappy --out outputs\\imp_bark.wav`

If exporting to Minecraft:
`python -m soundgen.generate --engine rfxgen --minecraft --namespace mymod --event mobs.imp.bark --prompt "imp bark" --post --polish --pro-preset creature.medium_growl --polish-profile creature_snappy`

## Available keys (copy/paste reference)

Pro preset keys (`--pro-preset`):

- `creature.small_chitter`
- `creature.medium_growl`
- `creature.large_roar`
- `creature.insectoid_buzz`
- `creature.ethereal_whisper`
- `creature.undead_rasp`
- `creature.slime_gurgle`
- `creature.serpent_hiss`
- `creature.goblin_bark`
- `creature.orc_brute_bark`
- `creature.wolf_snarl`
- `creature.ghost_wail`
- `creature.demon_scream`
- `creature.dragon_bellow`
- `creature.avian_screech`
- `creature.frog_croak`
- `creature.aquatic_gurgle`
- `creature.tiny_squeak`
- `creature.golem_grind`
- `env.cave_drone`
- `env.lava_rumble`
- `env.forest_wind`
- `env.magical_hum`
- `env.mechanical_ambience`
- `foley.bone_crack`
- `foley.flesh_impact`
- `foley.metal_scrape`
- `foley.footstep_variations`

Polish profile keys (`--polish-profile`):

- `ui_clean`
- `foley_punchy`
- `creature_gritty`
- `creature_low_end`
- `creature_snappy`
- `creature_hushed`
- `ambience_smooth`
- `ambience_warm_mono`
- `ambience_glue_open`
- `ambience_vhs`
- `ambience_loop_ready`
- `impact_hard_punch`
- `impact_soft_mid`
