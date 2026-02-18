# Project system (v2.4)

The project system is a lightweight way to:

- create/load a project
- track generated **and edited** versions per sound
- export a Minecraft-ready pack (ogg + sounds.json + subtitles + pack credits)

A project is just a folder containing a `sndbund_project.json` file plus generated audio under `project_audio/`.

---

## Sound pack project (UI / ambience / SFX)

Create a project:

```powershell
python -m soundgen.app project create --root projects\mymod_pack --kind soundpack --id mymod_pack --title "My Mod Pack" --namespace mymod --pack-root resourcepack
```

Add an item:

```powershell
python -m soundgen.app project add --root projects\mymod_pack --id ui.coin --category ui --engine rfxgen --prompt "coin pickup" --seconds 0.7 --event ui.coin --sound-path ui/coin --subtitle "Coin" --variants 5 --generate-arg --post
```

Build (generate + export into the project’s pack):

```powershell
python -m soundgen.app project build --root projects\mymod_pack
```

This writes:

- WAV variants under `projects\mymod_pack\project_audio\<item>\v0001\...`
- Minecraft pack output under `projects\mymod_pack\resourcepack\assets\<namespace>\...`
- Pack credits under `projects\mymod_pack\resourcepack\assets\<namespace>\soundgen_credits.json`

Open the editor on the **active** version of an item:

```powershell
python -m soundgen.app project edit --root projects\mymod_pack --id ui.coin
```

If you save an edited WAV elsewhere (or the editor created a new file), import it as a new version and re-export:

```powershell
python -m soundgen.app project import-edit --root projects\mymod_pack --id ui.coin --wav outputs\ui_coin_edit.wav --notes "Trim + click removal"
```

---

## Minecraft mob project mode (generate → edit → export)

Create a mob project:

```powershell
python -m soundgen.app project create --root projects\zombie --kind mob --id zombie --title "Zombie Soundset" --namespace mymod --pack-root resourcepack --mob zombie --subtitle-base "Zombie" --style "retro, crunchy"
```

Build the mob soundset (hurt/death/ambient/step) into the project pack and keep WAVs for editing:

```powershell
python -m soundgen.app project build --root projects\zombie
```

Then open the editor on a specific type’s active version (example: hurt):

```powershell
python -m soundgen.app project edit --root projects\zombie --id hurt
```

After editing, import the edited WAV as a new version (this also exports it into the project’s pack):

```powershell
python -m soundgen.app project import-edit --root projects\zombie --id hurt --wav outputs\zombie_hurt_edit.wav
```
