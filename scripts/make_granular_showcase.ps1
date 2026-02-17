Param(
	[int]$SecondsMs = 1500,
	[string]$OutDir = "outputs\\showcase"
)

$ErrorActionPreference = "Stop"

$seconds = [Math]::Max(0.1, $SecondsMs / 1000.0)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$python = "python"

& $python -m soundgen.generate --engine layered --seconds $seconds --seed 100 --prompt "insect chitter" --layered-granular-preset chitter --layered-granular-amount 0.40 --layered-granular-spray 0.60 --layered-granular-grain-ms 18 --post --out (Join-Path $OutDir "granular_chitter.wav")
& $python -m soundgen.generate --engine layered --seconds $seconds --seed 101 --prompt "creature rasp"   --layered-granular-preset rasp    --layered-granular-amount 0.30 --layered-granular-spray 0.35 --layered-granular-grain-ms 35 --post --out (Join-Path $OutDir "granular_rasp.wav")
& $python -m soundgen.generate --engine layered --seconds $seconds --seed 102 --prompt "wasp buzz"       --layered-granular-preset buzz    --layered-granular-amount 0.35 --layered-granular-spray 0.75 --layered-granular-grain-ms 10 --post --out (Join-Path $OutDir "granular_buzz.wav")
& $python -m soundgen.generate --engine layered --seconds $seconds --seed 103 --prompt "screechy scrape" --layered-granular-preset screech --layered-granular-amount 0.25 --layered-granular-spray 0.25 --layered-granular-grain-ms 70 --post --out (Join-Path $OutDir "granular_screech.wav")

Write-Host "\nWrote showcase snippets to: $OutDir"