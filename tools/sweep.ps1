# Measures tree brightness while the camera approaches the gallery trees, with smooth LOD bands
# and with hard switches, to show popping at the 50 m (mesh) and 100 m (impostor tier) boundaries.
param([string]$Out = "shots/lod_sweep.png")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$exe = ".\target\release\smart_imposter.exe"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$rows = @()
foreach ($mode in @("smooth", "hard")) {
    $extra = if ($mode -eq "hard") { @("--lod-band", "0.01", "--lod2-band", "0.01") } else { @() }
    for ($d = 125; $d -ge 30; $d -= 2.5) {
        $f = "shots/_sw_${mode}_$d.png"
        & $exe --gallery --size 960x540 --ss 1 --cam "0,8,$d,0,0" --frames 24 --shot $f @extra 2>&1 | Out-Null
        & $exe --gallery --size 960x540 --ss 1 --cam "0,8,$d,0,0" --frames 1 --debug 2 --shot "shots/_sw_n_$d.png" 2>&1 | Out-Null
        $rows += "$mode,$d,$f,shots/_sw_n_$d.png"
    }
}
$rows | Set-Content shots/_sweep.csv
& $py (Join-Path $PSScriptRoot "sweep_plot.py") shots/_sweep.csv $Out
