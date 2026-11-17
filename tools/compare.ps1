# Render the same camera as impostors (LOD 0) and as meshes (LOD 1000) for several debug modes
# and assemble a side-by-side grid. Usage: tools/compare.ps1 -Cam "0,8,60,0,0" -Out shots/cmp.png [-Gallery]
param(
    [string]$Cam = "0,9,60,0,0",
    [string]$Out = "shots/compare.png",
    [string]$Modes = "0,3,2,4",
    [int]$Frames = 24,
    [string]$Size = "1200x675",
    [int]$MeshLod = 400,
    [switch]$Gallery
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$exe = ".\target\release\smart_imposter.exe"
$extra = @()
if ($Gallery) { $extra += "--gallery" }
$files = @()
foreach ($m in $Modes.Split(",")) {
    foreach ($lod in @(0, $MeshLod)) {
        $f = "shots/_cmp_m${m}_l$(if ($lod -eq 0) { 'imp' } else { 'mesh' }).png"
        & $exe --size $Size --cam $Cam --frames $Frames --lod $lod --debug $m --shot $f @extra 2>&1 | Out-Null
        $files += $f
    }
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py (Join-Path $PSScriptRoot "grid.py") ($files -join ";") $Modes $Out
