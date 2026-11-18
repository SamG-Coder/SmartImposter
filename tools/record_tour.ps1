param([string]$Tour = "tours/showcase.txt", [string]$Pace = "40", [string]$Profile = "SmartImposter Showcase")
# Record a --tour with OBS Studio, then cut it with tools/tour_post.py.
#
#   powershell -File tools/record_tour.ps1 -Tour tours/showcase.txt
#   python tools/tour_post.py captures/<recording>.mkv tours/showcase.txt captures/showcase.mp4
#
# Needs an OBS profile and scene collection both named $Profile: 2560x1440 at 60 fps, recording
# to <repo>/captures (mkv), a Game Capture source of smart_imposter.exe, and the hotkeys
# Start Recording = Ctrl+Alt+Shift+F11, Stop Recording = Ctrl+Alt+Shift+F12. The tour runs paced
# (each frame on screen for $Pace ms, so OBS captures every one; tour_post.py drops the repeats).
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Keys {
    [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
    public static void Chord(byte key) {
        byte[] mods = { 0x11, 0x12, 0x10 }; // ctrl, alt, shift
        foreach (var m in mods) keybd_event(m, 0, 0, UIntPtr.Zero);
        keybd_event(key, 0, 0, UIntPtr.Zero);
        System.Threading.Thread.Sleep(120);
        keybd_event(key, 0, 2, UIntPtr.Zero);
        foreach (var m in mods) keybd_event(m, 0, 2, UIntPtr.Zero);
    }
}
"@
$capDir = Join-Path $root "captures"
New-Item -ItemType Directory -Force $capDir | Out-Null
$before = @(Get-ChildItem $capDir -Filter *.mkv -ErrorAction SilentlyContinue | ForEach-Object FullName)
$obsDir = "C:\Program Files\obs-studio\bin\64bit"
# OBS marks every run in .sentinel and asks about recovery after one that did not exit cleanly:
# clear stale marks (OBS is not running)
if (-not (Get-Process obs64 -ErrorAction SilentlyContinue)) {
    Remove-Item "$env:APPDATA\obs-studio\.sentinel\run_*" -Force -ErrorAction SilentlyContinue
}
$obsArgs = "--profile `"$Profile`" --collection `"$Profile`" --minimize-to-tray --disable-shutdown-check --disable-updater --disable-missing-files-check"
$obs = Start-Process -FilePath "$obsDir\obs64.exe" -WorkingDirectory $obsDir -ArgumentList $obsArgs -PassThru
Start-Sleep -Seconds 6
# the app first (game capture hooks it), then start recording
$app = Start-Process -FilePath (Join-Path $root "target\release\smart_imposter.exe") -WorkingDirectory $root `
    -ArgumentList "--globe --fullscreen --size 2560x1440 --tour $Tour --tour-pace $Pace" -PassThru `
    -RedirectStandardOutput (Join-Path $capDir "tour_run.log") -RedirectStandardError (Join-Path $capDir "tour_run.err")
Start-Sleep -Seconds 3
[Keys]::Chord(0x7A) # F11: start recording
Start-Sleep -Seconds 3
$rec = @(Get-ChildItem $capDir -Filter *.mkv | Where-Object { $before -notcontains $_.FullName })
"recording: " + ($rec | ForEach-Object Name)
if ($rec.Count -eq 0) {
    "OBS did not start recording: stopping"
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
$app.WaitForExit()
Start-Sleep -Seconds 2
[Keys]::Chord(0x7B) # F12: stop recording
Start-Sleep -Seconds 5
$rec = @(Get-ChildItem $capDir -Filter *.mkv | Where-Object { $before -notcontains $_.FullName })
$rec | ForEach-Object { "{0} {1:N0} MB" -f $_.FullName, ($_.Length / 1MB) }
# close OBS normally (WM_CLOSE; the recording is already stopped), forced only if it hangs
& taskkill.exe /PID $obs.Id | Out-Null
if (-not $obs.WaitForExit(20000)) {
    Stop-Process -Id $obs.Id -Force -ErrorAction SilentlyContinue
    Remove-Item "$env:APPDATA\obs-studio\.sentinel\run_*" -Force -ErrorAction SilentlyContinue
    "OBS had to be forced closed"
}
