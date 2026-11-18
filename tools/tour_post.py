"""Cut a recorded `--tour` (e.g. an OBS recording) into the finished showcase video.

    python tools/tour_post.py captures/recording.mkv tours/showcase.txt captures/showcase.mp4 [--realtime]

The tour shows black while each shot loads, and each shot fades in from and out to black: the
shots are found with ffmpeg's blackdetect in the recording.

A tour recorded with --tour-pace above 16.7 ms plays slower than real time, and a 60 fps
recording holds each rendered frame two or more times. Each shot is decoded and a frame kept
only where it differs from the last one kept (mean difference of a downscaled luma image: the
repeats differ by < 0.2 of 255, consecutive rendered frames by > 1); the kept frames play at
60 fps, which is real time again (the tour steps 1/60 s per frame). --realtime skips this for a
recording made at the default pace. Then a short dip to black between shots and a title on each
from the tour file (`| title | caption`): centred over the first and last shots, a lower third
over the rest. Output: H.264 CRF 15, 60 fps, no audio. Needs ffmpeg and numpy.
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np

FONT_LIGHT = "C\\:/Windows/Fonts/segoeuil.ttf"
FONT_SEMI = "C\\:/Windows/Fonts/seguisb.ttf"
FONT_REG = "C\\:/Windows/Fonts/segoeui.ttf"
DIP = 0.25  # seconds of black kept on each side of a shot
NEW_FRAME = 0.4  # mean luma difference (of 255, 1/4 resolution) that makes a frame a new one


def shots(path):
    out = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#")[0].strip()
        if line.startswith("shot "):
            parts = [p.strip() for p in line[5:].split("|")]
            out.append((float(parts[0].split()[0]), parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""))
    return out


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True, check=True)
    w, h, d = r.stdout.split()
    return int(w), int(h), float(d)


def segments(path):
    """Non-black stretches (start, end) in seconds."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vf", "blackdetect=d=0.3:pic_th=0.995:pix_th=0.03", "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)]
    total = probe(path)[2]
    segs, t = [], 0.0
    for a, b in blacks:
        if a - t > 0:
            segs.append((t, a))
        t = b
    if total - t > 0:
        segs.append((t, total))
    return [s for s in segs if s[1] - s[0] > 3.0]


def cut(rec, a, b, out, dedup):
    """rec[a, b] as a 60 fps intermediate (near lossless), each rendered frame once. Frames kept."""
    w, h, _ = probe(rec)
    n = w * h * 3 // 2  # yuv420p
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", rec, "-an", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "60", "-i", "-",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "8", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    last, kept, seen = None, 0, 0
    while True:
        buf = dec.stdout.read(n)
        if len(buf) < n:
            break
        seen += 1
        y = np.frombuffer(buf, np.uint8, w * h).reshape(h, w)[::4, ::4].astype(np.float32)
        if not dedup or last is None or np.abs(y - last).mean() > NEW_FRAME:
            enc.stdin.write(buf)
            last = y
            kept += 1
    enc.stdin.close()
    if enc.wait() != 0 or dec.wait() != 0:
        sys.exit(f"ffmpeg failed cutting {a:.2f}-{b:.2f}")
    return kept, seen


def gradients(tmp, w, h):
    """Title backings (RGBA PNGs): a soft horizontal band through the middle for the title cards,
    a soft darkening of the lower left for the lower thirds."""
    from PIL import Image

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    band = 0.42 * np.exp(-(((y - 0.47 * h) / (0.15 * h)) ** 2)) * (0.55 + 0.45 * np.exp(-(((x - 0.5 * w) / (0.42 * w)) ** 2)))
    s = np.clip((y - 0.62 * h) / (0.33 * h), 0, 1)
    corner = 0.55 * s * s * (3 - 2 * s) * np.exp(-((x / (0.5 * w)) ** 2))
    out = []
    for name, a in (("band", band), ("corner", corner)):
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        p = os.path.join(tmp, f"{name}.png")
        Image.fromarray(rgba, "RGBA").save(p)
        out.append(p)
    return out


def alpha(t0, t1, f=0.7):
    return f"if(lt(t,{t0}),0,if(lt(t,{t0 + f}),(t-{t0})/{f},if(lt(t,{t1 - f}),1,if(lt(t,{t1}),({t1}-t)/{f},0))))"


def text(tmp, k, s):
    p = os.path.join(tmp, f"t{k}.txt").replace("\\", "/")
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)
    return p.replace(":", "\\:")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rec, tour, out = args[:3]
    dedup = "--realtime" not in sys.argv
    plan = shots(tour)
    segs = segments(rec)
    print(f"{len(segs)} shots found in the recording, {len(plan)} in the tour")
    if len(segs) != len(plan):
        for a, b in segs:
            print(f"  {a:8.2f} - {b:8.2f}")
        sys.exit("shot count mismatch: check the recording (or the blackdetect thresholds)")
    tmp = tempfile.mkdtemp(prefix="tour_post_")
    clips, durs = [], []
    for i, ((a, b), (secs, title, _)) in enumerate(zip(segs, plan)):
        clip = os.path.join(tmp, f"shot{i:02d}.mkv")
        kept, seen = cut(rec, a, b, clip, dedup)
        clips.append(clip)
        durs.append(kept / 60)
        print(f"  {a:8.2f} - {b:8.2f}  {seen:5d} captured -> {kept:5d} frames = {kept / 60:6.2f} s (planned {secs:.0f} s)  {title}", flush=True)
    sh = "shadowcolor=black@0.5:shadowx=2:shadowy=2"
    w, h, _ = probe(clips[0])
    grad_band, grad_corner = gradients(tmp, w, h)
    parts, labels, grads, k = [], [], [], 0
    for i, (secs, title, caption) in enumerate(plan):
        dur = durs[i] + 2 * DIP
        # a short dip to black between shots (their own fades lead into it)
        f = f"[{i}:v]setpts=N/60/TB,tpad=start_duration={DIP}:stop_duration={DIP}:color=black,fps=60"
        # titles sit on a soft dark gradient that fades with them (legible over bright foliage)
        card = i == 0 or i == len(plan) - 1
        if card:
            t0, t1, fd = (2.0, 8.5, 1.0) if i == 0 else (max(dur - 9.0, 0.5), dur - 1.0, 1.0)
        else:
            t0, t1, fd = 1.4, 7.4, 0.7
        g = len(clips) + len(grads)
        grads.append((grad_band if card else grad_corner, dur))
        parts.append(
            f"[{g}:v]format=rgba,fade=t=in:st={max(t0 - 0.2, 0):.2f}:d={fd}:alpha=1,fade=t=out:st={max(t1 - fd + 0.2, 0):.2f}:d={fd}:alpha=1[g{i}]"
        )
        f += f"[b{i}];[b{i}][g{i}]overlay=0:0:shortest=1"
        if card:
            if i == 0:
                big, small = "SmartImposter", "The whole Earth at 1:1 scale, ray traced in real time"
            else:
                big, small = "SmartImposter", "Real-time hardware ray tracing  ·  one GPU  ·  2560 × 1440"
            f += f",drawtext=fontfile='{FONT_LIGHT}':textfile='{text(tmp, k, big)}':fontsize=h*0.085:fontcolor=white:{sh}:x=(w-tw)/2:y=h*0.385:alpha='{alpha(t0, t1, fd)}'"
            k += 1
            f += f",drawtext=fontfile='{FONT_REG}':textfile='{text(tmp, k, small)}':fontsize=h*0.027:fontcolor=white@0.95:{sh}:x=(w-tw)/2:y=h*0.515:alpha='{alpha(t0 + 0.5, t1, fd)}'"
            k += 1
        else:
            f += f",drawtext=fontfile='{FONT_SEMI}':textfile='{text(tmp, k, title)}':fontsize=h*0.042:fontcolor=white:{sh}:x=w*0.045:y=h*0.80:alpha='{alpha(t0, t1)}'"
            k += 1
            f += f",drawtext=fontfile='{FONT_REG}':textfile='{text(tmp, k, caption)}':fontsize=h*0.024:fontcolor=white@0.92:{sh}:x=w*0.045+2:y=h*0.80+h*0.058:alpha='{alpha(t0 + 0.35, t1)}'"
            k += 1
        parts.append(f + f"[v{i}]")
        labels.append(f"[v{i}]")
    graph = ";\n".join(parts) + ";\n" + "".join(labels) + f"concat=n={len(labels)}:v=1:a=0[out]"
    gpath = os.path.join(tmp, "graph.txt")
    with open(gpath, "w", encoding="utf-8") as f:
        f.write(graph)
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for c in clips:
        cmd += ["-i", c]
    for path, dur in grads:
        cmd += ["-loop", "1", "-framerate", "60", "-t", f"{dur:.3f}", "-i", path]
    cmd += ["-/filter_complex", gpath, "-map", "[out]", "-an", "-r", "60",
            "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {out} ({probe(out)[2]:.1f} s)")


if __name__ == "__main__":
    main()
