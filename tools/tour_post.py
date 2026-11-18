"""Cut a recorded `--tour` (e.g. an OBS recording) into the finished showcase video.

    python tools/tour_post.py captures/recording.mkv tours/showcase.txt captures/showcase.mp4

The tour shows black while each shot loads, and each shot fades in from and out to black. This
finds the shots with ffmpeg's blackdetect, keeps a short dip to black between them, and adds a
title to each from the tour file (`| title | caption`): centred over the first and last shots,
a lower third over the rest. Output: H.264, CRF 15, at the recording's size and frame rate, no
audio. Needs ffmpeg on the PATH.
"""

import os
import re
import subprocess
import sys
import tempfile

FONT_LIGHT = "C\\:/Windows/Fonts/segoeuil.ttf"
FONT_SEMI = "C\\:/Windows/Fonts/seguisb.ttf"
DIP = 0.25  # seconds of black kept on each side of a shot


def shots(path):
    out = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#")[0].strip()
        if line.startswith("shot "):
            parts = [p.strip() for p in line[5:].split("|")]
            out.append((float(parts[0].split()[0]), parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""))
    return out


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def segments(path):
    """Non-black stretches (start, end) in seconds."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vf", "blackdetect=d=0.1:pic_th=0.995:pix_th=0.03", "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)]
    total = probe_duration(path)
    segs, t = [], 0.0
    for a, b in blacks:
        if a - t > 0:
            segs.append((t, a))
        t = b
    if total - t > 0:
        segs.append((t, total))
    return [s for s in segs if s[1] - s[0] > 3.0]


def alpha(t0, t1, f=0.7):
    return f"if(lt(t,{t0}),0,if(lt(t,{t0 + f}),(t-{t0})/{f},if(lt(t,{t1 - f}),1,if(lt(t,{t1}),({t1}-t)/{f},0))))"


def text(tmp, k, s):
    p = os.path.join(tmp, f"t{k}.txt").replace("\\", "/")
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)
    return p.replace(":", "\\:")


def main():
    rec, tour, out = sys.argv[1:4]
    plan = shots(tour)
    segs = segments(rec)
    print(f"{len(segs)} shots found in the recording, {len(plan)} in the tour")
    for (a, b), (secs, title, _) in zip(segs, plan):
        print(f"  {a:8.2f} - {b:8.2f}  ({b - a:5.2f} s, planned {secs:.0f})  {title}")
    if len(segs) != len(plan):
        sys.exit("shot count mismatch: check the recording (or the blackdetect thresholds)")
    tmp = tempfile.mkdtemp(prefix="tour_post_")
    sh = "shadowcolor=black@0.45:shadowx=2:shadowy=2"
    parts, labels, k = [], [], 0
    for i, ((a, b), (secs, title, caption)) in enumerate(zip(segs, plan)):
        a, b = max(a - DIP, 0.0), b + DIP
        dur = b - a
        f = f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS"
        if i == 0 or i == len(plan) - 1:
            # centred title card
            t0, t1 = (2.0, 8.5) if i == 0 else (dur - 9.0, dur - 1.0)
            big, small = (title, caption) if i == 0 else ("SmartImposter", "Real-time hardware ray tracing  ·  one GPU  ·  2560 × 1440")
            if i == 0:
                big, small = "SmartImposter", "The whole Earth at 1:1 scale, ray traced in real time"
            f += f",drawtext=fontfile='{FONT_LIGHT}':textfile='{text(tmp, k, big)}':fontsize=h*0.085:fontcolor=white:{sh}:x=(w-tw)/2:y=h*0.40:alpha='{alpha(t0, t1, 1.0)}'"
            k += 1
            f += f",drawtext=fontfile='{FONT_LIGHT}':textfile='{text(tmp, k, small)}':fontsize=h*0.026:fontcolor=white@0.9:{sh}:x=(w-tw)/2:y=h*0.52:alpha='{alpha(t0 + 0.5, t1, 1.0)}'"
            k += 1
        else:
            t0, t1 = 1.4, 7.4
            f += f",drawtext=fontfile='{FONT_SEMI}':textfile='{text(tmp, k, title)}':fontsize=h*0.042:fontcolor=white:{sh}:x=w*0.045:y=h*0.80:alpha='{alpha(t0, t1)}'"
            k += 1
            f += f",drawtext=fontfile='{FONT_LIGHT}':textfile='{text(tmp, k, caption)}':fontsize=h*0.024:fontcolor=white@0.88:{sh}:x=w*0.045+2:y=h*0.80+h*0.058:alpha='{alpha(t0 + 0.35, t1)}'"
            k += 1
        parts.append(f + f"[v{i}]")
        labels.append(f"[v{i}]")
    graph = ";\n".join(parts) + ";\n" + "".join(labels) + f"concat=n={len(labels)}:v=1:a=0[out]"
    gpath = os.path.join(tmp, "graph.txt")
    with open(gpath, "w", encoding="utf-8") as f:
        f.write(graph)
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", rec, "-/filter_complex", gpath, "-map", "[out]", "-an",
           "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {out} ({probe_duration(out):.1f} s)")


if __name__ == "__main__":
    main()
