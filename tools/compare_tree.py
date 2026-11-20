"""Render the scan, generated mesh and live impostor with identical cameras and lighting."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def capture(exe, out, tag, args, views):
    prefix = out / tag
    cmd = [
        str(exe.resolve()),
        *args,
        "--views",
        ";".join(views),
        "--frames",
        "14",
        "--size",
        "640x640",
        "--ss",
        "1.5",
        "--sharpen",
        "0",
        "--dlss",
        "off",
        "--haze",
        "0",
        "--clouds",
        "off",
        "--shot",
        str(prefix) + ".png",
    ]
    with (out / (tag + ".log")).open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = [Path(f"{prefix}_{i}.png") for i in range(len(views))]
    log = (out / (tag + ".log")).read_text()
    if not all(p.exists() and f"Saved {p}" in log for p in result):
        raise RuntimeError(f"incomplete capture, see {tag}.log")
    return result


def metrics(ref, other):
    a, b = (np.asarray(Image.open(p).convert("L"), dtype=float) / 255 for p in (ref, other))

    def iou(x, y):
        return np.minimum(x, y).sum() / np.maximum(x, y).sum()

    def pool(x):
        return x.reshape(40, 16, 40, 16).mean(axis=(1, 3))

    if min(a.sum(), b.sum()) < 100:
        raise ValueError("empty specimen masks")
    return iou(pool(a), pool(b)), iou(a, b), b.sum() / a.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("species")
    parser.add_argument("variant")
    parser.add_argument("--out", type=Path, default=Path("shots/tree-comparison"))
    parser.add_argument("--height", type=float, default=16.5, help="specimen height in metres")
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--exe", type=Path, default=Path("target/release/smart_imposter.exe"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    h = args.height
    # Two whole-tree views and an elevated oblique camera at the mesh/impostor transition.
    cams = [
        f"0,{h * 0.51},{h * 1.30},0,0",
        f"{h * 1.30},{h * 0.51},0,-90,0",
        f"{h * 2.6},{h * 1.45},{h * 2.6},-45,-14.3",
    ]

    def views(lod):
        return [f"{cam},{lod},{debug}" for cam in cams for debug in (0, 12)]

    scan = capture(
        args.exe, out, "scan", ["--species", args.species, "--specimen", args.variant], views(1000)
    )
    gen = capture(
        args.exe,
        out,
        "grown",
        ["--procedural-tree", str(args.recipe), "--imp-grid", str(args.grid), "--imp-res", str(args.res)],
        views(1000) + views(0),
    )
    cols = [scan, gen[: len(scan)], gen[len(scan) :]]
    font = ImageFont.truetype("arial.ttf", 19)
    headings = ["Scanned mesh", "Procedural mesh", f"Live impostor  /  {args.grid} × {args.grid} views"]
    row_names = ["Front", "Side", "Elevated  /  LOD distance"]
    w = 368
    board = Image.new("RGB", (3 * w, 3 * (w + 29) + 40), (23, 29, 31))
    draw = ImageDraw.Draw(board)
    for col, title in enumerate(headings):
        draw.text((col * w + 10, 9), title, font=font, fill="white")
    for row, label in enumerate(row_names):
        for j, files in enumerate(cols):
            im = Image.open(files[2 * row]).convert("RGB")
            if row == 2:  # identical crop for all three, still rendered at the LOD distance
                im = im.crop((200, 200, 440, 440))
            y = 40 + row * (w + 29)
            board.paste(im.resize((w, w), Image.Resampling.LANCZOS), (w * j, y))
            draw.text((w * j + 10, y + w + 3), label, font=font, fill="#d6e4df")
    board.save(out / "comparison.jpg", quality=92)
    for title, pair in [
        ("scan -> procedural mesh", cols[:2]),
        ("procedural mesh -> impostor", cols[1:]),
        ("scan -> impostor", cols[::2]),
    ]:
        for row, label in enumerate(row_names):
            coarse, pixel, coverage = metrics(pair[0][2 * row + 1], pair[1][2 * row + 1])
            print(
                f"{title:29} {label:24}  pooled IoU {coarse:.3f}  pixel IoU {pixel:.3f}  coverage ratio {coverage:.3f}"
            )


if __name__ == "__main__":
    main()
