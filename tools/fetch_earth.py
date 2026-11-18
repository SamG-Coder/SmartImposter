"""Download the free global data for the 1:1 globe into assets/earth/ (resumable, parallel).

    python tools/fetch_earth.py vectors              # biomes, rivers, lakes (~1.5 GB)
    python tools/fetch_earth.py cover [--bbox ...]   # ESA WorldCover 2021, 10 m (124 GB worldwide)
    python tools/fetch_earth.py dem [--bbox ...]     # Copernicus GLO-30 (543 GB as published)
    python tools/fetch_earth.py dem --res 90         # Copernicus GLO-90 (67 GB as published)

--bbox south,west,north,east limits the raster sets to a region. --jobs sets parallel downloads.

Elevation tiles are converted to the engine's compact .dem format right after download (int16
metres, row deltas, deflate: ~5x smaller) by `smart_imposter --convert-dem`, and the published
GeoTIFF is deleted, so the disk only ever holds a few raw tiles at a time. Pass --keep-tif to
keep them. Every file lands as .part and is renamed when complete; finished files are skipped
on the next run, so the script can be stopped and restarted at any time.

Sources (all free; licences in docs/EARTH_PLAN.md):
  Copernicus DEM  https://registry.opendata.aws/copernicus-dem/
  ESA WorldCover  https://registry.opendata.aws/esa-worldcover-vito/
  RESOLVE Ecoregions 2017, HydroRIVERS / HydroLAKES v1.0 (hydrosheds.org)
"""

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EARTH = os.path.join(ROOT, "assets", "earth")
EXE = os.path.join(ROOT, "target", "release", "smart_imposter.exe" if os.name == "nt" else "smart_imposter")

VECTORS = [
    ("ecoregions", "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"),
    ("hydrorivers", "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_shp.zip"),
    ("hydrolakes", "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"),
]
DEM_BUCKET = "https://copernicus-dem-{res}m.s3.amazonaws.com"
COVER_BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

lock = threading.Lock()
stats = {"files": 0, "bytes": 0, "skipped": 0, "failed": 0, "start": time.time()}


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "smart-imposter-fetch/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url, dest, retries=4):
    """url -> dest via dest.part; returns True when dest is complete."""
    if os.path.exists(dest):
        with lock:
            stats["skipped"] += 1
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(retries):
        try:
            with get(url) as r, open(part, "wb") as f:
                size = int(r.headers.get("Content-Length", 0))
                n = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    n += len(b)
            if size and n != size:
                raise IOError(f"short read {n} / {size}")
            os.replace(part, dest)
            with lock:
                stats["files"] += 1
                stats["bytes"] += n
            return True
        except Exception as e:  # network hiccups: retry with backoff
            if attempt == retries - 1:
                print(f"  failed {url}: {e}", file=sys.stderr)
                with lock:
                    stats["failed"] += 1
            time.sleep(2 * (attempt + 1))
    if os.path.exists(part):
        os.remove(part)
    return False


def progress(total):
    s = stats
    done = s["files"] + s["skipped"] + s["failed"]
    rate = s["bytes"] / max(time.time() - s["start"], 1e-3) / 1e6
    print(f"\r  {done}/{total} files, {s['bytes'] / 1e9:.1f} GB at {rate:.0f} MB/s, {s['failed']} failed", end="", flush=True)


def in_bbox(lat, lon, size, bbox):
    if not bbox:
        return True
    s, w, n, e = bbox
    return lat < n and lat + size > s and lon < e and lon + size > w


def vectors(args):
    for name, url in VECTORS:
        d = os.path.join(EARTH, name)
        z = os.path.join(d, os.path.basename(url))
        print(f"{name}: {url}")
        if download(url, z) and not any(f.endswith(".shp") for _, _, fs in os.walk(d) for f in fs):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
            print(f"  unpacked into {d}")


def dem_tiles(res):
    with get(f"{DEM_BUCKET.format(res=res)}/tileList.txt") as r:
        names = [l.strip() for l in r.read().decode().splitlines() if l.strip()]
    out = []
    for n in names:
        m = re.search(r"_([NS])(\d\d)_00_([EW])(\d\d\d)_00_", n)
        lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
        lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
        out.append((n, lat, lon))
    return out


CONVERTER = os.path.join(ROOT, "target", "fetch_earth", os.path.basename(EXE))


def convert(paths):
    """Published GeoTIFFs -> .dem (the engine converts and deletes them). Runs a copy of the
    engine, so rebuilding it while a download runs does not collide with a conversion."""
    if paths:
        subprocess.run([CONVERTER, "--convert-dem", *paths], check=True, stdout=subprocess.DEVNULL)


def dem(args):
    if not args.keep_tif:
        if not os.path.exists(EXE):
            sys.exit(f"build the engine first (cargo build --release): {EXE} converts the tiles")
        os.makedirs(os.path.dirname(CONVERTER), exist_ok=True)
        shutil.copy2(EXE, CONVERTER)
    tiles = [t for t in dem_tiles(args.res) if in_bbox(t[1], t[2], 1, args.bbox)]
    d = os.path.join(EARTH, "glo30" if args.res == 30 else "glo90")
    todo = [t for t in tiles if not os.path.exists(os.path.join(d, t[0] + ".dem"))]
    print(f"GLO-{args.res}: {len(tiles)} tiles in range, {len(tiles) - len(todo)} already here -> {d}")
    batch = []

    def one(t):
        n = t[0]
        tif = os.path.join(d, n + ".tif")
        if download(f"{DEM_BUCKET.format(res=args.res)}/{n}/{n}.tif", tif) and not args.keep_tif:
            with lock:
                batch.append(tif)
                ready = list(batch) if len(batch) >= 16 else None
                if ready:
                    batch.clear()
            if ready:
                convert(ready)

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 4 == 0:
                progress(len(todo))
    convert(batch)
    progress(len(todo))
    print()


def cover(args):
    tiles = []
    for lat in range(-60, 84, 3):
        for lon in range(-180, 180, 3):
            if in_bbox(lat, lon, 3, args.bbox):
                ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
                ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                tiles.append(f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif")
    # only the tiles that exist (ocean tiles are not published)
    listed = set()
    token = None
    while True:
        url = f"{COVER_BUCKET}/?list-type=2&prefix=v200/2021/map/"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        with get(url) as r:
            x = r.read().decode()
        listed.update(re.findall(r"<Key>v200/2021/map/([^<]+_Map\.tif)</Key>", x))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            break
        token = m.group(1)
    tiles = [t for t in tiles if t in listed]
    d = os.path.join(EARTH, "worldcover")
    print(f"WorldCover: {len(tiles)} tiles in range -> {d}")
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for i, _ in enumerate(ex.map(lambda t: download(f"{COVER_BUCKET}/v200/2021/map/{t}", os.path.join(d, t)), tiles)):
            if i % 4 == 0:
                progress(len(tiles))
    progress(len(tiles))
    print()


def main():
    import urllib.parse  # noqa: F401  (cover listing)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["vectors", "cover", "dem"])
    p.add_argument("--bbox", type=lambda s: [float(v) for v in s.split(",")], help="south,west,north,east in degrees")
    p.add_argument("--res", type=int, choices=[30, 90], default=30)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--keep-tif", action="store_true")
    a = p.parse_args()
    {"vectors": vectors, "cover": cover, "dem": dem}[a.what](a)
    s = stats
    print(f"done: {s['files']} downloaded ({s['bytes'] / 1e9:.1f} GB), {s['skipped']} already present, {s['failed']} failed")


if __name__ == "__main__":
    import urllib.parse

    main()
