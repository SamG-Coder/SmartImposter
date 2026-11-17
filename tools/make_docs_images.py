"""Build the README images in docs/images from renders in shots/ (see README "Screenshots")."""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S = lambda n: os.path.join(ROOT, "shots", n)
OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)
try:
    FONT = ImageFont.truetype("arial.ttf", 30)
except OSError:
    FONT = ImageFont.load_default()


def label(img, text, xy=(16, 14)):
    d = ImageDraw.Draw(img)
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=FONT)
    d.rectangle((l - 8, t - 6, r + 8, b + 6), fill=(0, 0, 0))
    d.text((x, y), text, fill=(255, 255, 255), font=FONT)
    return img


def save(img, name, width=1600):
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.convert("RGB").save(os.path.join(OUT, name), quality=88, optimize=True)
    print("wrote", name, img.size)


def side_by_side(a, b, la, lb, gap=8):
    h = max(a.height, b.height)
    out = Image.new("RGB", (a.width + b.width + gap, h), (20, 20, 20))
    out.paste(label(a.copy(), la), (0, 0))
    out.paste(label(b.copy(), lb), (a.width + gap, 0))
    return out


op = lambda n: Image.open(S(n)).convert("RGB")
save(op("pub_hero.png"), "hero.jpg")
save(op("pub_overview.png"), "overview.jpg")
save(op("pub_canopy.png"), "canopy.jpg")
save(label(op("pub_lodview.png"), "LOD view: green = mesh, orange = high-detail impostor, red = impostor"), "lod_view.jpg")

a, b = op("cmp_lod50.png"), op("cmp_lod400.png")
box = (300, 60, 1100, 700)
save(side_by_side(a.crop(box), b.crop(box), "Impostors beyond 50 m (default)", "Every tree as real geometry"), "impostor_vs_mesh.jpg")

a, b = op("cmp_imp.png"), op("cmp_mesh.png")
box = (500, 60, 1000, 700)
save(side_by_side(a.crop(box), b.crop(box), "Impostor (forced, 25-35 m)", "Mesh"), "impostor_vs_mesh_close.jpg")

a, b = op("tex_front.png"), op("tex_back.png")
box = (350, 0, 1250, 1000)
save(side_by_side(a.crop(box), b.crop(box), "Front-lit", "Back-lit (leaf transmission)"), "leaves.jpg")

# Full 12x12 hemi-octahedral atlas of tier 0, variant 0: albedo and depth x coverage.
panels = [op(f"pub_atlas_0_{k}.png").resize((900, 900), Image.LANCZOS) for k in ("albedo", "depth")]
sheet = Image.new("RGB", (1808, 900), (20, 20, 20))
for i, (p, t) in enumerate(zip(panels, ["albedo (12x12 views)", "depth x coverage"])):
    sheet.paste(label(p, t), (i * 908, 0))
save(sheet, "atlas.jpg", width=1600)
