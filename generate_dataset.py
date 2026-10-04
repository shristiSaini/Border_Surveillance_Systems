import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import json
import csv

OUT_DIR = Path(__file__).parent / "ground_truth_dataset"

SIZE = 512                          
RNG  = np.random.default_rng(42)  

SAND  = (178, 162, 120)   
VEG   = ( 98, 128,  62)   
DENSE = ( 72, 108,  44)   
ROCK  = (145, 132, 105)   
ROAD  = (195, 180, 148)   
ROOF  = (180, 162, 130)   
CLOUD = (210, 215, 205)   

def perlin_like(size, scale=40, octaves=4, seed=0):
    
    rng2  = np.random.default_rng(seed)
    total = np.zeros((size, size), dtype=np.float32)
    amp, freq = 1.0, 1.0
    for _ in range(octaves):
        s     = max(2, int(size / (scale * freq)))
        noise = rng2.random((s, s)).astype(np.float32)
        up    = np.array(
            Image.fromarray((noise * 255).astype(np.uint8))
                 .resize((size, size), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        total += amp * up
        amp   *= 0.5
        freq  *= 2.0
    mn, mx = total.min(), total.max()
    return (total - mn) / (mx - mn + 1e-8)


def make_terrain(size=SIZE, seed=0):
    
    noise = perlin_like(size, scale=30, octaves=5, seed=seed)
    veg   = perlin_like(size, scale=15, octaves=3, seed=seed + 1)
    img   = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            n = noise[y, x]
            v = veg[y, x]
            if n > 0.72:                        # rocky outcrop
                c = tuple(
                    int(ROCK[i] + (n - 0.72) * 60 * ((-1) ** i))
                    for i in range(3)
                )
            elif v > 0.62:                      # dense vegetation
                blend = min(1.0, (v - 0.62) / 0.25)
                c = tuple(int(DENSE[i] * blend + VEG[i] * (1 - blend)) for i in range(3))
            elif v > 0.38:                      # sparse vegetation
                blend = (v - 0.38) / 0.24
                c = tuple(int(VEG[i] * blend + SAND[i] * (1 - blend)) for i in range(3))
            else:                               # bare sand / soil
                c = tuple(int(SAND[i] + (n - 0.5) * 40) for i in range(3))
            img[y, x] = np.clip(c, 0, 255)

    sensor_noise = RNG.integers(-6, 7, (size, size, 3))
    return np.clip(img.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)


def add_sensor_noise(arr, seed=0):
    """Add small per-pixel noise (simulates pass-to-pass sensor variance)."""
    rng2 = np.random.default_rng(seed)
    return np.clip(
        arr.astype(np.int16) + rng2.integers(-5, 6, arr.shape),
        0, 255,
    ).astype(np.uint8)


def draw_coord_overlay(img_arr):
    
    img = Image.fromarray(img_arr)
    d   = ImageDraw.Draw(img)
    d.text((6, SIZE - 18), "34.5200\u00b0N  69.1800\u00b0E", fill=(60, 220, 130))
    return np.array(img)


def seasonal_tint(arr, factor=(1.0, 1.02, 0.97)):
    """Apply a slight per-channel multiplier (mimics seasonal colour shift)."""
    out = arr.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= factor[c]
    return np.clip(out, 0, 255).astype(np.uint8)

def add_road(arr, x0, y0, length=320, width=10):
    """Paint a new horizontal road segment."""
    out = arr.copy()
    col = np.array(ROAD, dtype=np.int16) + np.array([10, 8, 6])
    strip = np.tile(col, (width, length, 1))
    noise = RNG.integers(-8, 9, (width, length, 3))
    out[y0:y0 + width, x0:x0 + length] = np.clip(strip + noise, 0, 255).astype(np.uint8)
    return out

def add_buildings(arr, cx, cy, n=6, seed=5):
    """Add a cluster of n building footprints around (cx, cy)."""
    rng2 = np.random.default_rng(seed)
    img  = Image.fromarray(arr.copy())
    d    = ImageDraw.Draw(img)
    for _ in range(n):
        bx = cx + int(rng2.integers(-55, 56))
        by = cy + int(rng2.integers(-55, 56))
        bw = int(rng2.integers(18, 38))
        bh = int(rng2.integers(14, 30))
        r, g, b = ROOF
        fill = (
            int(np.clip(r + rng2.integers(-15, 16), 0, 255)),
            int(np.clip(g + rng2.integers(-10, 11), 0, 255)),
            int(np.clip(b + rng2.integers(-8,   9), 0, 255)),
        )
        d.rectangle([bx, by, bx + bw, by + bh], fill=fill)
        d.rectangle([bx, by, bx + bw, by + bh], outline=(100, 90, 70), width=1)
    return np.array(img)

def clear_vegetation(arr, x0, y0, w=155, h=125):
    """Replace a vegetation patch with bare earth (simulates clearing)."""
    out     = arr.copy()
    cleared = np.full((h, w, 3), SAND, dtype=np.uint8)
    noise   = RNG.integers(-12, 13, (h, w, 3))
    out[y0:y0 + h, x0:x0 + w] = np.clip(
        cleared.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)
    return out

def add_staging_area(arr, cx, cy):
    """Add a vehicle/equipment staging area — flattened pad + parked vehicles."""
    img = Image.fromarray(arr.copy())
    d   = ImageDraw.Draw(img)
    d.rectangle([cx - 70, cy - 50, cx + 70, cy + 50], fill=(190, 175, 138))
    for vx, vy in [
        (cx - 45, cy - 20), (cx - 10, cy - 20), (cx + 25, cy - 20),
        (cx - 45, cy + 10), (cx - 10, cy + 10),
    ]:
        d.rectangle([vx, vy, vx + 18, vy + 10], fill=(80, 75, 65))
    return np.array(img)

def add_cloud(arr, cx, cy, radius=120, opacity=0.60, seed=7):
    """Overlay a semi-transparent Gaussian cloud blob onto the image."""
    rng2  = np.random.default_rng(seed)
    out   = arr.copy().astype(np.float32)
    Y, X  = np.ogrid[:SIZE, :SIZE]
    mask  = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * radius ** 2))
    mask  = np.clip(mask * 1.4, 0, 1)
    noise = rng2.random((SIZE, SIZE)) * 0.15
    mask  = np.clip(mask + noise, 0, 1)
    cloud = np.array(CLOUD, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - mask * opacity) + cloud[c] * mask * opacity
    return np.clip(out, 0, 255).astype(np.uint8)

def add_shadow_shift(arr, shift_x=30, shift_y=20, intensity=0.28):
    
    out    = arr.copy().astype(np.float32)
    Y, X   = np.mgrid[:SIZE, :SIZE]
    band   = np.sin((X - Y * 0.6 + shift_x) / 40.0) * 0.5 + 0.5
    band   = np.clip(band, 0, 1)
    shadow = 1.0 - band * intensity
    for c in range(3):
        out[:, :, c] *= shadow
    return np.clip(out, 0, 255).astype(np.uint8)

PAIRS = [
    dict(
        id="P01", category="action", label=1,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="New road construction along northern ridge",
        t2_fn=lambda a: add_road(a, x0=60, y0=210, length=320, width=10),
    ),
    dict(
        id="P02", category="action", label=1,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Building cluster expansion — 6 new structures",
        t2_fn=lambda a: add_buildings(a, cx=280, cy=300, n=6, seed=5),
    ),
    dict(
        id="P03", category="action", label=1,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Vegetation clearing — 155x125 px patch removed",
        t2_fn=lambda a: clear_vegetation(a, x0=160, y0=170, w=155, h=125),
    ),
    dict(
        id="P04", category="action", label=1,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Vehicle staging area appears in T2",
        t2_fn=lambda a: add_staging_area(a, cx=310, cy=200),
    ),
    dict(
        id="P05", category="baseline", label=0,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Stable agricultural terraces — no change",
        t2_fn=lambda a: seasonal_tint(add_sensor_noise(a, seed=20), (1.01, 1.0, 0.99)),
    ),
    dict(
        id="P06", category="baseline", label=0,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Urban residential block — no structural change",
        t2_fn=lambda a: add_sensor_noise(a, seed=30),
    ),
    dict(
        id="P07", category="baseline", label=0,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Rocky hillside — stable geology",
        t2_fn=lambda a: seasonal_tint(add_sensor_noise(a, seed=40), (0.99, 1.01, 1.0)),
    ),
    dict(
        id="P08", category="baseline", label=0,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Valley floor — no change, minor illumination delta",
        t2_fn=lambda a: add_sensor_noise(seasonal_tint(a, (1.0, 0.99, 1.01)), seed=50),
    ),
    dict(
        id="P09", category="tricky", label=0,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Partial cloud cover in T2 — no ground change",
        t2_fn=lambda a: add_cloud(a, cx=300, cy=180, radius=120, opacity=0.60, seed=7),
    ),
    dict(
        id="P10", category="tricky", label=0,
        date_t1="2026-03-20", date_t2="2026-04-11",
        description="Solar angle shift — diagonal shadow band, no structure change",
        t2_fn=lambda a: add_shadow_shift(a, shift_x=30, shift_y=20, intensity=0.28),
    ),
]

def generate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest      = []
    terrain_cache = {}   

    print("=" * 60)
    print("  Border Surveillance System — Dataset Generator")
    print(f"  Output  : {OUT_DIR}")
    print(f"  Pairs   : {len(PAIRS)}")
    print(f"  Size    : {SIZE} x {SIZE} px per patch")
    print("=" * 60)

    for i, pair in enumerate(PAIRS):
        pid  = pair["id"]
        seed = i * 7 + 3   

        print(f"\n  [{i+1:02d}/10]  {pid}  [{pair['category']:8s}]  label={pair['label']}")
        print(f"           {pair['description']}")
        if seed not in terrain_cache:
            terrain_cache[seed] = make_terrain(SIZE, seed=seed)
        base = terrain_cache[seed]

        t1_arr = draw_coord_overlay(add_sensor_noise(base, seed=seed + 100))

        t2_base = pair["t2_fn"](base.copy())
        t2_arr  = draw_coord_overlay(add_sensor_noise(t2_base, seed=seed + 200))

        t1_path = OUT_DIR / f"{pid}_T1_{pair['date_t1']}.png"
        t2_path = OUT_DIR / f"{pid}_T2_{pair['date_t2']}.png"
        Image.fromarray(t1_arr).save(t1_path)
        Image.fromarray(t2_arr).save(t2_path)
        print(f"           Saved: {t1_path.name}  |  {t2_path.name}")

        manifest.append({
            "patch_id":    pid,
            "category":    pair["category"],
            "label":       pair["label"],
            "label_text":  "CHANGE" if pair["label"] == 1 else "NO CHANGE",
            "date_t1":     pair["date_t1"],
            "date_t2":     pair["date_t2"],
            "description": pair["description"],
            "t1_filename": t1_path.name,
            "t2_filename": t2_path.name,
        })

    csv_path = OUT_DIR / "ground_truth_manifest.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    json_path = OUT_DIR / "ground_truth.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "project":    "Border Surveillance System",
                "region":     "Kabul Border",
                "coords":     {"lat": 34.52, "lon": 69.18},
                "satellite":  "Sentinel-2 (Optical - MSI)",
                "resolution": "10 m/px",
                "patch_size": SIZE,
                "pairs":      manifest,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(f"  Done.  {len(PAIRS)} pairs generated.")
    print(f"  Folder : {OUT_DIR}")
    print(f"  CSV    : ground_truth_manifest.csv")
    print(f"  JSON   : ground_truth.json")
    print("=" * 60)
    print("\n  You can now run:  python benchmark.py")

if __name__ == "__main__":
    generate()
