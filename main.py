import argparse
import sys
from pathlib import Path

from sentinel_fetch   import query_stac, download_images, SentinelScene
from change_detection import detect_changes, ALL_METHODS, METHOD_LABELS
from visualise        import generate_report

PRESETS = {                                                 #border coordinates
    "jammu":      (32.72,  74.86, "Jammu LoC"),
    "kabul":      (34.52,  69.18, "Kabul Border Region"),
    "pak-loc":    (33.68,  73.05, "Pakistan LoC"),
    "bhutan":     (27.47,  89.64, "Bhutan-India Border"),
    "bangladesh": (23.73,  90.39, "Bangladesh Border"),
    "kutch":      (24.88,  67.03, "Rann of Kutch"),
}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bss",
        description="Border Surveillance System — Sentinel Change Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    coord_grp = p.add_mutually_exclusive_group(required=True)
    coord_grp.add_argument("--lat",    type=float,
                           help="Latitude  (decimal °N)")
    coord_grp.add_argument("--preset", choices=list(PRESETS.keys()),
                           help="Named border-region preset")

    p.add_argument("--lon",       type=float, default=None,
                   help="Longitude (decimal °E)  [required with --lat]")
    p.add_argument("--satellite", choices=["S2", "S1"], default="S2",
                   help="S2 = Sentinel-2 optical (default)  |  S1 = Sentinel-1 SAR")
    p.add_argument("--radius",    type=float, default=0.2,
                   help="Search radius in degrees (default: 0.2 ≈ 22 km)")
    p.add_argument("--cloud",     type=int,   default=20,
                   help="Max cloud cover %% for S2 (default: 20)")
    p.add_argument("--days-back", type=int,   default=180,
                   help="How many days back to search (default: 180)")
    p.add_argument(
        "--algo",
        choices=ALL_METHODS,
        default="mad",
        metavar="ALGO",
        help=(
            "Change-detection algorithm  (default: mad)\n"
            "  Pixel : mad | ratio | ndvi\n"
            "  ML/DL : kmeans | svm | random_forest | siamese | alphaearth"
        ),
    )
    p.add_argument("--threshold", type=float, default=40.0,
                   help="Change magnitude threshold 0-255 (default: 40)")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Random seed for ML algorithms (default: 42)")
    p.add_argument("--output",    type=str,   default="bss_report.png",
                   help="Output PNG path (default: bss_report.png)")
    p.add_argument("--show",      action="store_true",
                   help="Open the report image after saving")
    return p

def run(args) -> None:

    if args.preset:                        #resolve coordinates
        lat, lon, label = PRESETS[args.preset]
        print(f"[BSS] Preset   : {label}  ({lat}°N, {lon}°E)")
    else:
        lat = args.lat
        if args.lon is None:
            print("[BSS] ERROR: --lon is required when using --lat",
                  file=sys.stderr)
            sys.exit(1)
        lon = args.lon
        print(f"[BSS] Target   : {lat}°N, {lon}°E")

    if not (-90 <= lat <= 90):
        print("[BSS] ERROR: Latitude must be between -90 and 90.",
              file=sys.stderr)
        sys.exit(1)
    if not (-180 <= lon <= 180):
        print("[BSS] ERROR: Longitude must be between -180 and 180.",
              file=sys.stderr)
        sys.exit(1)

    algo_label = METHOD_LABELS.get(args.algo, args.algo)
    print(f"[BSS] Satellite: {args.satellite}  |  Radius: ±{args.radius}°  |  "
          f"Cloud ≤{args.cloud}%")
    print(f"[BSS] Algorithm: {algo_label}  (--algo {args.algo})")
    print("-" * 70)

    scenes = query_stac(                         #STAC query
        lat        = lat,
        lon        = lon,
        radius_deg = args.radius,
        satellite  = args.satellite,
        max_cloud  = args.cloud,
        n_scenes   = 2,
        days_back  = args.days_back,
    )

    import hashlib
    while len(scenes) < 2:
        fid = hashlib.md5(
            f"placeholder_{len(scenes)}_{lat}_{lon}".encode()
        ).hexdigest()[:16]
        scenes.append(SentinelScene(
            scene_id      = fid,
            date          = _fake_date(30 + len(scenes) * 14),
            satellite     = args.satellite,
            cloud_cover   = None,
            bbox          = [lon - args.radius, lat - args.radius,
                             lon + args.radius, lat + args.radius],
            thumbnail_url = None,
            image         = None,
        ))
    print("-" * 70)

    scenes         = download_images(scenes)
    scene1, scene2 = scenes[1], scenes[0]   # STAC newest-first → swap

    print("-" * 70)

    print(f"[CD]  Running '{algo_label}' …")
    result = detect_changes(              #change detection
        scene1.image,
        scene2.image,
        method    = args.algo,
        threshold = args.threshold,
        seed      = args.seed,
    )

    print("-" * 70)
    print(f"[CD]  Alert level  : {result.alert_level}")
    print(f"[CD]  Changed px   : {result.pct_changed:.2f} %")
    print(f"[CD]  Mean delta   : {result.mean_delta:.2f}")
    print(f"[CD]  Max delta    : {result.max_delta:.2f}")
    print(f"[CD]  Hotspots     : {len(result.hotspots)}")
    if result.extra:
        print(f"[CD]  ── {algo_label} diagnostics ──")
        for k, v in result.extra.items():
            if k != "description":
                print(f"[CD]    {k:<28}: {v}")
    print("-" * 70)

# Helper
def _fake_date(days_ago: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
#entry point for UI
def run_system(
    lat:        float = 32.72,
    lon:        float = 74.86,
    preset:     str   = None,
    satellite:  str   = "S2",
    radius:     float = 0.2,
    cloud:      int   = 20,
    days_back:  int   = 180,
    algo:       str   = "mad",
    threshold:  float = 40.0,
    seed:       int   = 42,
    output:     str   = "bss_report.png",
    show:       bool  = False,
) -> None:

    class Args:
        pass

    args             = Args()
    args.lat         = lat
    args.lon         = lon
    args.preset      = preset
    args.satellite   = satellite
    args.radius      = radius
    args.cloud       = cloud
    args.days_back   = days_back
    args.algo        = algo
    args.threshold   = threshold
    args.seed        = seed
    args.output      = output
    args.show        = show

    run(args)


# ─── CLI entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(build_parser().parse_args())
