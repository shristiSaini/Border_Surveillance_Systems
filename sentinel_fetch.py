import requests
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional


# STAC endpoint
STAC_URL = "https://earth-search.aws.element84.com/v1/search"

COLLECTIONS = {
    "S2": "sentinel-2-l2a",   
    "S1": "sentinel-1-grd",   
}

IMG_SIZE = (512, 512)          

@dataclass
class SentinelScene:
    scene_id:    str
    date:        str
    satellite:   str
    cloud_cover: Optional[float]
    bbox:        list           # [minLon, minLat, maxLon, maxLat]
    thumbnail_url: Optional[str]
    image:       Optional[np.ndarray]  # H×W×3 uint8 RGB array, or None


# STAC query
def query_stac(
    lat: float,
    lon: float,
    radius_deg: float = 0.2,
    satellite: str = "S2",
    max_cloud: int = 20,
    n_scenes: int = 2,
    days_back: int = 180,
) -> list[SentinelScene]:
    bbox = [
        round(lon - radius_deg, 6),
        round(lat - radius_deg, 6),
        round(lon + radius_deg, 6),
        round(lat + radius_deg, 6),
    ]

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)
    datetime_range = (
        start_dt.strftime("%Y-%m-%dT%H:%M:%SZ") + "/" +
        end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    collection = COLLECTIONS[satellite]

    payload = {
        "collections": [collection],
        "bbox":        bbox,
        "datetime":    datetime_range,
        "sortby":      [{"field": "datetime", "direction": "desc"}],
        "limit":       n_scenes * 3,   # fetch extra; we'll filter below
    }

    if satellite == "S2":
        payload["query"] = {"eo:cloud_cover": {"lte": max_cloud}}

    print(f"[STAC] Querying {collection} …")
    print(f"       BBox  : {bbox}")
    print(f"       Period: {datetime_range}")

    try:
        resp = requests.post(STAC_URL, json=payload, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[STAC] Request failed: {exc}")
        print("[STAC] Falling back to simulated imagery.")
        return []

    features = resp.json().get("features", [])
    print(f"[STAC] Found {len(features)} matching scenes.")

    scenes = []
    for feat in features[:n_scenes]:
        props   = feat.get("properties", {})
        assets  = feat.get("assets", {})

        thumb = (
            assets.get("thumbnail", {}).get("href") or
            assets.get("preview",   {}).get("href") or
            assets.get("overview",  {}).get("href") or
            None
        )

        scenes.append(SentinelScene(
            scene_id    = feat.get("id", "unknown"),
            date        = props.get("datetime", "unknown")[:10],
            satellite   = satellite,
            cloud_cover = props.get("eo:cloud_cover"),
            bbox        = feat.get("bbox", bbox),
            thumbnail_url = thumb,
            image       = None,
        ))

    return scenes


# Download Image 
def download_images(scenes: list[SentinelScene]) -> list[SentinelScene]:
    
    for sc in scenes:
        if not sc.thumbnail_url:
            print(f"[IMG] No thumbnail URL for {sc.scene_id} — using placeholder.")
            sc.image = _placeholder(sc)
            continue

        print(f"[IMG] Downloading thumbnail for {sc.scene_id} …")
        try:
            r = requests.get(sc.thumbnail_url, timeout=30)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB").resize(IMG_SIZE)
            sc.image = np.array(img, dtype=np.uint8)
            print(f"[IMG] OK — shape {sc.image.shape}")
        except Exception as exc:
            print(f"[IMG] Download failed ({exc}) — using placeholder.")
            sc.image = _placeholder(sc)

    return scenes
# Placeholder
def _placeholder(sc: SentinelScene) -> np.ndarray:

    rng = np.random.default_rng(abs(hash(sc.scene_id)) % (2**32))

    H, W = IMG_SIZE

    img = np.zeros((H, W, 3), dtype=np.float32)
    for scale in [1, 2, 4, 8, 16]:
        noise = rng.random((H // scale + 2, W // scale + 2))
        from PIL import Image as _PIL
        noise_up = np.array(
            _PIL.fromarray((noise * 255).astype(np.uint8)).resize((W, H)),
            dtype=np.float32
        ) / 255.0
        img[:, :, 0] += noise_up / scale
        img[:, :, 1] += noise_up / scale
        img[:, :, 2] += noise_up / scale

    img = img / img.max()

    if sc.satellite == "S1":
        # SAR 
        grey = img[:, :, 0]
        img[:, :, 0] = grey * 0.7
        img[:, :, 1] = grey * 0.8
        img[:, :, 2] = grey * 1.0
    else:
        # S2 
        img[:, :, 0] = img[:, :, 0] * 0.6 + 0.15
        img[:, :, 1] = img[:, :, 1] * 0.7 + 0.20
        img[:, :, 2] = img[:, :, 2] * 0.4 + 0.05

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
