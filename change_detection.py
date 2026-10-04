import numpy as np
from dataclasses import dataclass, field

@dataclass
class ChangeResult:
    change_map:    np.ndarray          
    magnitude_map: np.ndarray          
    pct_changed:   float               
    mean_delta:    float               
    max_delta:     float               
    alert_level:   str                 
    hotspots:      list = field(default_factory=list)   
    algorithm:     str  = "mad"        
    extra:         dict = field(default_factory=dict)   

PIXEL_METHODS = ["mad", "ratio", "ndvi"]
ML_METHODS    = ["kmeans", "svm", "random_forest", "siamese", "alphaearth"]
ALL_METHODS   = PIXEL_METHODS + ML_METHODS

METHOD_LABELS = {
    "mad":           "Mean Absolute Difference",
    "ratio":         "Log-Ratio (SAR)",
    "ndvi":          "NDVI Vegetation Change",
    "kmeans":        "K-Means Clustering",
    "svm":           "Support Vector Machine",
    "random_forest": "Random Forest",
    "siamese":       "Siamese Network",
    "alphaearth":    "AlphaEarth Foundation Model",
}

def detect_changes(
    img1:      np.ndarray,
    img2:      np.ndarray,
    method:    str   = "mad",
    threshold: float = 40.0,
    seed:      int   = 42,
) -> ChangeResult:
   
    if method not in ALL_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {ALL_METHODS}"
        )

    if method in ML_METHODS:
        from algorithms import run_ml_algorithm
        result = run_ml_algorithm(img1, img2, algorithm=method,
                                  threshold=threshold, seed=seed)
        return ChangeResult(
            change_map    = result.change_map,
            magnitude_map = result.magnitude_map,
            pct_changed   = result.pct_changed,
            mean_delta    = result.mean_delta,
            max_delta     = result.max_delta,
            alert_level   = result.alert_level,
            hotspots      = result.hotspots,
            algorithm     = method,
            extra         = result.extra,
        )

    a = img1.astype(np.float32)
    b = img2.astype(np.float32)

    if method == "mad":
        magnitude = _mad(a, b)
    elif method == "ratio":
        magnitude = _log_ratio(a, b)
    else:                               # ndvi
        magnitude = _ndvi_diff(a, b)

    changed_mask = magnitude > threshold
    pct_changed  = float(changed_mask.mean() * 100)
    mean_delta   = float(magnitude.mean())
    max_delta    = float(magnitude.max())

    hotspots = _find_hotspots(magnitude, threshold)

    n_hotspots = len(hotspots)
    if n_hotspots <= 1:
        alert_level = "LOW"
    elif n_hotspots <= 4:
        alert_level = "MODERATE"
    else:
        alert_level = "HIGH"

    return ChangeResult(
        change_map    = _colorise(magnitude, threshold),
        magnitude_map = magnitude,
        pct_changed   = pct_changed,
        mean_delta    = mean_delta,
        max_delta     = max_delta,
        alert_level   = alert_level,
        hotspots      = hotspots,
        algorithm     = method,
        extra         = {},
    )

def _mad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).mean(axis=2)

def _log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps   = 1e-3
    ratio = np.abs(np.log(b + eps) - np.log(a + eps)).mean(axis=2)
    return ratio / ratio.max() * 255 if ratio.max() > 0 else ratio

def _ndvi_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps    = 1e-3
    ndvi_a = (a[:, :, 1] - a[:, :, 0]) / (a[:, :, 1] + a[:, :, 0] + eps)
    ndvi_b = (b[:, :, 1] - b[:, :, 0]) / (b[:, :, 1] + b[:, :, 0] + eps)
    return np.abs(ndvi_b - ndvi_a) * 127.5

def _colorise(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    H, W   = magnitude.shape
    out    = np.zeros((H, W, 3), dtype=np.uint8)
    m_norm = np.clip(magnitude / (magnitude.max() + 1e-6), 0, 1)

    no_change = m_norm < 0.08
    minor     = (m_norm >= 0.08) & (m_norm < 0.30)
    moderate  = (m_norm >= 0.30) & (m_norm < 0.60)
    high      = m_norm >= 0.60

    out[no_change] = [20, 30, 22]

    t = m_norm[minor]
    out[minor, 0] = (t * 80).astype(np.uint8)
    out[minor, 1] = (60 + t * 160).astype(np.uint8)
    out[minor, 2] = (t * 40).astype(np.uint8)

    t = m_norm[moderate]
    out[moderate, 0] = np.clip((200 + t * 40), 0, 255).astype(np.uint8)
    out[moderate, 1] = np.clip((130 - t * 80), 0, 255).astype(np.uint8)
    out[moderate, 2] = 10

    t = m_norm[high]
    out[high, 0] = np.clip((210 + t * 40), 0, 255).astype(np.uint8)
    out[high, 1] = 10
    out[high, 2] = 10

    return out


def _find_hotspots(
    magnitude:     np.ndarray,
    hot_threshold: float,
    min_size:      int = 30,
) -> list:
    H, W     = magnitude.shape
    visited  = np.zeros((H, W), dtype=bool)
    hot_mask = magnitude > hot_threshold
    hotspots = []

    for start_r in range(0, H, 4):
        for start_c in range(0, W, 4):
            if not hot_mask[start_r, start_c] or visited[start_r, start_c]:
                continue

            queue  = [(start_r, start_c)]
            pixels = []
            visited[start_r, start_c] = True

            while queue:
                r, c = queue.pop()
                pixels.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if hot_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            if len(pixels) < min_size:
                continue

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            cr   = int(np.mean(rows))
            cc   = int(np.mean(cols))
            rad  = int(max(max(rows) - min(rows), max(cols) - min(cols)) / 2 + 4)
            hotspots.append((cr, cc, rad))

    return hotspots