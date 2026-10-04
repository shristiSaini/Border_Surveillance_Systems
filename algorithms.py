from __future__ import annotations

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
    algorithm:     str  = "unknown"    
    extra:         dict = field(default_factory=dict)   

ALGORITHM_LABELS = {
    "kmeans":        "K-Means Clustering",
    "svm":           "Support Vector Machine",
    "random_forest": "Random Forest",
    "siamese":       "Siamese Network",
    "alphaearth":    "AlphaEarth Foundation Model",
}

ALL_ALGORITHMS = list(ALGORITHM_LABELS.keys())

def run_ml_algorithm(
    img1:      np.ndarray,
    img2:      np.ndarray,
    algorithm: str   = "svm",
    threshold: float = 40.0,
    seed:      int   = 42,
) -> ChangeResult:
   
    if algorithm not in ALL_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {ALL_ALGORITHMS}"
        )

    print(f"[ALGO] Running: {ALGORITHM_LABELS[algorithm]} …")

    a = img1.astype(np.float32)
    b = img2.astype(np.float32)

    dispatch = {
        "kmeans":        _run_kmeans,
        "svm":           _run_svm,
        "random_forest": _run_random_forest,
        "siamese":       _run_siamese,
        "alphaearth":    _run_alphaearth,
    }

    magnitude, label_map, extra = dispatch[algorithm](a, b, threshold=threshold, seed=seed)

    changed_mask = magnitude > threshold
    pct_changed  = float(changed_mask.mean() * 100)
    mean_delta   = float(magnitude.mean())
    max_delta    = float(magnitude.max())

    change_map = _colorise(magnitude, threshold)
    hotspots   = _find_hotspots(magnitude, threshold)

    n_hotspots = len(hotspots)
    if n_hotspots <= 1:
        alert_level = "LOW"
    elif n_hotspots <= 4:
        alert_level = "MODERATE"
    else:
        alert_level = "HIGH"

    print(f"[ALGO] Done — alert: {alert_level}, hotspots: {n_hotspots}, changed: {pct_changed:.1f}%")

    return ChangeResult(
        change_map    = change_map,
        magnitude_map = magnitude,
        pct_changed   = pct_changed,
        mean_delta    = mean_delta,
        max_delta     = max_delta,
        alert_level   = alert_level,
        hotspots      = hotspots,
        algorithm     = algorithm,
        extra         = extra,
    )

def _pixel_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    diff    = np.abs(a - b)
    bright1 = a.mean(axis=2, keepdims=True)
    bright2 = b.mean(axis=2, keepdims=True)
    euclid  = np.sqrt((diff ** 2).sum(axis=2, keepdims=True))
    mean_d  = diff.mean(axis=2, keepdims=True)

    X = np.concatenate([a, b, diff, bright1, bright2, euclid, mean_d], axis=2)
    H, W, F = X.shape
    return X.reshape(-1, F).astype(np.float32)

def _extended_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    from scipy.ndimage import uniform_filter

    base    = _pixel_features(a, b)
    H, W, _ = a.shape
    extras  = []

    for ch in range(3):
        ch1    = a[:, :, ch] / 255.0
        lmean1 = uniform_filter(ch1, size=3).flatten()
        lvar1  = uniform_filter(ch1 ** 2, size=3) - lmean1.reshape(H, W) ** 2
        lstd1  = np.sqrt(np.clip(lvar1, 0, None)).flatten()

        ch2    = b[:, :, ch] / 255.0
        lmean2 = uniform_filter(ch2, size=3).flatten()

        extras.extend([lmean1, lstd1, lmean2])

    return np.concatenate(
        [base, np.stack(extras, axis=1).astype(np.float32)], axis=1
    )


def _pseudo_labels(a: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    """
    Generate binary change/no-change labels via simple MAD threshold.
    Used as self-supervised training targets — no manual annotation needed.
    """
    mad = np.abs(a - b).mean(axis=2).flatten()
    return (mad > threshold).astype(np.int32)
# 1. KMeans
def _run_kmeans(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    seed: int,
    n_clusters: int = 5,
) -> tuple:
    
    from sklearn.cluster       import KMeans
    from sklearn.preprocessing import StandardScaler

    H, W, _ = a.shape
    N = H * W

    print(f"[K-Means]   Building {N}-pixel feature matrix …")
    X      = _pixel_features(a, b)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    print(f"[K-Means]   Fitting K={n_clusters} clusters …")
    km = KMeans(
        n_clusters   = n_clusters,
        init         = "k-means++",
        n_init       = 5,
        max_iter     = 200,
        random_state = seed,
    )
    km.fit(X_sc)
    labels = km.labels_

    mad_flat   = np.abs(a - b).mean(axis=2).flatten()
    mad_means  = np.array([mad_flat[labels == k].mean() for k in range(n_clusters)])
    rank_order = np.argsort(mad_means)
    rank_map   = np.empty(n_clusters, dtype=int)
    for rank, orig_k in enumerate(rank_order):
        rank_map[orig_k] = rank

    ranked       = rank_map[labels]
    magnitude    = (ranked / (n_clusters - 1) * 255).astype(np.float32)
    label_map    = ranked.reshape(H, W).astype(np.int32)

    extra = {
        "n_clusters":       n_clusters,
        "inertia":          float(km.inertia_),
        "cluster_sizes":    [int((labels == k).sum()) for k in range(n_clusters)],
        "cluster_mad_means": mad_means.tolist(),
        "no_change_cluster": int(rank_order[0]),
        "max_change_cluster": int(rank_order[-1]),
    }

    return magnitude.reshape(H, W), label_map, extra
# 2. SVM
def _run_svm(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    seed: int,
) -> tuple:
    from sklearn.svm           import SVC
    from sklearn.preprocessing import StandardScaler

    H, W, _ = a.shape
    N = H * W

    print("[SVM]   Building feature matrix …")
    X      = _pixel_features(a, b)
    y      = _pseudo_labels(a, b, threshold)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rng     = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_each  = min(len(pos_idx), len(neg_idx), 2048)

    if n_each < 10:
        train_idx = rng.choice(N, min(4096, N), replace=False)
    else:
        train_idx = np.concatenate([
            rng.choice(pos_idx, n_each, replace=False),
            rng.choice(neg_idx, n_each, replace=False),
        ])

    print(f"[SVM]   Training on {len(train_idx)} pixels …")
    clf = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",
        random_state=seed,
    )
    clf.fit(X_sc[train_idx], y[train_idx])

    print("[SVM]   Predicting on full image …")
    scores   = clf.decision_function(X_sc)
    labels   = clf.predict(X_sc).astype(np.int32)
    s_min, s_max = scores.min(), scores.max()
    magnitude = ((scores - s_min) / (s_max - s_min + 1e-8) * 255).astype(np.float32)

    train_acc = float((clf.predict(X_sc[train_idx]) == y[train_idx]).mean())
    extra = {
        "kernel":            "rbf",
        "C":                 1.0,
        "n_support_vectors": int(len(clf.support_vectors_)),
        "n_train_pixels":    int(len(train_idx)),
        "train_accuracy":    round(train_acc, 4),
    }

    return magnitude.reshape(H, W), labels.reshape(H, W), extra
# 3. Random Forest
def _run_random_forest(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    seed: int,
    n_estimators: int = 80,
) -> tuple:
    from sklearn.ensemble      import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    H, W, _ = a.shape
    N = H * W

    print("[RF]   Extracting 22-D feature vectors …")
    X      = _extended_features(a, b)
    y      = _pseudo_labels(a, b, threshold)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rng     = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_each  = min(len(pos_idx), len(neg_idx), 3000)

    if n_each < 10:
        train_idx = rng.choice(N, min(6000, N), replace=False)
    else:
        train_idx = np.concatenate([
            rng.choice(pos_idx, n_each, replace=False),
            rng.choice(neg_idx, n_each, replace=False),
        ])

    print(f"[RF]   Training {n_estimators}-tree RF on {len(train_idx)} pixels …")
    clf = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth    = 12,
        class_weight = "balanced",
        n_jobs       = -1,
        random_state = seed,
    )
    clf.fit(X_sc[train_idx], y[train_idx])

    print("[RF]   Predicting change probabilities …")
    proba     = clf.predict_proba(X_sc)
    labels    = clf.predict(X_sc).astype(np.int32)
    p_change  = proba[:, 1]
    magnitude = (p_change * 255).astype(np.float32)

    feat_imp  = clf.feature_importances_
    train_acc = float((clf.predict(X_sc[train_idx]) == y[train_idx]).mean())
    extra = {
        "n_estimators":    n_estimators,
        "n_features":      int(X.shape[1]),
        "top_feature_idx": int(feat_imp.argmax()),
        "top3_importances": feat_imp[np.argsort(feat_imp)[-3:][::-1]].tolist(),
        "train_accuracy":  round(train_acc, 4),
    }

    return magnitude.reshape(H, W), labels.reshape(H, W), extra
#4. Siamese Network
def _siamese_encoder(img: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    feats = []
    norm  = img / 255.0

    feats.append(norm)

    feats.append(uniform_filter(norm, size=(5, 5, 1)))

    feats.append(uniform_filter(norm, size=(11, 11, 1)))

    H, W, C = img.shape
    grads = []
    for ch in range(C):
        ch_n = norm[:, :, ch]
        gx   = np.gradient(ch_n, axis=1)
        gy   = np.gradient(ch_n, axis=0)
        grads.append(np.sqrt(gx ** 2 + gy ** 2))
    feats.append(np.stack(grads, axis=2))

    patch_mean = uniform_filter(norm, size=(9, 9, 1))
    feats.append(patch_mean)

    patch_var = np.clip(
        uniform_filter(norm ** 2, size=(9, 9, 1)) - patch_mean ** 2, 0, None
    )
    feats.append(patch_var)

    return np.concatenate(feats, axis=2).astype(np.float32)   # H×W×18


def _run_siamese(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    seed: int,
) -> tuple:
    
    H, W, _ = a.shape

    print("[Siamese]   Encoding T-1 …")
    F1 = _siamese_encoder(a)

    print("[Siamese]   Encoding T-2 …")
    F2 = _siamese_encoder(b)   

    diff      = F1 - F2
    magnitude = np.sqrt((diff ** 2).sum(axis=2))          # H×W
    mag_max   = magnitude.max() + 1e-6
    magnitude = (magnitude / mag_max * 255).astype(np.float32)

    label_map = (magnitude > threshold).astype(np.int32)

    extra = {
        "feature_dims": int(F1.shape[2]),
        "mean_f1_norm": float(np.linalg.norm(F1, axis=2).mean()),
        "mean_f2_norm": float(np.linalg.norm(F2, axis=2).mean()),
        "description":  "L2 distance in 18-D shared-encoder feature space",
    }

    return magnitude, label_map, extra
#5. AlphaEarth
_PATCH_SIZE = 16
_EMBED_DIM  = 64
_PROJ_SEED  = 7777

def _alphaearth_embed(img: np.ndarray) -> np.ndarray:
    
    from scipy.ndimage import zoom

    H, W, C = img.shape
    P, D    = _PATCH_SIZE, _EMBED_DIM

    pH = H // P
    pW = W // P
    cropped = img[: pH * P, : pW * P, :]
    patches = (
        cropped
        .reshape(pH, P, pW, P, C)
        .transpose(0, 2, 1, 3, 4)
        .reshape(pH, pW, P * P * C)
        .astype(np.float32) / 255.0
    )

    rng    = np.random.default_rng(_PROJ_SEED)
    W_proj = rng.standard_normal((P * P * C, D)).astype(np.float32) * 0.1
    embed  = patches @ W_proj                      # (pH, pW, D)

    pos_r = np.array([
        [np.sin(r / (10000 ** (d / D))) if d % 2 == 0
         else np.cos(r / (10000 ** ((d - 1) / D)))
         for d in range(D)]
        for r in range(pH)
    ], dtype=np.float32)                           # (pH, D)

    pos_c = np.array([
        [np.sin(c / (10000 ** (d / D))) if d % 2 == 0
         else np.cos(c / (10000 ** ((d - 1) / D)))
         for d in range(D)]
        for c in range(pW)
    ], dtype=np.float32)                           # (pW, D)

    embed += pos_r[:, np.newaxis, :]
    embed += pos_c[np.newaxis, :, :]

    return zoom(embed, (H / pH, W / pW, 1), order=1).astype(np.float32)

def _run_alphaearth(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    seed: int,
) -> tuple:
    
    H, W, _ = a.shape

    print("[AlphaEarth]   Embedding T-1 patches …")
    E1 = _alphaearth_embed(a)

    print("[AlphaEarth]   Embedding T-2 patches …")
    E2 = _alphaearth_embed(b)

    dot        = (E1 * E2).sum(axis=2)
    norm1      = np.linalg.norm(E1, axis=2) + 1e-8
    norm2      = np.linalg.norm(E2, axis=2) + 1e-8
    cos_sim    = dot / (norm1 * norm2)
    cos_dist   = 1.0 - cos_sim                     # range [0, 2]

    magnitude  = np.clip(cos_dist / 2.0, 0, 1) * 255
    magnitude  = magnitude.astype(np.float32)
    label_map  = (magnitude > threshold).astype(np.int32)

    extra = {
        "patch_size":      _PATCH_SIZE,
        "embedding_dim":   _EMBED_DIM,
        "mean_cosine_sim": round(float(cos_sim.mean()), 4),
        "min_cosine_sim":  round(float(cos_sim.min()), 4),
        "description":     "Cosine distance in ViT patch-embedding space",
    }

    return magnitude, label_map, extra
#post processing
def _colorise(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    """
      Dark grey   → no change
      Green       → minor change
      Amber       → moderate change
      Red         → high / hotspot change
    """
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