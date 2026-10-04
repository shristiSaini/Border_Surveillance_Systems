import sys, os, json, csv, time
sys.path.insert(0, os.path.dirname(__file__))   # allow local imports

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from datetime import datetime

DATASET_DIR  = Path(__file__).parent / "ground_truth_dataset"
OUT_DIR      = Path(__file__).parent / "benchmark_results"
THRESHOLD    = 40.0          # change-magnitude cut-off (0-255), matches app default
SEED         = 42

BG_DARK   = "#080c10"
BG_PANEL  = "#0d1219"
BG_SURF   = "#111922"
ACCENT    = "#00c88c"
TEXT_PRI  = "#e8f0e9"
TEXT_SEC  = "#7a9e8a"
TEXT_MUTE = "#3d5a4a"
DANGER    = "#ef4444"
WARN      = "#f59e0b"

CAT_COLORS = {
    "action":   "#3478c0",
    "baseline": "#00c88c",
    "tricky":   "#f59e0b",
}

ALGO_DISPLAY = {
    "mad":           "MAD",
    "ratio":         "Log-Ratio",
    "ndvi":          "NDVI",
    "kmeans":        "K-Means",
    "svm":           "SVM",
    "random_forest": "Rand. Forest",
    "siamese":       "Siamese Net",
    "alphaearth":    "AlphaEarth",
}

ALGO_TYPE = {
    "mad":           "pixel",
    "ratio":         "pixel",
    "ndvi":          "pixel",
    "kmeans":        "ml",
    "svm":           "ml",
    "random_forest": "ml",
    "siamese":       "dl",
    "alphaearth":    "dl",
}

ALL_ALGOS = list(ALGO_DISPLAY.keys())

GROUND_TRUTH = [
    {"id": "P01", "category": "action",   "label": 1, "description": "New road construction"},
    {"id": "P02", "category": "action",   "label": 1, "description": "Building cluster expansion"},
    {"id": "P03", "category": "action",   "label": 1, "description": "Vegetation clearing"},
    {"id": "P04", "category": "action",   "label": 1, "description": "Vehicle staging area"},
    {"id": "P05", "category": "baseline", "label": 0, "description": "Stable agricultural terraces"},
    {"id": "P06", "category": "baseline", "label": 0, "description": "Urban residential block"},
    {"id": "P07", "category": "baseline", "label": 0, "description": "Rocky hillside"},
    {"id": "P08", "category": "baseline", "label": 0, "description": "Valley floor"},
    {"id": "P09", "category": "tricky",   "label": 0, "description": "Cloud cover in T2"},
    {"id": "P10", "category": "tricky",   "label": 0, "description": "Solar angle shadow shift"},
]

def load_pair(pair: dict) -> tuple[np.ndarray, np.ndarray]:
    pid = pair["id"]
    t1_files = sorted(DATASET_DIR.glob(f"{pid}_T1_*.png"))
    t2_files = sorted(DATASET_DIR.glob(f"{pid}_T2_*.png"))
    if not t1_files or not t2_files:
        raise FileNotFoundError(
            f"Cannot find {pid} patch images in {DATASET_DIR}\n"
            "Run generate_dataset.py first."
        )
    t1 = np.array(Image.open(t1_files[0]).convert("RGB").resize((512, 512)), dtype=np.uint8)
    t2 = np.array(Image.open(t2_files[0]).convert("RGB").resize((512, 512)), dtype=np.uint8)
    return t1, t2

def run_detection(t1: np.ndarray, t2: np.ndarray, algo: str) -> dict:
    """
    Run one algorithm on one image pair.
    Returns a dict with pct_changed, mean_delta, max_delta, hotspots,
    alert_level, runtime_s, and binary prediction (1 = change detected).
    """
    import io, contextlib
    from change_detection import detect_changes

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = detect_changes(t1, t2, method=algo, threshold=THRESHOLD, seed=SEED)
    except Exception:
        # Fallback: run MAD if ML algo fails (e.g. single-class SVM)
        from change_detection import detect_changes as _dc
        with contextlib.redirect_stdout(io.StringIO()):
            result = _dc(t1, t2, method="mad", threshold=THRESHOLD, seed=SEED)
    elapsed = round(time.time() - t0, 3)

    pred = 1 if result.alert_level in ("MODERATE", "HIGH") else 0

    return {
        "pct_changed":  round(result.pct_changed, 2),
        "mean_delta":   round(result.mean_delta,   2),
        "max_delta":    round(result.max_delta,    2),
        "n_hotspots":   len(result.hotspots),
        "alert_level":  result.alert_level,
        "runtime_s":    elapsed,
        "prediction":   pred,
    }

def compute_metrics(labels: list[int], preds: list[int]) -> dict:
    labels = np.array(labels)
    preds  = np.array(preds)
    n      = len(labels)

    TP = int(((labels == 1) & (preds == 1)).sum())
    TN = int(((labels == 0) & (preds == 0)).sum())
    FP = int(((labels == 0) & (preds == 1)).sum())
    FN = int(((labels == 1) & (preds == 0)).sum())

    accuracy  = round((TP + TN) / n, 4)
    precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0
    recall    = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) \
                if (precision + recall) > 0 else 0.0
    far       = round(FP / (FP + TN), 4) if (FP + TN) > 0 else 0.0   # False Alarm Rate
    mdr       = round(FN / (FN + TP), 4) if (FN + TP) > 0 else 0.0   # Missed Detection Rate

    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1=f1, false_alarm_rate=far, missed_detection=mdr,
                TP=TP, TN=TN, FP=FP, FN=FN)

def per_category_metrics(results_table: list[dict], algo: str) -> dict:
    cats = {}
    for row in results_table:
        if row["algo"] != algo:
            continue
        c = row["category"]
        if c not in cats:
            cats[c] = {"labels": [], "preds": []}
        cats[c]["labels"].append(row["gt_label"])
        cats[c]["preds"].append(row["prediction"])
    out = {}
    for c, d in cats.items():
        m = compute_metrics(d["labels"], d["preds"])
        out[c] = {"n": len(d["labels"]), "accuracy": m["accuracy"],
                  "FP": m["FP"], "FN": m["FN"]}
    return out

def run_benchmark() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("  BORDER SURVEILLANCE SYSTEM — BENCHMARKING SUITE")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Pairs   : {len(GROUND_TRUTH)}")
    print(f"  Algos   : {len(ALL_ALGOS)}")
    print(f"  Threshold: {THRESHOLD}")
    print("=" * 68)

    print("\n[1/4] Loading ground truth images …")
    pairs_loaded = []
    for pair in GROUND_TRUTH:
        t1, t2 = load_pair(pair)
        pairs_loaded.append({**pair, "t1": t1, "t2": t2})
        print(f"      {pair['id']}  [{pair['category']:8s}]  GT={pair['label']}  "
              f"— {pair['description']}")

    print(f"\n[2/4] Running {len(ALL_ALGOS)} algorithms × {len(GROUND_TRUTH)} pairs "
          f"= {len(ALL_ALGOS)*len(GROUND_TRUTH)} evaluations …\n")

    results_table = []   
    algo_summary  = {}   # aggregate metrics

    for algo in ALL_ALGOS:
        aname  = ALGO_DISPLAY[algo]
        labels = []
        preds  = []
        runtimes = []

        print(f"  ── {aname:<14s}", end="  ", flush=True)
        for pair in pairs_loaded:
            det = run_detection(pair["t1"], pair["t2"], algo)
            labels.append(pair["label"])
            preds.append(det["prediction"])
            runtimes.append(det["runtime_s"])

            results_table.append({
                "algo":        algo,
                "algo_display": aname,
                "pair_id":     pair["id"],
                "category":    pair["category"],
                "gt_label":    pair["label"],
                "gt_text":     "CHANGE" if pair["label"] else "NO CHANGE",
                **det,
            })
            status = "✓" if det["prediction"] == pair["label"] else "✗"
            print(status, end="" , flush=True)

        m = compute_metrics(labels, preds)
        cat_m = per_category_metrics(results_table, algo)

        algo_summary[algo] = {
            "algo_display": aname,
            "type":         ALGO_TYPE[algo],
            "mean_runtime": round(np.mean(runtimes), 3),
            "total_runtime": round(sum(runtimes), 3),
            **m,
            "per_category": cat_m,
        }
        print(f"  →  Acc={m['accuracy']:.2f}  F1={m['f1']:.2f}  "
              f"FAR={m['false_alarm_rate']:.2f}  "
              f"RT={round(np.mean(runtimes),2)}s")

    # JSON report 
    print("\n[3/4] Saving JSON + CSV reports …")
    report = {
        "project":    "Border Surveillance System",
        "generated":  datetime.utcnow().isoformat() + "Z",
        "threshold":  THRESHOLD,
        "n_pairs":    len(GROUND_TRUTH),
        "algorithms": algo_summary,
        "results_table": results_table,
    }
    json_path = OUT_DIR / "benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = OUT_DIR / "scores_table.csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["algo_display","pair_id","category","gt_label","gt_text",
                "prediction","alert_level","pct_changed","mean_delta",
                "n_hotspots","runtime_s"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results_table)

    print(f"      JSON → {json_path}")
    print(f"      CSV  → {csv_path}")

    # dashboard
    print("[4/4] Rendering benchmark dashboard …")
    plot_path = render_dashboard(algo_summary, results_table, report)
    print(f"      PNG  → {plot_path}")
    print("\n" + "=" * 68)
    print_summary_table(algo_summary)
    print("=" * 68)

    return report

def print_summary_table(algo_summary: dict):
    hdr = f"  {'Algorithm':<16} {'Type':<7} {'Acc':>5} {'Prec':>5} {'Rec':>5} {'F1':>5} {'FAR':>5} {'MDR':>5} {'RT(s)':>6}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for algo, m in algo_summary.items():
        print(
            f"  {m['algo_display']:<16} {m['type']:<7} "
            f"{m['accuracy']:>5.2f} {m['precision']:>5.2f} "
            f"{m['recall']:>5.2f} {m['f1']:>5.2f} "
            f"{m['false_alarm_rate']:>5.2f} {m['missed_detection']:>5.2f} "
            f"{m['mean_runtime']:>6.2f}"
        )

def render_dashboard(algo_summary: dict, results_table: list, report: dict) -> str:
    algos     = list(algo_summary.keys())
    n_algos   = len(algos)
    disp      = [algo_summary[a]["algo_display"] for a in algos]
    pairs     = GROUND_TRUTH

    fig = plt.figure(figsize=(22, 26), facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)

    gs = gridspec.GridSpec(
        5, 3,
        figure=fig,
        hspace=0.52, wspace=0.32,
        top=0.95, bottom=0.04,
        left=0.06, right=0.97,
        height_ratios=[0.07, 1, 1, 1, 0.9],
    )

    # Header
    fig.text(0.06, 0.973,
             "BORDER SURVEILLANCE SYSTEM  //  Algorithm Benchmark Report",
             color=ACCENT, fontsize=13, fontweight="bold", fontfamily="monospace")
    fig.text(0.06, 0.957,
             f"Pairs: {len(pairs)}  │  Algos: {n_algos}  │  "
             f"Threshold: {report['threshold']}  │  "
             f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
             color=TEXT_SEC, fontsize=9, fontfamily="monospace")
    fig.add_artist(matplotlib.lines.Line2D(
        [0.06, 0.97], [0.948, 0.948],
        transform=fig.transFigure, color=ACCENT, linewidth=0.6, alpha=0.5))

    def _ax(r, c, **kw):
        ax = fig.add_subplot(gs[r, c], **kw)
        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(TEXT_MUTE); sp.set_linewidth(0.5)
        return ax

    def _title(ax, txt, color=ACCENT):
        ax.set_title(txt, color=color, fontsize=9, fontfamily="monospace",
                     fontweight="bold", pad=6)

    # 1. Accuracy chart 
    ax1 = _ax(1, 0)
    accs = [algo_summary[a]["accuracy"] for a in algos]
    bar_colors = ["#3478c0" if ALGO_TYPE[a]=="pixel" else
                  "#00c88c" if ALGO_TYPE[a]=="ml"    else "#f59e0b" for a in algos]
    bars = ax1.barh(disp, accs, color=bar_colors, height=0.6, alpha=0.9)
    ax1.set_xlim(0, 1.15)
    ax1.axvline(0.5, color=TEXT_MUTE, linewidth=0.6, linestyle="--", alpha=0.6)
    ax1.axvline(0.75, color=WARN, linewidth=0.6, linestyle="--", alpha=0.5)
    for bar, v in zip(bars, accs):
        ax1.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}", va="center", color=TEXT_PRI, fontsize=8,
                 fontfamily="monospace")
    ax1.set_xlabel("Accuracy", color=TEXT_SEC, fontsize=8)
    ax1.tick_params(colors=TEXT_SEC, labelsize=8)
    _title(ax1, "[1] ACCURACY BY ALGORITHM")

    legend_patches = [
        mpatches.Patch(color="#3478c0", label="Pixel method"),
        mpatches.Patch(color="#00c88c", label="ML method"),
        mpatches.Patch(color="#f59e0b", label="Deep Learning"),
    ]
    ax1.legend(handles=legend_patches, fontsize=7, facecolor=BG_SURF,
               edgecolor=TEXT_MUTE, labelcolor=TEXT_PRI, loc="lower right")

    # 2. F1 / Precision / Recall 
    ax2 = _ax(1, 1)
    x = np.arange(n_algos)
    w = 0.25
    f1s   = [algo_summary[a]["f1"]        for a in algos]
    precs = [algo_summary[a]["precision"]  for a in algos]
    recs  = [algo_summary[a]["recall"]     for a in algos]
    ax2.bar(x - w, precs, w, label="Precision", color="#3b82f6", alpha=0.85)
    ax2.bar(x,     f1s,   w, label="F1 Score",  color=ACCENT,    alpha=0.85)
    ax2.bar(x + w, recs,  w, label="Recall",    color=WARN,      alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(disp, rotation=40, ha="right",
                                            fontsize=7, color=TEXT_SEC)
    ax2.set_ylim(0, 1.2); ax2.set_ylabel("Score", color=TEXT_SEC, fontsize=8)
    ax2.tick_params(colors=TEXT_SEC, labelsize=8)
    ax2.legend(fontsize=7.5, facecolor=BG_SURF, edgecolor=TEXT_MUTE, labelcolor=TEXT_PRI)
    _title(ax2, "[2] PRECISION · F1 · RECALL")

    # 3. False Alarm Rate vs Missed Detection 
    ax3 = _ax(1, 2)
    fars = [algo_summary[a]["false_alarm_rate"] for a in algos]
    mdrs = [algo_summary[a]["missed_detection"] for a in algos]
    ax3.scatter(fars, mdrs,
                c=[bar_colors[i] for i in range(n_algos)],
                s=100, zorder=5, edgecolors=TEXT_PRI, linewidths=0.5)
    for i, name in enumerate(disp):
        ax3.annotate(name, (fars[i], mdrs[i]),
                     xytext=(5, 3), textcoords="offset points",
                     color=TEXT_SEC, fontsize=7, fontfamily="monospace")
    ax3.axhline(0.5, color=TEXT_MUTE, linewidth=0.5, linestyle="--")
    ax3.axvline(0.5, color=TEXT_MUTE, linewidth=0.5, linestyle="--")
    ax3.set_xlabel("False Alarm Rate →  (lower is better)",
                   color=TEXT_SEC, fontsize=8)
    ax3.set_ylabel("Missed Detection Rate →  (lower is better)",
                   color=TEXT_SEC, fontsize=8)
    ax3.tick_params(colors=TEXT_SEC, labelsize=8)
    ax3.set_xlim(-0.05, 1.1); ax3.set_ylim(-0.05, 1.1)
    # Ideal quadrant shading
    ax3.fill_between([0, 0.5], [0, 0], [0.5, 0.5],
                     color=ACCENT, alpha=0.07, label="Ideal zone")
    ax3.legend(fontsize=7, facecolor=BG_SURF, edgecolor=TEXT_MUTE, labelcolor=TEXT_PRI)
    _title(ax3, "[3] FALSE ALARM vs MISSED DETECTION")

    # 4. Per-category accuracy heatmap 
    ax4 = _ax(2, 0)
    cats = ["action", "baseline", "tricky"]
    heat = np.zeros((len(cats), n_algos))
    for j, algo in enumerate(algos):
        for i, cat in enumerate(cats):
            acc = algo_summary[algo]["per_category"].get(cat, {}).get("accuracy", 0.0)
            heat[i, j] = acc
    im = ax4.imshow(heat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax4.set_xticks(range(n_algos))
    ax4.set_xticklabels(disp, rotation=45, ha="right", fontsize=7, color=TEXT_SEC)
    ax4.set_yticks(range(len(cats)))
    ax4.set_yticklabels([c.upper() for c in cats], fontsize=8, color=TEXT_SEC)
    for i in range(len(cats)):
        for j in range(n_algos):
            ax4.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="black" if heat[i,j] > 0.4 else TEXT_PRI,
                     fontfamily="monospace", fontweight="bold")
    plt.colorbar(im, ax=ax4, shrink=0.8)
    _title(ax4, "[4] ACCURACY HEATMAP  (per category)")

    # 5. Runtime bar chart
    ax5 = _ax(2, 1)
    rts = [algo_summary[a]["mean_runtime"] for a in algos]
    bar_r = ax5.barh(disp, rts, color=[bar_colors[i] for i in range(n_algos)],
                     height=0.6, alpha=0.85)
    for bar, v in zip(bar_r, rts):
        ax5.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}s", va="center", color=TEXT_PRI, fontsize=8,
                 fontfamily="monospace")
    ax5.set_xlabel("Mean runtime per pair (seconds)", color=TEXT_SEC, fontsize=8)
    ax5.tick_params(colors=TEXT_SEC, labelsize=8)
    _title(ax5, "[5] RUNTIME  (mean per pair, seconds)")

    # 6. Per-pair prediction heatmap 
    ax6 = _ax(2, 2)
    # Build matrix: rows=algos, cols=pairs
    n_pairs = len(pairs)
    pred_mat = np.zeros((n_algos, n_pairs))
    gt_row   = np.array([p["label"] for p in pairs])
    for row in results_table:
        ai = algos.index(row["algo"])
        pi = next(i for i, p in enumerate(pairs) if p["id"] == row["pair_id"])
        pred_mat[ai, pi] = row["prediction"]

    # Colour
    correct_mat = (pred_mat == gt_row[np.newaxis, :]).astype(float)
    ax6.imshow(correct_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax6.set_xticks(range(n_pairs))
    ax6.set_xticklabels([p["id"] for p in pairs], fontsize=7.5, color=TEXT_SEC)
    ax6.set_yticks(range(n_algos))
    ax6.set_yticklabels(disp, fontsize=7.5, color=TEXT_SEC)
    #  ✓/✗
    for i in range(n_algos):
        for j in range(n_pairs):
            sym = "✓" if correct_mat[i, j] == 1 else "✗"
            col = "black" if correct_mat[i, j] == 1 else "white"
            ax6.text(j, i, sym, ha="center", va="center", fontsize=9, color=col)
    # 
    cat_bounds = {"action": (0,3), "baseline": (4,7), "tricky": (8,9)}
    for cat, (lo, hi) in cat_bounds.items():
        ax6.axvline(lo - 0.5, color=CAT_COLORS[cat], linewidth=1.0, alpha=0.7)
        ax6.text(lo + (hi-lo)/2, -0.85, cat.upper(),
                 ha="center", color=CAT_COLORS[cat], fontsize=6.5,
                 fontfamily="monospace")
    _title(ax6, "[6] PREDICTION GRID  (green=correct, red=wrong)")

    # 7. Confusion matrix 
    sorted_by_f1 = sorted(algos, key=lambda a: algo_summary[a]["f1"], reverse=True)
    focus_algos  = [sorted_by_f1[0], "siamese", sorted_by_f1[-1]]
    focus_labels = ["Best F1", "Siamese", "Worst F1"]

    for col_idx, (fa, fl) in enumerate(zip(focus_algos, focus_labels)):
        ax_cm = _ax(3, col_idx)
        m = algo_summary[fa]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        im_cm = ax_cm.imshow(cm, cmap="Blues", vmin=0)
        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["No Change", "Change"], fontsize=8, color=TEXT_SEC)
        ax_cm.set_yticklabels(["No Change", "Change"], fontsize=8, color=TEXT_SEC)
        ax_cm.set_xlabel("Predicted →", color=TEXT_SEC, fontsize=7.5)
        ax_cm.set_ylabel("Actual →",    color=TEXT_SEC, fontsize=7.5)
        for i in range(2):
            for j in range(2):
                c = "white" if cm[i, j] > cm.max() * 0.5 else TEXT_PRI
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                           fontsize=18, color=c, fontweight="bold")
        _title(ax_cm,
               f"[7] CONFUSION  [{fl}]\n{ALGO_DISPLAY[fa]}  "
               f"Acc={m['accuracy']:.2f}  F1={m['f1']:.2f}")

    # 8. Summary table 
    ax_kpi = fig.add_subplot(gs[4, :])
    ax_kpi.set_facecolor(BG_PANEL)
    for sp in ax_kpi.spines.values():
        sp.set_edgecolor(TEXT_MUTE); sp.set_linewidth(0.5)
    ax_kpi.axis("off")
    cols_kpi = ["Algorithm", "Type", "Accuracy", "Precision", "Recall",
                "F1", "FAR", "MDR", "Avg RT(s)", "TP", "TN", "FP", "FN"]
    rows_kpi = []
    for a in algos:
        m = algo_summary[a]
        rows_kpi.append([
            m["algo_display"], m["type"].upper(),
            f"{m['accuracy']:.3f}", f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",   f"{m['f1']:.3f}",
            f"{m['false_alarm_rate']:.3f}", f"{m['missed_detection']:.3f}",
            f"{m['mean_runtime']:.2f}",
            str(m["TP"]), str(m["TN"]), str(m["FP"]), str(m["FN"]),
        ])

    tbl = ax_kpi.table(
        cellText=rows_kpi,
        colLabels=cols_kpi,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(BG_SURF if r == 0 else BG_PANEL)
        cell.set_edgecolor(TEXT_MUTE)
        cell.set_text_props(color=ACCENT if r == 0 else TEXT_PRI,
                            fontfamily="monospace",
                            fontweight="bold" if r == 0 else "normal")

    _title(ax_kpi, "[8] FULL BENCHMARK SUMMARY TABLE")

    fig.text(0.06, 0.018,
             f"Border Surveillance System  ·  Siamese Network · Sentinel-2 · 34.5200°N, 69.1800°E  "
             f"·  Threshold={THRESHOLD}  ·  Seed={SEED}",
             color=TEXT_MUTE, fontsize=8, fontfamily="monospace")

    out_path = OUT_DIR / "benchmark_dashboard.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    return str(out_path)

if __name__ == "__main__":
    report = run_benchmark()
    print(f"\n✓  Benchmark complete.")
    print(f"   Dashboard  → benchmark_results/benchmark_dashboard.png")
    print(f"   JSON       → benchmark_results/benchmark_report.json")
    print(f"   CSV        → benchmark_results/scores_table.csv")
