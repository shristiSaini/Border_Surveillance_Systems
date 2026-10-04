import numpy as np
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

from change_detection import ChangeResult
#colour
BG_DARK   = "#080c10"
BG_PANEL  = "#0d1219"
BG_SURF   = "#111922"
ACCENT    = "#00c88c"
TEXT_PRI  = "#e8f0e9"
TEXT_SEC  = "#7a9e8a"
TEXT_MUTE = "#3d5a4a"
DANGER    = "#ef4444"
WARN      = "#f59e0b"
INFO      = "#3b82f6"

ALERT_COLORS = {
    "LOW":      ("#4ade80", "#1a3a1a"),
    "MODERATE": ("#f59e0b", "#2a1f00"),
    "HIGH":     ("#ef4444", "#2a0000"),
}


def generate_report(
    scene1,           # older pass
    scene2,           # recent pass
    result: ChangeResult,
    lat: float,
    lon: float,
    radius_deg: float,
    output_path: str = "bss_report.png",
    show: bool = False,
) -> str:

    fig = plt.figure(figsize=(16, 10), facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)
    # grid layout
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.38,
        wspace=0.18,
        top=0.88, bottom=0.07,
        left=0.04, right=0.97,
        height_ratios=[0.06, 1, 0.22],
    )

    ax_t1   = fig.add_subplot(gs[1, 0])    # Pass T-1
    ax_t2   = fig.add_subplot(gs[1, 1])    # Pass T-2
    ax_diff = fig.add_subplot(gs[1, 2])    # Change map
    ax_stat = fig.add_subplot(gs[2, :])    # Statistics bar (full width)
    #Header
    bbox = [
        round(lon - radius_deg, 4), round(lat - radius_deg, 4),
        round(lon + radius_deg, 4), round(lat + radius_deg, 4),
    ]
    fig.text(
        0.04, 0.95,
        "BORDER SURVEILLANCE SYSTEM  //  Sentinel Change Detection",
        color=ACCENT, fontsize=12, fontweight="bold",
        fontfamily="monospace",
    )
    fig.text(
        0.04, 0.915,
        f"Target: {lat:.5f}°N  {lon:.5f}°E   │   "
        f"BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]   │   "
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        color=TEXT_SEC, fontsize=8.5, fontfamily="monospace",
    )

    sat_label = (
        f"Satellite: {'Sentinel-2 (Optical · MSI)' if scene1.satellite == 'S2' else 'Sentinel-1 (SAR · GRD)'}"
        f"   │   Resolution: 10 m/px"
    )
    fig.text(0.04, 0.893, sat_label, color=TEXT_MUTE, fontsize=8, fontfamily="monospace")

    fig.add_artist(matplotlib.patches.FancyArrowPatch(              #accent line
        (0.04, 0.883), (0.97, 0.883),
        transform=fig.transFigure,
        arrowstyle="-",
        color=ACCENT,
        linewidth=0.6,
        alpha=0.4,
    ))
    #helper - style
    def style_ax(ax, title, subtitle, border_color=ACCENT):
        ax.set_facecolor(BG_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title,   color=border_color, fontsize=8.5,
                     fontfamily="monospace", fontweight="bold", pad=5)
        ax.set_xlabel(subtitle, color=TEXT_SEC, fontsize=7.5,
                      fontfamily="monospace", labelpad=4)

    # T1
    if scene1.image is not None:
        ax_t1.imshow(scene1.image, aspect="auto")
    _draw_coord_overlay(ax_t1, lat, lon, bbox)
    style_ax(ax_t1,
             f"PASS T-1  ·  {scene1.date}",
             f"ID: {scene1.scene_id[:28]}…"
             if len(scene1.scene_id) > 28 else f"ID: {scene1.scene_id}")

    # T2
    if scene2.image is not None:
        ax_t2.imshow(scene2.image, aspect="auto")
    _draw_coord_overlay(ax_t2, lat, lon, bbox)
    style_ax(ax_t2,
             f"PASS T-2  ·  {scene2.date}",
             f"ID: {scene2.scene_id[:28]}…"
             if len(scene2.scene_id) > 28 else f"ID: {scene2.scene_id}")

    # Change map 
    ax_diff.imshow(result.change_map, aspect="auto")

    # hotspot circles
    H_img = result.change_map.shape[0]
    W_img = result.change_map.shape[1]
    for idx, (cr, cc, rad) in enumerate(result.hotspots[:8]):
        circle = plt.Circle(
            (cc, cr), rad,
            color=DANGER, fill=False, linewidth=1.2, linestyle="--",
        )
        ax_diff.add_patch(circle)
        ax_diff.text(
            cc, cr - rad - 4,
            f"H{idx+1}", color=DANGER,
            fontsize=7, fontfamily="monospace", ha="center",
        )

    _draw_coord_overlay(ax_diff, lat, lon, bbox)

    alert_fg, alert_bg = ALERT_COLORS[result.alert_level]
    style_ax(ax_diff,
             f"CHANGE MAP  ·  ALERT: {result.alert_level}",
             f"MAE · changed: {result.pct_changed:.1f}%  · hotspots: {len(result.hotspots)}",
             border_color=alert_fg)

    # Legends on change map 
    patches = [
        mpatches.Patch(color="#141e14", label="No change"),
        mpatches.Patch(color="#4ade80", label="Minor"),
        mpatches.Patch(color="#f59e0b", label="Moderate"),
        mpatches.Patch(color="#ef4444", label="High"),
    ]
    ax_diff.legend(
        handles=patches,
        loc="lower right",
        fontsize=6.5,
        facecolor=BG_PANEL,
        edgecolor=TEXT_MUTE,
        labelcolor=TEXT_PRI,
    )

    # Statistics 
    ax_stat.set_facecolor(BG_SURF)
    ax_stat.set_xticks([]); ax_stat.set_yticks([])
    for spine in ax_stat.spines.values():
        spine.set_edgecolor(TEXT_MUTE)
        spine.set_linewidth(0.5)

    stats = [
        ("CHANGED PIXELS",  f"{result.pct_changed:.1f} %"),
        ("MEAN DELTA",       f"{result.mean_delta:.1f}"),
        ("MAX DELTA",        f"{result.max_delta:.1f}"),
        ("HOTSPOT CLUSTERS", str(len(result.hotspots))),
        ("ALERT LEVEL",      result.alert_level),
        ("PASS INTERVAL",    _interval_str(scene1.date, scene2.date)),
    ]

    for i, (lbl, val) in enumerate(stats):
        x = 0.01 + i * (1.0 / len(stats))
        ax_stat.text(x + 0.005, 0.72, lbl,
                     transform=ax_stat.transAxes,
                     color=TEXT_MUTE, fontsize=7.5, fontfamily="monospace",
                     fontweight="bold")
        # Colour alert 
        if lbl == "ALERT LEVEL":
            val_color, _ = ALERT_COLORS.get(val, (ACCENT, BG_SURF))
        else:
            val_color = ACCENT
        ax_stat.text(x + 0.005, 0.22, val,
                     transform=ax_stat.transAxes,
                     color=val_color, fontsize=16, fontfamily="monospace",
                     fontweight="bold")

    # Dividers
    for i in range(1, len(stats)):
        xd = i / len(stats)
        ax_stat.axvline(xd, color=TEXT_MUTE, linewidth=0.4, alpha=0.5)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(output_path)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)

    if show:
        try:
            import subprocess, sys
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(out)])
        except Exception:
            pass

    print(f"[VIS] Report saved → {out.resolve()}")
    return str(out.resolve())

#helper
def _draw_coord_overlay(ax, lat, lon, bbox):
    """Stamps coordinate information onto an image axis."""
    coord_str = f"{lat:.4f}°N  {lon:.4f}°E"
    ax.text(
        0.02, 0.03, coord_str,
        transform=ax.transAxes,
        color=ACCENT, fontsize=7, fontfamily="monospace",
        bbox=dict(boxstyle="square,pad=0.3", fc="black", ec=ACCENT,
                  alpha=0.75, linewidth=0.6),
    )


def _interval_str(date1: str, date2: str) -> str:
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        days = abs((d2 - d1).days)
        return f"{days} days"
    except Exception:
        return "N/A"
