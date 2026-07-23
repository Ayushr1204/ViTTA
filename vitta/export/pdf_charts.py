"""
Matplotlib chart generators for PDF report — matches web dashboard charts exactly.
All charts use the ViTTA green accent palette.

IMPORTANT: All figures use a fixed 4.5×3.0 figsize and are saved WITHOUT
bbox_inches='tight' so that the output image dimensions are deterministic.
This guarantees uniform row heights in the PDF two-column layout.
"""
from __future__ import annotations
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (matches app.js) ──────────────────────────────────────
GREEN = "#15803D"
GREEN_L = "#22C55E"
COLORS = ["#6366F1","#10B981","#F59E0B","#EF4444","#06B6D4","#EC4899",
           "#84CC16","#8B5CF6","#3B82F6","#D946EF","#14B8A6","#F97316"]
BEH_COLORS = {"Disciplined":"#10B981","Speeding":"#EF4444","Slow":"#F59E0B",
    "Erratic":"#8B5CF6","Aggressive Braking":"#DC2626","Tailgating":"#F97316",
    "Lane Weaving":"#06B6D4","Stopped/Idling":"#6B7280"}
DIR_COLORS = {"Northbound":"#3B82F6","Southbound":"#EF4444",
    "Eastbound":"#F59E0B","Westbound":"#8B5CF6","Stationary":"#6B7280"}
DIR_ANGLES = {"Northbound":0,"Eastbound":90,"Southbound":180,"Westbound":270,"Stationary":315}
TEXT = "#4B5563"
GRID = "#E5E7EB"

# Standard figure size for all charts — produces ~61mm height at 92mm width
_FIG_W, _FIG_H = 4.5, 3.0


def _ax(ax, xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=TEXT)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=TEXT)


def _save(fig, d, name):
    """Save figure at exactly the figsize dimensions (no bbox_inches='tight')."""
    p = d / f"{name}.png"
    fig.savefig(str(p), dpi=180, facecolor="white", edgecolor="none")
    plt.close(fig)
    return p


# 1. Vehicle Composition (Donut) — matches chartClassDist
def chart_composition(class_counts: Dict[str,int], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    labels = sorted(class_counts.keys())
    values = [class_counts[c] for c in labels]
    cols = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    wedges, texts, autos = ax.pie(values, autopct="%1.1f%%", startangle=90,
        colors=cols, pctdistance=0.78, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
        textprops={"fontsize":7, "color":"white", "fontweight":"bold"})
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.88, 0.5),
             fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.68, top=0.96, bottom=0.05)
    return _save(fig, out, "chart_composition")


# 2. Speed Distribution (Histogram 1 m/s bins) — matches chartSpeedHist
def chart_speed_hist(speeds: List[float], unit: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if not speeds:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out, "chart_speed")
    bins = np.arange(0, math.ceil(max(speeds)) + 1, 1.0)
    ax.hist(speeds, bins=bins, color=GREEN, alpha=0.85, edgecolor="#0F5132", linewidth=0.8)
    _ax(ax, f"Speed ({unit})", "Vehicle Count")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.18)
    return _save(fig, out, "chart_speed")


# 3. Direction Breakdown (Polar) — matches chartDirection
def chart_direction(dir_counts: Dict[str,int], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), subplot_kw=dict(projection="polar"))
    labels = sorted(dir_counts.keys())
    values = [dir_counts[d] for d in labels]
    angles = [math.radians(90 - DIR_ANGLES.get(d, 0)) for d in labels]  # convert to math convention
    cols = [DIR_COLORS.get(d, "#6B7280") for d in labels]
    width = math.radians(50)
    bars = ax.bar(angles, values, width=width, color=cols, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise
    ax.set_thetagrids([0,90,180,270], ["N","E","S","W"], fontsize=9, color=TEXT)
    ax.tick_params(axis="y", labelsize=7, colors="#9CA3AF")
    ax.set_facecolor("white")
    ax.grid(color=GRID, linewidth=0.4, alpha=0.5)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05)
    return _save(fig, out, "chart_direction")


# 4. Class Speed Comparison (Horizontal Bar) — matches chartClassSpeed
def chart_class_speed(class_speeds: Dict[str,List[float]], unit: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    classes = sorted(class_speeds.keys())
    avgs = [sum(class_speeds[c])/len(class_speeds[c]) for c in classes]
    counts = [len(class_speeds[c]) for c in classes]
    cols = [COLORS[i % len(COLORS)] for i in range(len(classes))]
    bars = ax.barh(classes, avgs, color=cols, edgecolor="white", linewidth=1, height=0.55)
    for bar, val, n in zip(bars, avgs, counts):
        ax.text(bar.get_width() + max(avgs)*0.02, bar.get_y()+bar.get_height()/2,
                f"{val:.1f} {unit} (n={n})", va="center", fontsize=7, color=TEXT)
    _ax(ax, f"Avg Speed ({unit})", "")
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.22, right=0.96, top=0.95, bottom=0.18)
    return _save(fig, out, "chart_class_speed")


# 5. Behaviour Distribution (Horizontal Bar) — matches chartBehaviour
def chart_behaviour(beh_counts: Dict[str,int], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    sorted_items = sorted(beh_counts.items(), key=lambda x: x[1])
    labels = [i[0] for i in sorted_items]
    values = [i[1] for i in sorted_items]
    total = sum(values) or 1
    cols = [BEH_COLORS.get(l, "#6B7280") for l in labels]
    bars = ax.barh(labels, values, color=cols, edgecolor="white", linewidth=1, height=0.6)
    for bar, val in zip(bars, values):
        pct = val/total*100
        ax.text(bar.get_width()+max(values)*0.02, bar.get_y()+bar.get_height()/2,
                f"{val} ({pct:.0f}%)", va="center", fontsize=7, color=TEXT)
    _ax(ax, "Vehicle Count", "")
    fig.subplots_adjust(left=0.30, right=0.96, top=0.95, bottom=0.18)
    return _save(fig, out, "chart_behaviour")


# 6. Speed vs Headway (Scatter) — matches chartSpeedHeadway
def chart_speed_headway(speeds, headways, classes, unit, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if not speeds:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out, "chart_scatter")
    unique = sorted(set(classes))
    for i, cls in enumerate(unique):
        idx = [j for j, c in enumerate(classes) if c == cls]
        sx = [headways[j] for j in idx]
        sy = [speeds[j] for j in idx]
        ax.scatter(sx, sy, s=12, alpha=0.6, color=COLORS[i % len(COLORS)],
                   edgecolors="white", linewidth=0.3, label=cls)
    # P95 axis limits
    sh = sorted(headways)
    ss = sorted(speeds)
    p95h = sh[int(len(sh)*0.95)] if len(sh) > 5 else max(headways)
    p95s = ss[int(len(ss)*0.95)] if len(ss) > 5 else max(speeds)
    ax.set_xlim(0, max(p95h*1.3, 2))
    ax.set_ylim(0, p95s*1.3)
    _ax(ax, "Time Headway (s)", f"Speed ({unit})")
    ax.legend(fontsize=6, frameon=False, ncol=min(4, len(unique)), loc="upper right")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.18)
    return _save(fig, out, "chart_scatter")


# 7. Traffic Density Over Time (Area) — matches chartDensity
def chart_density(labels, values, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if not labels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out, "chart_density")
    dl, dv = list(labels), list(values)
    if len(dl) > 200:
        step = max(1, len(dl)//200)
        dl, dv = dl[::step], dv[::step]
    ax.fill_between(dl, dv, alpha=0.12, color=GREEN)
    ax.plot(dl, dv, color=GREEN, linewidth=2)
    _ax(ax, "Time (seconds)", "Active Vehicles")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.18)
    return _save(fig, out, "chart_density")


# 8. Headway Distribution (Histogram)
def chart_headway_hist(headways: List[float], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if not headways:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out, "chart_headway")
    valid_h = [h for h in headways if h < 60]
    bins = np.arange(0, math.ceil(max(valid_h)) + 1, 1.0)
    ax.hist(valid_h, bins=bins, color=GREEN, alpha=0.85, edgecolor="#0F5132", linewidth=0.8)
    _ax(ax, "Time Headway (s)", "Vehicle Count")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.18)
    return _save(fig, out, "chart_headway")
