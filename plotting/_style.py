"""
Shared style, paths, and utilities for Figures 2–6.

Unified color palette:
  - Normal/structured:  #1f77b4 (blue)
  - Scrambled/random:   #aec7e8 (light blue / gray-blue)
  - Feedback:           #7b2d8e (purple)
  - Static S:           #d62728 (red)
  - Static L:           #2ca02c (green)
  - Piecewise/Schedule: #ff7f0e (orange)
  - Natural OP marker:  #d62728 (red dashed)
  - Stability optimum:  #2ca02c (green star)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS = RESULTS_DIR
TRC_BASELINES = RESULTS / "trc_baselines"
TRC_RESULTS = RESULTS / "trc_results"
SCRAMBLED = RESULTS / "scrambled"
OUT = BASE

# ── Fonts ────────────────────────────────────────────────────────────────
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK = 10
FONT_ANNOT = 9
DPI = 250

# ── Colors ───────────────────────────────────────────────────────────────
C_NORMAL   = "#1f77b4"
C_SCRAMBLE = "#aec7e8"
C_FEEDBACK = "#7b2d8e"
C_STATIC_S = "#d62728"
C_STATIC_L = "#2ca02c"
C_PIECEWISE = "#ff7f0e"
C_RANDOM   = "#8c564b"
C_NATOP    = "#d62728"
C_STABOPT  = "#2ca02c"


def academic_ax(ax):
    """Apply clean academic style to an axis."""
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def save_fig(fig, name):
    path_png = OUT / f"{name}.png"
    path_pdf = OUT / f"{name}.pdf"
    fig.savefig(path_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path_png}")
    print(f"  Saved: {path_pdf}")
