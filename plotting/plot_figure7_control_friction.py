"""
Figure 7 — The Failure of Fixed-Setpoint Control

Shows that fixed-τ DOM (PD controller) collapses at T=100 with 40% failure
rate (10/25)—worse than the uncontrolled baseline (4/25 = 16%).

Data source: results/dom_exp1/exp1_phase2_results.json
  - baseline:   25 runs, failure_rate 4/25
  - static_0.2: 25 runs, failure_rate 8/25
  - dom_damped:  25 runs, failure_rate 10/25 (fixed-setpoint)

Panels:
  Single panel: STR trajectories showing baseline vs fixed-τ DOM collapse
  with failure rate annotations.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_FILE = RESULTS_DIR / "exp1" / "exp1_phase2_results.json"
OUT_FILE = BASE / "figure7_control_friction.png"

# ── Style ──────────────────────────────────────────────────────────────
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK = 10
FONT_ANNOT = 9
DPI = 250

C_BASELINE = "#1f77b4"
C_STATIC   = "#ff7f0e"
C_DOM      = "#d62728"
C_BAND     = "#27AE60"


def academic_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def avg_trace(runs, max_len=100):
    """Average STR traces across runs, padding to max_len."""
    traces = []
    for r in runs:
        tr = r["str_trace"]
        valid = [s for s in tr if s is not None]
        if not valid:
            continue
        padded = []
        for s in tr[:max_len]:
            padded.append(s if s is not None else valid[-1])
        while len(padded) < max_len:
            padded.append(valid[-1])
        traces.append(padded)
    if not traces:
        return np.zeros(max_len), np.zeros(max_len)
    arr = np.array(traces)
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    print("Figure 7: The Failure of Fixed-Setpoint Control")

    with open(DATA_FILE) as f:
        data = json.load(f)

    config = data["config"]
    conditions = data["conditions"]
    tau = config["tau"]

    # Extract conditions
    bl_runs = conditions["baseline"]["runs"]
    bl_summary = conditions["baseline"]["summary"]
    dom_runs = conditions["dom_damped"]["runs"]
    dom_summary = conditions["dom_damped"]["summary"]

    bl_mean, bl_std = avg_trace(bl_runs)
    dom_mean, dom_std = avg_trace(dom_runs)
    t_axis = np.arange(len(bl_mean))

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    academic_ax(ax)

    # Stability band
    band_margin = 0.07
    ax.axhspan(tau - band_margin, tau + band_margin, color=C_BAND,
               alpha=0.08, label="Stability band")
    ax.axhline(tau, color=C_BAND, ls="--", lw=0.8, alpha=0.4)

    # STR traces
    ax.plot(t_axis, bl_mean, color=C_BASELINE, lw=2, label="Baseline (no control)",
            alpha=0.9, zorder=3)
    ax.fill_between(t_axis, bl_mean - 0.5 * bl_std, bl_mean + 0.5 * bl_std,
                    color=C_BASELINE, alpha=0.1)

    ax.plot(t_axis, dom_mean, color=C_DOM, lw=2, label="Fixed-τ DOM (PD controller)",
            alpha=0.9, zorder=3)
    ax.fill_between(t_axis, dom_mean - 0.5 * dom_std, dom_mean + 0.5 * dom_std,
                    color=C_DOM, alpha=0.1)

    # Mark T=60 boundary (where fixed-τ still appears OK)
    ax.axvline(60, color="#888", ls=":", lw=1, alpha=0.5)
    ax.text(61, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "$T=60$", fontsize=8, color="#666", fontstyle="italic")

    # Failure rate annotations
    bl_fail = bl_summary["failure_rate"]
    dom_fail = dom_summary["failure_rate"]

    # Box at top-right with failure rates
    textstr = (f"Failure rate at $T=100$:\n"
               f"  Baseline:      {bl_fail} = {int(bl_fail.split('/')[0])/int(bl_fail.split('/')[1])*100:.0f}%\n"
               f"  Fixed-τ DOM: {dom_fail} = {int(dom_fail.split('/')[0])/int(dom_fail.split('/')[1])*100:.0f}%")
    props = dict(boxstyle="round,pad=0.5", facecolor="#FFF3F3",
                 edgecolor=C_DOM, alpha=0.9)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=props, fontfamily="monospace")

    # Control friction subtitle (below title)
    ax.text(0.5, 1.01, "Control friction accumulates over long horizons →  fixed-setpoint collapse",
            transform=ax.transAxes, fontsize=9, ha="center",
            color=C_DOM, fontstyle="italic")

    ax.set_xlabel("Generation step $t$", fontsize=FONT_LABEL)
    ax.set_ylabel("STR$_t$", fontsize=FONT_LABEL)
    ax.set_title("Figure 7 — The Failure of Fixed-Setpoint Control",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="0.8")

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT_FILE).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
