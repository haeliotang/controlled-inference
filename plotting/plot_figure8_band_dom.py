"""
Figure 8 — Boundary-Based Control Reduces Instability Without Continuous Correction

Data source: results/dom_exp2/exp2_results.json
  - baseline:   25 runs, fail_rate 13/25 (52%)
  - dom_fixed:  25 runs, fail_rate 8/25 (32%)
  - dom_band:   25 runs, fail_rate 5/25 (20%)

Three panels:
  (A) STR traces: Band DOM stays within band; baseline and fixed-τ drift
  (B) Control activation: fixed-τ is continuous; Band is sparse (~79% zero)
  (C) Collapse rate comparison: bar chart
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
DATA_FILE = RESULTS_DIR / "exp2" / "exp2_results.json"
OUT_FILE = BASE / "figure8_band_dom.png"

# ── Style ──────────────────────────────────────────────────────────────
FONT_TITLE = 12
FONT_LABEL = 10
FONT_TICK = 9
FONT_ANNOT = 9
DPI = 250

C_BASELINE = "#1f77b4"
C_FIXED    = "#ff7f0e"
C_BAND     = "#2ca02c"
C_BAND_FILL = "#D5F5E3"


def academic_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def avg_trace(runs, key="str_trace", max_len=100):
    """Average traces across runs, padding to max_len."""
    traces = []
    for r in runs:
        tr = r[key]
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


def compute_zero_control_fraction(runs, neutral=0.5, tol=0.01):
    """Compute fraction of steps where α_t ≈ 0.5 (no control)."""
    total_steps = 0
    zero_steps = 0
    for r in runs:
        alphas = r.get("alpha_trace", [])
        for a in alphas:
            if a is not None:
                total_steps += 1
                if abs(a - neutral) < tol:
                    zero_steps += 1
    return zero_steps / max(total_steps, 1)


def main():
    print("Figure 8: Boundary-Based Control (Band DOM)")

    with open(DATA_FILE) as f:
        data = json.load(f)

    bl_runs = data["baseline"]["runs"]
    fx_runs = data["dom_fixed"]["runs"]
    bd_runs = data["dom_band"]["runs"]

    bl_summary = data["baseline"]["summary"]
    fx_summary = data["dom_fixed"]["summary"]
    bd_summary = data["dom_band"]["summary"]

    # Compute traces
    bl_mean, bl_std = avg_trace(bl_runs)
    fx_mean, fx_std = avg_trace(fx_runs)
    bd_mean, bd_std = avg_trace(bd_runs)
    t_axis = np.arange(len(bl_mean))

    # Alpha traces (control intensity)
    fx_alpha_mean, _ = avg_trace(fx_runs, key="alpha_trace")
    bd_alpha_mean, _ = avg_trace(bd_runs, key="alpha_trace")

    # Zero-control fraction for Band
    zero_frac = compute_zero_control_fraction(bd_runs)

    # ── Figure: 3 panels ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5),
                                         gridspec_kw={"width_ratios": [1.3, 1.0, 0.8]})
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2, ax3):
        academic_ax(ax)

    # ── Panel A: STR Trajectories ──
    # Stability band (calibrated from baseline median)
    tau = bl_mean.mean()
    margin = 0.07
    ax1.axhspan(tau - margin, tau + margin, color=C_BAND, alpha=0.08,
                label="Stability band")
    ax1.axhline(tau, color=C_BAND, ls="--", lw=0.8, alpha=0.4)

    ax1.plot(t_axis, bl_mean, color=C_BASELINE, lw=1.8, label="Baseline",
             alpha=0.9, zorder=3)
    ax1.fill_between(t_axis, bl_mean - 0.5 * bl_std, bl_mean + 0.5 * bl_std,
                     color=C_BASELINE, alpha=0.08)

    ax1.plot(t_axis, fx_mean, color=C_FIXED, lw=1.8, label="Fixed-τ DOM",
             alpha=0.9, zorder=3)
    ax1.fill_between(t_axis, fx_mean - 0.5 * fx_std, fx_mean + 0.5 * fx_std,
                     color=C_FIXED, alpha=0.08)

    ax1.plot(t_axis, bd_mean, color=C_BAND, lw=2.2, label="Band DOM",
             alpha=0.95, zorder=4)
    ax1.fill_between(t_axis, bd_mean - 0.5 * bd_std, bd_mean + 0.5 * bd_std,
                     color=C_BAND, alpha=0.08)

    ax1.set_xlabel("Generation step $t$", fontsize=FONT_LABEL)
    ax1.set_ylabel("STR$_t$", fontsize=FONT_LABEL)
    ax1.set_title("(A)  STR Trajectories", fontsize=FONT_TITLE, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9, edgecolor="0.8")

    # ── Panel B: Control Activation ──
    # Show |α_t - 0.5| as control intensity (0 = neutral, higher = more intervention)
    fx_intensity = np.abs(fx_alpha_mean - 0.5)
    bd_intensity = np.abs(bd_alpha_mean - 0.5)

    ax2.plot(t_axis, fx_intensity, color=C_FIXED, lw=1.5,
             label="Fixed-τ (continuous)", alpha=0.9)
    ax2.plot(t_axis, bd_intensity, color=C_BAND, lw=1.5,
             label="Band (sparse)", alpha=0.9)
    ax2.axhline(0, color="#aaa", lw=0.5, ls="-")

    # Annotate zero-control fraction (bottom-left, clear of legend)
    ax2.text(0.97, 0.08, f"~{zero_frac*100:.0f}% zero input",
             transform=ax2.transAxes, fontsize=10, ha="right",
             color=C_BAND, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BAND_FILL,
                       edgecolor=C_BAND, alpha=0.8))

    ax2.set_xlabel("Generation step $t$", fontsize=FONT_LABEL)
    ax2.set_ylabel("Control intensity $|\\alpha_t - 0.5|$", fontsize=FONT_LABEL)
    ax2.set_title("(B)  Control Activation", fontsize=FONT_TITLE, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9, edgecolor="0.8")

    # ── Panel C: Collapse Rate Bar Chart ──
    labels = ["Baseline", "Fixed-τ", "Band"]
    fail_rates_raw = [bl_summary["fail_rate"], fx_summary["fail_rate"],
                      bd_summary["fail_rate"]]
    fail_pcts = []
    for fr in fail_rates_raw:
        num, den = fr.split("/")
        fail_pcts.append(int(num) / int(den) * 100)

    colors = [C_BASELINE, C_FIXED, C_BAND]
    bars = ax3.bar(labels, fail_pcts, color=colors, edgecolor="black",
                   linewidth=0.6, alpha=0.85, width=0.6)

    # Value labels on bars
    for bar, fr, pct in zip(bars, fail_rates_raw, fail_pcts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{fr}\n({pct:.0f}%)", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    ax3.set_ylabel("Collapse Rate (%)", fontsize=FONT_LABEL)
    ax3.set_title("(C)  Collapse Rate", fontsize=FONT_TITLE, fontweight="bold")
    ax3.set_ylim(0, 70)

    fig.suptitle("Figure 8 — Boundary-Based Control Reduces Instability "
                 "Without Continuous Correction",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT_FILE).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
