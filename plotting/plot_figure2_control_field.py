"""
Figure 2 — Control Field over Trajectory Geometry

Three panels:
  (A) Mean STR vs λ — monotonic bidirectional control
  (B) Var(STR) vs λ — non-monotonic stability landscape
  (C) Normal vs Scrambled — directional causality
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from _style import (
    TRC_BASELINES, SCRAMBLED, FONT_TITLE, FONT_LABEL, FONT_ANNOT,
    C_NORMAL, C_SCRAMBLE, C_NATOP, C_STABOPT,
    academic_ax, save_fig,
)


def main():
    print("Figure 2: Control Field Existence")

    # Load baseline data
    with open(TRC_BASELINES / "trc_baselines.json") as f:
        baselines = json.load(f)
    summaries = baselines["summaries"]
    lambdas = sorted([float(k) for k in summaries.keys()])
    means = [summaries[str(l)]["mean_str"] for l in lambdas]
    vars_ = [summaries[str(l)]["var_str"] for l in lambdas]

    # Load scrambled data
    with open(SCRAMBLED / "scrambled_results.json") as f:
        scrambled = json.load(f)
    s_lambdas = sorted([float(k) for k in scrambled["normal"].keys()])
    s_normal = [scrambled["normal"][str(l)]["mean_str"] for l in s_lambdas]
    s_scramble = [scrambled["scrambled"][str(l)]["mean_str"] for l in s_lambdas]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2, ax3):
        academic_ax(ax)

    # Panel A: Mean STR vs λ (monotonic control)
    ax1.plot(lambdas, means, "o-", color=C_NORMAL, lw=2, ms=6, zorder=3)
    ax1.axvline(0.0, color=C_NATOP, ls=":", lw=1, alpha=0.5)
    ax1.set_xlabel("λ (control strength)", fontsize=FONT_LABEL)
    ax1.set_ylabel("Mean STR", fontsize=FONT_LABEL)
    ax1.set_title("(A) Controllability", fontsize=FONT_TITLE, fontweight="bold")
    ax1.annotate("Contraction\n(+λ)", xy=(0.15, means[-1]),
                 xytext=(0.15, means[-1] - 0.035), fontsize=8,
                 color=C_NORMAL, ha="center", va="top")
    ax1.annotate("Expansion\n(−λ)", xy=(-0.15, means[0]),
                 xytext=(-0.15, means[0] + 0.005), fontsize=8,
                 color=C_NORMAL, ha="center", va="bottom")

    # Panel B: Var(STR) vs λ (stability landscape preview)
    ax2.plot(lambdas, vars_, "s-", color="#9467bd", lw=2, ms=6, zorder=3)
    ax2.axvline(0.0, color=C_NATOP, ls=":", lw=1, alpha=0.5)
    min_idx = np.argmin(vars_)
    ax2.scatter([lambdas[min_idx]], [vars_[min_idx]], s=120, color=C_STABOPT,
                marker="*", zorder=5)
    ax2.annotate("Stability\nOptimum", xy=(lambdas[min_idx], vars_[min_idx]),
                 xytext=(lambdas[min_idx] - 0.08, vars_[min_idx] + 0.001),
                 arrowprops=dict(arrowstyle="->", color=C_STABOPT, lw=1.2),
                 fontsize=FONT_ANNOT, color=C_STABOPT, fontweight="bold")
    ax2.set_xlabel("λ (control strength)", fontsize=FONT_LABEL)
    ax2.set_ylabel("Var(STR$_t$)", fontsize=FONT_LABEL)
    ax2.set_title("(B) Stability Landscape", fontsize=FONT_TITLE, fontweight="bold")

    # Panel C: Normal vs Scrambled (the killer)
    ax3.plot(s_lambdas, s_normal, "o-", color=C_NORMAL, lw=2.5, ms=7,
             label="Structured (normal)", zorder=3)
    ax3.plot(s_lambdas, s_scramble, "D--", color=C_SCRAMBLE, lw=2, ms=7,
             label="Scrambled (norm-preserved)", zorder=3)
    ax3.axvline(0.0, color=C_NATOP, ls=":", lw=1, alpha=0.5)
    ax3.fill_between(s_lambdas, s_normal, s_scramble,
                     alpha=0.12, color=C_NORMAL, zorder=1)
    ax3.set_xlabel("λ (control strength)", fontsize=FONT_LABEL)
    ax3.set_ylabel("Mean STR", fontsize=FONT_LABEL)
    ax3.set_title("(C) Directional Causality", fontsize=FONT_TITLE, fontweight="bold")
    ax3.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="0.8")

    fig.suptitle("Figure 2 — Control Field over Trajectory Geometry",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "figure2_control_field")


if __name__ == "__main__":
    main()
