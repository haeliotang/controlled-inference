"""
Figure 6 — Feedback as an Independent Control Primitive

Two panels:
  (A) Time-series: STR under static M_S, static M_L, random switching,
      and state-dependent feedback (F-only) — all with frozen weights.
  (B) Variance bar chart comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from _style import (
    TRC_RESULTS, FONT_TITLE, FONT_LABEL,
    C_FEEDBACK, C_STATIC_S, C_STATIC_L, C_RANDOM,
    academic_ax, save_fig,
)


def main():
    print("Figure 6: Feedback Causality (F-only)")

    with open(TRC_RESULTS / "f_only_results.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [2.5, 1]})
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        academic_ax(ax)

    configs = [
        ("static_s", C_STATIC_S, "Static $M_S$ (contraction)", "--", 1.8),
        ("static_l", C_STATIC_L, "Static $M_L$ (expansion)",   "--", 1.8),
        ("random",   C_RANDOM,   "Random switching",            ":",  1.8),
        ("f_only",   C_FEEDBACK, "Feedback (state-dependent)",  "-",  2.8),
    ]

    # Panel A: Time series
    for key, color, label, ls, lw in configs:
        traj = np.array(data[key]["avg_traj"])
        T = len(traj)
        ax1.plot(np.arange(T), traj, ls, color=color, lw=lw, label=label, zorder=3)

    # Target band
    target = data["f_only"]["mean"]
    band = 0.012
    ax1.axhspan(target - band, target + band, color=C_FEEDBACK, alpha=0.06,
                zorder=0)
    ax1.axhline(target, color=C_FEEDBACK, ls="--", lw=0.8, alpha=0.4)

    ax1.set_xlabel("Generation step $t$", fontsize=FONT_LABEL)
    ax1.set_ylabel("STR$_t$", fontsize=FONT_LABEL)
    ax1.set_title("(A) Trajectory Dynamics Under Frozen Weights",
                  fontsize=FONT_TITLE, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9, edgecolor="0.8")

    ax1.annotate("Fast drift ↑", xy=(42, data["static_s"]["avg_traj"][-5]),
                 fontsize=8, color=C_STATIC_S, ha="right", fontstyle="italic")
    ax1.annotate("Slow drift ↑", xy=(42, data["static_l"]["avg_traj"][-5]),
                 fontsize=8, color=C_STATIC_L, ha="right", fontstyle="italic")

    # Panel B: Variance bar chart
    methods = ["Static $M_S$", "Static $M_L$", "Random", "Feedback"]
    variances = [data[k]["var"] for k in ["static_s", "static_l", "random", "f_only"]]
    bar_colors = [C_STATIC_S, C_STATIC_L, C_RANDOM, C_FEEDBACK]

    bars = ax2.bar(methods, variances, color=bar_colors, edgecolor="black",
                   linewidth=0.6, alpha=0.85, width=0.65)
    ax2.set_ylabel("Trajectory Variance", fontsize=FONT_LABEL)
    ax2.set_title("(B) Variance Comparison", fontsize=FONT_TITLE, fontweight="bold")
    ax2.tick_params(axis="x", rotation=25, labelsize=8)

    bars[-1].set_edgecolor(C_FEEDBACK)
    bars[-1].set_linewidth(2)

    for bar, v in zip(bars, variances):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.suptitle("Figure 6 — Feedback as an Independent Control Primitive",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "figure6_feedback_causality")


if __name__ == "__main__":
    main()
