"""
Figure 5 — Perturbation Recovery (Necessity of Feedback)

Single panel: STR trajectories under perturbation at t=15.
Only feedback control recovers; open-loop policies fail.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from _style import (
    TRC_RESULTS, FONT_TITLE, FONT_LABEL, FONT_ANNOT,
    C_FEEDBACK, C_PIECEWISE, C_STABOPT,
    academic_ax, save_fig,
)


def main():
    print("Figure 5: Perturbation Recovery")

    with open(TRC_RESULTS / "perturbation_results.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    academic_ax(ax)

    configs = [
        ("oracle",         C_FEEDBACK,  "Feedback (gated oracle)", "-",  2.5),
        ("piecewise",      C_PIECEWISE, "Scheduled (piecewise)",   "--", 2.0),
        ("gated_feedback",  C_STABOPT,  "Gated feedback",          "-.", 2.0),
    ]

    for key, color, label, ls, lw in configs:
        traj = np.array(data[key]["avg_traj"])
        T = len(traj)
        ax.plot(np.arange(T), traj, ls, color=color, lw=lw, label=label, zorder=3)

    # Mark perturbation point
    perturb_t = 15
    ax.axvline(perturb_t, color="#e377c2", ls="--", lw=1.5, alpha=0.7)
    ax.annotate("Perturbation\ninjected", xy=(perturb_t, ax.get_ylim()[1] * 0.95),
                xytext=(perturb_t + 3, ax.get_ylim()[1] * 0.95),
                arrowprops=dict(arrowstyle="->", color="#e377c2", lw=1.2),
                fontsize=FONT_ANNOT, color="#e377c2", fontweight="bold",
                ha="left", va="top")

    # Recovery annotation
    ax.annotate("Recovery", xy=(25, data["oracle"]["avg_traj"][25]),
                xytext=(28, data["oracle"]["avg_traj"][25] - 0.025),
                arrowprops=dict(arrowstyle="->", color=C_FEEDBACK, lw=1),
                fontsize=9, color=C_FEEDBACK, fontstyle="italic",
                ha="left", va="top")

    ax.set_xlabel("Generation step $t$", fontsize=FONT_LABEL)
    ax.set_ylabel("STR$_t$", fontsize=FONT_LABEL)
    ax.set_title("Figure 5 — Perturbation Recovery (Necessity of Feedback)",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor="0.8")

    plt.tight_layout()
    save_fig(fig, "figure5_perturbation")


if __name__ == "__main__":
    main()
