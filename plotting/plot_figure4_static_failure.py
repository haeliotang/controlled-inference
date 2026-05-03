"""
Figure 4 — Failure of Static and Scheduled Control

Single panel: time-series of STR under static, piecewise-scheduled, and
feedback control, showing that open-loop policies cannot maintain stability.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from _style import (
    TRC_RESULTS, FONT_TITLE, FONT_LABEL, FONT_ANNOT,
    C_STATIC_S, C_PIECEWISE, C_FEEDBACK,
    academic_ax, save_fig,
)


def main():
    print("Figure 4: Temporal Structure")

    # Load per-step trajectories
    trajs = {}
    for name in ["static", "piecewise", "feedback"]:
        with open(TRC_RESULTS / f"trajectories_{name}.json") as f:
            data = json.load(f)
        all_str = np.array([entry["str"] for entry in data])
        trajs[name] = {
            "mean": all_str.mean(axis=0),
            "std": all_str.std(axis=0),
        }

    T = len(trajs["static"]["mean"])
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    academic_ax(ax)

    for name, color, label, ls in [
        ("static",    C_STATIC_S,  "Static control (λ=0)",          "-"),
        ("piecewise", C_PIECEWISE, "Scheduled control (piecewise)", "--"),
        ("feedback",  C_FEEDBACK,  "Feedback control",              "-"),
    ]:
        m = trajs[name]["mean"]
        s = trajs[name]["std"]
        ax.plot(timesteps, m, ls, color=color, lw=2.5, label=label, zorder=3)
        ax.fill_between(timesteps, m - 0.5 * s, m + 0.5 * s,
                        alpha=0.12, color=color, zorder=1)

    # Target band
    target = trajs["static"]["mean"].mean()
    band_w = 0.01
    ax.axhspan(target - band_w, target + band_w, color="#ddd", alpha=0.3,
               zorder=0, label="Target regime band")
    ax.axhline(target, color="#bbb", ls="--", lw=0.8, alpha=0.5)

    ax.set_xlabel("Generation step $t$", fontsize=FONT_LABEL)
    ax.set_ylabel("STR$_t$", fontsize=FONT_LABEL)
    ax.set_title("Figure 4 — Failure of Static and Scheduled Control",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="0.8")

    ax.annotate("Drift →", xy=(T - 3, trajs["static"]["mean"][-1]),
                fontsize=9, color=C_STATIC_S, ha="right",
                fontstyle="italic")

    plt.tight_layout()
    save_fig(fig, "figure4_temporal_structure")


if __name__ == "__main__":
    main()
