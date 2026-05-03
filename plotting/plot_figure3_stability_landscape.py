"""
Figure 3 — Stability–Operability Landscape

Three panels:
  (A) Order Parameter: Mean STR vs λ with natural OP marked
  (B) Fluctuation Landscape: Var(STR) vs λ, dissociation arrow
  (C) Regime Structure: Violin plots of STR distribution per λ
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from _style import (
    TRC_BASELINES, FONT_TITLE, FONT_LABEL, FONT_ANNOT,
    C_NORMAL, C_NATOP, C_STABOPT,
    academic_ax, save_fig,
)


def main():
    print("Figure 3: Regime Structure")

    # Load summaries
    with open(TRC_BASELINES / "trc_baselines.json") as f:
        baselines = json.load(f)
    summaries = baselines["summaries"]
    lambdas = sorted([float(k) for k in summaries.keys()])
    means = [summaries[str(l)]["mean_str"] for l in lambdas]
    vars_ = [summaries[str(l)]["var_str"] for l in lambdas]

    # Load per-λ trajectory distributions for violin
    all_str_per_lambda = {}
    for lam in lambdas:
        fname = TRC_BASELINES / f"trajectories_lambda_{lam}.json"
        with open(fname) as f:
            data = json.load(f)
        all_vals = []
        for entry in data:
            all_vals.extend(entry["str"])
        all_str_per_lambda[lam] = all_vals

    fig = plt.figure(figsize=(14, 4.5))
    fig.patch.set_facecolor("white")

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for ax in (ax1, ax2, ax3):
        academic_ax(ax)

    nat_idx = lambdas.index(0.0)
    min_var_idx = int(np.argmin(vars_))

    # Panel A: Order Parameter
    ax1.plot(lambdas, means, "o-", color=C_NORMAL, lw=2, ms=6, zorder=3)
    ax1.axvline(0.0, color=C_NATOP, ls=":", lw=1, alpha=0.5)
    ax1.scatter([0.0], [means[nat_idx]], s=80, color=C_NATOP, marker="D", zorder=5)
    ax1.annotate("Natural\nOperating Point", xy=(0.0, means[nat_idx]),
                 xytext=(0.07, means[nat_idx] - 0.015),
                 arrowprops=dict(arrowstyle="->", color=C_NATOP, lw=1.2),
                 fontsize=FONT_ANNOT, color=C_NATOP, fontweight="bold")
    ax1.set_xlabel("λ", fontsize=FONT_LABEL)
    ax1.set_ylabel("Mean STR (order parameter)", fontsize=FONT_LABEL)
    ax1.set_title("(A) Order Parameter", fontsize=FONT_TITLE, fontweight="bold")

    # Panel B: Fluctuation Landscape
    ax2.plot(lambdas, vars_, "s-", color="#9467bd", lw=2, ms=6, zorder=3)
    ax2.scatter([lambdas[min_var_idx]], [vars_[min_var_idx]], s=150,
                color=C_STABOPT, marker="*", zorder=5)
    ax2.scatter([0.0], [vars_[nat_idx]], s=80, color=C_NATOP, marker="D", zorder=5)
    ax2.annotate("Stability\nOptimum", xy=(lambdas[min_var_idx], vars_[min_var_idx]),
                 xytext=(lambdas[min_var_idx] - 0.09, vars_[min_var_idx] + 0.0012),
                 arrowprops=dict(arrowstyle="->", color=C_STABOPT, lw=1.2),
                 fontsize=FONT_ANNOT, color=C_STABOPT, fontweight="bold")
    ax2.annotate("Natural OP",
                 xy=(0.0, vars_[nat_idx]),
                 xytext=(0.04, vars_[nat_idx] + 0.0012),
                 arrowprops=dict(arrowstyle="->", color=C_NATOP, lw=1.2),
                 fontsize=FONT_ANNOT, color=C_NATOP, fontweight="bold")
    # Dissociation arrow
    ax2.annotate("", xy=(lambdas[min_var_idx], vars_[min_var_idx] + 0.0002),
                 xytext=(0.0, vars_[nat_idx] - 0.0002),
                 arrowprops=dict(arrowstyle="<->", color="#e377c2", lw=1.5, ls="--"))
    mid_x = (lambdas[min_var_idx] + 0.0) / 2
    mid_y = (vars_[min_var_idx] + vars_[nat_idx]) / 2
    ax2.text(mid_x + 0.04, mid_y, "Dissociation",
             fontsize=8, color="#e377c2", fontstyle="italic", ha="left")
    ax2.set_xlabel("λ", fontsize=FONT_LABEL)
    ax2.set_ylabel("Var(STR$_t$)", fontsize=FONT_LABEL)
    ax2.set_title("(B) Fluctuation Landscape", fontsize=FONT_TITLE, fontweight="bold")

    # Panel C: Violin plots
    violin_data = [all_str_per_lambda[l] for l in lambdas]
    positions = list(range(len(lambdas)))

    n = len(lambdas)
    colors = []
    for i, l in enumerate(lambdas):
        if l < 0:
            t = abs(l) / 0.2
            colors.append(plt.cm.Blues(0.3 + 0.5 * t))
        elif l > 0:
            t = l / 0.2
            colors.append(plt.cm.Reds(0.3 + 0.5 * t))
        else:
            colors.append("#999999")

    parts = ax3.violinplot(violin_data, positions=positions, showmeans=True,
                           showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
        pc.set_alpha(0.75)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)

    ax3.set_xticks(positions)
    ax3.set_xticklabels([f"{l:.2f}" for l in lambdas], fontsize=7, rotation=45)
    ax3.set_xlabel("λ", fontsize=FONT_LABEL)
    ax3.set_ylabel("STR distribution", fontsize=FONT_LABEL)
    ax3.set_title("(C) Regime Structure", fontsize=FONT_TITLE, fontweight="bold")

    ax3.axvline(positions[nat_idx], color=C_NATOP, ls=":", lw=1, alpha=0.5)

    ax3.text(1.0, ax3.get_ylim()[1] * 0.98, "Exploratory",
             fontsize=7, color=plt.cm.Blues(0.7), ha="center", va="top",
             fontstyle="italic")
    ax3.text(positions[nat_idx], ax3.get_ylim()[1] * 0.98, "Unstable\nNatural",
             fontsize=7, color="#666", ha="center", va="top",
             fontstyle="italic")
    ax3.text(len(lambdas) - 2, ax3.get_ylim()[1] * 0.98, "Coherent",
             fontsize=7, color=plt.cm.Reds(0.7), ha="center", va="top",
             fontstyle="italic")

    fig.suptitle("Figure 3 — Stability\u2013Operability Landscape",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "figure3_regime_structure")


if __name__ == "__main__":
    main()
