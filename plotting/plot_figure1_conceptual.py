"""
Figure 1 — Inference as a Controlled Dynamical System

Conceptual diagram with four panels:
  (A) Open-loop inference: drifting, unstable trajectory
  (B) STR as state observable: coherent vs fragmented
  (C) Control loop block diagram
  (D) Closed-loop inference: bounded, stable trajectory
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

DARK_BG    = "#ffffff"
PANEL_BG   = "#ffffff"
GRID_COLOR = "#e5e7eb"
TEXT_COLOR = "#111827"
ACCENT_RED = "#d73a49"
ACCENT_GREEN = "#2ea043"
ACCENT_BLUE = "#0366d6"
ACCENT_PURPLE = "#6f42c1"
ACCENT_ORANGE = "#d18616"
ACCENT_YELLOW = "#b08800"
MUTED_GRAY = "#4b5563"
BAND_GREEN = "#2ea04320"

fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)

# ── Layout: 2x2 grid ─────────────────────────────────────────
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                      left=0.06, right=0.97, top=0.92, bottom=0.06)

# ══════════════════════════════════════════════════════════════
# Panel (A): Open-loop — drifting trajectory
# ══════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0], facecolor=PANEL_BG)
ax_a.set_title("(A)  Uncontrolled Inference (Open Loop)",
               color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=12)

np.random.seed(42)
T = 80
# Generate a drifting trajectory with increasing variance
t = np.linspace(0, 1, T)
drift = 0.8 * t**1.5
noise = np.cumsum(np.random.randn(T) * 0.06)
traj_x = t
traj_y = 0.5 + drift + noise

# Draw faint "ideal" band
ax_a.axhspan(0.35, 0.65, color=ACCENT_GREEN, alpha=0.08)
ax_a.axhline(0.5, color=ACCENT_GREEN, alpha=0.25, ls="--", lw=0.8)

# Draw the drifting trajectory
for i in range(len(traj_x) - 1):
    progress = i / len(traj_x)
    color_intensity = 0.4 + 0.6 * progress
    ax_a.plot(traj_x[i:i+2], traj_y[i:i+2],
              color=ACCENT_RED, alpha=color_intensity, lw=1.8)

# Mark start and drift
ax_a.plot(traj_x[0], traj_y[0], 'o', color=ACCENT_GREEN, ms=7, zorder=5)
ax_a.plot(traj_x[-1], traj_y[-1], 'o', color=ACCENT_RED, ms=7, zorder=5)
ax_a.annotate("$h_1$", (traj_x[0], traj_y[0]), color=ACCENT_GREEN,
              fontsize=10, fontweight="bold", xytext=(-15, -18),
              textcoords="offset points")
ax_a.annotate("$h_T$", (traj_x[-1], traj_y[-1]), color=ACCENT_RED,
              fontsize=10, fontweight="bold", xytext=(5, -18),
              textcoords="offset points")

# Drift annotation
ax_a.annotate("Drift", xy=(0.75, traj_y[60]), xytext=(0.55, 1.55),
              color=ACCENT_RED, fontsize=10, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.5),
              ha="center")
ax_a.text(0.5, 0.3, "Stable regime", color=ACCENT_GREEN, alpha=0.5,
          fontsize=8, ha="center", fontstyle="italic")

ax_a.set_xlim(-0.05, 1.05)
ax_a.set_ylim(0.0, 2.0)
ax_a.set_xlabel("Generation step $t$", color=MUTED_GRAY, fontsize=9)
ax_a.set_ylabel("Representation space", color=MUTED_GRAY, fontsize=9)
ax_a.tick_params(colors=MUTED_GRAY, labelsize=7)
for spine in ax_a.spines.values():
    spine.set_color(GRID_COLOR)

# Bottom label
ax_a.text(0.5, -0.16, "No control → trajectory drift",
          transform=ax_a.transAxes, ha="center", fontsize=9,
          color=ACCENT_RED, fontstyle="italic")

# ══════════════════════════════════════════════════════════════
# Panel (B): STR as Observable — two contrasting trajectories
# ══════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1], facecolor=PANEL_BG)
ax_b.set_title("(B)  Trajectory Geometry as State Observable",
               color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=12)

np.random.seed(7)
T2 = 60
t2 = np.linspace(0, 1, T2)

# Coherent trajectory (high STR)
coherent_y = 0.7 + 0.15 * np.sin(2 * np.pi * t2) + np.random.randn(T2) * 0.02
# Fragmented trajectory (low STR)
fragmented_y = 0.3 + np.cumsum(np.random.randn(T2) * 0.04)
fragmented_y = fragmented_y - fragmented_y[0] + 0.3

ax_b.plot(t2, coherent_y, color=ACCENT_GREEN, lw=2.2, alpha=0.9,
          label="High STR (coherent)")
ax_b.plot(t2, fragmented_y, color=ACCENT_ORANGE, lw=1.8, alpha=0.8,
          ls="-", label="Low STR (fragmented)")

# STR annotation arrows
ax_b.annotate("STR ≈ 1.0", xy=(0.65, coherent_y[39]),
              xytext=(0.78, 1.0), color=ACCENT_GREEN, fontsize=9,
              fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.2))
ax_b.annotate("STR ≈ 0.3", xy=(0.65, fragmented_y[39]),
              xytext=(0.78, -0.05), color=ACCENT_ORANGE, fontsize=9,
              fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=1.2))

# Observable box
bbox_props = dict(boxstyle="round,pad=0.4", fc=ACCENT_BLUE, alpha=0.15,
                  ec=ACCENT_BLUE, lw=1.2)
ax_b.text(0.5, 0.9, "Observable:  STR($h_1 \\dots h_t$)",
          transform=ax_b.transAxes, ha="center", va="center",
          fontsize=10, color=ACCENT_BLUE, fontweight="bold",
          bbox=bbox_props)

ax_b.set_xlim(-0.05, 1.05)
ax_b.set_ylim(-0.15, 1.35)
ax_b.set_xlabel("Generation step $t$", color=MUTED_GRAY, fontsize=9)
ax_b.set_ylabel("Trajectory geometry", color=MUTED_GRAY, fontsize=9)
ax_b.tick_params(colors=MUTED_GRAY, labelsize=7)
for spine in ax_b.spines.values():
    spine.set_color(GRID_COLOR)

ax_b.text(0.5, -0.16, "STR measures trajectory geometry (state of the system)",
          transform=ax_b.transAxes, ha="center", fontsize=9,
          color=ACCENT_BLUE, fontstyle="italic")

# ══════════════════════════════════════════════════════════════
# Panel (C): Control Loop Block Diagram
# ══════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 0], facecolor=PANEL_BG)
ax_c.set_title("(C)  Feedback Control Loop",
               color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=12)
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 7)
ax_c.set_aspect("equal")
ax_c.axis("off")

def draw_block(ax, x, y, w, h, label, color, sublabel=None):
    """Draw a rounded rectangle block with label."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, alpha=0.2,
                          edgecolor=color, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=10,
            color=color, fontweight="bold")
    if sublabel:
        ax.text(x, y - 0.3, sublabel, ha="center", va="center",
                fontsize=8, color=color, alpha=0.7)

def draw_arrow(ax, x1, y1, x2, y2, color, label=None, label_offset=(0, 0.3)):
    """Draw an arrow between two points with optional label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                mutation_scale=15))
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

# Blocks
draw_block(ax_c, 5, 6, 3.0, 1.0, "Controller", ACCENT_PURPLE, sublabel="$F$")
draw_block(ax_c, 5, 3.5, 3.0, 1.0, "Inference Dynamics", ACCENT_BLUE, sublabel="Model")
draw_block(ax_c, 5, 1, 3.0, 1.0, "State Observer", ACCENT_GREEN, sublabel="STR")

# Arrows: Controller → Dynamics (control signal λ_t)
draw_arrow(ax_c, 5, 5.5, 5, 4.0, ACCENT_PURPLE,
           label="$\\lambda_t$  (control signal)", label_offset=(1.8, 0))

# Arrows: Dynamics → Observer (hidden states)
draw_arrow(ax_c, 5, 3.0, 5, 1.5, ACCENT_BLUE,
           label="$h_t$  (hidden states)", label_offset=(1.6, 0))

# Arrows: Observer → Controller (feedback — goes around the side)
# Left side path: Observer bottom → left → Controller left
ax_c.annotate("", xy=(3.5, 6), xytext=(3.5, 1),
              arrowprops=dict(arrowstyle="-|>", color=ACCENT_GREEN, lw=1.8,
                              connectionstyle="arc3,rad=0.4",
                              mutation_scale=15))
ax_c.text(1.5, 3.5, "$\\mathrm{STR}_t$\n(feedback)",
          ha="center", va="center", fontsize=8,
          color=ACCENT_GREEN, fontweight="bold")

# Operators annotation on the right
ax_c.text(8.5, 4.8, "$M_S$  (contraction)", color=ACCENT_YELLOW,
          fontsize=8, ha="left", fontstyle="italic")
ax_c.text(8.5, 4.3, "$M_L$  (expansion)", color=ACCENT_ORANGE,
          fontsize=8, ha="left", fontstyle="italic")

# "Closed Loop" label
bbox_cl = dict(boxstyle="round,pad=0.3", fc=ACCENT_PURPLE, alpha=0.1,
               ec=ACCENT_PURPLE, lw=1)
ax_c.text(5, -0.3, "Closed-loop: control action depends on current state",
          ha="center", va="center", fontsize=8.5, color=ACCENT_PURPLE,
          fontstyle="italic", bbox=bbox_cl)

# ══════════════════════════════════════════════════════════════
# Panel (D): Controlled Inference — bounded trajectory
# ══════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 1], facecolor=PANEL_BG)
ax_d.set_title("(D)  Controlled Inference (Closed Loop)",
               color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=12)

np.random.seed(21)
T3 = 80
t3 = np.linspace(0, 1, T3)

# Stable operating band
band_center = 0.5
band_width = 0.15
ax_d.axhspan(band_center - band_width, band_center + band_width,
             color=ACCENT_GREEN, alpha=0.12, label="Stable operating regime")
ax_d.axhline(band_center, color=ACCENT_GREEN, alpha=0.3, ls="--", lw=0.8)

# Controlled trajectory — bounded oscillation
controlled_y = band_center + 0.12 * np.sin(3 * np.pi * t3)
# Add small noise but keep bounded
noise_c = np.random.randn(T3) * 0.03
controlled_y += noise_c
# Clip to simulate feedback bounding
controlled_y = np.clip(controlled_y, band_center - band_width * 1.1,
                       band_center + band_width * 1.1)

# Add a perturbation at step 40, then recovery
perturb_idx = 40
controlled_y[perturb_idx] += 0.35  # big spike
# Exponential recovery
for i in range(perturb_idx + 1, min(perturb_idx + 15, T3)):
    recovery_rate = 0.55
    controlled_y[i] = band_center + (controlled_y[i-1] - band_center) * recovery_rate
    controlled_y[i] += np.random.randn() * 0.01

# Draw trajectory
for i in range(len(t3) - 1):
    if i == perturb_idx:
        ax_d.plot(t3[i:i+2], controlled_y[i:i+2],
                  color=ACCENT_RED, lw=2.5, alpha=0.9)
    else:
        ax_d.plot(t3[i:i+2], controlled_y[i:i+2],
                  color=ACCENT_GREEN, lw=1.8, alpha=0.85)

# Mark perturbation
ax_d.annotate("Perturbation", xy=(t3[perturb_idx], controlled_y[perturb_idx]),
              xytext=(t3[perturb_idx] + 0.08, 1.05),
              color=ACCENT_RED, fontsize=9, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.3))

# Mark recovery
ax_d.annotate("Recovery", xy=(t3[perturb_idx + 8], controlled_y[perturb_idx + 8]),
              xytext=(t3[perturb_idx + 8] + 0.12, 0.85),
              color=ACCENT_GREEN, fontsize=9, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.3))

# Start and end markers
ax_d.plot(t3[0], controlled_y[0], 'o', color=ACCENT_GREEN, ms=7, zorder=5)
ax_d.plot(t3[-1], controlled_y[-1], 'o', color=ACCENT_GREEN, ms=7, zorder=5)
ax_d.annotate("$h_1$", (t3[0], controlled_y[0]), color=ACCENT_GREEN,
              fontsize=10, fontweight="bold", xytext=(-15, -18),
              textcoords="offset points")
ax_d.annotate("$h_T$", (t3[-1], controlled_y[-1]), color=ACCENT_GREEN,
              fontsize=10, fontweight="bold", xytext=(5, -18),
              textcoords="offset points")

# Band labels
ax_d.text(0.15, band_center + band_width + 0.03, "Upper bound",
          color=ACCENT_GREEN, fontsize=7, alpha=0.6)
ax_d.text(0.15, band_center - band_width - 0.06, "Lower bound",
          color=ACCENT_GREEN, fontsize=7, alpha=0.6)

ax_d.set_xlim(-0.05, 1.05)
ax_d.set_ylim(0.0, 1.2)
ax_d.set_xlabel("Generation step $t$", color=MUTED_GRAY, fontsize=9)
ax_d.set_ylabel("Representation space", color=MUTED_GRAY, fontsize=9)
ax_d.tick_params(colors=MUTED_GRAY, labelsize=7)
for spine in ax_d.spines.values():
    spine.set_color(GRID_COLOR)

ax_d.text(0.5, -0.16, "Feedback → bounded dynamics + perturbation recovery",
          transform=ax_d.transAxes, ha="center", fontsize=9,
          color=ACCENT_GREEN, fontstyle="italic")

# ── Main title ────────────────────────────────────────────────
fig.suptitle("Figure 1 — Inference as a Controlled Dynamical System",
             color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97)

# ── Save ──────────────────────────────────────────────────────
out = str(Path(__file__).parent / "figure1_conceptual.png")
fig.savefig(out, dpi=300, facecolor=DARK_BG, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), facecolor=DARK_BG, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
