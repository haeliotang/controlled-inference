"""
Figure 9 — Geometric Recovery Does Not Imply Semantic Restoration

Uses precomputed data (no model inference required):
  - exp3_v3_results.json: STR traces and semantic analysis
  - baseline_cossim.json: precomputed intra-trajectory CosSim baseline

Panel A: STR Impulse Response (both baseline and Band recover)
Panel B: Output text trajectory (math → code switch)
Panel C: Semantic Shift Quantification (CosSim + Mode Switch bars)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR = RESULTS_DIR / "exp3"
RESULTS_FILE = OUT_DIR / "exp3_v3_results.json"
BASELINE_COSSIM_FILE = OUT_DIR / "baseline_cossim.json"
FIG_FILE = Path(__file__).parent / "figure9_decoupling.png"

MAX_NEW_TOK = 100
SHOCK_STEP = 50


def avg_trace(runs, max_len=MAX_NEW_TOK):
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


def parse_rate(s):
    parts = s.split("/")
    return int(parts[0]) / int(parts[1]) * 100


def main():
    print("Figure 9: Geometric Recovery Does Not Imply Semantic Restoration")

    # Load precomputed baseline CosSim
    with open(BASELINE_COSSIM_FILE) as f:
        bl_data = json.load(f)
    baseline_cossim = bl_data["avg_intra_trajectory_cossim"]

    # Load exp3 v3 results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    baseline_runs = results["baseline_shock"]["runs"]
    band_runs = results["band_shock"]["runs"]
    baseline_summary = results["baseline_shock"]["summary"]
    band_summary = results["band_shock"]["summary"]

    # STR traces
    bl_mean, bl_std = avg_trace(baseline_runs)
    bd_mean, bd_std = avg_trace(band_runs)
    t_axis = np.arange(len(bl_mean))

    # Semantic data
    bl_cos = baseline_summary.get("avg_cosine_sim", 0)
    bd_cos = band_summary.get("avg_cosine_sim", 0)
    bl_ms = parse_rate(baseline_summary["mode_switch_rate"])
    bd_ms = parse_rate(band_summary["mode_switch_rate"])

    # Find representative text previews
    bl_text_pre, bl_text_post = "", ""
    for r in baseline_runs:
        sem = r.get("semantic", {})
        if sem.get("pre_mode") == "math" and sem.get("post_mode") == "code":
            bl_text_pre = sem.get("pre_text_preview", "")[:100]
            bl_text_post = sem.get("post_text_preview", "")[:100]
            break

    # ── Color palette ──
    C_BL = "#E74C3C"
    C_BD = "#2980B9"
    C_REF = "#27AE60"
    C_SHOCK = "#8E44AD"
    BG_COLOR = "#FAFAFA"

    fig = plt.figure(figsize=(16, 5.5), dpi=150)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.32)
    fig.patch.set_facecolor("white")

    # ── Panel A: STR Impulse Response ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG_COLOR)

    tau = 0.7003
    tau_low, tau_high = tau - 0.07, tau + 0.07
    ax1.axhspan(tau_low, tau_high, alpha=0.15, color="#27AE60", label="Stability Band")
    ax1.axhline(tau, color="#27AE60", ls="--", alpha=0.4, lw=0.8)
    ax1.axvline(SHOCK_STEP, color=C_SHOCK, ls="--", lw=1.5, alpha=0.7, label="Shock injection")

    ax1.plot(t_axis, bl_mean, color=C_BL, lw=1.8, label="Baseline", alpha=0.9)
    ax1.fill_between(t_axis, bl_mean - bl_std, bl_mean + bl_std, color=C_BL, alpha=0.1)
    ax1.plot(t_axis, bd_mean, color=C_BD, lw=1.8, label="Band DOM", alpha=0.9)
    ax1.fill_between(t_axis, bd_mean - bd_std, bd_mean + bd_std, color=C_BD, alpha=0.1)

    ax1.set_xlabel("Generation Step $t$", fontsize=10)
    ax1.set_ylabel("STR (Straightness Ratio)", fontsize=10)
    ax1.set_title("(A)  STR Impulse Response", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.set_xlim(0, MAX_NEW_TOK - 1)
    ax1.set_ylim(0.45, 0.95)

    ax1.annotate("Both recover!\nBut to where?",
                 xy=(75, tau), xytext=(78, 0.88),
                 fontsize=8, fontstyle="italic", color="#555",
                 arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

    # ── Panel B: Text Trajectory ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG_COLOR)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("(B)  Output Trajectory", fontsize=11, fontweight="bold", pad=10)

    phases = [
        (0.92, "Before Shock $(t < 50)$", "#2ECC71", "math reasoning"),
        (0.55, "Shock Injection $(t = 50)$", C_SHOCK, "structural conflict"),
        (0.18, "After Shock $(t > 50)$", "#E74C3C", "code generation"),
    ]

    for y_pos, label, color, mode in phases:
        ax2.text(0.05, y_pos + 0.06, label, fontsize=9, fontweight="bold", color=color,
                 transform=ax2.transAxes)
        ax2.text(0.08, y_pos - 0.01, f"[{mode}]", fontsize=7, fontstyle="italic",
                 color="#888", transform=ax2.transAxes)

    def wrap_text(text, max_chars=40):
        lines = []
        while len(text) > max_chars:
            idx = text.rfind(" ", 0, max_chars)
            if idx == -1:
                idx = max_chars
            lines.append(text[:idx])
            text = text[idx:].lstrip()
        if text:
            lines.append(text)
        return "\n".join(lines[:2])

    if bl_text_pre:
        ax2.text(0.08, y_pos + 0.67, wrap_text(bl_text_pre[:60]),
                 fontsize=8, fontfamily="monospace", color="#333",
                 transform=ax2.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F5", alpha=0.6))

    shock_preview = 'Wait. The previous reasoning\nis fundamentally flawed...\nRewrite as Python code...'
    ax2.text(0.08, y_pos + 0.32, shock_preview,
             fontsize=8, fontfamily="monospace", color=C_SHOCK,
             transform=ax2.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5EEF8", alpha=0.6))

    if bl_text_post:
        ax2.text(0.08, y_pos - 0.05, wrap_text(bl_text_post.strip()[:60]),
                 fontsize=8, fontfamily="monospace", color="#C0392B",
                 transform=ax2.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", alpha=0.6))

    for y1, y2 in [(0.80, 0.72), (0.43, 0.35)]:
        ax2.annotate("", xy=(0.04, y2), xytext=(0.04, y1),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="#AAA", lw=1.2))

    # ── Panel C: Semantic Shift Quantification ──
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(BG_COLOR)

    x_pos = np.array([0, 1, 2.5])

    cossim_vals = [baseline_cossim, bl_cos, bd_cos]
    cossim_colors = [C_REF, C_BL, C_BD]
    cossim_labels = ["No Shock\n(reference)", "Baseline\n+ Shock", "Band DOM\n+ Shock"]

    bars = ax3.bar(x_pos, cossim_vals, width=0.6, color=cossim_colors, alpha=0.8,
                   edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, cossim_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cossim_labels, fontsize=8)
    ax3.set_ylabel("Embedding Cosine Similarity", fontsize=10)
    ax3.set_title("(C)  Semantic Shift", fontsize=11, fontweight="bold", pad=10)
    ax3.set_ylim(0.6, 1.02)

    ax3.axhline(baseline_cossim, color=C_REF, ls=":", lw=1, alpha=0.5)

    ax3.text(0.5, 0.12,
             f"Mode Switch Rate:\n"
             f"Baseline: {bl_ms:.0f}%  |  Band: {bd_ms:.0f}%",
             transform=ax3.transAxes, fontsize=8.5, ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7",
                       edgecolor="#F39C12", alpha=0.9))

    drop = baseline_cossim - bl_cos
    ax3.annotate(f"$\\Delta$ = {drop:.3f}\nsemantic shift",
                 xy=(1, bl_cos), xytext=(1.73, 0.72),
                 fontsize=7.5, color="#E74C3C", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=0.8))

    fig.suptitle(
        "Figure 9 — Geometric Recovery Does Not Imply Semantic Restoration",
        fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(FIG_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(str(FIG_FILE).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {FIG_FILE}")


if __name__ == "__main__":
    main()
