"""
Killer Figure C: The Decoupling of Stability and Semantics
==========================================================

This script:
  1. Computes intra-trajectory CosSim baseline (no shock) as reference
  2. Reads Exp3 v3 results JSON
  3. Generates a 3-panel figure (NeurIPS-ready)

Panel A: STR Impulse Response (both recover)
Panel B: Output text trajectory (math → code switch)
Panel C: Semantic Shift Quantification (CosSim + Mode Switch bars)
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# sys.path configured for local imports
from train_phase1 import LLAMA_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUT_DIR = Path(__file__).parent / "results_dom" / "exp3"
RESULTS_FILE = OUT_DIR / "exp3_v3_results.json"
FIG_FILE = OUT_DIR / "figure_c_decoupling.png"

PROMPTS = [
    "Question: If a train travels 120 km in 2 hours and then 180 km in 3 hours, what is its average speed? Let's think step by step.",
    "Solve the equation: 3(x - 4) + 2 = 5x - 7. Let's think step by step.",
    "A baker has 50 apples. He uses 5 apples for each pie and makes 6 pies. Then he buys 20 more apples. How many apples does he have now? Let's think step by step.",
]

MAX_NEW_TOK = 100
SHOCK_STEP = 50


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Compute Intra-Trajectory CosSim Baseline (no shock)
# ═══════════════════════════════════════════════════════════════════════
def compute_baseline_cossim(model, tokenizer, device, n_runs=3):
    """Generate WITHOUT shock, split in half, compute CosSim.
    This gives us the reference value for 'normal' intra-trajectory similarity.
    """
    print("\n  Computing intra-trajectory CosSim baseline (no shock)...")
    cos_sims = []

    for i, prompt in enumerate(PROMPTS):
        for run in range(n_runs):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            # Simple greedy-ish generation (no shock, no control)
            for t in range(MAX_NEW_TOK):
                with torch.no_grad():
                    out = model(input_ids, use_cache=False)
                logits = out.logits[:, -1, :] / 0.7
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_tok], dim=-1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
                if str(device) == "mps" and (t + 1) % 20 == 0:
                    torch.mps.empty_cache()

            gen_len = input_ids.shape[1] - prompt_len
            if gen_len < 10:
                continue

            # Split at midpoint and compute CosSim
            mid = prompt_len + gen_len // 2
            with torch.no_grad():
                out = model(input_ids, use_cache=False, output_hidden_states=True)
                H = out.hidden_states[-1][0]  # (seq_len, hidden_dim)

            pre_emb = H[prompt_len:mid].float().mean(dim=0)
            post_emb = H[mid:].float().mean(dim=0)
            cos = torch.nn.functional.cosine_similarity(
                pre_emb.unsqueeze(0), post_emb.unsqueeze(0)
            ).item()
            cos_sims.append(cos)
            print(f"    P{i+1}R{run+1}: CosSim={cos:.4f} (gen_len={gen_len})")

    avg = sum(cos_sims) / max(len(cos_sims), 1)
    print(f"\n  Baseline Intra-Trajectory CosSim: {avg:.4f} (n={len(cos_sims)})")
    return avg, cos_sims


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Generate Figure C
# ═══════════════════════════════════════════════════════════════════════
def plot_figure_c(results, baseline_cossim, baseline_cossims_list):
    """Generate the 3-panel Decoupling figure."""

    # Extract data from results
    baseline_runs = results["baseline_shock"]["runs"]
    band_runs = results["band_shock"]["runs"]
    baseline_summary = results["baseline_shock"]["summary"]
    band_summary = results["band_shock"]["summary"]

    # ── Collect STR traces (average across runs) ──
    def avg_trace(runs, max_len=MAX_NEW_TOK):
        traces = []
        for r in runs:
            tr = r["str_trace"]
            # Pad to max_len with last valid value
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

    bl_mean, bl_std = avg_trace(baseline_runs)
    bd_mean, bd_std = avg_trace(band_runs)
    t_axis = np.arange(len(bl_mean))

    # ── Collect semantic data ──
    bl_cos = baseline_summary.get("avg_cosine_sim", 0)
    bd_cos = band_summary.get("avg_cosine_sim", 0)

    # Parse mode switch rates
    def parse_rate(s):
        parts = s.split("/")
        return int(parts[0]) / int(parts[1]) * 100

    bl_ms = parse_rate(baseline_summary["mode_switch_rate"])
    bd_ms = parse_rate(band_summary["mode_switch_rate"])

    # ── Collect text previews ──
    # Find a representative run with mode switch
    bl_text_pre, bl_text_post = "", ""
    bd_text_pre, bd_text_post = "", ""
    for r in baseline_runs:
        sem = r.get("semantic", {})
        if sem.get("pre_mode") == "math" and sem.get("post_mode") == "code":
            bl_text_pre = sem.get("pre_text_preview", "")[:100]
            bl_text_post = sem.get("post_text_preview", "")[:100]
            break
    for r in band_runs:
        sem = r.get("semantic", {})
        if sem.get("pre_mode") == "math" and sem.get("post_mode") == "code":
            bd_text_pre = sem.get("pre_text_preview", "")[:100]
            bd_text_post = sem.get("post_text_preview", "")[:100]
            break

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE
    # ═══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 5.5), dpi=150)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.32)

    # ── Color palette ──
    C_BL = "#E74C3C"    # Baseline red
    C_BD = "#2980B9"    # Band blue
    C_REF = "#27AE60"   # Reference green
    C_SHOCK = "#8E44AD"  # Shock purple
    C_BAND_FILL = "#D5F5E3"
    BG_COLOR = "#FAFAFA"

    fig.patch.set_facecolor("white")

    # ── Panel A: STR Impulse Response ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG_COLOR)

    # Draw band region
    tau = 0.7003
    tau_low, tau_high = tau - 0.07, tau + 0.07
    ax1.axhspan(tau_low, tau_high, alpha=0.15, color="#27AE60", label="Stability Band")
    ax1.axhline(tau, color="#27AE60", ls="--", alpha=0.4, lw=0.8)

    # Shock line
    ax1.axvline(SHOCK_STEP, color=C_SHOCK, ls="--", lw=1.5, alpha=0.7, label="Shock injection")

    # STR traces with confidence bands
    ax1.plot(t_axis, bl_mean, color=C_BL, lw=1.8, label="Baseline", alpha=0.9)
    ax1.fill_between(t_axis, bl_mean - bl_std, bl_mean + bl_std, color=C_BL, alpha=0.1)
    ax1.plot(t_axis, bd_mean, color=C_BD, lw=1.8, label="Band DOM", alpha=0.9)
    ax1.fill_between(t_axis, bd_mean - bd_std, bd_mean + bd_std, color=C_BD, alpha=0.1)

    ax1.set_xlabel("Generation Step $t$", fontsize=10)
    ax1.set_ylabel("STR (Spatiotemporal Repulsion)", fontsize=10)
    ax1.set_title("(A)  STR Impulse Response", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax1.set_xlim(0, MAX_NEW_TOK - 1)
    ax1.set_ylim(0.45, 0.95)

    # Annotate the key insight
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

    # Phase labels
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

    # Show actual text snippets (truncated)
    def wrap_text(text, max_chars=55):
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
        ax2.text(0.08, y_pos + 0.67, wrap_text(bl_text_pre[:80]),
                 fontsize=6.5, fontfamily="monospace", color="#333",
                 transform=ax2.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F5", alpha=0.6))

    shock_preview = 'Wait. The previous reasoning is\nfundamentally flawed... Rewrite as\nPython code: def solve(): ...'
    ax2.text(0.08, y_pos + 0.32, shock_preview,
             fontsize=6.5, fontfamily="monospace", color=C_SHOCK,
             transform=ax2.transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5EEF8", alpha=0.6))

    if bl_text_post:
        ax2.text(0.08, y_pos - 0.05, wrap_text(bl_text_post[:80]),
                 fontsize=6.5, fontfamily="monospace", color="#C0392B",
                 transform=ax2.transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", alpha=0.6))

    # Arrows between phases
    for y1, y2 in [(0.86, 0.63), (0.49, 0.26)]:
        ax2.annotate("", xy=(0.5, y2), xytext=(0.5, y1),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="#AAA", lw=1.2))

    # ── Panel C: Semantic Shift Quantification ──
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(BG_COLOR)

    x_pos = np.array([0, 1, 2.5])
    width = 0.35

    # CosSim bars
    cossim_vals = [baseline_cossim, bl_cos, bd_cos]
    cossim_colors = [C_REF, C_BL, C_BD]
    cossim_labels = ["No Shock\n(reference)", "Baseline\n+ Shock", "Band DOM\n+ Shock"]

    bars = ax3.bar(x_pos, cossim_vals, width=0.6, color=cossim_colors, alpha=0.8,
                   edgecolor="white", linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, cossim_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cossim_labels, fontsize=8)
    ax3.set_ylabel("Embedding Cosine Similarity", fontsize=10)
    ax3.set_title("(C)  Semantic Shift", fontsize=11, fontweight="bold", pad=10)
    ax3.set_ylim(0.6, 1.02)

    # Draw a horizontal line at the reference level
    ax3.axhline(baseline_cossim, color=C_REF, ls=":", lw=1, alpha=0.5)

    # Add mode switch annotation
    ax3.text(0.5, 0.12,
             f"Mode Switch Rate:\n"
             f"Baseline: {bl_ms:.0f}%  |  Band: {bd_ms:.0f}%",
             transform=ax3.transAxes, fontsize=8.5, ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7",
                       edgecolor="#F39C12", alpha=0.9))

    # Bracket annotation showing the drop
    drop = baseline_cossim - bl_cos
    ax3.annotate(f"$\\Delta$ = {drop:.3f}\nsemantic shift",
                 xy=(1, bl_cos), xytext=(1.7, 0.72),
                 fontsize=7.5, color="#E74C3C", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=0.8))

    # ── Suptitle ──
    fig.suptitle(
        "Figure C: Decoupling of Dynamical Stability and Semantic Consistency",
        fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(FIG_FILE, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\n  Figure saved to: {FIG_FILE}")
    return FIG_FILE


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model for baseline CosSim computation
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    # Step 1: Baseline CosSim
    baseline_avg, baseline_list = compute_baseline_cossim(model, tokenizer, device, n_runs=3)

    # Free model memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    # Step 2: Load v3 results and plot
    print(f"\n  Loading results from {RESULTS_FILE}...")
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    fig_path = plot_figure_c(results, baseline_avg, baseline_list)

    # Save baseline data
    baseline_data = {
        "avg_intra_trajectory_cossim": baseline_avg,
        "individual_values": [round(v, 4) for v in baseline_list],
        "n_samples": len(baseline_list),
    }
    with open(OUT_DIR / "baseline_cossim.json", "w") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"  Baseline CosSim data saved.")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Intra-trajectory CosSim (no shock): {baseline_avg:.4f}")
    print(f"  Baseline + Shock CosSim:            {results['baseline_shock']['summary']['avg_cosine_sim']:.4f}")
    print(f"  Band DOM + Shock CosSim:            {results['band_shock']['summary']['avg_cosine_sim']:.4f}")
    print(f"  Drop (no shock -> baseline shock):   {baseline_avg - results['baseline_shock']['summary']['avg_cosine_sim']:.4f}")
    print(f"  Drop (no shock -> band shock):       {baseline_avg - results['band_shock']['summary']['avg_cosine_sim']:.4f}")
    print(f"  Mode Switch: Baseline {results['baseline_shock']['summary']['mode_switch_rate']}, "
          f"Band {results['band_shock']['summary']['mode_switch_rate']}")
    print(f"\n  Figure: {fig_path}")


if __name__ == "__main__":
    main()
