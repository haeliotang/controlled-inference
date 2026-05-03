"""
Experiment: KV-Cache Ablation (Exp-KV-Minimal)

Tests whether DOM control is effective with KV-cache ON.

Already available data:
  1. Baseline + KV-ON   → from baseline results (standard inference)
  2. Baseline + KV-OFF  → from dom_results.json (baseline runs)
  3. Band DOM + KV-OFF  → from dom_results.json (DOM runs)
  4. Band DOM + KV-ON   ← THIS EXPERIMENT

If (4) ≈ (3): KV-cache doesn't affect DOM → paper can simplify
If (4) ≠ (3): KV-cache matters → paper must discuss as technical requirement

Architecture note:
  DOM injects control into the FINAL residual stream (after all attention).
  With KV-cache, prior layers' K/V are cached and NOT recomputed.
  The control signal affects: H_controlled → RMSNorm → lm_head → logits → next token
  But the NEXT step's attention uses fresh K/V for the new position only.
  So DOM + KV-cache should still work at the logit level, just without
  the indirect re-attention benefit.

Usage:
    cd code/experiments
    TRANSFORMERS_OFFLINE=1 python3 ../Controlled Inference/exp_kv_cache.py
"""

import json, sys, time, os
from pathlib import Path

# Fix for transformers 5.6.0: local path loading
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup (same as dom_prototype.py) ──
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR = Path(__file__).parent.parent / "results"
# sys.path configured for local imports
# sys.path configured for local imports


from str_loss import _pairwise_dist_sq
from train_phase1 import (
    apply_lora, LoRALinear, load_data, collate_fn, estimate_sigma,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, MAX_LEN, BATCH_SIZE,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from dom_prototype import (
    StateBuffer, PDController, dom_control, extract_last_layer_lora,
    TEST_PROMPTS, GAMMA, PD_KAPPA, PD_BETA, PD_TAU, WINDOW_K, MAX_NEW_TOK,
)

OUT_DIR = Path(__file__).parent / "results_kv_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# DOM generation with KV-cache ON (incremental forward)
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_with_dom_kv_on(model, tokenizer, prompt, A_S, B_S, A_L, B_L,
                             sigma, lora_scaling=1.0, gamma=GAMMA,
                             max_new_tokens=MAX_NEW_TOK, device="mps"):
    """Generate with DOM control + KV-cache ON.

    Key difference from dom_prototype.py:
    - Uses past_key_values for incremental decoding
    - Each step only forwards the latest token through the model
    - DOM control still modifies the final hidden state before lm_head
    """
    controller = PDController()
    state = StateBuffer(window_k=WINDOW_K, sigma=sigma)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    str_log = []
    alpha_log = []
    entropy_log = []
    past_kv = None

    for t in range(max_new_tokens):
        if t == 0:
            # First step: full forward (no cache yet)
            outputs = model(input_ids, output_hidden_states=True,
                            use_cache=True)
        else:
            # Incremental: only forward the latest token
            outputs = model(input_ids[:, -1:],
                            past_key_values=past_kv,
                            output_hidden_states=True,
                            use_cache=True)

        past_kv = outputs.past_key_values
        H_base = outputs.hidden_states[-1]  # (1, seq_or_1, D)

        # State: update buffer with last position hidden
        state.update(H_base[:, -1, :])
        str_t = state.compute_str(device)

        if str_t is None:
            logits = outputs.logits[:, -1:, :].float()
            alpha_t = 0.5
        else:
            alpha_t = controller.step(str_t)
            # DOM actuator: modify final hidden state
            # For KV-cache mode, H_base is (1, 1, D) for t > 0
            H_ctrl = dom_control(H_base, A_S, B_S, A_L, B_L,
                                 alpha_t, gamma=gamma,
                                 lora_scaling=lora_scaling)
            H_normed = model.model.norm(H_ctrl[:, -1:, :])
            logits = model.lm_head(H_normed).float()

            # NOTE: We do NOT inject the controlled hidden state back into
            # the KV cache. The cache retains the "uncontrolled" K/V.
            # This is the architectural limitation being tested.

        probs = F.softmax(logits.squeeze(0), dim=-1)
        ent = -(probs * F.log_softmax(logits.squeeze(0), dim=-1)).sum(-1)
        entropy_log.append(ent.item())

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        str_log.append(str_t)
        alpha_log.append(alpha_t)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if str(device) == "mps" and (t + 1) % 10 == 0:
            torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "str_trace": str_log,
        "alpha_trace": alpha_log,
        "entropy_trace": entropy_log,
        "n_tokens": len(alpha_log),
    }


@torch.no_grad()
def generate_baseline_kv_on(model, tokenizer, prompt, sigma=1.0,
                             max_new_tokens=MAX_NEW_TOK, device="mps"):
    """Baseline generation with KV-cache ON."""
    state = StateBuffer(window_k=WINDOW_K, sigma=sigma)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    str_log = []
    entropy_log = []
    past_kv = None

    for t in range(max_new_tokens):
        if t == 0:
            outputs = model(input_ids, output_hidden_states=True,
                            use_cache=True)
        else:
            outputs = model(input_ids[:, -1:],
                            past_key_values=past_kv,
                            output_hidden_states=True,
                            use_cache=True)

        past_kv = outputs.past_key_values
        H_base = outputs.hidden_states[-1]

        state.update(H_base[:, -1, :])
        str_t = state.compute_str(device)
        str_log.append(str_t)

        logits = outputs.logits[:, -1:, :].float()
        probs = F.softmax(logits.squeeze(0), dim=-1)
        ent = -(probs * F.log_softmax(logits.squeeze(0), dim=-1)).sum(-1)
        entropy_log.append(ent.item())

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if str(device) == "mps" and (t + 1) % 10 == 0:
            torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "str_trace": str_log,
        "entropy_trace": entropy_log,
        "n_tokens": len(str_log),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def stats(vals):
    if not vals:
        return {"mean": 0, "var": 0, "max": 0, "min": 0}
    t = torch.tensor(vals)
    return {
        "mean": round(t.mean().item(), 4),
        "var": round(t.var().item(), 6),
        "max": round(t.max().item(), 4),
        "min": round(t.min().item(), 4),
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load pre-trained LoRA operators ──
    lora_S_path = RESULTS_DIR / "lora_S.pt"
    lora_L_path = RESULTS_DIR / "lora_L.pt"

    if not lora_S_path.exists() or not lora_L_path.exists():
        print("ERROR: Pre-trained LoRA weights not found.")
        print(f"  Expected: {lora_S_path}")
        print(f"  Expected: {lora_L_path}")
        print("  Run dom_prototype.py first to train operators.")
        return

    print(f"Loading LoRA operators from {RESULTS_DIR}")

    # Load base model and apply LoRA to extract structure
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, dtype=torch.float16)
    model.to(device)

    # Apply LoRA to get the structure, then load S weights
    apply_lora(model, LORA_RANK, LORA_ALPHA)
    lora_S_state = torch.load(lora_S_path, map_location=device, weights_only=True)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_S_state:
                module.lora_A.data.copy_(lora_S_state[a_key].to(device))
                module.lora_B.data.copy_(lora_S_state[b_key].to(device))
    lora_S_info = extract_last_layer_lora(model)

    # Reload model for L weights
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, dtype=torch.float16)
    model.to(device)
    apply_lora(model, LORA_RANK, LORA_ALPHA)
    lora_L_state = torch.load(lora_L_path, map_location=device, weights_only=True)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_L_state:
                module.lora_A.data.copy_(lora_L_state[a_key].to(device))
                module.lora_B.data.copy_(lora_L_state[b_key].to(device))
    lora_L_info = extract_last_layer_lora(model)

    # Get sigma from existing results
    dom_results_path = RESULTS_DIR / "dom_results.json"
    with open(dom_results_path) as f:
        existing = json.load(f)
    sigma = existing["config"]["sigma"]
    scaling = LORA_ALPHA / LORA_RANK

    A_S = lora_S_info["A"].to(device)
    B_S = lora_S_info["B"].to(device)
    A_L = lora_L_info["A"].to(device)
    B_L = lora_L_info["B"].to(device)

    print(f"σ = {sigma:.1f}, scaling = {scaling}")
    print(f"A_S: {A_S.shape}, B_S: {B_S.shape}")

    # ── Reload clean base model for inference ──
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, dtype=torch.float16)
    model.to(device)
    model.eval()

    # ══════════════════════════════════════════════════════════
    # Run 4 conditions on all 3 prompts
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  KV-Cache Ablation Experiment")
    print("  Conditions: Baseline±KV, DOM±KV")
    print("=" * 60)

    all_results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─' * 50}")
        print(f"  Prompt {i + 1}/{len(TEST_PROMPTS)}")
        print(f"{'─' * 50}")
        print(f"  {prompt[:80]}...")

        # Condition 1: Baseline + KV-OFF (re-run for consistency)
        from dom_prototype import generate_baseline as gen_base_off
        base_off = gen_base_off(model, tokenizer, prompt,
                                sigma=sigma, device=str(device))

        # Condition 2: Baseline + KV-ON
        base_on = generate_baseline_kv_on(model, tokenizer, prompt,
                                           sigma=sigma, device=str(device))

        # Condition 3: DOM + KV-OFF (re-run for consistency)
        from dom_prototype import generate_with_dom as gen_dom_off
        dom_off = gen_dom_off(model, tokenizer, prompt,
                              A_S, B_S, A_L, B_L,
                              sigma=sigma, lora_scaling=scaling,
                              gamma=GAMMA, device=str(device))

        # Condition 4: DOM + KV-ON (THE NEW CONDITION)
        dom_on = generate_with_dom_kv_on(model, tokenizer, prompt,
                                          A_S, B_S, A_L, B_L,
                                          sigma=sigma, lora_scaling=scaling,
                                          gamma=GAMMA, device=str(device))

        # Collect stats
        def get_stats(result):
            strs = [s for s in result["str_trace"] if s is not None]
            return {
                "str_stats": stats(strs),
                "entropy_mean": round(sum(result["entropy_trace"]) /
                                      max(len(result["entropy_trace"]), 1), 4),
                "n_tokens": result["n_tokens"],
                "text_preview": result["text"][:200],
            }

        result = {
            "prompt_id": i,
            "base_kv_off": get_stats(base_off),
            "base_kv_on": get_stats(base_on),
            "dom_kv_off": get_stats(dom_off),
            "dom_kv_on": get_stats(dom_on),
        }
        if dom_off.get("alpha_trace"):
            result["dom_kv_off"]["alpha_stats"] = stats(dom_off["alpha_trace"])
        if dom_on.get("alpha_trace"):
            result["dom_kv_on"]["alpha_stats"] = stats(dom_on["alpha_trace"])

        all_results.append(result)

        # Print comparison table
        print(f"\n  {'Metric':<20} {'Base-OFF':>12} {'Base-ON':>12} "
              f"{'DOM-OFF':>12} {'DOM-ON':>12}")
        print(f"  {'─' * 68}")
        for cond, label in [("base_kv_off", "Base-OFF"), ("base_kv_on", "Base-ON"),
                             ("dom_kv_off", "DOM-OFF"), ("dom_kv_on", "DOM-ON")]:
            pass  # printed below
        print(f"  {'STR mean':<20} "
              f"{result['base_kv_off']['str_stats']['mean']:>12.4f} "
              f"{result['base_kv_on']['str_stats']['mean']:>12.4f} "
              f"{result['dom_kv_off']['str_stats']['mean']:>12.4f} "
              f"{result['dom_kv_on']['str_stats']['mean']:>12.4f}")
        print(f"  {'STR var':<20} "
              f"{result['base_kv_off']['str_stats']['var']:>12.6f} "
              f"{result['base_kv_on']['str_stats']['var']:>12.6f} "
              f"{result['dom_kv_off']['str_stats']['var']:>12.6f} "
              f"{result['dom_kv_on']['str_stats']['var']:>12.6f}")
        print(f"  {'Entropy':<20} "
              f"{result['base_kv_off']['entropy_mean']:>12.4f} "
              f"{result['base_kv_on']['entropy_mean']:>12.4f} "
              f"{result['dom_kv_off']['entropy_mean']:>12.4f} "
              f"{result['dom_kv_on']['entropy_mean']:>12.4f}")

        if device.type == "mps":
            torch.mps.empty_cache()

    # ── Summary Analysis ──
    print("\n" + "=" * 60)
    print("  ANALYSIS: KV-Cache Impact on DOM Control")
    print("=" * 60)

    # Compare DOM-OFF vs DOM-ON across prompts
    dom_off_vars = [r["dom_kv_off"]["str_stats"]["var"] for r in all_results]
    dom_on_vars = [r["dom_kv_on"]["str_stats"]["var"] for r in all_results]
    base_off_vars = [r["base_kv_off"]["str_stats"]["var"] for r in all_results]
    base_on_vars = [r["base_kv_on"]["str_stats"]["var"] for r in all_results]

    avg_dom_off_var = sum(dom_off_vars) / len(dom_off_vars)
    avg_dom_on_var = sum(dom_on_vars) / len(dom_on_vars)
    avg_base_off_var = sum(base_off_vars) / len(base_off_vars)
    avg_base_on_var = sum(base_on_vars) / len(base_on_vars)

    print(f"\n  Avg STR variance:")
    print(f"    Baseline KV-OFF: {avg_base_off_var:.6f}")
    print(f"    Baseline KV-ON:  {avg_base_on_var:.6f}")
    print(f"    DOM KV-OFF:      {avg_dom_off_var:.6f}")
    print(f"    DOM KV-ON:       {avg_dom_on_var:.6f}")

    # Key comparison: DOM effectiveness with/without KV-cache
    dom_off_reduction = (1.0 - avg_dom_off_var / max(avg_base_off_var, 1e-8)) * 100
    dom_on_reduction = (1.0 - avg_dom_on_var / max(avg_base_on_var, 1e-8)) * 100

    print(f"\n  DOM effectiveness (STR var reduction vs baseline):")
    print(f"    KV-OFF: {dom_off_reduction:+.1f}%")
    print(f"    KV-ON:  {dom_on_reduction:+.1f}%")

    diff = abs(avg_dom_on_var - avg_dom_off_var)
    rel_diff = diff / max(avg_dom_off_var, 1e-8)

    if rel_diff < 0.15:
        conclusion = "KV_NEUTRAL"
        print(f"\n  ✅ CONCLUSION: KV-cache has MINIMAL impact on DOM ({rel_diff:.1%} difference)")
        print(f"     → Paper can simplify: KV-cache setting is implementation detail")
    elif dom_on_reduction > 0 and dom_off_reduction > 0:
        conclusion = "KV_DEGRADES"
        print(f"\n  ⚠️  CONCLUSION: KV-cache DEGRADES DOM effectiveness")
        print(f"     KV-OFF: {dom_off_reduction:.1f}% reduction, KV-ON: {dom_on_reduction:.1f}%")
        print(f"     → Paper must discuss KV-cache OFF as technical requirement")
    else:
        conclusion = "KV_BREAKS"
        print(f"\n  ❌ CONCLUSION: KV-cache BREAKS DOM control")
        print(f"     → DOM requires full forward recomputation")

    # Save results
    output = {
        "experiment": "KV-Cache Ablation (Exp-KV-Minimal)",
        "model": "Llama-3.2-3B",
        "config": {
            "gamma": GAMMA, "kappa": PD_KAPPA, "beta": PD_BETA,
            "tau": PD_TAU, "sigma": sigma, "window_k": WINDOW_K,
        },
        "conditions": {
            "base_kv_off": "Baseline, use_cache=False",
            "base_kv_on": "Baseline, use_cache=True",
            "dom_kv_off": "DOM control, use_cache=False",
            "dom_kv_on": "DOM control, use_cache=True",
        },
        "prompts": all_results,
        "summary": {
            "avg_str_var": {
                "base_kv_off": round(avg_base_off_var, 6),
                "base_kv_on": round(avg_base_on_var, 6),
                "dom_kv_off": round(avg_dom_off_var, 6),
                "dom_kv_on": round(avg_dom_on_var, 6),
            },
            "dom_effectiveness_pct": {
                "kv_off": round(dom_off_reduction, 1),
                "kv_on": round(dom_on_reduction, 1),
            },
            "kv_impact_relative_diff": round(rel_diff, 4),
            "conclusion": conclusion,
        }
    }

    out_path = OUT_DIR / "kv_cache_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    print("=" * 60)


if __name__ == "__main__":
    main()
