"""
Final Phase 2 Experiment: Gated Regime Switching (Non-linear Orchestration)
Runs the `gated_feedback` protocol and appends to trc_results.json.

Hypothesis: Control signals are competitive, not additive. 
We use a sigmoid gate to switch between two distinct regimes:
1. Exploratory Regime (λ_L) when STR is low (safe/coherent)
2. Coherent Regime (λ_S) when STR is high (diverging/hallucination risk)
"""
import json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from train_phase1 import (
    apply_lora, load_data, collate_fn, estimate_sigma,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, BATCH_SIZE, LR, OUT_DIR
)
from trc_train import (
    train_one_protocol, compute_windowed_str_seq, NUM_EVAL_PROMPTS, OUT_TRC, TARGET_STR
)
from inference_str_logger import eval_trajectories

# Gated Control Config
LAMBDA_S = 0.15    # Strong convergence regime
LAMBDA_L = -0.05   # Exploratory regime
GATE_GAIN = 50.0   # Steepness of the sigmoid (high = almost discrete switch)

def compute_gated_protocol_loss(str_vals, device):
    """Compute STR loss using non-linear sigmoid gating."""
    N = len(str_vals)
    if N < 2:
        return torch.tensor(0.0, device=device)
    
    # Detach state to prevent gradient flow through the controller itself
    state = str_vals.detach()
    
    # Sigmoid gate: approaches 1 when STR > TARGET_STR (needs convergence)
    # approaches 0 when STR < TARGET_STR (safe to explore)
    gate = torch.sigmoid(GATE_GAIN * (state - TARGET_STR))
    
    # Competitive Orchestration
    lam = gate * LAMBDA_S + (1.0 - gate) * LAMBDA_L
    
    # Causal time alignment: λ_{t-1} applies to STR_t
    return (lam[:-1] * str_vals[1:]).mean()

# Monkey-patch the loss function in trc_train for this specific run
import trc_train
original_compute_loss = trc_train.compute_protocol_loss

def patched_compute_loss(str_vals, protocol, device):
    if protocol == "gated_feedback":
        return compute_gated_protocol_loss(str_vals, device)
    return original_compute_loss(str_vals, protocol, device)

trc_train.compute_protocol_loss = patched_compute_loss


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_TRC.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)

    prompts = []
    for item in data[split:split + NUM_EVAL_PROMPTS]:
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        p = text.split("Answer:")[0] + "Answer:" if "Answer:" in text else text
        prompts.append(p)

    # Estimate sigma
    tmp = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16).to(device)
    apply_lora(tmp, LORA_RANK, LORA_ALPHA)
    sigma = estimate_sigma(tmp, DataLoader(
        data[:20], batch_size=1, collate_fn=collate_fn), device)
    del tmp
    if device.type == "mps": torch.mps.empty_cache()
    print(f"Sigma: {sigma:.1f}")

    # Train gated_feedback
    model, _ = train_one_protocol("gated_feedback", tokenizer, train_loader, sigma, device)

    print("  Logging inference trajectories...")
    res = eval_trajectories(model, tokenizer, prompts, sigma, device)
    s = res["summary"]
    print(f"    mean(STR_t)={s['mean_str']:.4f}  Var(STR_t)={s['var_str']:.6f}")
    print(f"    mean(Ent_t)={s['mean_entropy']:.4f}")

    # Save trajectories
    with open(OUT_TRC / "trajectories_gated_feedback.json", "w") as f:
        json.dump(res["trajectories"], f)

    # Update results
    results_file = OUT_TRC / "trc_results.json"
    if results_file.exists():
        with open(results_file) as f:
            final = json.load(f)
    else:
        final = {"protocols": {}, "target_str": TARGET_STR,
                 "oracle_var": 0.005128, "unconstrained_var": 0.003937}

    final["protocols"]["gated_feedback"] = s
    with open(results_file, "w") as f:
        json.dump(final, f, indent=2)

    # Final comparison block
    print(f"\n{'='*60}")
    print(f"  ULTIMATE REGIME ORCHESTRATION vs ALL")
    print(f"{'='*60}")
    print(f"  Oracle (λ=0):     Var=0.005128  mean=0.6336")
    print(f"  Unconstrained:    Var=0.003937  mean=0.6122")
    print(f"  Target band:      mean ∈ [{TARGET_STR-0.012:.4f}, {TARGET_STR+0.012:.4f}]")
    print(f"  {'─'*56}")
    
    for name, ps in final["protocols"].items():
        in_band = abs(ps["mean_str"] - TARGET_STR) < 0.012
        beat = ps["var_str"] < 0.005128
        flag = "✓" if (in_band and beat) else "✗"
        print(f"  {flag} {name:16s}  Var={ps['var_str']:.6f}  mean={ps['mean_str']:.4f}")

    del model
    if device.type == "mps": torch.mps.empty_cache()

if __name__ == "__main__":
    main()
