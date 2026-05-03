"""
Final Anchor Experiment: Regime Recovery under Perturbation
Tests if the system exhibits true closed-loop dynamical response.

Step 1: Train Oracle (λ=0), Piecewise, and Gated Feedback.
Step 2: Generate with transient perturbation (noise added to token embedding at t=15).
Step 3: Analyze recovery dynamics.
"""
import json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from train_phase1 import (
    apply_lora, load_data, collate_fn, estimate_sigma,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, BATCH_SIZE, LR
)
from trc_train import train_one_protocol, NUM_EVAL_PROMPTS, OUT_TRC, TARGET_STR
from inference_str_logger import compute_window_str, TEMPERATURE, WINDOW_K

# Config
PERTURB_STEP = 15
EPSILON = 2.0  # Noise multiplier
MAX_NEW_TOKENS = 50

# Monkey-patch gated loss again for training
import trc_gated
import trc_train
original_compute_loss = trc_train.compute_protocol_loss
def patched_compute_loss(str_vals, protocol, device):
    if protocol == "oracle":
        return torch.tensor(0.0, device=device, requires_grad=True)
    if protocol == "gated_feedback":
        return trc_gated.compute_gated_protocol_loss(str_vals, device)
    return original_compute_loss(str_vals, protocol, device)
trc_train.compute_protocol_loss = patched_compute_loss


@torch.no_grad()
def generate_perturbed(model, prompt_ids, sigma, device):
    """Generate with transient noise injected at PERTURB_STEP."""
    h_buffer = []
    str_traj = []

    # Initial forward
    out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    past_kv = out.past_key_values
    h_t = out.hidden_states[-1][:, -1, :].float().squeeze(0)
    h_buffer.append(h_t)
    logits = out.logits[:, -1, :].float()

    embed_layer = model.get_input_embeddings()

    for step in range(MAX_NEW_TOKENS):
        next_tok = torch.multinomial(F.softmax(logits / TEMPERATURE, dim=-1), 1)
        
        # Inject transient noise at specific step
        inputs_embeds = embed_layer(next_tok)
        if step == PERTURB_STEP:
            noise = torch.randn_like(inputs_embeds) * EPSILON
            inputs_embeds = inputs_embeds + noise
        
        out = model(inputs_embeds=inputs_embeds, past_key_values=past_kv,
                    output_hidden_states=True, use_cache=True)
        past_kv = out.past_key_values
        h_t = out.hidden_states[-1][:, -1, :].float().squeeze(0)

        h_buffer.append(h_t)
        if len(h_buffer) > WINDOW_K:
            h_buffer.pop(0)

        s = compute_window_str(h_buffer, sigma, device)
        if s is not None:
            str_traj.append(s)

        logits = out.logits[:, -1, :].float()

    return str_traj

@torch.no_grad()
def eval_perturbed(model, tokenizer, prompts, sigma, device):
    model.eval()
    all_trajs = []
    for i, p in enumerate(prompts):
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(device)
        traj = generate_perturbed(model, enc["input_ids"], sigma, device)
        all_trajs.append(traj)
    return all_trajs

def analyze_recovery(trajectories, baseline_mean):
    """Calculate average trajectory and recovery metrics."""
    # Average across prompts
    n_steps = len(trajectories[0])
    avg_traj = [sum(t[i] for t in trajectories)/len(trajectories) for i in range(n_steps)]
    
    # Pre-perturbation mean
    pre_mean = sum(avg_traj[:PERTURB_STEP]) / PERTURB_STEP
    
    # Overshoot
    max_after = max(avg_traj[PERTURB_STEP:PERTURB_STEP+10])
    overshoot = max_after - baseline_mean
    
    # Recovery time (steps to get back within 0.01 of baseline mean)
    recovery_time = -1
    for i in range(PERTURB_STEP+1, n_steps):
        if abs(avg_traj[i] - baseline_mean) < 0.01:
            recovery_time = i - PERTURB_STEP
            break
            
    # Recovery stability (variance of steps 30-50)
    rec_vals = avg_traj[30:]
    rec_mean = sum(rec_vals)/len(rec_vals)
    rec_var = sum((x - rec_mean)**2 for x in rec_vals) / len(rec_vals)
    
    return avg_traj, overshoot, recovery_time, rec_var


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    prompts = []
    for item in data[split:split + NUM_EVAL_PROMPTS]:
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        p = text.split("Answer:")[0] + "Answer:" if "Answer:" in text else text
        prompts.append(p)

    tmp = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16).to(device)
    apply_lora(tmp, LORA_RANK, LORA_ALPHA)
    sigma = estimate_sigma(tmp, DataLoader(data[:20], batch_size=1, collate_fn=collate_fn), device)
    del tmp
    if device.type == "mps": torch.mps.empty_cache()

    protocols = ["oracle", "piecewise", "gated_feedback"]
    results = {}
    
    for proto in protocols:
        print(f"\n--- Testing {proto} ---")
        model, _ = train_one_protocol(proto, tokenizer, train_loader, sigma, device)
            
        print(f"  Evaluating perturbation on {proto}...")
        trajs = eval_perturbed(model, tokenizer, prompts, sigma, device)
        avg_traj, os, rt, rv = analyze_recovery(trajs, TARGET_STR)
        
        results[proto] = {
            "avg_traj": avg_traj,
            "overshoot": os,
            "recovery_time": rt,
            "recovery_var": rv
        }
        
        del model
        if device.type == "mps": torch.mps.empty_cache()

    # Save results
    OUT_TRC.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRC / "perturbation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  PERTURBATION RECOVERY RESULTS (t=15, eps={EPSILON})")
    print(f"{'='*60}")
    print(f"  {'Protocol':<15} | {'Overshoot':>10} | {'Rec. Time':>9} | {'Rec. Var':>10}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}")
    for p in protocols:
        r = results[p]
        rt_str = str(r['recovery_time']) if r['recovery_time'] != -1 else "FAILED"
        print(f"  {p:<15} | {r['overshoot']:10.4f} | {rt_str:>9} | {r['recovery_var']:10.6f}")

if __name__ == "__main__":
    main()
