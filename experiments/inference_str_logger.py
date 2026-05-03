"""
Inference STR Trajectory Logger — Step 1 of TRC pipeline.

For each static λ model (from lambda sweep), generates text autoregressively
and records per-step windowed STR_t and entropy_t trajectories.

Outputs:
- Per-λ: full STR_t/entropy_t trajectories, mean, var
- Oracle λ* computation (constrained: closest mean to target)
- Foundation data for perturbation test (Step 3)
"""

import json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from str_loss import _pairwise_dist_sq
from train_phase1 import (
    apply_lora, load_data, collate_fn, estimate_sigma,
    compute_str_loss_on_trajectories,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, MAX_LEN, BATCH_SIZE, LR, OUT_DIR
)
from lambda_sweep import extract_hidden_real

# === Config ===
LAMBDAS = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]
WINDOW_K = 4          # sliding window for local trajectory geometry
TAU_W = 3.0           # temporal decay for window STR
MAX_NEW_TOKENS = 50   # tokens to generate per prompt
TEMPERATURE = 0.7
NUM_PROMPTS = 20      # eval prompts for trajectory logging
OUT_TRC = OUT_DIR / "trc_baselines"


# === Core: windowed STR at step t ===
def compute_window_str(h_buffer, sigma, device):
    """Compute STR over a sliding window of hidden states."""
    if len(h_buffer) < 2:
        return None
    window = torch.stack(h_buffer).float()        # (W, D)
    d_sq = _pairwise_dist_sq(window)              # (W, W)
    K_vals = torch.exp(-d_sq / (2 * sigma ** 2))  # Gaussian kernel
    W_size = window.shape[0]
    idx = torch.arange(W_size, device=device, dtype=torch.float32)
    dt = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    W_mat = 1.0 - torch.exp(-dt / TAU_W)         # temporal weight
    return ((W_mat * K_vals).sum() / W_mat.sum()).item()


# === Custom autoregressive generation with trajectory tracking ===
@torch.no_grad()
def generate_with_trajectory(model, prompt_ids, sigma, device):
    """
    Generate tokens one-by-one using KV cache.
    At each step, record windowed STR_t and logit entropy_t.
    """
    h_buffer = []
    str_traj, ent_traj = [], []

    # Initial forward: process entire prompt
    out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    past_kv = out.past_key_values
    h_t = out.hidden_states[-1][:, -1, :].float().squeeze(0)
    h_buffer.append(h_t)
    logits = out.logits[:, -1, :].float()

    generated = []
    for _ in range(MAX_NEW_TOKENS):
        # Record entropy
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * F.log_softmax(logits, dim=-1)).sum(-1).item()
        ent_traj.append(ent)

        # Sample
        next_tok = torch.multinomial(F.softmax(logits / TEMPERATURE, dim=-1), 1)
        generated.append(next_tok.item())

        # Forward with KV cache (only the new token)
        out = model(next_tok, past_key_values=past_kv,
                    output_hidden_states=True, use_cache=True)
        past_kv = out.past_key_values
        h_t = out.hidden_states[-1][:, -1, :].float().squeeze(0)

        # Rolling window
        h_buffer.append(h_t)
        if len(h_buffer) > WINDOW_K:
            h_buffer.pop(0)

        # Windowed STR
        s = compute_window_str(h_buffer, sigma, device)
        if s is not None:
            str_traj.append(s)

        logits = out.logits[:, -1, :].float()

    return generated, {"str": str_traj, "entropy": ent_traj}


# === Evaluate one λ model ===
@torch.no_grad()
def eval_trajectories(model, tokenizer, prompts, sigma, device):
    """Run trajectory logging on all prompts and compute summary stats."""
    model.eval()
    all_str, all_ent = [], []
    trajectories = []

    for i, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_LEN).to(device)
        _, traj = generate_with_trajectory(model, enc["input_ids"], sigma, device)
        trajectories.append(traj)
        all_str.extend(traj["str"])
        all_ent.extend(traj["entropy"])
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(prompts)} prompts done")

    def stats(vals):
        if not vals:
            return 0.0, 0.0
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        return round(m, 6), round(v, 6)

    m_str, v_str = stats(all_str)
    m_ent, v_ent = stats(all_ent)

    return {
        "trajectories": trajectories,
        "summary": {
            "mean_str": m_str, "var_str": v_str,
            "mean_entropy": m_ent, "var_entropy": v_ent,
            "n_prompts": len(prompts), "n_str_samples": len(all_str),
        }
    }


# === Train one λ model (same recipe as lambda_sweep) ===
def train_lambda_model(lam, model, lora_params, train_loader, sigma, device):
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=len(train_loader))
    model.train()
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, attention_mask=mask, output_hidden_states=True, labels=ids)
        loss = out.loss
        if lam != 0.0:
            h = out.hidden_states[-1]
            traj = extract_hidden_real(h, mask)
            if traj.shape[0] >= 2:
                loss = loss + lam * compute_str_loss_on_trajectories([traj], sigma)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()
        if device.type == "mps" and (step + 1) % 10 == 0:
            torch.mps.empty_cache()
    print(f"  Trained 1 epoch in {time.time()-t0:.1f}s")


# === Full pipeline for one λ ===
def run_one_lambda(lam, tokenizer, train_loader, prompts, device):
    print(f"\n{'='*60}\n  λ = {lam}\n{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)
    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)
    sigma = estimate_sigma(model, DataLoader(
        load_data(tokenizer)[:20], batch_size=1, collate_fn=collate_fn), device)
    print(f"  Sigma: {sigma:.1f}")

    train_lambda_model(lam, model, lora_params, train_loader, sigma, device)

    print("  Logging inference trajectories...")
    result = eval_trajectories(model, tokenizer, prompts, sigma, device)
    s = result["summary"]
    print(f"    mean(STR_t)={s['mean_str']:.4f}  Var(STR_t)={s['var_str']:.6f}")
    print(f"    mean(Ent_t)={s['mean_entropy']:.4f}  Var(Ent_t)={s['var_entropy']:.6f}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


# === Oracle λ* computation ===
def compute_oracle(summaries, target_str, epsilon=0.01):
    """Find λ with lowest Var(STR_t) subject to |mean - target| < ε."""
    candidates = []
    for lam_str, s in summaries.items():
        if abs(s["mean_str"] - target_str) < epsilon:
            candidates.append((float(lam_str), s["var_str"], s["mean_str"]))
    if not candidates:
        # Relax: pick closest mean
        candidates = [(float(k), v["var_str"], v["mean_str"]) for k, v in summaries.items()]
        candidates.sort(key=lambda x: abs(x[2] - target_str))
        candidates = candidates[:3]
    candidates.sort(key=lambda x: x[1])  # sort by var
    return {"lambda_star": candidates[0][0],
            "var_star": candidates[0][1],
            "mean_star": candidates[0][2],
            "target": target_str,
            "all_candidates": candidates}


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

    # Prepare eval prompts (question part only)
    prompts = []
    for item in data[split:split + NUM_PROMPTS]:
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        p = text.split("Answer:")[0] + "Answer:" if "Answer:" in text else text
        prompts.append(p)
    print(f"Prepared {len(prompts)} eval prompts")

    # Run all λ values
    summaries = {}
    all_results = {}
    for lam in LAMBDAS:
        res = run_one_lambda(lam, tokenizer, train_loader, prompts, device)
        lam_key = str(lam)
        summaries[lam_key] = res["summary"]
        # Save trajectories separately (large)
        traj_file = OUT_TRC / f"trajectories_lambda_{lam_key}.json"
        with open(traj_file, "w") as f:
            json.dump(res["trajectories"], f)
        # Save running summary
        with open(OUT_TRC / "summaries.json", "w") as f:
            json.dump(summaries, f, indent=2)

    # Compute target_STR from baseline (λ=0)
    target_str = summaries["0.0"]["mean_str"]
    print(f"\n{'='*60}")
    print(f"  TARGET STR (from baseline λ=0): {target_str:.6f}")
    print(f"{'='*60}")

    # Oracle λ* (constrained)
    oracle = compute_oracle(summaries, target_str)
    print(f"  Oracle λ* = {oracle['lambda_star']}")
    print(f"  Oracle Var(STR_t) = {oracle['var_star']:.6f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  {'λ':>6} | {'mean(STR)':>10} | {'Var(STR)':>10} | {'mean(Ent)':>10} | {'Var(Ent)':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for lam in LAMBDAS:
        s = summaries[str(lam)]
        print(f"  {lam:>6} | {s['mean_str']:>10.4f} | {s['var_str']:>10.6f} | "
              f"{s['mean_entropy']:>10.4f} | {s['var_entropy']:>10.6f}")

    # Save final results
    final = {"summaries": summaries, "target_str": target_str, "oracle": oracle}
    with open(OUT_TRC / "trc_baselines.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {OUT_TRC / 'trc_baselines.json'}")


if __name__ == "__main__":
    main()
