"""
TRC (Temporal Regime Composition) Training — Phase 2 Core Experiment.

Three protocols:
1. Static:   λ = constant
2. Piecewise: λ = linear ramp over sequence
3. Feedback:  λ_t = K * EMA(target - STR_t)  [core innovation]

Victory: Var(STR_t)_feedback < Oracle (0.005128), approaching 0.003937.
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
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, MAX_LEN, BATCH_SIZE, LR, OUT_DIR
)
from lambda_sweep import extract_hidden_real
from inference_str_logger import eval_trajectories, WINDOW_K, TAU_W

# === TRC Config ===
TARGET_STR = 0.6336      # from baseline (λ=0) inference trajectory
K_GAIN = 0.1             # proportional gain
BETA = 0.9               # EMA smoothing
LAMBDA_CLAMP = 0.3       # max |λ_t|
STATIC_LAMBDA = 0.05     # for static protocol
PIECEWISE_END = 0.1      # ramp: 0 → this value
K_PHASE = 0.08           # phase ramp bias for phase_feedback
NUM_EVAL_PROMPTS = 20
OUT_TRC = OUT_DIR / "trc_results"


def compute_windowed_str_seq(traj, sigma):
    """Per-position windowed STR with gradients. Returns (N,) tensor."""
    T = traj.shape[0]
    if T < WINDOW_K:
        return torch.zeros(0, device=traj.device)
    results = []
    dev = traj.device
    for t in range(WINDOW_K - 1, T):
        w = traj[t - WINDOW_K + 1: t + 1]            # (K, D)
        d_sq = _pairwise_dist_sq(w)                   # (K, K)
        K_vals = torch.exp(-d_sq / (2 * sigma ** 2))
        idx = torch.arange(w.shape[0], device=dev, dtype=torch.float32)
        dt = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        W = 1.0 - torch.exp(-dt / TAU_W)
        results.append((W * K_vals).sum() / W.sum())
    return torch.stack(results)


def compute_protocol_loss(str_vals, protocol, device):
    """Compute λ-weighted STR loss for a given protocol."""
    N = len(str_vals)
    if N < 2:
        return torch.tensor(0.0, device=device)

    if protocol == "static":
        lam = torch.full((N,), STATIC_LAMBDA, device=device)

    elif protocol == "piecewise":
        lam = torch.linspace(0.0, PIECEWISE_END, N, device=device)

    elif protocol == "feedback":
        err = TARGET_STR - str_vals.detach()
        e_sm = torch.zeros(N, device=device)
        e_sm[0] = err[0]
        for i in range(1, N):
            e_sm[i] = BETA * e_sm[i - 1] + (1 - BETA) * err[i]
        lam = (K_GAIN * e_sm).clamp(-LAMBDA_CLAMP, LAMBDA_CLAMP)

    elif protocol == "phase_feedback":
        # State feedback + global phase prior
        err = TARGET_STR - str_vals.detach()
        e_sm = torch.zeros(N, device=device)
        e_sm[0] = err[0]
        for i in range(1, N):
            e_sm[i] = BETA * e_sm[i - 1] + (1 - BETA) * err[i]
        phase = torch.linspace(0.0, 1.0, N, device=device)
        lam = (K_GAIN * e_sm + K_PHASE * phase).clamp(-LAMBDA_CLAMP, LAMBDA_CLAMP)

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    # Causal: λ_{t-1} → STR_t
    return (lam[:-1] * str_vals[1:]).mean()


def train_one_protocol(protocol, tokenizer, train_loader, sigma, device):
    """Train model under one TRC protocol, return trained model."""
    print(f"\n{'='*60}\n  Protocol: {protocol}\n{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)
    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)

    opt = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=20, num_training_steps=len(train_loader))

    model.train()
    total_ce, total_str, n_steps = 0.0, 0.0, 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        out = model(ids, attention_mask=mask, output_hidden_states=True, labels=ids)
        loss_ce = out.loss

        # Windowed STR per position
        traj = extract_hidden_real(out.hidden_states[-1], mask)
        str_vals = compute_windowed_str_seq(traj, sigma) if traj.shape[0] >= WINDOW_K else None

        if str_vals is not None and len(str_vals) >= 2:
            loss_str = compute_protocol_loss(str_vals, protocol, device)
            loss = loss_ce + loss_str
            total_str += loss_str.item()
        else:
            loss = loss_ce

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        sched.step()
        total_ce += loss_ce.item()
        n_steps += 1

        if device.type == "mps" and (step + 1) % 10 == 0:
            torch.mps.empty_cache()

    dt = time.time() - t0
    print(f"  Trained in {dt:.1f}s | CE={total_ce/n_steps:.4f} | STR_loss={total_str/n_steps:.4f}")
    return model, sigma


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

    # Eval prompts
    prompts = []
    for item in data[split:split + NUM_EVAL_PROMPTS]:
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        p = text.split("Answer:")[0] + "Answer:" if "Answer:" in text else text
        prompts.append(p)

    # Estimate sigma from untrained model
    tmp_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16).to(device)
    apply_lora(tmp_model, LORA_RANK, LORA_ALPHA)
    sigma = estimate_sigma(tmp_model, DataLoader(
        data[:20], batch_size=1, collate_fn=collate_fn), device)
    del tmp_model
    if device.type == "mps":
        torch.mps.empty_cache()
    print(f"Sigma: {sigma:.1f}")
    print(f"Target STR: {TARGET_STR}")

    # === Train & evaluate each protocol ===
    protocols = ["static", "piecewise", "feedback", "phase_feedback"]
    results = {}

    for proto in protocols:
        model, sig = train_one_protocol(proto, tokenizer, train_loader, sigma, device)

        print(f"  Logging inference trajectories...")
        res = eval_trajectories(model, tokenizer, prompts, sigma, device)
        s = res["summary"]
        print(f"    mean(STR_t)={s['mean_str']:.4f}  Var(STR_t)={s['var_str']:.6f}")
        print(f"    mean(Ent_t)={s['mean_entropy']:.4f}")

        results[proto] = s

        # Save trajectories
        with open(OUT_TRC / f"trajectories_{proto}.json", "w") as f:
            json.dump(res["trajectories"], f)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # === Comparison table ===
    print(f"\n{'='*60}")
    print(f"  TRC RESULTS vs ORACLE BASELINES")
    print(f"{'='*60}")
    print(f"  Oracle (λ=0):     Var={0.005128:.6f}  mean={0.6336:.4f}")
    print(f"  Unconstrained:    Var={0.003937:.6f}  mean={0.6122:.4f}")
    print(f"  Target band:      mean ∈ [{TARGET_STR-0.012:.4f}, {TARGET_STR+0.012:.4f}]")
    print(f"  {'─'*56}")
    for proto in protocols:
        s = results[proto]
        in_band = abs(s["mean_str"] - TARGET_STR) < 0.012
        beat_oracle = s["var_str"] < 0.005128
        flag = "✓" if (in_band and beat_oracle) else "✗"
        print(f"  {flag} {proto:12s}  Var={s['var_str']:.6f}  mean={s['mean_str']:.4f}"
              f"  {'IN BAND' if in_band else 'OUT'}")

    # Save
    final = {"protocols": results, "target_str": TARGET_STR,
             "oracle_var": 0.005128, "unconstrained_var": 0.003937}
    with open(OUT_TRC / "trc_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {OUT_TRC / 'trc_results.json'}")


if __name__ == "__main__":
    main()
