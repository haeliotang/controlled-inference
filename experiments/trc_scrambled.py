"""
Scrambled Gradient Baseline Experiment

Goal: Prove that STR's control effect comes from gradient *direction*,
not gradient *magnitude*.

Protocol:
  For a set of λ values, compare:
    1. Normal STR training — gradients preserve direction
    2. Scrambled STR training — gradient norms preserved, but directions
       are randomly permuted across dimensions

  If both produce the same effect → STR is just noise/regularization.
  If only normal works → STR's effect is directional (causal control field).

Output: results_phase1/scrambled/scrambled_results.json
"""

import json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from str_loss import _pairwise_dist_sq
from train_phase1 import (
    apply_lora, load_data, collate_fn, estimate_sigma,
    compute_str_on_trajectories, compute_str_loss_on_trajectories,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, TAU, MAX_LEN, BATCH_SIZE, EPOCHS, LR, OUT_DIR
)

SCRAMBLED_DIR = OUT_DIR / "scrambled"
LAMBDAS = [-0.1, -0.05, 0.0, 0.05, 0.1]
N_EVAL_PROMPTS = 20


def extract_hidden_real(hidden, mask):
    """Extract valid positions based on mask from hidden state."""
    h = hidden[0]
    m = mask[0].bool()
    return h[m]


def scramble_str_gradients(lora_params):
    """Scramble gradient directions while preserving per-parameter gradient norms.

    For each LoRA parameter that has a gradient, we:
      1. Record the original gradient norm
      2. Replace the gradient with a random tensor of the same shape
      3. Rescale the random tensor to have the original norm

    This destroys directional information while preserving gradient energy.
    """
    for p in lora_params:
        if p.grad is not None:
            original_norm = p.grad.norm()
            if original_norm > 0:
                random_dir = torch.randn_like(p.grad)
                random_norm = random_dir.norm()
                if random_norm > 0:
                    p.grad.copy_(random_dir * (original_norm / random_norm))


@torch.no_grad()
def evaluate_str_distribution(model, tokenizer, dataloader, sigma, device):
    """Collect per-step STR values across prompts for distribution analysis."""
    model.eval()
    all_str_values = []
    count = 0

    for batch in dataloader:
        if count >= N_EVAL_PROMPTS:
            break
        ids = batch["input_ids"][:1].to(device)
        mask = batch["attention_mask"][:1].to(device)

        outputs = model(ids, attention_mask=mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        traj = extract_hidden_real(hidden, mask)
        if traj.shape[0] >= 2:
            str_val = compute_str_on_trajectories([traj], sigma).item()
            all_str_values.append(str_val)
        count += 1

    import numpy as np
    arr = np.array(all_str_values)
    return {
        "mean_str": float(arr.mean()),
        "var_str": float(arr.var()),
        "std_str": float(arr.std()),
        "n_samples": len(all_str_values),
        "values": all_str_values,
    }


def train_and_eval(lam, scramble, tokenizer, train_loader, eval_loader, device):
    """Train with given λ (normal or scrambled) and evaluate STR distribution."""
    tag = f"λ={lam}, {'scrambled' if scramble else 'normal'}"
    print(f"\n{'='*50}\n  {tag}\n{'='*50}")

    model = __import__("transformers").AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16
    )
    model.to(device)

    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)
    sigma = estimate_sigma(model, eval_loader, device)

    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=len(train_loader)
    )

    model.train()
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        outputs = model(ids, attention_mask=mask, output_hidden_states=True, labels=ids)
        loss_ce = outputs.loss

        if lam != 0.0:
            hidden = outputs.hidden_states[-1]
            traj = extract_hidden_real(hidden, mask)
            if traj.shape[0] >= 2:
                loss_str = compute_str_loss_on_trajectories([traj], sigma)
                loss = loss_ce + lam * loss_str
            else:
                loss = loss_ce
        else:
            loss = loss_ce

        optimizer.zero_grad()
        loss.backward()

        # KEY: scramble gradients before optimizer step
        if scramble and lam != 0.0:
            scramble_str_gradients(lora_params)

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()

        if device.type == "mps" and (step + 1) % 10 == 0:
            torch.mps.empty_cache()

    dt = time.time() - t0
    print(f"  Trained 1 epoch in {dt:.1f}s")

    # Evaluate
    print("  Evaluating STR distribution...")
    result = evaluate_str_distribution(model, tokenizer, eval_loader, sigma, device)
    print(f"    mean_str={result['mean_str']:.4f}, var_str={result['var_str']:.6f}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    SCRAMBLED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        LLAMA_PATH, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(data[split:], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    results = {"normal": {}, "scrambled": {}}

    for lam in LAMBDAS:
        # Normal
        res_n = train_and_eval(lam, scramble=False, tokenizer=tokenizer,
                               train_loader=train_loader, eval_loader=eval_loader,
                               device=device)
        results["normal"][str(lam)] = res_n

        # Scrambled (skip λ=0 since scrambling has no effect there)
        if lam != 0.0:
            res_s = train_and_eval(lam, scramble=True, tokenizer=tokenizer,
                                   train_loader=train_loader, eval_loader=eval_loader,
                                   device=device)
            results["scrambled"][str(lam)] = res_s
        else:
            results["scrambled"][str(lam)] = res_n  # identical at λ=0

        # Incremental save
        with open(SCRAMBLED_DIR / "scrambled_results_temp.json", "w") as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(SCRAMBLED_DIR / "scrambled_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("  Scrambled Baseline Summary")
    print(f"{'='*60}")
    print(f"{'λ':>8} | {'Normal mean':>12} | {'Scrambled mean':>14} | {'Δmean':>8}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}")

    baseline_mean = results["normal"]["0.0"]["mean_str"]
    for lam in LAMBDAS:
        n_mean = results["normal"][str(lam)]["mean_str"]
        s_mean = results["scrambled"][str(lam)]["mean_str"]
        delta_n = n_mean - baseline_mean
        delta_s = s_mean - baseline_mean
        print(f"{lam:>8} | {n_mean:>12.4f} | {s_mean:>14.4f} | N:{delta_n:>+.4f} S:{delta_s:>+.4f}")

    print(f"\nSaved: {SCRAMBLED_DIR / 'scrambled_results.json'}")


if __name__ == "__main__":
    main()
