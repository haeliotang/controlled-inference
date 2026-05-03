"""
Experiment: Lambda Sweep Phase Diagram

Goal: Prove that STR acts as a "Regime Control Field".
We sweep lambda over [-0.2, 0.2]. For each lambda:
1. Train S-only LoRA for 1 epoch.
2. Evaluate metrics to characterize the "Phase / Regime":
   - Trajectory Coherence (STR value)
   - Local Alignment (Cosine Coherence: cos(h_t, h_{t-1}))
   - Uncertainty (Logit Entropy)
   - Sampling Dispersion (Unique Ratio at T=0.7, N=5)
   
Look for non-linear bifurcations or clusters indicating Phase Transitions.
"""

import json, sys, time, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from str_loss import _pairwise_dist_sq
from train_phase1 import (
    apply_lora, load_data, collate_fn, estimate_sigma,
    compute_str_on_trajectories, compute_str_loss_on_trajectories,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, TAU, MAX_LEN, BATCH_SIZE, EPOCHS, LR, OUT_DIR
)

SWEEP_DIR = OUT_DIR / "lambda_sweep"
LAMBDAS = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]
MAX_EVAL_GEN = 20 # Only generate for 20 samples to save time
NUM_SAMPLES = 5

def extract_hidden_real(hidden, mask):
    """Extract valid positions based on mask from hidden state."""
    h = hidden[0]
    m = mask[0].bool()
    return h[m]

def compute_cosine_coherence(hidden_traj):
    """Compute mean cosine similarity between adjacent hidden states."""
    if hidden_traj.shape[0] < 2:
        return 0.0
    h1 = hidden_traj[:-1]
    h2 = hidden_traj[1:]
    cos_sim = F.cosine_similarity(h1, h2, dim=-1)
    return cos_sim.mean().item()

@torch.no_grad()
def evaluate_regime(model, tokenizer, dataloader, sigma, device):
    model.eval()
    entropies, margins, str_vals, coherences = [], [], [], []
    count = 0
    
    # 1. Deterministic metrics (Entropy, Margin, STR, Cosine Coherence)
    for batch in dataloader:
        if count >= 30: # Limit eval set to 30 for speed
            break
        ids = batch["input_ids"][:1].to(device)
        mask = batch["attention_mask"][:1].to(device)
        
        outputs = model(ids, attention_mask=mask, output_hidden_states=True)
        logits = outputs.logits[0].float()
        hidden = outputs.hidden_states[-1]
        
        seq_len = int(mask.sum().item())
        real_logits = logits[:seq_len - 1]
        
        if real_logits.shape[0] < 2:
            count += 1
            continue
            
        probs = F.softmax(real_logits, dim=-1)
        log_probs = F.log_softmax(real_logits, dim=-1)
        ent = -(probs * log_probs).sum(dim=-1).mean().item()
        top2 = real_logits.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()
        
        entropies.append(ent)
        margins.append(margin)
        
        traj = extract_hidden_real(hidden, mask)
        if traj.shape[0] >= 2:
            str_val = compute_str_on_trajectories([traj], sigma).item()
            str_vals.append(str_val)
            coherences.append(compute_cosine_coherence(traj))
            
        count += 1
        
    # 2. Diversity metrics (Generation: unique ratio)
    unique_ratios = []
    for i, batch in enumerate(dataloader):
        if i >= MAX_EVAL_GEN:
            break
            
        ids = batch["input_ids"][:1].to(device)
        
        # Get the prompt text
        full_text = tokenizer.decode(ids[0], skip_special_tokens=True)
        if "Answer:" in full_text:
            prompt_text = full_text.split("Answer:")[0] + "Answer:"
        else:
            prompt_text = full_text
            
        enc = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_ids = enc["input_ids"]
        
        gen_texts = []
        for _ in range(NUM_SAMPLES):
            out = model.generate(
                prompt_ids, max_new_tokens=15,
                temperature=0.7, do_sample=True, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            new_toks = out[0][prompt_ids.shape[1]:]
            text = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            gen_texts.append(text)
            
        unique_texts = len(set(gen_texts))
        unique_ratios.append(unique_texts / NUM_SAMPLES)
        
    def mean(lst): return sum(lst)/max(len(lst), 1)
    
    return {
        "entropy": round(mean(entropies), 4),
        "margin": round(mean(margins), 4),
        "str": round(mean(str_vals), 4),
        "coherence": round(mean(coherences), 4),
        "unique_ratio": round(mean(unique_ratios), 4),
    }

def run_lambda(lam, tokenizer, train_loader, eval_loader, device):
    print(f"\n{'='*50}\n  Running λ = {lam}\n{'='*50}")
    
    model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)
    
    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)
    
    # Estimate sigma (use 4 samples)
    sigma = estimate_sigma(model, eval_loader, device)
    
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=len(train_loader))
    
    model.train()
    
    total_loss = 0
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
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if device.type == "mps" and (step + 1) % 10 == 0:
            torch.mps.empty_cache()
            
    dt = time.time() - t0
    print(f"  Trained 1 epoch in {dt:.1f}s")
    
    # Eval
    print("  Evaluating regime metrics...")
    metrics = evaluate_regime(model, tokenizer, eval_loader, sigma, device)
    print(f"    STR: {metrics['str']:.4f} | Coherence: {metrics['coherence']:.4f}")
    print(f"    Ent: {metrics['entropy']:.4f} | Unique: {metrics['unique_ratio']:.3f}")
    
    del model
    if device.type == "mps": torch.mps.empty_cache()
        
    return metrics

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(data[split:], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    results = {}
    for lam in LAMBDAS:
        res = run_lambda(lam, tokenizer, train_loader, eval_loader, device)
        results[str(lam)] = res
        
        with open(SWEEP_DIR / "sweep_results_temp.json", "w") as f:
            json.dump(results, f, indent=2)
            
    with open(SWEEP_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to {SWEEP_DIR / 'sweep_results.json'}")

if __name__ == "__main__":
    main()
