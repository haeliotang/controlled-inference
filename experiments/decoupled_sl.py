"""
Experiment: Decoupled S+L Experiments

Is the S+L interference an architectural collision (same matrices) 
or a fundamental manifold conflict (incompatible mechanisms)?

Configs:
1. L: CE loss on q_proj, v_proj, latents.
2. S: CE + STR loss on q_proj, v_proj.
3. SL-Decoupled: v_proj & latents get CE. q_proj gets CE + STR.
4. SL-Crossed: q_proj & latents get CE. v_proj gets CE + STR.
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

from latent_tokens import LatentTokenInjector, compute_lm_loss_with_latent, find_question_end, extract_hidden_for_str
from train_phase1 import (
    LoRALinear, load_data, collate_fn, estimate_sigma,
    compute_str_loss_on_trajectories,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, MAX_LEN, BATCH_SIZE, EPOCHS, LR, OUT_DIR
)
from verify_eval_fairness import eval_without_latent

OUT_DECOUPLED = OUT_DIR / "decoupled_sl"
LAMBDA_STR = 0.1
LATENT_K = 8

CONFIGS = {
    "L":            {"use_str": False, "use_latent": True,  "desc": "L-only baseline"},
    "S":            {"use_str": True,  "use_latent": False, "desc": "S-only baseline"},
    "SL-Decoupled": {"use_str": True,  "use_latent": True,  "desc": "v=CE, q=CE+STR"},
    "SL-Crossed":   {"use_str": True,  "use_latent": True,  "desc": "q=CE, v=CE+STR"},
}

def apply_lora_named(model, rank=8, alpha=16.0):
    lora_params = []
    for name, module in list(model.named_modules()):
        for target in ("q_proj", "v_proj"):
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                lora_mod = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, name.split(".")[-1], lora_mod)
                lora_params.append((f"{name}.lora_A", lora_mod.lora_A))
                lora_params.append((f"{name}.lora_B", lora_mod.lora_B))
    for p in model.parameters():
        p.requires_grad = False
    for n, p in lora_params:
        p.requires_grad = True
    return lora_params

def run_config(cfg_name, cfg, tokenizer, train_loader, eval_loader, device):
    print(f"\n{'='*60}")
    print(f"  {cfg_name}: {cfg['desc']}")
    print(f"{'='*60}")

    outdir = OUT_DECOUPLED / cfg_name
    outdir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)

    named_params = apply_lora_named(model, LORA_RANK, LORA_ALPHA)
    
    injector = None
    if cfg["use_latent"]:
        hidden_dim = model.config.hidden_size
        injector = LatentTokenInjector(K=LATENT_K, hidden_dim=hidden_dim)
        injector.init_from_embeddings(model.get_input_embeddings())
        injector.to(device)
        for n, p in injector.named_parameters():
            if p.requires_grad:
                named_params.append((f"latent_{n}", p))

    trainable_params = [p for n, p in named_params]
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

    sigma = estimate_sigma(model, eval_loader, device)
    print(f"  Sigma: {sigma:.1f}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(50, total_steps//5), num_training_steps=total_steps)

    model.train()
    if injector: injector.train()
    
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        if cfg["use_latent"]:
            split_pos = find_question_end(ids, tokenizer)
            embeds, new_mask, lat_mask = injector(ids, mask, model.get_input_embeddings(), split_pos)
            outputs = model(inputs_embeds=embeds, attention_mask=new_mask, output_hidden_states=True)
            loss_ce = compute_lm_loss_with_latent(outputs.logits, ids, lat_mask)
        else:
            outputs = model(ids, attention_mask=mask, output_hidden_states=True, labels=ids)
            loss_ce = outputs.loss
            new_mask = mask
            lat_mask = torch.zeros_like(mask, dtype=torch.bool)

        loss_str_val = 0.0
        if cfg["use_str"]:
            hidden = outputs.hidden_states[-1]
            trajs = extract_hidden_for_str(hidden, new_mask, lat_mask, include_latent=cfg["use_latent"])
            if trajs:
                loss_str = compute_str_loss_on_trajectories(trajs, sigma)
                loss_str_val = loss_str.item()
            else:
                loss_str = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss_str = torch.tensor(0.0, device=device, requires_grad=True)

        optimizer.zero_grad()

        # Gradient Routing
        grad_ce = torch.autograd.grad(loss_ce, trainable_params, retain_graph=True, allow_unused=True)
        
        if cfg["use_str"] and loss_str_val != 0.0:
            grad_str = torch.autograd.grad(loss_str, trainable_params, allow_unused=True)
        else:
            grad_str = [None] * len(trainable_params)

        for (name, p), gc, gs in zip(named_params, grad_ce, grad_str):
            if gc is None: gc = torch.zeros_like(p)
            if gs is None: gs = torch.zeros_like(p)
            
            if cfg_name == "L":
                p.grad = gc
            elif cfg_name == "S":
                p.grad = gc + LAMBDA_STR * gs
            elif cfg_name == "SL-Decoupled":
                if "q_proj" in name:
                    p.grad = gc + LAMBDA_STR * gs
                else:
                    p.grad = gc
            elif cfg_name == "SL-Crossed":
                if "v_proj" in name:
                    p.grad = gc + LAMBDA_STR * gs
                else:
                    p.grad = gc

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if device.type == "mps" and (step + 1) % 10 == 0:
            torch.mps.empty_cache()

    dt = time.time() - t0
    print(f"  Training done: {dt:.1f}s")

    print("  Evaluating WITHOUT latent tokens (Fair Eval)...")
    metrics = eval_without_latent(model, eval_loader, device)
    print(f"  POST: entropy={metrics['entropy']:.4f} margin={metrics['margin']:.2f}")

    del model
    if injector: del injector
    if device.type == "mps": torch.mps.empty_cache()

    return metrics

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DECOUPLED.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(data[split:], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    results = {}
    for cfg_name in ["L", "S", "SL-Decoupled", "SL-Crossed"]:
        res = run_config(cfg_name, CONFIGS[cfg_name], tokenizer, train_loader, eval_loader, device)
        results[cfg_name] = res
        with open(OUT_DECOUPLED / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  DECOUPLED S+L RESULTS (Fair Eval - No Latent)")
    print(f"{'='*50}")
    print(f"  {'Config':<15} {'Entropy':>10}")
    for k, v in results.items():
        print(f"  {k:<15} {v['entropy']:>10.4f}")

if __name__ == "__main__":
    main()
