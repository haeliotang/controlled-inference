"""
Experiment: Why is S+L sub-additive?

Two quick experiments (SL config only):
  1. SL-weak:     λ=0.01 (10x weaker STR)
  2. SL-indirect: include_latent=False (STR only on real tokens)

Baseline comparison from previous run:
  baseline POST entropy = 0.8995
  L alone  POST entropy = 0.8568  ← target to beat
"""

import json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from latent_tokens import (
    LatentTokenInjector, compute_lm_loss_with_latent,
    extract_hidden_for_str, find_question_end,
)
from str_loss import _pairwise_dist_sq, _temporal_weight
from train_phase1 import (
    LoRALinear, apply_lora, load_data, collate_fn,
    compute_str_on_trajectories, compute_str_loss_on_trajectories,
    estimate_sigma, evaluate,
    LLAMA_PATH, LATENT_K, LORA_RANK, LORA_ALPHA, TAU, MAX_LEN,
    BATCH_SIZE, EPOCHS, LR, OUT_DIR,
)

DIAG_CONFIGS = {
    "SL-weak": {
        "use_str": True, "use_latent": True,
        "lambda_str": 0.01,          # 10x weaker
        "include_latent_in_str": True,
        "desc": "S+L, λ=0.01",
    },
    "SL-indirect": {
        "use_str": True, "use_latent": True,
        "lambda_str": 0.1,
        "include_latent_in_str": False,  # STR only on real tokens
        "desc": "S+L, STR on real only",
    },
}


def run_diag(cfg_name, cfg, tokenizer, train_loader, eval_loader, device):
    print(f"\n{'='*60}")
    print(f"  {cfg_name}: {cfg['desc']}")
    print(f"  λ={cfg['lambda_str']}, include_latent={cfg['include_latent_in_str']}")
    print(f"{'='*60}")

    outdir = OUT_DIR / cfg_name
    outdir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)

    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)
    trainable_params = list(lora_params)

    hidden_dim = model.config.hidden_size
    injector = LatentTokenInjector(K=LATENT_K, hidden_dim=hidden_dim)
    injector.init_from_embeddings(model.get_input_embeddings())
    injector.to(device)
    trainable_params.extend(list(injector.parameters()))

    print(f"  Trainable: {sum(p.numel() for p in trainable_params):,}")

    sigma = estimate_sigma(model, eval_loader, device)
    print(f"  sigma = {sigma:.1f}")

    # Use the same eval cfg format as train_phase1
    eval_cfg = {"use_latent": True, "use_str": True}
    pre = evaluate(model, eval_loader, injector, tokenizer, sigma, device, eval_cfg)
    print(f"  PRE: entropy={pre['entropy']:.4f}")

    # Train
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(50, total_steps // 5),
        num_training_steps=total_steps)

    model.train()
    injector.train()
    lam = cfg["lambda_str"]
    include_lat = cfg["include_latent_in_str"]
    total_loss, step_count = 0, 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        split_pos = find_question_end(ids, tokenizer)
        embeds, new_mask, lat_mask = injector(ids, mask,
            model.get_input_embeddings(), split_pos)
        outputs = model(inputs_embeds=embeds, attention_mask=new_mask,
                       output_hidden_states=True)
        loss_ce = compute_lm_loss_with_latent(outputs.logits, ids, lat_mask)

        hidden = outputs.hidden_states[-1]
        trajs = extract_hidden_for_str(hidden, new_mask, lat_mask,
                                        include_latent=include_lat)
        if trajs:
            loss_str = compute_str_loss_on_trajectories(trajs, sigma)
            loss_total = loss_ce + lam * loss_str
        else:
            loss_total = loss_ce

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_total.item()
        step_count += 1
        if (step + 1) % 50 == 0:
            print(f"    step {step+1}: loss={total_loss/step_count:.4f}")
        if device.type == "mps" and (step + 1) % 5 == 0:
            torch.mps.empty_cache()

    dt = time.time() - t0
    print(f"  Done: {dt:.0f}s")

    post = evaluate(model, eval_loader, injector, tokenizer, sigma, device, eval_cfg)
    print(f"  POST: entropy={post['entropy']:.4f} margin={post['margin']:.2f}")

    result = {
        "config": cfg_name, "desc": cfg["desc"],
        "lambda": cfg["lambda_str"],
        "include_latent_in_str": cfg["include_latent_in_str"],
        "pre": pre, "post": post,
        "delta_entropy": round(post["entropy"] - pre["entropy"], 4),
    }
    with open(outdir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    del model, injector
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(data[split:], batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(data[:split])} train, {len(data[split:])} eval")

    results = {}
    for name in ["SL-weak", "SL-indirect"]:
        results[name] = run_diag(name, DIAG_CONFIGS[name], tokenizer,
                                  train_loader, eval_loader, device)

    # Compare with previous results
    prev = OUT_DIR / "summary.json"
    if prev.exists():
        with open(prev) as f:
            prev_data = json.load(f)
    else:
        prev_data = {}

    print(f"\n{'='*60}")
    print("  DIAGNOSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Previous results:")
    for k in ["baseline", "S", "L", "SL"]:
        if k in prev_data:
            print(f"    {k:12s}  POST entropy = {prev_data[k]['post']['entropy']:.4f}")
    print(f"\n  Diagnostic results:")
    for k, r in results.items():
        print(f"    {k:12s}  POST entropy = {r['post']['entropy']:.4f}")
    print()

    L_ent = prev_data.get("L", {}).get("post", {}).get("entropy", 999)
    for k, r in results.items():
        if r["post"]["entropy"] < L_ent:
            print(f"  🔥 {k} beats L-alone ({r['post']['entropy']:.4f} < {L_ent:.4f})")
            print(f"     → This is the correct S+L interaction mode!")
        else:
            print(f"  ❌ {k}: {r['post']['entropy']:.4f} ≥ L-alone {L_ent:.4f}")

    print(f"{'='*60}")

    with open(OUT_DIR / "diagnosis.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
