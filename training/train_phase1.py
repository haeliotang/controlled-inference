"""
Experiment: S+L Joint Training — Llama-3.2-3B

4-config experiment matrix testing super-additive interaction:
  Baseline (no S, no L) / S only / L only / S+L

Core metric: effect(S+L) - [effect(S) + effect(L)] > 0?

Usage:
    cd code/training
    TRANSFORMERS_OFFLINE=1 python3 train_phase1.py
"""

import json, math, sys, time
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
from str_loss import str_loss, _pairwise_dist_sq, _temporal_weight

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
LLAMA_PATH = str(ROOT / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-3B"
                        "/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062")
TRUTHFULQA_JSON = Path(__file__).parent / "data" / "truthfulqa_gpt2.json"
OUT_DIR = Path(__file__).parent / "results_phase1"

# ── Hyperparameters ────────────────────────────────────────────────────
LATENT_K     = 8       # number of latent tokens
LORA_RANK    = 8
LORA_ALPHA   = 16
LAMBDA_STR   = 0.1
TAU          = 10.0
MAX_LEN      = 96
BATCH_SIZE   = 1
EPOCHS       = 1
LR           = 2e-4
MAX_STR_TOK  = 24

# ── Configs ────────────────────────────────────────────────────────────
CONFIGS = {
    "baseline": {"use_str": False, "use_latent": False, "desc": "CE only"},
    "S":        {"use_str": True,  "use_latent": False, "desc": "STR+ only"},
    "L":        {"use_str": False, "use_latent": True,  "desc": "Latent only"},
    "SL":       {"use_str": True,  "use_latent": True,  "desc": "STR+ + Latent"},
}


# ── Manual LoRA (from train_3b.py) ────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank=8, alpha=16.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False
        dev = original.weight.device
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank,
                                                device=dev, dtype=torch.float32) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features,
                                                device=dev, dtype=torch.float32))

    def forward(self, x):
        base = self.original(x)
        lora_out = (x.float() @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora_out.to(base.dtype)


def apply_lora(model, rank=8, alpha=16.0):
    lora_params = []
    for name, module in list(model.named_modules()):
        for target in ("q_proj", "v_proj"):
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                lora_mod = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, name.split(".")[-1], lora_mod)
                lora_params.extend([lora_mod.lora_A, lora_mod.lora_B])
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True
    return lora_params


# ── Data ───────────────────────────────────────────────────────────────
def load_data(tokenizer, max_len=MAX_LEN):
    with open(TRUTHFULQA_JSON) as f:
        raw = json.load(f)
    samples = [s for s in raw["samples"] if s["label"] in (0, 1)]
    out = []
    for item in samples:
        text = f"Question: {item['question']}\nAnswer: {item['output']}"
        enc = tokenizer(text, truncation=True, max_length=max_len,
                       padding="max_length", return_tensors="pt")
        out.append({
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": item["label"],
        })
    return out


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


# ── STR computation ───────────────────────────────────────────────────
def compute_str_on_trajectories(trajectories, sigma, tau=TAU):
    """Compute mean STR over a list of (T, D) trajectories."""
    vals = []
    for h in trajectories:
        T = h.shape[0]
        if T < 2:
            continue
        if T > MAX_STR_TOK:
            idx = torch.linspace(0, T - 1, MAX_STR_TOK).long()
            h = h[idx]
            T = MAX_STR_TOK
        h_f32 = h.float()
        if h_f32.requires_grad:
            h_norm = F.layer_norm(h_f32, h_f32.shape[-1:])
        else:
            h_norm = F.layer_norm(h_f32.detach(), h_f32.shape[-1:])
        d2 = _pairwise_dist_sq(h_norm)
        K = torch.exp(-d2 / (sigma ** 2))
        W = _temporal_weight(T, tau, h.device)
        vals.append((K * W).mean())
    if not vals:
        dev = trajectories[0].device if trajectories else torch.device("cpu")
        return torch.tensor(0.0, device=dev, requires_grad=True)
    return torch.stack(vals).mean()


def compute_str_loss_on_trajectories(trajectories, sigma, tau=TAU):
    """Differentiable STR loss (negative STR) for training."""
    return -compute_str_on_trajectories(trajectories, sigma, tau)


# ── Sigma estimation ──────────────────────────────────────────────────
@torch.no_grad()
def estimate_sigma(model, dataloader, device, max_samples=4):
    model.eval()
    dists = []
    count = 0
    for batch in dataloader:
        if count >= max_samples:
            break
        ids = batch["input_ids"][:1].to(device)
        mask = batch["attention_mask"][:1].to(device)
        out = model(ids, attention_mask=mask, output_hidden_states=True)
        h = out.hidden_states[-1][0].float()
        seq_len = mask[0].sum().item()
        h = h[:int(seq_len)]
        if h.shape[0] < 2:
            continue
        d2 = _pairwise_dist_sq(h)
        nonzero = d2[d2 > 0]
        if nonzero.numel() > 0:
            dists.append(torch.sqrt(nonzero).cpu())
        count += 1
        if device.type == "mps":
            torch.mps.empty_cache()
    if not dists:
        return 1.0
    return torch.median(torch.cat(dists)).item()


# ── Training loop ─────────────────────────────────────────────────────
def run_config(cfg_name, cfg, tokenizer, train_loader, eval_loader, device):
    print(f"\n{'='*60}")
    print(f"  {cfg_name}: {cfg['desc']}")
    print(f"  S={cfg['use_str']}, L={cfg['use_latent']}")
    print(f"{'='*60}")

    outdir = OUT_DIR / cfg_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)

    # LoRA
    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)
    trainable_params = list(lora_params)

    # Latent tokens
    injector = None
    if cfg["use_latent"]:
        hidden_dim = model.config.hidden_size
        injector = LatentTokenInjector(K=LATENT_K, hidden_dim=hidden_dim)
        injector.init_from_embeddings(model.get_input_embeddings())
        injector.to(device)
        trainable_params.extend(list(injector.parameters()))

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable params: {n_trainable:,}")

    # Sigma
    print("  Estimating sigma...")
    sigma = estimate_sigma(model, eval_loader, device)
    print(f"  sigma = {sigma:.1f}")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(50, total_steps // 5),
        num_training_steps=total_steps)

    # Pre-train eval
    pre_metrics = evaluate(model, eval_loader, injector, tokenizer, sigma, device, cfg)
    print(f"  PRE: entropy={pre_metrics['entropy']:.4f} margin={pre_metrics['margin']:.2f}")

    # Train
    model.train()
    if injector:
        injector.train()
    total_loss, total_ce, total_str, step_count = 0, 0, 0, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            if cfg["use_latent"]:
                split_pos = find_question_end(ids, tokenizer)
                embeds, new_mask, lat_mask = injector(ids, mask,
                    model.get_input_embeddings(), split_pos)
                outputs = model(inputs_embeds=embeds, attention_mask=new_mask,
                               output_hidden_states=True)
                loss_ce = compute_lm_loss_with_latent(outputs.logits, ids, lat_mask)
            else:
                outputs = model(ids, attention_mask=mask,
                               output_hidden_states=True, labels=ids)
                loss_ce = outputs.loss
                new_mask = mask
                lat_mask = torch.zeros_like(mask, dtype=torch.bool)

            # STR loss
            loss_str_val = 0.0
            if cfg["use_str"]:
                hidden = outputs.hidden_states[-1]
                trajs = extract_hidden_for_str(hidden, new_mask, lat_mask,
                                                include_latent=cfg["use_latent"])
                if trajs:
                    loss_str = compute_str_loss_on_trajectories(trajs, sigma)
                    loss_total = loss_ce + LAMBDA_STR * loss_str
                    loss_str_val = loss_str.item()
                else:
                    loss_total = loss_ce
            else:
                loss_total = loss_ce

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss_total.item()
            total_ce += loss_ce.item()
            total_str += loss_str_val
            step_count += 1

            if (step + 1) % 20 == 0:
                print(f"    step {step+1}: loss={total_loss/step_count:.4f} "
                      f"ce={total_ce/step_count:.4f} str={total_str/step_count:.4f}")

            if device.type == "mps" and (step + 1) % 5 == 0:
                torch.mps.empty_cache()

    dt = time.time() - t0
    print(f"  Training done: {dt:.0f}s ({dt/60:.1f}min)")

    # Post-train eval
    post_metrics = evaluate(model, eval_loader, injector, tokenizer, sigma, device, cfg)
    print(f"  POST: entropy={post_metrics['entropy']:.4f} margin={post_metrics['margin']:.2f}")

    # Save
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    torch.save(lora_state, outdir / "lora_weights.pt")

    if injector:
        torch.save(injector.state_dict(), outdir / "latent_weights.pt")

    result = {
        "config": cfg_name, "desc": cfg["desc"],
        "n_trainable": n_trainable, "sigma": sigma,
        "train_time_s": round(dt, 1),
        "pre": pre_metrics, "post": post_metrics,
        "delta_entropy": round(post_metrics["entropy"] - pre_metrics["entropy"], 4),
        "delta_margin": round(post_metrics["margin"] - pre_metrics["margin"], 2),
    }
    with open(outdir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    del model
    if injector:
        del injector
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


# ── Evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataloader, injector, tokenizer, sigma, device, cfg,
             max_samples=30):
    model.eval()
    if injector:
        injector.eval()

    entropies, margins, top5s, max_probs, str_vals = [], [], [], [], []
    count = 0

    for batch in dataloader:
        if count >= max_samples:
            break
        ids = batch["input_ids"][:1].to(device)
        mask = batch["attention_mask"][:1].to(device)

        if cfg["use_latent"] and injector is not None:
            split_pos = find_question_end(ids, tokenizer)
            embeds, new_mask, lat_mask = injector(ids, mask,
                model.get_input_embeddings(), split_pos)
            outputs = model(inputs_embeds=embeds, attention_mask=new_mask,
                           output_hidden_states=True)
            logits = outputs.logits
            seq_len = int(new_mask.sum().item())
        else:
            outputs = model(ids, attention_mask=mask, output_hidden_states=True)
            logits = outputs.logits
            new_mask = mask
            lat_mask = torch.zeros_like(mask, dtype=torch.bool)
            seq_len = int(mask.sum().item())

        # Logit stats (on non-latent positions only)
        real_mask = new_mask[0].bool() & ~lat_mask[0]
        real_logits = logits[0][real_mask][:-1].float()
        if real_logits.shape[0] < 2:
            count += 1
            continue

        probs = F.softmax(real_logits, dim=-1)
        log_probs = F.log_softmax(real_logits, dim=-1)
        ent = -(probs * log_probs).sum(dim=-1).mean().item()
        top2 = real_logits.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()
        t5 = probs.topk(5, dim=-1).values.sum(dim=-1).mean().item()
        mp = probs.max(dim=-1).values.mean().item()

        entropies.append(ent)
        margins.append(margin)
        top5s.append(t5)
        max_probs.append(mp)

        # STR on last hidden
        hidden = outputs.hidden_states[-1]
        trajs = extract_hidden_for_str(hidden, new_mask, lat_mask,
                                        include_latent=cfg.get("use_latent", False))
        if trajs:
            str_val = compute_str_on_trajectories(trajs, sigma).item()
            str_vals.append(str_val)

        count += 1
        if device.type == "mps" and count % 10 == 0:
            torch.mps.empty_cache()

    model.train()
    if injector:
        injector.train()

    def mean(lst):
        return sum(lst) / max(len(lst), 1)

    return {
        "entropy": round(mean(entropies), 4),
        "margin": round(mean(margins), 2),
        "top5_mass": round(mean(top5s), 4),
        "max_prob": round(mean(max_probs), 4),
        "str_mean": round(mean(str_vals), 4) if str_vals else None,
        "n_samples": count,
    }


# ── Summary ────────────────────────────────────────────────────────────
def print_summary(results):
    print(f"\n{'='*72}")
    print("  PHASE 1: S+L INTERACTION RESULTS")
    print(f"{'='*72}")
    print(f"  {'Config':<10} {'Δ Entropy':>10} {'Δ Margin':>10} "
          f"{'Post Ent':>10} {'Post Margin':>12}")
    print(f"  {'─'*52}")
    for name, r in results.items():
        print(f"  {name:<10} {r['delta_entropy']:>+10.4f} {r['delta_margin']:>+10.2f} "
              f"{r['post']['entropy']:>10.4f} {r['post']['margin']:>12.2f}")

    # Super-additivity test
    if all(k in results for k in ["baseline", "S", "L", "SL"]):
        print(f"\n  ── Super-Additivity Test ──")
        s_effect = results["S"]["post"]["entropy"] - results["baseline"]["post"]["entropy"]
        l_effect = results["L"]["post"]["entropy"] - results["baseline"]["post"]["entropy"]
        sl_effect = results["SL"]["post"]["entropy"] - results["baseline"]["post"]["entropy"]
        interaction = sl_effect - (s_effect + l_effect)

        print(f"  S effect (entropy):  {s_effect:+.4f}")
        print(f"  L effect (entropy):  {l_effect:+.4f}")
        print(f"  S+L effect:          {sl_effect:+.4f}")
        print(f"  Predicted (S + L):   {s_effect + l_effect:+.4f}")
        print(f"  Interaction (超加性): {interaction:+.4f}")

        if interaction < -0.01:  # negative entropy = more confident
            print(f"\n  🔥 SUPER-ADDITIVE INTERACTION DETECTED")
            print(f"     → S+L produces {abs(interaction):.4f} more entropy reduction")
            print(f"       than S and L independently combined")
            print(f"     → Proceed to Phase 2")
        elif abs(interaction) < 0.005:
            print(f"\n  ⚠️  ADDITIVE ONLY — no interaction detected")
            print(f"     → S and L effects are independent")
            print(f"     → Review L implementation before Phase 2")
        else:
            print(f"\n  ❓ SUB-ADDITIVE — components may interfere")
            print(f"     → Diagnose interference mechanism")

    print(f"{'='*72}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"K={LATENT_K}, λ={LAMBDA_STR}, LoRA rank={LORA_RANK}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_data, eval_data = data[:split], data[split:]
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)
    print(f"Data: {len(train_data)} train, {len(eval_data)} eval")

    results = {}
    for cfg_name in ["baseline", "S", "L", "SL"]:
        results[cfg_name] = run_config(
            cfg_name, CONFIGS[cfg_name], tokenizer,
            train_loader, eval_loader, device)

    print_summary(results)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
