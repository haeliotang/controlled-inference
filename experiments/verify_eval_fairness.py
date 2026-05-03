"""
Verify eval fairness: are L/SL improvements from LoRA or from latent tokens at eval time?

Test: Load L-trained and SL-trained LoRA weights, evaluate WITHOUT latent tokens.
If entropy is still lower than baseline → LoRA genuinely improved.
If entropy equals baseline → improvement was latent-token-dependent (confound).

Expected reference values (from Phase 1 run):
  baseline POST entropy = 0.8995  (eval WITHOUT latent)
  S        POST entropy = 0.8724  (eval WITHOUT latent)
  L        POST entropy = 0.8568  (eval WITH latent)
  SL       POST entropy = 0.8878  (eval WITH latent)
"""

import json, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from str_loss import _pairwise_dist_sq
from train_phase1 import (
    LoRALinear, apply_lora, load_data, collate_fn,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, BATCH_SIZE, MAX_LEN, OUT_DIR,
)


def load_lora_weights(model, weights_path):
    """Load saved LoRA weights into an already LoRA-injected model."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state and b_key in state:
                module.lora_A.data.copy_(state[a_key])
                module.lora_B.data.copy_(state[b_key])
                loaded += 1
    return loaded


@torch.no_grad()
def eval_without_latent(model, dataloader, device, max_samples=30):
    """
    Evaluate model WITHOUT any latent token injection.
    Same code path as baseline/S evaluation in train_phase1.py evaluate().
    """
    model.eval()
    entropies, margins = [], []
    count = 0

    for batch in dataloader:
        if count >= max_samples:
            break
        ids = batch["input_ids"][:1].to(device)
        mask = batch["attention_mask"][:1].to(device)

        # Standard forward — NO latent tokens
        outputs = model(ids, attention_mask=mask)
        logits = outputs.logits[0].float()
        seq_len = int(mask.sum().item())

        # Same slicing as evaluate() for non-latent path
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
        count += 1

        if device.type == "mps" and count % 10 == 0:
            torch.mps.empty_cache()

    return {
        "entropy": round(sum(entropies) / max(len(entropies), 1), 4),
        "margin": round(sum(margins) / max(len(margins), 1), 2),
        "n_samples": count,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    eval_loader = DataLoader(data[split:], batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)
    print(f"Eval samples: {len(data[split:])}")

    # Configs to test: load their LoRA weights, eval WITHOUT latent
    configs_to_test = ["baseline", "S", "L", "SL"]
    results = {}

    for cfg_name in configs_to_test:
        weights_path = OUT_DIR / cfg_name / "lora_weights.pt"
        if not weights_path.exists():
            print(f"  {cfg_name}: weights not found, skipping")
            continue

        print(f"\n  Loading {cfg_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
        model.to(device)
        apply_lora(model, LORA_RANK, LORA_ALPHA)
        n = load_lora_weights(model, weights_path)
        print(f"    Loaded {n} LoRA modules")

        print(f"    Evaluating WITHOUT latent tokens...")
        metrics = eval_without_latent(model, eval_loader, device)
        results[cfg_name] = metrics
        print(f"    entropy={metrics['entropy']:.4f}  margin={metrics['margin']:.2f}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("  EVAL FAIRNESS VERIFICATION")
    print(f"  All configs evaluated WITHOUT latent tokens")
    print(f"{'='*60}")
    print(f"  {'Config':<12} {'No-Latent Ent':>14} {'Original Ent':>14} {'Note'}")
    print(f"  {'─'*56}")

    # Original entropy values (from Phase 1 with their native eval mode)
    original = {"baseline": 0.8995, "S": 0.8724, "L": 0.8568, "SL": 0.8878}

    for name in configs_to_test:
        if name not in results:
            continue
        orig = original.get(name, "N/A")
        no_lat = results[name]["entropy"]
        if name in ("L", "SL"):
            note = "was WITH latent" if orig != no_lat else "same"
        else:
            note = "no latent in orig"
        print(f"  {name:<12} {no_lat:>14.4f} {orig:>14.4f} {note}")

    print(f"\n  Interpretation:")
    if "L" in results:
        l_no_lat = results["L"]["entropy"]
        b_no_lat = results["baseline"]["entropy"]
        if l_no_lat < b_no_lat - 0.01:
            print(f"  ✅ L LoRA genuinely improves model (no-latent {l_no_lat:.4f} < baseline {b_no_lat:.4f})")
        elif l_no_lat > b_no_lat + 0.01:
            print(f"  ⚠️  L LoRA HURTS without latent tokens ({l_no_lat:.4f} > {b_no_lat:.4f})")
            print(f"     → L improvement was latent-dependent (eval confound)")
        else:
            print(f"  ➡️  L LoRA neutral without latent ({l_no_lat:.4f} ≈ {b_no_lat:.4f})")
            print(f"     → L improvement was partially latent-dependent")
    print(f"{'='*60}")

    with open(OUT_DIR / "eval_fairness.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
