"""
Run ONLY the phase_feedback protocol (incremental experiment).
Reuses sigma + prompts from main TRC script, appends to trc_results.json.
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
    train_one_protocol, NUM_EVAL_PROMPTS, OUT_TRC, TARGET_STR
)
from inference_str_logger import eval_trajectories

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

    # Train phase_feedback
    model, _ = train_one_protocol("phase_feedback", tokenizer, train_loader, sigma, device)

    print("  Logging inference trajectories...")
    res = eval_trajectories(model, tokenizer, prompts, sigma, device)
    s = res["summary"]
    print(f"    mean(STR_t)={s['mean_str']:.4f}  Var(STR_t)={s['var_str']:.6f}")
    print(f"    mean(Ent_t)={s['mean_entropy']:.4f}")

    # Save trajectories
    with open(OUT_TRC / "trajectories_phase_feedback.json", "w") as f:
        json.dump(res["trajectories"], f)

    # Update results
    results_file = OUT_TRC / "trc_results.json"
    if results_file.exists():
        with open(results_file) as f:
            final = json.load(f)
    else:
        final = {"protocols": {}, "target_str": TARGET_STR,
                 "oracle_var": 0.005128, "unconstrained_var": 0.003937}

    final["protocols"]["phase_feedback"] = s
    with open(results_file, "w") as f:
        json.dump(final, f, indent=2)

    # Comparison
    print(f"\n{'='*60}")
    print(f"  PHASE-AWARE FEEDBACK vs ALL")
    print(f"{'='*60}")
    for name, ps in final["protocols"].items():
        in_band = abs(ps["mean_str"] - TARGET_STR) < 0.012
        beat = ps["var_str"] < 0.005128
        flag = "✓" if (in_band and beat) else "✗"
        print(f"  {flag} {name:16s}  Var={ps['var_str']:.6f}  mean={ps['mean_str']:.4f}")

    del model
    if device.type == "mps": torch.mps.empty_cache()

if __name__ == "__main__":
    main()
