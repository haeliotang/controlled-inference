"""
Final Gap Experiment: F-only Control Primitive
Isolates Feedback (F) as an independent causal variable by switching between
frozen contraction (M_S) and expansion (M_L) models during autoregressive generation.
"""
import json, sys, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
# sys.path configured for local imports

from train_phase1 import (
    load_data, collate_fn, estimate_sigma,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, BATCH_SIZE, LR, LoRALinear, apply_lora
)
from trc_train import NUM_EVAL_PROMPTS, OUT_TRC, TARGET_STR
from inference_str_logger import compute_window_str, TEMPERATURE, WINDOW_K
from lambda_sweep import extract_hidden_real, compute_str_loss_on_trajectories

# Config
LAMBDA_S = 0.15
LAMBDA_L = -0.05
MAX_NEW_TOKENS = 50

# --- Training Single Models ---
def train_and_extract_lora(lam, tokenizer, train_loader, sigma, device):
    """Train a static lambda model and return its LoRA weights."""
    print(f"\nTraining static model with lambda = {lam}")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16).to(device)
    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)
    
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=len(train_loader))
    
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
            
    print(f"  Trained in {time.time()-t0:.1f}s")
    
    # Extract weights
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.detach().clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.detach().clone()
            
    del model
    if device.type == "mps": torch.mps.empty_cache()
    return lora_state

# --- Dual LoRA Injection for Inference ---
class DualLoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank, alpha, state_S, state_L, prefix):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        
        # We assume base weights are frozen
        self.lora_A_S = nn.Parameter(state_S[f"{prefix}.lora_A"])
        self.lora_B_S = nn.Parameter(state_S[f"{prefix}.lora_B"])
        self.lora_A_L = nn.Parameter(state_L[f"{prefix}.lora_A"])
        self.lora_B_L = nn.Parameter(state_L[f"{prefix}.lora_B"])
        
        # Frozen
        self.lora_A_S.requires_grad = False
        self.lora_B_S.requires_grad = False
        self.lora_A_L.requires_grad = False
        self.lora_B_L.requires_grad = False
        for p in self.original.parameters():
            p.requires_grad = False
            
        # 0 for M_S, 1 for M_L
        self.selector = 0

    def forward(self, x):
        base = self.original(x)
        if self.selector == 0:
            lora_out = (x.float() @ self.lora_A_S @ self.lora_B_S) * self.scaling
        else:
            lora_out = (x.float() @ self.lora_A_L @ self.lora_B_L) * self.scaling
        return base + lora_out.to(base.dtype)

def apply_dual_lora(model, state_S, state_L, rank=8, alpha=16.0):
    for name, module in list(model.named_modules()):
        for target in ("q_proj", "v_proj"):
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                lora_mod = DualLoRALinear(module, rank, alpha, state_S, state_L, name)
                setattr(parent, name.split(".")[-1], lora_mod)
    for p in model.parameters():
        p.requires_grad = False

def set_dual_selector(model, selector_val):
    """Sets selector for all DualLoRALinear modules. 0 = M_S, 1 = M_L"""
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            module.selector = selector_val


# --- Generation with Control ---
@torch.no_grad()
def generate_f_controlled(model, prompt_ids, sigma, device, mode="f_only"):
    h_buffer = []
    str_traj = []

    # Initial forward: use baseline (S)
    set_dual_selector(model, 0)
    out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    past_kv = out.past_key_values
    h_t = out.hidden_states[-1][:, -1, :].float().squeeze(0)
    h_buffer.append(h_t)
    logits = out.logits[:, -1, :].float()

    for step in range(MAX_NEW_TOKENS):
        # Determine current STR
        current_str = None
        if len(h_buffer) >= 2:
            current_str = compute_window_str(h_buffer, sigma, device)
            
        # Selection Logic
        if mode == "static_s":
            sel = 0
        elif mode == "static_l":
            sel = 1
        elif mode == "random":
            sel = random.choice([0, 1])
        elif mode == "f_only":
            if current_str is None:
                sel = 0 # Default start
            else:
                sel = 1 if current_str > TARGET_STR else 0
        
        set_dual_selector(model, sel)

        # Sample
        next_tok = torch.multinomial(F.softmax(logits / TEMPERATURE, dim=-1), 1)

        # Forward
        out = model(next_tok, past_key_values=past_kv,
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
def evaluate_modes(model, tokenizer, prompts, sigma, device):
    model.eval()
    modes = ["static_s", "static_l", "random", "f_only"]
    results = {m: [] for m in modes}
    
    for i, p in enumerate(prompts):
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(device)
        for m in modes:
            traj = generate_f_controlled(model, enc["input_ids"], sigma, device, mode=m)
            results[m].append(traj)
    return results

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 1. Train models and get weights
    tmp = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16).to(device)
    apply_lora(tmp, LORA_RANK, LORA_ALPHA)
    sigma = estimate_sigma(tmp, DataLoader(data[:20], batch_size=1, collate_fn=collate_fn), device)
    del tmp
    if device.type == "mps": torch.mps.empty_cache()

    state_S = train_and_extract_lora(LAMBDA_S, tokenizer, train_loader, sigma, device)
    state_L = train_and_extract_lora(LAMBDA_L, tokenizer, train_loader, sigma, device)

    # 2. Build Dual Model
    print("\nBuilding Dual LoRA Model for Inference...")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16).to(device)
    apply_dual_lora(model, state_S, state_L, LORA_RANK, LORA_ALPHA)
    
    prompts = []
    for item in data[split:split + NUM_EVAL_PROMPTS]:
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        p = text.split("Answer:")[0] + "Answer:" if "Answer:" in text else text
        prompts.append(p)

    # 3. Evaluate Modes
    print("Evaluating modes...")
    trajs = evaluate_modes(model, tokenizer, prompts, sigma, device)
    
    # 4. Analyze & Save
    final_results = {}
    print(f"\n{'='*60}")
    print(f"  FEEDBACK CONTROL RESULTS")
    print(f"{'='*60}")
    print(f"  {'Mode':<15} | {'Mean(STR)':>10} | {'Var(STR)':>10}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
    
    for m in ["static_s", "static_l", "random", "f_only"]:
        t_list = trajs[m]
        # Average trajectory
        n_steps = len(t_list[0])
        avg_traj = [sum(t[i] for t in t_list)/len(t_list) for i in range(n_steps)]
        
        # Flatten all values for Var
        all_vals = [val for t in t_list for val in t]
        mean_val = sum(all_vals)/len(all_vals)
        var_val = sum((x - mean_val)**2 for x in all_vals)/len(all_vals)
        
        final_results[m] = {
            "avg_traj": avg_traj,
            "mean": mean_val,
            "var": var_val
        }
        print(f"  {m:<15} | {mean_val:10.4f} | {var_val:10.6f}")

    OUT_TRC.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRC / "f_only_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved results to {OUT_TRC / 'f_only_results.json'}")

if __name__ == "__main__":
    main()
