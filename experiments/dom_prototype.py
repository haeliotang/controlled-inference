"""
Controlled Inference: Dynamic Operator Mixing (DOM) Prototype

Implements the four-module closed-loop control system:
  Dynamics  → H_{t+1} = f(H_t, x_t)
  State     → STR_t = WindowSTR(H_{t-K:t}, σ)
  Controller→ α_t = PD(STR_t)
  Actuator  → u_t = γ · [α_t·ΔH_S + (1-α_t)·ΔH_L]

Usage:
    cd code/experiments
    TRANSFORMERS_OFFLINE=1 python3 dom_prototype.py
"""

import json, sys, time, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# ── Path setup ─────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
# sys.path configured for local imports


from str_loss import _pairwise_dist_sq
from train_phase1 import (
    apply_lora, LoRALinear, load_data, collate_fn, estimate_sigma,
    compute_str_loss_on_trajectories,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, MAX_LEN, BATCH_SIZE, LR,
)
from lambda_sweep import extract_hidden_real

OUT_DIR = Path(__file__).parent / "results_dom"

# ── DOM Hyperparameters ────────────────────────────────────────────────
LAMBDA_S    = 0.15    # strong contraction penalty for ΔW_S
LAMBDA_L    = -0.05   # negative penalty (expansion) for ΔW_L
WINDOW_K    = 4       # sliding window for STR
TAU_W       = 3.0     # temporal decay for windowed STR
GAMMA       = 0.05    # control gain (relative to base norm)
PD_KAPPA    = 10.0    # proportional gain
PD_BETA     = 2.0     # derivative gain
PD_TAU      = 0.63    # target STR (healthy manifold)
PD_EPS      = 0.05    # clamping epsilon
MAX_NEW_TOK = 60      # tokens to generate


# ═══════════════════════════════════════════════════════════════════════
# Module 1: State Buffer (STR first-class citizen)
# ═══════════════════════════════════════════════════════════════════════
class StateBuffer:
    """Sliding-window state buffer for online STR computation."""

    def __init__(self, window_k=WINDOW_K, sigma=1.0):
        self.window_k = window_k
        self.sigma = sigma
        self.h_buffer = []
        self.str_history = []

    def update(self, h_t):
        """Push new hidden state (detached), maintain window."""
        self.h_buffer.append(h_t.detach().squeeze(0))  # (D,)
        if len(self.h_buffer) > self.window_k:
            self.h_buffer.pop(0)

    def compute_str(self, device):
        """Compute windowed STR from current buffer."""
        if len(self.h_buffer) < 2:
            return None
        window = torch.stack(self.h_buffer).float()        # (W, D)
        d_sq = _pairwise_dist_sq(window)                   # (W, W)
        K_vals = torch.exp(-d_sq / (2 * self.sigma ** 2))  # Gaussian kernel
        W_size = window.shape[0]
        idx = torch.arange(W_size, device=device, dtype=torch.float32)
        dt = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        W_mat = 1.0 - torch.exp(-dt / TAU_W)              # temporal weight
        s = ((W_mat * K_vals).sum() / W_mat.sum()).item()
        self.str_history.append(s)
        return s

    def reset(self):
        self.h_buffer.clear()
        self.str_history.clear()


# ═══════════════════════════════════════════════════════════════════════
# Module 2: PD Controller
# ═══════════════════════════════════════════════════════════════════════
class PDController:
    """Proportional-Derivative controller in the latent space."""

    def __init__(self, kappa=PD_KAPPA, beta=PD_BETA, tau=PD_TAU, eps=PD_EPS):
        self.kappa = kappa
        self.beta = beta
        self.tau = tau
        self.eps = eps
        self.prev_error = 0.0

    def step(self, str_val: float) -> float:
        e = str_val - self.tau
        de = e - self.prev_error
        raw = self.kappa * e + self.beta * de
        alpha = torch.sigmoid(torch.tensor(raw)).item()
        alpha = max(self.eps, min(1.0 - self.eps, alpha))
        self.prev_error = e
        return alpha

    def reset(self):
        self.prev_error = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Module 3: LoRA Operator Extraction
# ═══════════════════════════════════════════════════════════════════════
def extract_last_layer_lora(model):
    """Extract (A, B, scaling) from the last transformer layer's q_proj LoRA.

    Returns dict with keys 'A', 'B', 'scaling'.
    Raises ValueError if last layer q_proj is not LoRA-wrapped.
    """
    last_layer = model.model.layers[-1]
    q_proj = last_layer.self_attn.q_proj
    if not isinstance(q_proj, LoRALinear):
        raise ValueError("Last layer q_proj is not LoRA-wrapped. "
                         "Did you call apply_lora() first?")
    return {
        "A": q_proj.lora_A.detach().clone(),  # (in_features, rank)
        "B": q_proj.lora_B.detach().clone(),  # (rank, out_features)
        "scaling": q_proj.scaling,
    }


def save_lora_weights(model, path):
    """Save all LoRA weights to a checkpoint file."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    torch.save(state, path)
    return state


# ═══════════════════════════════════════════════════════════════════════
# Module 4: DOM Actuator (Control Input Injection)
# ═══════════════════════════════════════════════════════════════════════
def dom_control(H_base, A_S, B_S, A_L, B_L, alpha, gamma=GAMMA,
                lora_scaling=1.0):
    """Inject control input u_t into the final residual stream.

    u_t = γ · [α·ΔH_S + (1-α)·ΔH_L] / base_scale

    Uses RELATIVE scaling (not L2 normalization) for physical correctness.
    """
    # Compute LoRA residuals: ΔH = H @ A @ B * scaling
    delta_S = (H_base.float() @ A_S @ B_S) * lora_scaling
    delta_L = (H_base.float() @ A_L @ B_L) * lora_scaling
    # Relative scaling: control signal proportional to base magnitude
    base_scale = H_base.float().norm(dim=-1, keepdim=True).mean() + 1e-6
    u_t = gamma * (alpha * delta_S + (1.0 - alpha) * delta_L) / base_scale
    return H_base + u_t.to(H_base.dtype)


# ═══════════════════════════════════════════════════════════════════════
# Operator Training: Train ΔW_S and ΔW_L with opposite λ
# ═══════════════════════════════════════════════════════════════════════
def train_operator(lam, tokenizer, train_loader, device, tag="S"):
    """Train a single LoRA operator with given λ (STR penalty weight).

    λ > 0  → contraction operator (ΔW_S): penalizes high STR variance
    λ < 0  → expansion operator (ΔW_L): encourages STR exploration
    """
    print(f"\n{'─'*50}")
    print(f"  Training Operator {tag} (λ={lam})")
    print(f"{'─'*50}")

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)

    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)

    # Sigma calibration (warmup)
    eval_loader = DataLoader(load_data(tokenizer)[:10], batch_size=1,
                             collate_fn=collate_fn)
    sigma = estimate_sigma(model, eval_loader, device)
    print(f"  σ = {sigma:.1f}")

    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20,
        num_training_steps=len(train_loader))

    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        out = model(ids, attention_mask=mask,
                    output_hidden_states=True, labels=ids)
        loss = out.loss

        if lam != 0.0:
            hidden = out.hidden_states[-1]
            traj = extract_hidden_real(hidden, mask)
            if traj.shape[0] >= 2:
                loss = loss + lam * compute_str_loss_on_trajectories(
                    [traj], sigma)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if device.type == "mps" and (step + 1) % 10 == 0:
            torch.mps.empty_cache()

    dt = time.time() - t0
    print(f"  Done: {dt:.0f}s, avg_loss={total_loss/max(len(train_loader),1):.4f}")

    # Extract last-layer LoRA before deleting model
    lora_info = extract_last_layer_lora(model)

    # Save full LoRA weights
    save_path = OUT_DIR / f"lora_{tag}.pt"
    save_lora_weights(model, save_path)
    print(f"  Saved: {save_path}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return lora_info, sigma


# ═══════════════════════════════════════════════════════════════════════
# Generation Loop (DOM-controlled autoregressive)
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_with_dom(model, tokenizer, prompt, A_S, B_S, A_L, B_L,
                      sigma, lora_scaling=1.0, gamma=GAMMA,
                      max_new_tokens=MAX_NEW_TOK, device="mps"):
    """Generate with DOM feedback control.

    Causal chain per step:
      f(H_t, x_t) → H_base → STR_t → e_t → α_t → u_t → H_controlled → logits
    """
    controller = PDController()
    state = StateBuffer(window_k=WINDOW_K, sigma=sigma)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    str_log = []
    alpha_log = []
    entropy_log = []

    for t in range(max_new_tokens):
        # ── Dynamics: base model forward (NO KV cache) ──
        outputs = model(input_ids, output_hidden_states=True,
                        use_cache=False)
        H_base = outputs.hidden_states[-1]  # (1, seq, D)

        # ── State: update buffer, compute STR_t ──
        state.update(H_base[:, -1, :])
        str_t = state.compute_str(device)

        if str_t is None:
            # Warmup phase: not enough history for STR
            logits = outputs.logits[:, -1:, :].float()
            alpha_t = 0.5  # neutral
        else:
            # ── Controller: STR_t → α_t ──
            alpha_t = controller.step(str_t)
            # ── Actuator: inject control into final residual ──
            H_ctrl = dom_control(H_base, A_S, B_S, A_L, B_L,
                                 alpha_t, gamma=gamma,
                                 lora_scaling=lora_scaling)
            # CRITICAL: apply RMSNorm before lm_head (Llama architecture)
            H_normed = model.model.norm(H_ctrl[:, -1:, :])
            logits = model.lm_head(H_normed).float()

        # Record entropy
        probs = F.softmax(logits.squeeze(0), dim=-1)
        ent = -(probs * F.log_softmax(logits.squeeze(0), dim=-1)).sum(-1)
        entropy_log.append(ent.item())

        # Greedy decode
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        str_log.append(str_t)
        alpha_log.append(alpha_t)

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Memory pressure relief
        if str(device) == "mps" and (t + 1) % 10 == 0:
            torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "str_trace": str_log,
        "alpha_trace": alpha_log,
        "entropy_trace": entropy_log,
        "n_tokens": len(alpha_log),
    }


@torch.no_grad()
def generate_baseline(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOK,
                      sigma=1.0, device="mps"):
    """Baseline generation (no DOM control) with STR logging."""
    state = StateBuffer(window_k=WINDOW_K, sigma=sigma)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    str_log = []
    entropy_log = []

    for t in range(max_new_tokens):
        outputs = model(input_ids, output_hidden_states=True,
                        use_cache=False)
        H_base = outputs.hidden_states[-1]

        state.update(H_base[:, -1, :])
        str_t = state.compute_str(device)
        str_log.append(str_t)

        logits = outputs.logits[:, -1:, :].float()
        probs = F.softmax(logits.squeeze(0), dim=-1)
        ent = -(probs * F.log_softmax(logits.squeeze(0), dim=-1)).sum(-1)
        entropy_log.append(ent.item())

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if str(device) == "mps" and (t + 1) % 10 == 0:
            torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "str_trace": str_log,
        "entropy_trace": entropy_log,
        "n_tokens": len(str_log),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════
TEST_PROMPTS = [
    "Question: A train travels 60 miles per hour for 2 hours, then 40 miles "
    "per hour for 3 hours. What is the total distance?\nAnswer: Let me think "
    "step by step.",
    "Question: If a store offers a 20% discount on a $150 item, and then "
    "charges 8% tax on the discounted price, what is the final price?\n"
    "Answer: Let me work through this carefully.",
    "Question: Sarah has 3 times as many apples as Tom. Together they have "
    "48 apples. How many does each person have?\nAnswer: I'll solve this "
    "step by step.",
]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"γ={GAMMA}, κ={PD_KAPPA}, β={PD_BETA}, τ={PD_TAU}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)

    # ── Step 1: Train dual operators ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: Train Dual Operators (ΔW_S and ΔW_L)")
    print("=" * 60)

    lora_S, sigma_S = train_operator(LAMBDA_S, tokenizer, train_loader,
                                      device, tag="S")
    lora_L, sigma_L = train_operator(LAMBDA_L, tokenizer, train_loader,
                                      device, tag="L")
    sigma = (sigma_S + sigma_L) / 2.0
    scaling = LORA_ALPHA / LORA_RANK

    A_S = lora_S["A"].to(device)
    B_S = lora_S["B"].to(device)
    A_L = lora_L["A"].to(device)
    B_L = lora_L["B"].to(device)

    print(f"\n  Operators extracted.")
    print(f"  A_S: {A_S.shape}, B_S: {B_S.shape}")
    print(f"  A_L: {A_L.shape}, B_L: {B_L.shape}")
    print(f"  σ (averaged): {sigma:.1f}")

    # ── Step 2: Load clean base model for inference ───────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: DOM Inference (Baseline vs Controlled)")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    all_results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─'*50}")
        print(f"  Prompt {i+1}/{len(TEST_PROMPTS)}")
        print(f"{'─'*50}")
        print(f"  {prompt[:80]}...")

        # Baseline (no control)
        base = generate_baseline(model, tokenizer, prompt,
                                 sigma=sigma, device=str(device))

        # DOM-controlled
        dom = generate_with_dom(model, tokenizer, prompt,
                                A_S, B_S, A_L, B_L,
                                sigma=sigma, lora_scaling=scaling,
                                gamma=GAMMA, device=str(device))

        # Statistics
        base_strs = [s for s in base["str_trace"] if s is not None]
        dom_strs = [s for s in dom["str_trace"] if s is not None]

        def stats(vals):
            if not vals:
                return {"mean": 0, "var": 0, "max": 0, "min": 0}
            t = torch.tensor(vals)
            return {
                "mean": round(t.mean().item(), 4),
                "var": round(t.var().item(), 6),
                "max": round(t.max().item(), 4),
                "min": round(t.min().item(), 4),
            }

        result = {
            "prompt_id": i,
            "baseline": {
                "text": base["text"],
                "str_stats": stats(base_strs),
                "entropy_mean": round(sum(base["entropy_trace"]) /
                                      max(len(base["entropy_trace"]), 1), 4),
                "n_tokens": base["n_tokens"],
            },
            "dom": {
                "text": dom["text"],
                "str_stats": stats(dom_strs),
                "alpha_stats": stats(dom["alpha_trace"]),
                "entropy_mean": round(sum(dom["entropy_trace"]) /
                                      max(len(dom["entropy_trace"]), 1), 4),
                "n_tokens": dom["n_tokens"],
            },
        }
        all_results.append(result)

        # Print comparison
        bs = result["baseline"]["str_stats"]
        ds = result["dom"]["str_stats"]
        da = result["dom"]["alpha_stats"]
        print(f"\n  {'Metric':<20} {'Baseline':>12} {'DOM':>12}")
        print(f"  {'─'*44}")
        print(f"  {'STR mean':<20} {bs['mean']:>12.4f} {ds['mean']:>12.4f}")
        print(f"  {'STR var':<20} {bs['var']:>12.6f} {ds['var']:>12.6f}")
        print(f"  {'STR max':<20} {bs['max']:>12.4f} {ds['max']:>12.4f}")
        print(f"  {'Entropy mean':<20} "
              f"{result['baseline']['entropy_mean']:>12.4f} "
              f"{result['dom']['entropy_mean']:>12.4f}")
        print(f"  {'α mean':<20} {'N/A':>12} {da['mean']:>12.4f}")
        print(f"  {'α range':<20} {'N/A':>12} "
              f"[{da['min']:.2f}, {da['max']:.2f}]")

        if device.type == "mps":
            torch.mps.empty_cache()

    # ── Step 3: Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3: Summary")
    print("=" * 60)

    # Aggregate across prompts
    all_base_var = [r["baseline"]["str_stats"]["var"] for r in all_results]
    all_dom_var = [r["dom"]["str_stats"]["var"] for r in all_results]

    avg_base_var = sum(all_base_var) / len(all_base_var)
    avg_dom_var = sum(all_dom_var) / len(all_dom_var)

    print(f"\n  Avg STR variance:  Baseline={avg_base_var:.6f}  "
          f"DOM={avg_dom_var:.6f}")

    if avg_dom_var < avg_base_var:
        reduction = (1.0 - avg_dom_var / max(avg_base_var, 1e-8)) * 100
        print(f"  🔥 DOM reduces STR variance by {reduction:.1f}%")
        print(f"     → Feedback control stabilizes inference trajectory")
    else:
        print(f"  ⚠️  DOM did not reduce STR variance.")
        print(f"     → Check γ, κ, β tuning or operator quality")

    # Save
    with open(OUT_DIR / "dom_results.json", "w") as f:
        json.dump({
            "config": {
                "gamma": GAMMA, "kappa": PD_KAPPA, "beta": PD_BETA,
                "tau": PD_TAU, "lambda_S": LAMBDA_S, "lambda_L": LAMBDA_L,
                "window_k": WINDOW_K, "sigma": sigma,
            },
            "prompts": all_results,
            "summary": {
                "avg_base_str_var": round(avg_base_var, 6),
                "avg_dom_str_var": round(avg_dom_var, 6),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT_DIR / 'dom_results.json'}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
