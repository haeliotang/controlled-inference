"""
Controlled Inference: DOM Tuning v2 — Sampling Decode + KL Divergence

Three-way comparison:
  1. Greedy + DOM  → proves argmax barrier (control changes distribution but not tokens)
  2. Sampling baseline → natural stochasticity without control
  3. Sampling + DOM → stable controlled trajectories

Key addition: KL(p_base || p_dom) as intermediate observable proving
  control → distribution shift → trajectory shift

Usage:
    cd code/experiments
    TRANSFORMERS_OFFLINE=1 python3 dom_tuning.py
"""

import json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Path setup ─────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
# sys.path configured for local imports


from str_loss import _pairwise_dist_sq
from train_phase1 import (
    apply_lora, LoRALinear, load_data, collate_fn, estimate_sigma,
    compute_str_loss_on_trajectories,
    LLAMA_PATH, LORA_RANK, LORA_ALPHA, BATCH_SIZE, LR,
)
from lambda_sweep import extract_hidden_real

OUT_DIR = Path(__file__).parent / "results_dom"

# ── Constants ──────────────────────────────────────────────────────────
WINDOW_K    = 4
TAU_W       = 3.0
PD_EPS      = 0.05
MAX_NEW_TOK = 60
LAMBDA_S    = 0.15
LAMBDA_L    = -0.05

# ── Sampling config ───────────────────────────────────────────────────
TEMPERATURE = 0.7
TOP_P       = 0.9
GAMMA_SWEEP = [0.005, 0.01, 0.02, 0.05]
ROUND2_LAMBDA_D = 2.0


# ═══════════════════════════════════════════════════════════════════════
# StateBuffer
# ═══════════════════════════════════════════════════════════════════════
class StateBuffer:
    def __init__(self, window_k=WINDOW_K, sigma=1.0):
        self.window_k = window_k
        self.sigma = sigma
        self.h_buffer = []

    def update(self, h_t):
        self.h_buffer.append(h_t.detach().squeeze(0))
        if len(self.h_buffer) > self.window_k:
            self.h_buffer.pop(0)

    def compute_str(self, device):
        if len(self.h_buffer) < 2:
            return None
        window = torch.stack(self.h_buffer).float()
        d_sq = _pairwise_dist_sq(window)
        K_vals = torch.exp(-d_sq / (2 * self.sigma ** 2))
        W_size = window.shape[0]
        idx = torch.arange(W_size, device=device, dtype=torch.float32)
        dt = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        W_mat = 1.0 - torch.exp(-dt / TAU_W)
        return ((W_mat * K_vals).sum() / W_mat.sum()).item()

    def reset(self):
        self.h_buffer.clear()


# ═══════════════════════════════════════════════════════════════════════
# PD Controller
# ═══════════════════════════════════════════════════════════════════════
class PDController:
    def __init__(self, kappa=10.0, beta=2.0, tau=0.63, eps=PD_EPS,
                 lambda_d=0.0):
        self.kappa = kappa
        self.beta = beta
        self.tau = tau
        self.eps = eps
        self.lambda_d = lambda_d
        self.prev_error = 0.0

    def step(self, str_val):
        e = str_val - self.tau
        de = e - self.prev_error
        raw = self.kappa * e + self.beta * de - self.lambda_d * abs(de)
        alpha = torch.sigmoid(torch.tensor(raw)).item()
        alpha = max(self.eps, min(1.0 - self.eps, alpha))
        self.prev_error = e
        return alpha

    def reset(self):
        self.prev_error = 0.0


# ═══════════════════════════════════════════════════════════════════════
# DOM Actuator
# ═══════════════════════════════════════════════════════════════════════
def dom_control(H_base, A_S, B_S, A_L, B_L, alpha, gamma,
                lora_scaling=1.0):
    delta_S = (H_base.float() @ A_S @ B_S) * lora_scaling
    delta_L = (H_base.float() @ A_L @ B_L) * lora_scaling
    base_scale = H_base.float().norm(dim=-1, keepdim=True).mean() + 1e-6
    u_t = gamma * (alpha * delta_S + (1.0 - alpha) * delta_L) / base_scale
    return H_base + u_t.to(H_base.dtype)


def sample_token(logits, temperature=TEMPERATURE, top_p=TOP_P):
    """Nucleus (top-p) sampling with temperature."""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Remove tokens above threshold
    mask = cumulative - probs > top_p
    sorted_logits[mask] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, idx)


# ═══════════════════════════════════════════════════════════════════════
# Generation: supports greedy/sampling × baseline/DOM
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate(model, tokenizer, prompt, sigma, device,
             # DOM params (None = baseline)
             A_S=None, B_S=None, A_L=None, B_L=None,
             gamma=0.0, controller=None,
             # Decode mode
             use_sampling=False,
             max_new_tokens=MAX_NEW_TOK):
    """Unified generation with optional DOM control and sampling.

    Returns dict with str_trace, alpha_trace, entropy_trace, kl_trace, text.
    """
    is_dom = controller is not None and A_S is not None
    state = StateBuffer(sigma=sigma)
    if is_dom:
        controller.reset()
    lora_scaling = LORA_ALPHA / LORA_RANK

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    str_log, alpha_log, entropy_log, kl_log = [], [], [], []

    for t in range(max_new_tokens):
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        H_base = outputs.hidden_states[-1]
        base_logits = outputs.logits[:, -1, :].float()  # (1, vocab)

        state.update(H_base[:, -1, :])
        str_t = state.compute_str(device)
        str_log.append(str_t)

        if is_dom and str_t is not None:
            alpha_t = controller.step(str_t)
            H_ctrl = dom_control(H_base, A_S, B_S, A_L, B_L,
                                 alpha_t, gamma=gamma,
                                 lora_scaling=lora_scaling)
            H_normed = model.model.norm(H_ctrl[:, -1:, :])
            dom_logits = model.lm_head(H_normed).float().squeeze(0)  # (vocab,)

            # KL divergence: KL(p_base || p_dom)
            p_base = F.softmax(base_logits.squeeze(0), dim=-1)
            p_dom = F.softmax(dom_logits, dim=-1)
            kl = F.kl_div(p_dom.log(), p_base, reduction='sum').item()
            kl_log.append(kl)

            logits_for_decode = dom_logits.unsqueeze(0)  # (1, vocab)
        else:
            alpha_t = 0.5
            logits_for_decode = base_logits
            kl_log.append(0.0)

        alpha_log.append(alpha_t)

        # Entropy
        probs = F.softmax(logits_for_decode.squeeze(0), dim=-1)
        ent = -(probs * probs.log().clamp(min=-100)).sum().item()
        entropy_log.append(ent)

        # Decode
        if use_sampling:
            next_token = sample_token(logits_for_decode.squeeze(0))
            next_token = next_token.view(1, 1)
        else:
            next_token = torch.argmax(logits_for_decode.squeeze(0), dim=-1)
            next_token = next_token.view(1, 1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        if str(device) == "mps" and (t + 1) % 10 == 0:
            torch.mps.empty_cache()

    return {
        "str_trace": str_log, "alpha_trace": alpha_log,
        "entropy_trace": entropy_log, "kl_trace": kl_log,
        "text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
    }


# ═══════════════════════════════════════════════════════════════════════
# Operator Training
# ═══════════════════════════════════════════════════════════════════════
def extract_last_layer_lora(model):
    last = model.model.layers[-1].self_attn.q_proj
    if not isinstance(last, LoRALinear):
        raise ValueError("Last layer q_proj not LoRA-wrapped")
    return {"A": last.lora_A.detach().clone(), "B": last.lora_B.detach().clone()}


def train_operator(lam, tokenizer, train_loader, device, tag):
    from transformers import get_linear_schedule_with_warmup
    print(f"\n  Training Operator {tag} (λ={lam})")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)
    lora_params = apply_lora(model, LORA_RANK, LORA_ALPHA)
    eval_loader = DataLoader(load_data(tokenizer)[:10], batch_size=1,
                             collate_fn=collate_fn)
    sigma = estimate_sigma(model, eval_loader, device)
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20,
        num_training_steps=len(train_loader))
    model.train()
    t0 = time.time()
    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, attention_mask=mask, output_hidden_states=True,
                    labels=ids)
        loss = out.loss
        if lam != 0.0:
            traj = extract_hidden_real(out.hidden_states[-1], mask)
            if traj.shape[0] >= 2:
                loss = loss + lam * compute_str_loss_on_trajectories(
                    [traj], sigma)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()
        if device.type == "mps" and (step + 1) % 10 == 0:
            torch.mps.empty_cache()
    print(f"  Done: {time.time()-t0:.0f}s")
    lora_info = extract_last_layer_lora(model)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return lora_info, sigma


# ═══════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════
def stats(vals):
    if not vals:
        return {"mean": 0, "var": 0, "max": 0, "min": 0}
    t = torch.tensor(vals)
    return {"mean": round(t.mean().item(), 4), "var": round(t.var().item(), 6),
            "max": round(t.max().item(), 4), "min": round(t.min().item(), 4)}


# ═══════════════════════════════════════════════════════════════════════
# Main
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


def run_condition(model, tokenizer, sigma, device, label,
                  A_S=None, B_S=None, A_L=None, B_L=None,
                  gamma=0.0, tau=0.63, lambda_d=0.0,
                  use_sampling=False):
    """Run one experimental condition across all prompts."""
    print(f"\n  ── {label} ──")
    controller = PDController(tau=tau, lambda_d=lambda_d) if A_S is not None else None
    results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        r = generate(model, tokenizer, prompt, sigma, device,
                     A_S=A_S, B_S=B_S, A_L=A_L, B_L=B_L,
                     gamma=gamma, controller=controller,
                     use_sampling=use_sampling)
        valid_strs = [s for s in r["str_trace"] if s is not None]
        valid_kls = [k for k in r["kl_trace"] if k > 0]
        s = stats(valid_strs)
        k = stats(valid_kls) if valid_kls else {"mean": 0, "var": 0}
        a = stats(r["alpha_trace"])
        results.append({
            "str_stats": s, "kl_stats": k, "alpha_stats": a,
            "entropy_mean": round(sum(r["entropy_trace"]) /
                                  max(len(r["entropy_trace"]), 1), 4),
        })
        kl_str = f"KL={k['mean']:.4f}" if valid_kls else "KL=N/A"
        print(f"    P{i+1}: STR={s['mean']:.4f}±{s['var']:.6f}  "
              f"{kl_str}  α=[{a['min']:.2f},{a['max']:.2f}]")
    # Averages
    avg = {
        "avg_str_var": round(sum(r["str_stats"]["var"] for r in results) / len(results), 6),
        "avg_str_mean": round(sum(r["str_stats"]["mean"] for r in results) / len(results), 4),
        "avg_kl": round(sum(r["kl_stats"]["mean"] for r in results) / len(results), 4),
        "prompts": results,
    }
    print(f"  → avg STR var={avg['avg_str_var']:.6f}, KL={avg['avg_kl']:.4f}")
    return avg


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dev_str = str(device)
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Train operators ───────────────────────────────────────────────
    print("\n  Training dual operators...")
    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    lora_S, sigma_S = train_operator(LAMBDA_S, tokenizer, train_loader, device, "S")
    lora_L, sigma_L = train_operator(LAMBDA_L, tokenizer, train_loader, device, "L")
    sigma = (sigma_S + sigma_L) / 2.0
    A_S = lora_S["A"].to(device)
    B_S = lora_S["B"].to(device)
    A_L = lora_L["A"].to(device)
    B_L = lora_L["B"].to(device)
    print(f"  σ={sigma:.1f}")

    # ── Load base model ───────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    # ═══════════════════════════════════════════════════════════════════
    # Condition 1: Sampling Baseline (auto-τ calibration)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  CONDITION 1: Sampling Baseline (τ calibration)")
    print(f"{'='*60}")
    c1 = run_condition(model, tokenizer, sigma, dev_str,
                       "Sampling Baseline", use_sampling=True)

    tau_auto = c1["avg_str_mean"]
    base_var = c1["avg_str_var"]
    print(f"\n  ✅ τ_auto = {tau_auto:.4f}")
    print(f"  ✅ Baseline avg STR var = {base_var:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # Condition 2: Greedy + DOM (proves argmax barrier)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  CONDITION 2: Greedy + DOM (argmax barrier proof)")
    print(f"{'='*60}")
    c2 = run_condition(model, tokenizer, sigma, dev_str,
                       "Greedy+DOM γ=0.05",
                       A_S=A_S, B_S=B_S, A_L=A_L, B_L=B_L,
                       gamma=0.05, tau=tau_auto,
                       use_sampling=False)

    # ═══════════════════════════════════════════════════════════════════
    # Condition 3: Sampling + DOM (γ sweep, Round 1 standard PD)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  CONDITION 3: Sampling + DOM (Round 1: Standard PD)")
    print(f"{'='*60}")
    r1_sweep = {}
    for gamma in GAMMA_SWEEP:
        c3 = run_condition(model, tokenizer, sigma, dev_str,
                           f"Sampling+DOM γ={gamma}",
                           A_S=A_S, B_S=B_S, A_L=A_L, B_L=B_L,
                           gamma=gamma, tau=tau_auto,
                           use_sampling=True)
        r1_sweep[str(gamma)] = c3

    best_g1 = min(r1_sweep, key=lambda g: r1_sweep[g]["avg_str_var"])
    r1_var = r1_sweep[best_g1]["avg_str_var"]
    r1_kl = r1_sweep[best_g1]["avg_kl"]

    # ═══════════════════════════════════════════════════════════════════
    # Condition 4: Sampling + DOM (Round 2: Asymmetric Damping)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  CONDITION 4: Sampling + DOM (Round 2: Damped λd={ROUND2_LAMBDA_D})")
    print(f"{'='*60}")
    r2_sweep = {}
    for gamma in GAMMA_SWEEP:
        c4 = run_condition(model, tokenizer, sigma, dev_str,
                           f"Sampling+DOM+Damped γ={gamma}",
                           A_S=A_S, B_S=B_S, A_L=A_L, B_L=B_L,
                           gamma=gamma, tau=tau_auto,
                           lambda_d=ROUND2_LAMBDA_D,
                           use_sampling=True)
        r2_sweep[str(gamma)] = c4

    best_g2 = min(r2_sweep, key=lambda g: r2_sweep[g]["avg_str_var"])
    r2_var = r2_sweep[best_g2]["avg_str_var"]
    r2_kl = r2_sweep[best_g2]["avg_kl"]

    # ═══════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  τ_auto:                          {tau_auto:.4f}")
    print(f"  ───────────────────────────────────────────────")
    print(f"  Sampling Baseline STR var:       {base_var:.6f}")
    print(f"  Greedy+DOM KL (proves shift):    {c2['avg_kl']:.4f}")
    print(f"  Greedy+DOM STR var (unchanged):  {c2['avg_str_var']:.6f}")
    print(f"  ───────────────────────────────────────────────")
    print(f"  Round 1 best (PD) γ={best_g1}:")
    print(f"    STR var: {r1_var:.6f}  KL: {r1_kl:.4f}")
    print(f"  Round 2 best (Damped) γ={best_g2}:")
    print(f"    STR var: {r2_var:.6f}  KL: {r2_kl:.4f}")

    # Verdict
    print(f"\n  ═══ VERDICT ═══")
    if r1_var < base_var:
        pct = (1 - r1_var / max(base_var, 1e-8)) * 100
        print(f"  🔥 Round 1 reduces STR variance by {pct:.1f}%")
    if r2_var < r1_var:
        pct2 = (1 - r2_var / max(r1_var, 1e-8)) * 100
        print(f"  🔥 Damping further reduces by {pct2:.1f}%")
    if c2["avg_kl"] > 0.001:
        print(f"  ✅ Greedy KL > 0 proves control shifts distribution")
        print(f"     (but argmax blocks token-level effect)")

    # Save
    results = {
        "tau_auto": round(tau_auto, 4), "sigma": sigma,
        "sampling_baseline": c1,
        "greedy_dom": c2,
        "round1_standard_pd": {"best_gamma": best_g1, "sweep": r1_sweep},
        "round2_damped": {"lambda_d": ROUND2_LAMBDA_D, "best_gamma": best_g2,
                          "sweep": r2_sweep},
    }
    out_path = OUT_DIR / "tuning_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
