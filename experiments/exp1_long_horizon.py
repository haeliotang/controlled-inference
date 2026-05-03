"""
Controlled Inference: Experiment 1 — Long-Horizon Trajectory Stability

Proves: "Without feedback, trajectories inevitably drift.
         With feedback, they remain bounded."

Three conditions × 5 prompts × 5 runs = 75 generation passes.
Frozen params from dom_tuning.py Round 2: γ=0.02, λ_d=2.0.
Phase 2: T=100 (extended horizon for drift accumulation).

Conditions:
  1. Baseline        — sampling, no control
  2. Static α=0.2    — operator mixing, expansion bias, no feedback
  3. DOM (PD+Damped) — full closed-loop feedback control

Metrics (per-token):
  - STR trajectory
  - Drift index: |STR_t - τ|
  - Cumulative drift: running mean of drift
  - Failure step: first t where drift > 0.15 (~3σ)
  - KL divergence (control conditions only)
  - α trajectory

Usage:
    cd code/experiments
    TRANSFORMERS_OFFLINE=1 python3 exp1_long_horizon.py
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

OUT_DIR = Path(__file__).parent / "results_dom" / "exp1"

# ── Constants (FROZEN from tuning) ─────────────────────────────────────
WINDOW_K    = 4
TAU_W       = 3.0
PD_EPS      = 0.05
MAX_NEW_TOK = 100      # Phase 2: extended horizon for drift accumulation.
LAMBDA_S    = 0.15
LAMBDA_L    = -0.05

# Decode
TEMPERATURE = 0.7
TOP_P       = 0.9

# Controller (frozen from Round 2 best)
GAMMA       = 0.02
LAMBDA_D    = 2.0
KAPPA       = 10.0
BETA        = 2.0

# Experiment
N_RUNS      = 5        # Runs per condition-prompt pair (more for stability)
DRIFT_THRESHOLD = 0.15 # ~3σ for failure detection
BOUNDED_DELTA   = 0.10 # Boundedness ratio threshold: |STR-τ| < δ


# ═══════════════════════════════════════════════════════════════════════
# StateBuffer (identical to dom_tuning.py)
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
# Controllers
# ═══════════════════════════════════════════════════════════════════════
class PDController:
    """PD controller with optional asymmetric damping."""
    def __init__(self, kappa=KAPPA, beta=BETA, tau=0.63, eps=PD_EPS,
                 lambda_d=LAMBDA_D):
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


class StaticController:
    """Fixed-α controller: operator mixing without feedback."""
    def __init__(self, alpha=0.5):
        self.fixed_alpha = alpha

    def step(self, str_val):
        return self.fixed_alpha

    def reset(self):
        pass


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
    mask = cumulative - probs > top_p
    sorted_logits[mask] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, idx)


# ═══════════════════════════════════════════════════════════════════════
# Unified generation (same as dom_tuning v2)
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate(model, tokenizer, prompt, sigma, device,
             A_S=None, B_S=None, A_L=None, B_L=None,
             gamma=0.0, controller=None,
             max_new_tokens=MAX_NEW_TOK):
    """Generate with sampling. Returns per-token traces."""
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
        base_logits = outputs.logits[:, -1, :].float()

        state.update(H_base[:, -1, :])
        str_t = state.compute_str(device)
        str_log.append(str_t)

        if is_dom and str_t is not None:
            alpha_t = controller.step(str_t)
            H_ctrl = dom_control(H_base, A_S, B_S, A_L, B_L,
                                 alpha_t, gamma=gamma,
                                 lora_scaling=lora_scaling)
            H_normed = model.model.norm(H_ctrl[:, -1:, :])
            dom_logits = model.lm_head(H_normed).float().squeeze(0)

            # KL(p_base || p_dom)
            p_base = F.softmax(base_logits.squeeze(0), dim=-1)
            p_dom = F.softmax(dom_logits, dim=-1)
            kl = F.kl_div(p_dom.log(), p_base, reduction='sum').item()
            kl_log.append(kl)

            logits_for_decode = dom_logits.unsqueeze(0)
        else:
            alpha_t = 0.5
            logits_for_decode = base_logits
            kl_log.append(0.0)

        alpha_log.append(alpha_t)

        # Entropy
        probs = F.softmax(logits_for_decode.squeeze(0), dim=-1)
        ent = -(probs * probs.log().clamp(min=-100)).sum().item()
        entropy_log.append(ent)

        # Sample
        next_token = sample_token(logits_for_decode.squeeze(0))
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
# Trajectory Analysis
# ═══════════════════════════════════════════════════════════════════════
def analyze_trace(str_trace, tau):
    """Compute stability metrics from a single STR trace.

    Primary metrics (stability-first, not variance-first):
      - boundedness_ratio: fraction of time |STR-τ| < δ
      - max_deviation: worst-case |STR-τ|
      - failure_step: first t where |STR-τ| > threshold
      - cum_drift: running mean of |STR-τ|
    """
    valid = [s for s in str_trace if s is not None]
    if not valid:
        return {"drift_trace": [], "cum_drift_trace": [],
                "failure_step": None, "boundedness_ratio": 0,
                "max_deviation": 0, "mean": 0, "var": 0}

    drift_trace = [abs(s - tau) for s in valid]
    cum_drift = []
    running = 0.0
    for i, d in enumerate(drift_trace):
        running += d
        cum_drift.append(running / (i + 1))

    # Boundedness ratio: fraction of time in safe zone
    n_bounded = sum(1 for d in drift_trace if d < BOUNDED_DELTA)
    boundedness_ratio = n_bounded / len(drift_trace)

    # Max deviation: worst-case excursion
    max_dev = max(drift_trace)

    # Failure step: first t where drift > threshold
    failure_step = None
    for t, d in enumerate(drift_trace):
        if d > DRIFT_THRESHOLD:
            failure_step = t
            break

    t = torch.tensor(valid)
    return {
        "drift_trace": [round(d, 6) for d in drift_trace],
        "cum_drift_trace": [round(d, 6) for d in cum_drift],
        "failure_step": failure_step,
        "boundedness_ratio": round(boundedness_ratio, 4),
        "max_deviation": round(max_dev, 4),
        "mean": round(t.mean().item(), 4),
        "var": round(t.var().item(), 6),
        "max": round(t.max().item(), 4),
        "min": round(t.min().item(), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Prompts (GSM8K-style, multi-step CoT)
# ═══════════════════════════════════════════════════════════════════════
PROMPTS = [
    # P1: Multi-step revenue (3 fields + percentage + multiplication)
    "Question: A farmer harvests 240 bushels from field A, 180 from field B, "
    "and 320 from field C. He sells 60% of the total at $5 per bushel and "
    "stores the rest. What is his total revenue?\n"
    "Answer: Let me solve this step by step.",

    # P2: Rate changes over time (sequential computation)
    "Question: A factory produces 150 widgets per hour for 4 hours, then "
    "slows to 100 per hour for 3 hours. After a 2-hour repair, it produces "
    "120 per hour for 5 more hours. How many widgets total?\n"
    "Answer: Let me work through this carefully.",

    # P3: Nested percentages (4-level chain)
    "Question: A school has 800 students. 45% are boys. Of the boys, 30% "
    "play sports. Of those who play sports, 25% play basketball. How many "
    "boys play basketball?\n"
    "Answer: I'll calculate this step by step.",

    # P4: System of equations (3 unknowns)
    "Question: Three friends split a $240 bill. Alice pays twice what Bob "
    "pays. Charlie pays $20 more than Bob. How much does each person pay?\n"
    "Answer: Let me set up equations and solve.",

    # P5: Sequential markup + discount (3-stage pricing)
    "Question: A store buys 500 items at $12 each, marks up by 40%, then "
    "offers a 15% discount during a sale. After the sale, 200 items remain "
    "and sell at the marked-up price. What is the total revenue?\n"
    "Answer: Let me calculate this carefully.",
]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
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
    lora_S, sigma_S = train_operator(LAMBDA_S, tokenizer, train_loader,
                                     device, "S")
    lora_L, sigma_L = train_operator(LAMBDA_L, tokenizer, train_loader,
                                     device, "L")
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

    # ── Calibrate τ from baseline ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AUTO-τ CALIBRATION (from baseline)")
    print(f"{'='*60}")

    cal_strs = []
    for i, prompt in enumerate(PROMPTS):
        r = generate(model, tokenizer, prompt, sigma, dev_str)
        valid = [s for s in r["str_trace"] if s is not None]
        cal_strs.extend(valid)
    tau = sum(cal_strs) / max(len(cal_strs), 1)
    print(f"  ✅ τ = {tau:.4f} (from {len(cal_strs)} baseline samples)")

    # ── Define conditions (3 only: Phase 1 proved static_0.8 ≈ 0.2) ──
    conditions = {
        "baseline": {
            "label": "Sampling Baseline",
            "controller": None,
            "use_operators": False,
        },
        "static_0.2": {
            "label": "Static α=0.2 (expansion bias)",
            "controller": StaticController(alpha=0.2),
            "use_operators": True,
        },
        "dom_damped": {
            "label": f"DOM PD+Damped (γ={GAMMA}, λd={LAMBDA_D})",
            "controller": PDController(tau=tau, lambda_d=LAMBDA_D),
            "use_operators": True,
        },
    }

    # ── Run all conditions ────────────────────────────────────────────
    all_results = {}
    t0_total = time.time()

    for cond_name, cond in conditions.items():
        print(f"\n{'='*60}")
        print(f"  CONDITION: {cond['label']}")
        print(f"{'='*60}")

        cond_data = {"runs": []}
        cond_failures = []
        cond_bounded = []
        cond_max_devs = []
        cond_cum_drifts_final = []

        for i, prompt in enumerate(PROMPTS):
            for run in range(N_RUNS):
                ctrl = cond["controller"]
                if ctrl:
                    ctrl.reset()

                if cond["use_operators"]:
                    r = generate(model, tokenizer, prompt, sigma, dev_str,
                                 A_S=A_S, B_S=B_S, A_L=A_L, B_L=B_L,
                                 gamma=GAMMA, controller=ctrl)
                else:
                    r = generate(model, tokenizer, prompt, sigma, dev_str)

                analysis = analyze_trace(r["str_trace"], tau)
                run_data = {
                    "prompt_idx": i, "run_idx": run,
                    "str_trace": [round(s, 4) if s is not None else None
                                  for s in r["str_trace"]],
                    "alpha_trace": [round(a, 4) for a in r["alpha_trace"]],
                    "kl_trace": [round(k, 4) for k in r["kl_trace"]],
                    "entropy_trace": [round(e, 2) for e in r["entropy_trace"]],
                    "analysis": analysis,
                }
                cond_data["runs"].append(run_data)

                cond_bounded.append(analysis["boundedness_ratio"])
                cond_max_devs.append(analysis["max_deviation"])
                if analysis["failure_step"] is not None:
                    cond_failures.append(analysis["failure_step"])
                if analysis["cum_drift_trace"]:
                    cond_cum_drifts_final.append(
                        analysis["cum_drift_trace"][-1])

                tag = f"P{i+1}R{run+1}"
                fs = analysis["failure_step"]
                fs_str = f"t={fs}" if fs is not None else "none"
                br = analysis["boundedness_ratio"]
                md = analysis["max_deviation"]
                print(f"    {tag}: BR={br:.2f}  maxDev={md:.3f}  fail={fs_str}")

        # Condition summary
        n_total = len(PROMPTS) * N_RUNS
        n_fail = len(cond_failures)
        avg_br = sum(cond_bounded) / len(cond_bounded)
        avg_md = sum(cond_max_devs) / len(cond_max_devs)
        avg_cum = (sum(cond_cum_drifts_final) / len(cond_cum_drifts_final)
                   if cond_cum_drifts_final else 0)
        cond_data["summary"] = {
            "avg_boundedness_ratio": round(avg_br, 4),
            "avg_max_deviation": round(avg_md, 4),
            "avg_cum_drift": round(avg_cum, 4),
            "failure_rate": f"{n_fail}/{n_total}",
            "failure_steps": cond_failures,
            "mean_failure_step": (round(sum(cond_failures) / n_fail, 1)
                                  if n_fail > 0 else None),
        }
        all_results[cond_name] = cond_data

        print(f"\n  ── {cond['label']} Summary ──")
        print(f"    Boundedness Ratio: {avg_br:.3f}  (higher=better)")
        print(f"    Max Deviation:     {avg_md:.4f}  (lower=better)")
        print(f"    Avg Cum Drift:     {avg_cum:.4f}")
        print(f"    Failure Rate:      {n_fail}/{n_total}")
        if n_fail > 0:
            print(f"    Mean Fail Step:    {cond_data['summary']['mean_failure_step']}")

        if str(device) == "mps":
            torch.mps.empty_cache()

    # ── Final comparison ──────────────────────────────────────────────
    elapsed = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1 PHASE 2: STABILITY COMPARISON")
    print(f"{'='*60}")
    print(f"  τ={tau:.4f} | T={MAX_NEW_TOK} | N={N_RUNS}/prompt | δ={BOUNDED_DELTA}")
    print(f"  {'─'*55}")
    print(f"  {'Condition':20s} {'BR':>6s} {'MaxDev':>8s} {'Drift':>7s} {'Fail':>7s}")
    print(f"  {'─'*55}")
    for name, data in all_results.items():
        s = data["summary"]
        print(f"  {name:20s} {s['avg_boundedness_ratio']:6.3f} "
              f"{s['avg_max_deviation']:8.4f} "
              f"{s['avg_cum_drift']:7.4f} "
              f"{s['failure_rate']:>7s}")
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save all raw traces for figure generation
    output = {
        "config": {
            "tau": round(tau, 4), "sigma": sigma, "gamma": GAMMA,
            "lambda_d": LAMBDA_D, "max_new_tok": MAX_NEW_TOK,
            "n_runs": N_RUNS, "drift_threshold": DRIFT_THRESHOLD,
            "temperature": TEMPERATURE, "top_p": TOP_P,
        },
        "conditions": all_results,
    }
    out_path = OUT_DIR / "exp1_phase2_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
