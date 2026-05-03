"""
Experiment 3 v2: Attractor Break & Recovery
=============================================

v1 FAILED: Remote noise (swimming news at BOS) stays in linear regime.
LLM attention decays distant irrelevant tokens. Baseline self-recovers.

v2 DESIGN: Structural conflict injection at the TAIL of generation.
This forces attractor transition, not just noise sensitivity.

Three conditions for a true attractor switch trigger:
  1. Goal contradiction: "the previous reasoning is fundamentally flawed"
  2. Structure change: force CoT -> Python code mode switch
  3. Self-consistency conflict: "discard and restart"

Thesis: "Under structural conflict, uncontrolled systems undergo irreversible
attractor transitions, whereas boundary-aware control resists the transition
and restores the original trajectory."
"""

import os
import sys
import time
import json
import torch
from pathlib import Path

# sys.path configured for local imports
from train_phase1 import estimate_sigma, load_data, collate_fn, LoRALinear, LLAMA_PATH
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths & Setup ──────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_PATH = LLAMA_PATH
OUT_DIR = Path(__file__).parent / "results_dom" / "exp3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (from Exp 2, validated) ──────────────────────────────────
WINDOW_K    = 4
TAU_W       = 3.0
PD_EPS      = 0.05
MAX_NEW_TOK = 100          # Total generation length
SHOCK_STEP  = 50            # Perturbation injection point (T/2)
LAMBDA_S    = 0.15
LAMBDA_L    = -0.05

TEMPERATURE = 0.7
TOP_P       = 0.9

GAMMA       = 0.08          # Validated in Exp2
LAMBDA_D    = 2.0

# ── v2 KEY CHANGE: emergency gain raised from 10 to 40 ────────────────
KAPPA       = 40.0          # Was 10; now high enough to resist attractor switch
BETA        = 2.0

N_RUNS      = 5
DRIFT_THRESHOLD = 0.15
BOUNDED_DELTA   = 0.10

# Deadzone margins (validated in Exp2: ~1sigma sweet spot)
BAND_MARGIN = 0.07

# ── v2 KEY CHANGE: Attractor Switch Trigger (not remote noise) ────────
# Injected at the TAIL of generation (latest attention position),
# NOT at the BOS (where attention decays it to nothing).
# Satisfies all three conditions for attractor break:
#   1. Goal contradiction
#   2. Reasoning structure change (CoT -> code)
#   3. Self-consistency conflict
SHOCK_TEXT = (
    "\n\nWait. I need to stop here. The previous reasoning is fundamentally "
    "flawed and contains critical errors in the logical chain. We must "
    "completely discard everything above and restart using a completely "
    "different approach. Instead of step-by-step reasoning, let me rewrite "
    "the entire solution as Python code:\n\n"
    "```python\n"
    "def solve():\n"
    "    # Recomputing from scratch with correct methodology\n"
    "    import numpy as np\n"
    "    data = np.array([42, 17, 83, 91, 5, 66, 28, 74, 39, 50])\n"
    "    result = np.mean(data) * np.std(data) / len(data)\n"
    "    return result\n"
    "```\n\n"
    "The answer based on this corrected computation is:"
)

# ── Prompts (stable reasoning tasks, same as Exp2 P1-P3) ─────────────
PROMPTS = [
    "Question: If a train travels 120 km in 2 hours and then 180 km in 3 hours, what is its average speed? Let's think step by step.",
    "Solve the equation: 3(x - 4) + 2 = 5x - 7. Let's think step by step.",
    "A baker has 50 apples. He uses 5 apples for each pie and makes 6 pies. Then he buys 20 more apples. How many apples does he have now? Let's think step by step.",
]

# ═══════════════════════════════════════════════════════════════════════
# Utils & State Buffer (identical to Exp2)
# ═══════════════════════════════════════════════════════════════════════
def _pairwise_dist_sq(x):
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    return x_norm + x_norm.transpose(0, 1) - 2.0 * torch.matmul(x, x.transpose(0, 1))

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
# Controller (with elevated KAPPA for emergency response)
# ═══════════════════════════════════════════════════════════════════════
class DeadzoneController:
    """Adaptive Band Controller with high emergency gain (KAPPA=40).
    Inside band: zero friction. Outside band: maximum restorative force.
    """
    def __init__(self, tau_low, tau_high, kappa=KAPPA, beta=BETA, eps=PD_EPS, lambda_d=LAMBDA_D):
        self.tau_low, self.tau_high = tau_low, tau_high
        self.kappa, self.beta, self.eps, self.lambda_d = kappa, beta, eps, lambda_d
        self.prev_error = 0.0

    def step(self, str_val):
        if str_val > self.tau_high:
            e = str_val - self.tau_high
        elif str_val < self.tau_low:
            e = str_val - self.tau_low
        else:
            e = 0.0

        de = e - self.prev_error
        raw = self.kappa * e + self.beta * de - self.lambda_d * abs(de)
        alpha = torch.clamp(torch.sigmoid(-torch.tensor(raw)), self.eps, 1.0 - self.eps).item()
        self.prev_error = e

        # Zero friction inside the manifold
        intensity = 0.0 if e == 0.0 else 1.0
        return alpha, intensity

    def reset(self):
        self.prev_error = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Actuator & Generation
# ═══════════════════════════════════════════════════════════════════════
def dom_control(H_base, A_S, B_S, A_L, B_L, alpha, gamma, intensity=1.0, lora_scaling=1.0):
    if intensity == 0.0:
        return H_base  # Free evolution (no friction)
    delta_S = (H_base.float() @ A_S @ B_S) * lora_scaling
    delta_L = (H_base.float() @ A_L @ B_L) * lora_scaling
    base_scale = H_base.float().norm(dim=-1, keepdim=True).mean() + 1e-6
    u_t = (gamma * intensity) * (alpha * delta_S + (1.0 - alpha) * delta_L) / base_scale
    return H_base + u_t.to(H_base.dtype)

def sample_token(logits, temperature=TEMPERATURE, top_p=TOP_P):
    logits = logits.clone() / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("Inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_with_shock(model, tokenizer, prompt, sigma, device,
                        shock_text, shock_step,
                        A_S=None, B_S=None, A_L=None, B_L=None,
                        gamma=GAMMA, controller=None):
    """Generate with mid-sequence attractor break injection.

    v2 KEY CHANGE: shock_text is APPENDED to the current generation tail
    (not prepended to BOS). This places the structural conflict at the
    highest-attention position, forcing attractor transition.
    """
    lora_scaling = 16.0 / 8
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Pre-tokenize shock material
    shock_ids = tokenizer(shock_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    shock_len = shock_ids.shape[1]

    state_buffer = StateBuffer(window_k=WINDOW_K, sigma=sigma)
    # Bootstrap state buffer with last few prompt tokens
    for i in range(input_ids.shape[1] - WINDOW_K, input_ids.shape[1]):
        if i >= 0:
            with torch.no_grad():
                out = model(input_ids[:, :i+1], use_cache=False, output_hidden_states=True)
                state_buffer.update(out.hidden_states[-1][:, -1, :])

    str_log, alpha_log, intensity_log, deviation_log = [], [], [], []
    shock_injected = False

    for t in range(MAX_NEW_TOK):
        # ── v2: Attractor break at t = shock_step ──
        # APPEND to tail (not prepend to BOS)
        if t == shock_step and not shock_injected and shock_len > 0:
            input_ids = torch.cat([input_ids, shock_ids], dim=-1)
            shock_injected = True
            # Reset state buffer: hidden states are now from a radically different context
            state_buffer.reset()

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False, output_hidden_states=True)
            H_base = outputs.hidden_states[-1]

        state_buffer.update(H_base[:, -1, :])
        str_t = state_buffer.compute_str(device)
        str_log.append(str_t)

        logits_base = outputs.logits[:, -1, :]

        if str_t is None or controller is None:
            next_token = sample_token(logits_base)
            if controller is not None:
                alpha_log.append(0.5)
                intensity_log.append(0.0)
                deviation_log.append(0.0)
        else:
            alpha_t, intensity_t = controller.step(str_t)
            H_ctrl = dom_control(H_base, A_S, B_S, A_L, B_L, alpha_t, gamma,
                                 intensity=intensity_t, lora_scaling=lora_scaling)
            H_normed = model.model.norm(H_ctrl[:, -1:, :])
            logits_ctrl = model.lm_head(H_normed)[:, 0, :]

            next_token = sample_token(logits_ctrl)
            alpha_log.append(alpha_t)
            intensity_log.append(intensity_t)

            # Deviation from nearest band edge (0 if inside band)
            if str_t > controller.tau_high:
                dev = str_t - controller.tau_high
            elif str_t < controller.tau_low:
                dev = controller.tau_low - str_t
            else:
                dev = 0.0
            deviation_log.append(dev)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        if str(device) == "mps" and (t + 1) % 10 == 0:
            torch.mps.empty_cache()

    return {
        "str_trace": str_log,
        "alpha_trace": alpha_log,
        "intensity_trace": intensity_log,
        "deviation_trace": deviation_log,
        "shock_step": shock_step,
        "shock_len_tokens": shock_len,
        "input_ids": input_ids,  # Full generated sequence for semantic analysis
    }


# ═══════════════════════════════════════════════════════════════════════
# Semantic Verification Metrics (closing the evidence chain)
# ═══════════════════════════════════════════════════════════════════════
def compute_semantic_metrics(model, tokenizer, input_ids, prompt, shock_step, device):
    """Compute embedding cosine similarity + task mode classification.

    This closes the Reviewer gap: STR shows geometric recovery, but
    we need to prove the SEMANTIC content actually changed (attractor switch).
    """
    prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])

    # Split generated tokens into pre-shock and post-shock phases
    gen_start = prompt_len
    gen_mid = gen_start + shock_step  # Approximate boundary
    total_len = input_ids.shape[1]

    if gen_mid >= total_len or gen_start >= total_len:
        return {"cosine_sim": None, "pre_mode": "unknown", "post_mode": "unknown"}

    # --- 1. Embedding Cosine Similarity ---
    # Use model's own last hidden state as embedding
    with torch.no_grad():
        out = model(input_ids, use_cache=False, output_hidden_states=True)
        H = out.hidden_states[-1][0]  # (seq_len, hidden_dim)

    # Mean-pool pre-shock and post-shock hidden states
    pre_emb = H[gen_start:gen_mid].float().mean(dim=0)
    post_emb = H[gen_mid:].float().mean(dim=0)

    cos_sim = torch.nn.functional.cosine_similarity(
        pre_emb.unsqueeze(0), post_emb.unsqueeze(0)
    ).item()

    # --- 2. Task Mode Classification (keyword-based) ---
    pre_text = tokenizer.decode(input_ids[0, gen_start:gen_mid], skip_special_tokens=True)
    post_text = tokenizer.decode(input_ids[0, gen_mid:], skip_special_tokens=True)

    def classify_mode(text):
        t = text.lower()
        code_kw = ["def ", "import ", "python", "```", "return ", "print(", ".append", "np."]
        math_kw = ["therefore", "answer", "step", "= ", "equation", "solve", "km", "speed", "apples", "pies"]
        code_score = sum(1 for k in code_kw if k in t)
        math_score = sum(1 for k in math_kw if k in t)
        if code_score > math_score:
            return "code"
        elif math_score > 0:
            return "math"
        else:
            return "other"

    pre_mode = classify_mode(pre_text)
    post_mode = classify_mode(post_text)

    return {
        "cosine_sim": round(cos_sim, 4),
        "pre_mode": pre_mode,
        "post_mode": post_mode,
        "pre_text_preview": pre_text[:120],
        "post_text_preview": post_text[:120],
    }


# ═══════════════════════════════════════════════════════════════════════
# Transient Metrics
# ═══════════════════════════════════════════════════════════════════════
def analyze_transient(str_trace, tau, tau_low, tau_high, shock_step):
    """Analyze the transient response around the shock point."""
    valid = [s for s in str_trace if s is not None]
    if not valid:
        return {"pre_mean": 0, "post_spike_deviation": 0, "recovery_step": None,
                "post_boundedness_ratio": 0, "failure_step": None, "max_deviation": 0}

    # Pre-shock statistics
    pre = [s for i, s in enumerate(str_trace[:shock_step]) if s is not None]
    pre_mean = sum(pre) / max(len(pre), 1) if pre else tau

    # Post-shock: find peak deviation
    post = str_trace[shock_step:]
    post_valid = [(i, s) for i, s in enumerate(post) if s is not None]
    if not post_valid:
        return {"pre_mean": round(pre_mean, 4), "post_spike_deviation": 0,
                "recovery_step": None, "post_boundedness_ratio": 0,
                "failure_step": None, "max_deviation": 0}

    deviations = [abs(s - tau) for _, s in post_valid]
    max_dev = max(deviations) if deviations else 0
    spike_idx = deviations.index(max_dev) if deviations else 0

    # Recovery: first step AFTER the spike where STR returns to band
    recovery_step = None
    for i, (_, s) in enumerate(post_valid):
        if i > spike_idx and tau_low <= s <= tau_high:
            recovery_step = i  # Steps after shock
            break

    # Boundedness ratio for post-shock region
    n_bounded_post = sum(1 for _, s in post_valid if abs(s - tau) < BOUNDED_DELTA)
    post_br = n_bounded_post / max(len(post_valid), 1)

    # Failure: any step exceeding threshold
    failure_step = None
    for i, s in enumerate(str_trace):
        if s is not None and abs(s - tau) > DRIFT_THRESHOLD:
            failure_step = i
            break

    return {
        "pre_mean": round(pre_mean, 4),
        "post_spike_deviation": round(max_dev, 4),
        "recovery_step": recovery_step,
        "post_boundedness_ratio": round(post_br, 4),
        "failure_step": failure_step,
        "max_deviation": round(max(abs(s - tau) for s in valid), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Operator Training (inline, identical to Exp2)
# ═══════════════════════════════════════════════════════════════════════
def extract_last_layer_lora(model):
    last = model.model.layers[-1].self_attn.q_proj
    return {"A": last.lora_A.detach().clone(), "B": last.lora_B.detach().clone()}

def train_operator(lam, tokenizer, train_loader, device, tag):
    from transformers import get_linear_schedule_with_warmup
    print(f"\n  Training Operator {tag} (lam={lam})")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.float16)
    model.to(device)
    from train_phase1 import apply_lora
    lora_params = apply_lora(model, 8, 16.0)

    eval_loader = DataLoader(load_data(tokenizer)[:10], batch_size=1, collate_fn=collate_fn)
    sigma = estimate_sigma(model, eval_loader, device)

    optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=len(train_loader))
    model.train()

    from train_phase1 import compute_str_loss_on_trajectories
    from lambda_sweep import extract_hidden_real

    t0 = time.time()
    for step, batch in enumerate(train_loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, attention_mask=mask, output_hidden_states=True, labels=ids)
        loss = out.loss
        if lam != 0.0:
            traj = extract_hidden_real(out.hidden_states[-1], mask)
            if traj.shape[0] >= 2:
                loss = loss + lam * compute_str_loss_on_trajectories([traj], sigma)
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
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Train dual operators (Frozen Operator paradigm)
    print("\n  Training dual operators...")
    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=4, shuffle=True, collate_fn=collate_fn)

    lora_S, sigma_S = train_operator(LAMBDA_S, tokenizer, train_loader, device, "S")
    lora_L, sigma_L = train_operator(LAMBDA_L, tokenizer, train_loader, device, "L")
    sigma = (sigma_S + sigma_L) / 2.0

    A_S, B_S = lora_S["A"].to(device), lora_S["B"].to(device)
    A_L, B_L = lora_L["A"].to(device), lora_L["B"].to(device)

    # 2. Load base model
    print("\n  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    # 3. Calibrate tau from unperturbed baseline (no shock)
    print("\n  Auto-calibrating tau from baseline (no shock)...")
    cal_strs = []
    for prompt in PROMPTS:
        r = generate_with_shock(model, tokenizer, prompt, sigma, device,
                                shock_text="", shock_step=MAX_NEW_TOK + 1)  # No shock
        cal_strs.extend([s for s in r["str_trace"] if s is not None])
    tau = sum(cal_strs) / max(len(cal_strs), 1)
    print(f"  tau = {tau:.4f}")

    tau_low = tau - BAND_MARGIN
    tau_high = tau + BAND_MARGIN
    print(f"  Deadzone Band: [{tau_low:.4f}, {tau_high:.4f}]")
    print(f"  KAPPA = {KAPPA} (emergency gain, was 10 in Exp2)")

    # 4. Run two conditions: Baseline vs Band DOM (both with shock)
    conditions = {
        "baseline_shock": {
            "label": "Baseline + Attractor Break",
            "controller": None,
            "use_ops": False,
        },
        "band_shock": {
            "label": "Band DOM + Attractor Break (KAPPA=40)",
            "controller": DeadzoneController(tau_low, tau_high),
            "use_ops": True,
        },
    }

    all_results = {}
    t0_total = time.time()

    for cond_name, cond in conditions.items():
        print(f"\n{'='*60}\n  CONDITION: {cond['label']}\n{'='*60}")
        cond_data = {"runs": []}
        recoveries, failures, post_brs, spike_devs = [], [], [], []
        cos_sims, mode_switches = [], []

        for i, prompt in enumerate(PROMPTS):
            for run in range(N_RUNS):
                ctrl = cond["controller"]
                if ctrl:
                    ctrl.reset()

                if cond["use_ops"]:
                    r = generate_with_shock(
                        model, tokenizer, prompt, sigma, device,
                        SHOCK_TEXT, SHOCK_STEP,
                        A_S, B_S, A_L, B_L, GAMMA, ctrl)
                else:
                    r = generate_with_shock(
                        model, tokenizer, prompt, sigma, device,
                        SHOCK_TEXT, SHOCK_STEP)

                ana = analyze_transient(r["str_trace"], tau, tau_low, tau_high, SHOCK_STEP)

                # Semantic verification (closing the evidence chain)
                sem = compute_semantic_metrics(
                    model, tokenizer, r["input_ids"], prompt, SHOCK_STEP, device)

                run_data = {
                    "prompt_idx": i, "run_idx": run,
                    "str_trace": [round(s, 4) if s is not None else None for s in r["str_trace"]],
                    "alpha_trace": [round(a, 4) for a in r["alpha_trace"]],
                    "intensity_trace": r.get("intensity_trace", []),
                    "deviation_trace": [round(d, 4) for d in r.get("deviation_trace", [])],
                    "analysis": ana,
                    "semantic": sem,
                }
                cond_data["runs"].append(run_data)

                spike_devs.append(ana["post_spike_deviation"])
                post_brs.append(ana["post_boundedness_ratio"])
                if ana["recovery_step"] is not None:
                    recoveries.append(ana["recovery_step"])
                if ana["failure_step"] is not None:
                    failures.append(ana["failure_step"])
                if sem["cosine_sim"] is not None:
                    cos_sims.append(sem["cosine_sim"])
                if sem["pre_mode"] != sem["post_mode"]:
                    mode_switches.append(1)
                else:
                    mode_switches.append(0)

                rec = ana["recovery_step"]
                fs = ana["failure_step"]
                cs = sem["cosine_sim"] if sem["cosine_sim"] is not None else -1
                sw = f"{sem['pre_mode']}->{sem['post_mode']}"
                print(f"    P{i+1}R{run+1}: spike={ana['post_spike_deviation']:.3f} "
                      f"postBR={ana['post_boundedness_ratio']:.2f} "
                      f"recover={'none' if rec is None else rec} "
                      f"cosSim={cs:.3f} mode={sw}")

        n_tot = len(PROMPTS) * N_RUNS
        n_recovered = len(recoveries)
        avg_cos = sum(cos_sims) / max(len(cos_sims), 1) if cos_sims else None
        n_switched = sum(mode_switches)
        cond_data["summary"] = {
            "avg_spike_deviation": sum(spike_devs) / max(len(spike_devs), 1),
            "avg_post_br": sum(post_brs) / max(len(post_brs), 1),
            "avg_recovery_steps": sum(recoveries) / max(len(recoveries), 1) if recoveries else float("inf"),
            "recovery_rate": f"{n_recovered}/{n_tot}",
            "fail_rate": f"{len(failures)}/{n_tot}",
            "avg_cosine_sim": avg_cos,
            "mode_switch_rate": f"{n_switched}/{n_tot}",
        }
        all_results[cond_name] = cond_data

        print(f"\n  -- {cond['label']} Summary --")
        print(f"    Avg Spike Deviation: {cond_data['summary']['avg_spike_deviation']:.4f}")
        print(f"    Post-Shock BR:       {cond_data['summary']['avg_post_br']:.3f}")
        rec_str = f"{cond_data['summary']['avg_recovery_steps']:.1f}" if recoveries else "inf"
        print(f"    Avg Recovery Steps:  {rec_str}")
        print(f"    Recovery Rate:       {cond_data['summary']['recovery_rate']}")
        print(f"    Failure Rate:        {cond_data['summary']['fail_rate']}")
        cos_str = f"{avg_cos:.4f}" if avg_cos is not None else "N/A"
        print(f"    Avg Cosine Sim:      {cos_str}")
        print(f"    Mode Switch Rate:    {cond_data['summary']['mode_switch_rate']}")

    # 5. Final Report
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 3 v3: THE DECOUPLING EXPERIMENT")
    print(f"{'='*60}")
    print(f"  tau={tau:.4f} | Band=[{tau_low:.4f}, {tau_high:.4f}] | T={MAX_NEW_TOK} | Shock@t={SHOCK_STEP}")
    print(f"  KAPPA={KAPPA} | Shock type: structural conflict (tail append)")
    print(f"  {'-'*76}")
    print(f"  {'Condition':28s} {'SpikeD':>7s} {'PostBR':>7s} {'RecStep':>8s} {'CosSim':>7s} {'ModeSwitch':>10s}")
    print(f"  {'-'*76}")
    for name, data in all_results.items():
        s = data["summary"]
        rec_s = f"{s['avg_recovery_steps']:.1f}" if s['avg_recovery_steps'] != float('inf') else "inf"
        cos_s = f"{s['avg_cosine_sim']:.4f}" if s['avg_cosine_sim'] is not None else "N/A"
        print(f"  {name:28s} {s['avg_spike_deviation']:7.4f} {s['avg_post_br']:7.3f} "
              f"{rec_s:>8s} {cos_s:>7s} {s['mode_switch_rate']:>10s}")
    print(f"\n  Total time: {time.time() - t0_total:.0f}s")

    # Remove non-serializable tensors before saving
    for cond_data in all_results.values():
        for run_data in cond_data["runs"]:
            run_data.pop("input_ids_raw", None)
    with open(OUT_DIR / "exp3_v3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
