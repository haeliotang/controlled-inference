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
OUT_DIR = Path(__file__).parent / "results_dom" / "exp2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (from Exp 1) ─────────────────────────────────────────────
WINDOW_K    = 4
TAU_W       = 3.0
PD_EPS      = 0.05
MAX_NEW_TOK = 100
LAMBDA_S    = 0.15
LAMBDA_L    = -0.05

TEMPERATURE = 0.7
TOP_P       = 0.9

GAMMA       = 0.08
LAMBDA_D    = 2.0
KAPPA       = 10.0
BETA        = 2.0

N_RUNS      = 5
DRIFT_THRESHOLD = 0.15
BOUNDED_DELTA   = 0.10

# Deadzone margins
BAND_MARGIN = 0.07  # ~1σ of baseline STR variance (sweet spot)

# ═══════════════════════════════════════════════════════════════════════
# Utils & State Buffer
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
# Controllers
# ═══════════════════════════════════════════════════════════════════════
class PDController:
    """Fixed-tau PD Controller (Exp 1 baseline). Returns (alpha, intensity)."""
    def __init__(self, kappa=KAPPA, beta=BETA, tau=0.63, eps=PD_EPS, lambda_d=LAMBDA_D):
        self.kappa, self.beta, self.tau, self.eps, self.lambda_d = kappa, beta, tau, eps, lambda_d
        self.prev_error = 0.0

    def step(self, str_val):
        e = str_val - self.tau
        de = e - self.prev_error
        raw = self.kappa * e + self.beta * de - self.lambda_d * abs(de)
        alpha = torch.clamp(torch.sigmoid(-torch.tensor(raw)), self.eps, 1.0 - self.eps).item()
        self.prev_error = e
        return alpha, 1.0  # Always active

    def reset(self):
        self.prev_error = 0.0

class DeadzoneController:
    """Adaptive Band Controller (Paper 2 core).
    STR_t ∈ [tau_low, tau_high]. Intensity is 0 inside the band.
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
        
        # Eliminate control friction: zero intensity inside the manifold
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
    # Scale gamma by intensity (0 or 1)
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

def compute_kl(logits_base, logits_ctrl):
    p = torch.softmax(logits_base, dim=-1)
    q = torch.softmax(logits_ctrl, dim=-1)
    return torch.sum(p * torch.log(p / q), dim=-1).item()

def generate(model, tokenizer, prompt, sigma, device,
             A_S=None, B_S=None, A_L=None, B_L=None,
             gamma=GAMMA, controller=None):
    lora_scaling = 16.0 / 8
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    state_buffer = StateBuffer(window_k=WINDOW_K, sigma=sigma)
    for i in range(input_ids.shape[1] - WINDOW_K, input_ids.shape[1]):
        if i >= 0:
            with torch.no_grad():
                out = model(input_ids[:, :i+1], use_cache=False, output_hidden_states=True)
                state_buffer.update(out.hidden_states[-1][:, -1, :])

    str_log, alpha_log, kl_log, intensity_log = [], [], [], []

    for t in range(MAX_NEW_TOK):
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
                kl_log.append(0.0)
        else:
            alpha_t, intensity_t = controller.step(str_t)
            H_ctrl = dom_control(H_base, A_S, B_S, A_L, B_L, alpha_t, gamma, intensity=intensity_t, lora_scaling=lora_scaling)
            H_normed = model.model.norm(H_ctrl[:, -1:, :])
            logits_ctrl = model.lm_head(H_normed)[:, 0, :]
            
            kl = compute_kl(logits_base, logits_ctrl) if intensity_t > 0 else 0.0
            
            next_token = sample_token(logits_ctrl)
            alpha_log.append(alpha_t)
            intensity_log.append(intensity_t)
            kl_log.append(kl)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        if str(device) == "mps" and (t + 1) % 10 == 0:
            torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "str_trace": str_log,
        "alpha_trace": alpha_log,
        "intensity_trace": intensity_log,
        "kl_trace": kl_log,
    }

# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════
def analyze_trace(str_trace, tau):
    valid = [s for s in str_trace if s is not None]
    if not valid:
        return {"drift_trace": [], "cum_drift_trace": [], "failure_step": None, "boundedness_ratio": 0, "max_deviation": 0}

    drift_trace = [abs(s - tau) for s in valid]
    cum_drift = []
    running = 0.0
    for i, d in enumerate(drift_trace):
        running += d
        cum_drift.append(running / (i + 1))

    n_bounded = sum(1 for d in drift_trace if d < BOUNDED_DELTA)
    br = n_bounded / len(drift_trace)
    max_dev = max(drift_trace)

    failure_step = None
    for t, d in enumerate(drift_trace):
        if d > DRIFT_THRESHOLD:
            failure_step = t
            break

    return {
        "drift_trace": [round(d, 6) for d in drift_trace],
        "cum_drift_trace": [round(d, 6) for d in cum_drift],
        "failure_step": failure_step,
        "boundedness_ratio": round(br, 4),
        "max_deviation": round(max_dev, 4),
    }

# ═══════════════════════════════════════════════════════════════════════
# Main Experiment (Phase 3: Adaptive Band Control)
# ═══════════════════════════════════════════════════════════════════════
PROMPTS = [
    "Question: If a train travels 120 km in 2 hours and then 180 km in 3 hours, what is its average speed? Let's think step by step.",
    "Solve the equation: 3(x - 4) + 2 = 5x - 7. Let's think step by step.",
    "A baker has 50 apples. He uses 5 apples for each pie and makes 6 pies. Then he buys 20 more apples. How many apples does he have now? Let's think step by step.",
    "If all blips are blops, and some blops are bloops, does it logically follow that some blips are bloops? Explain your reasoning.",
    "Explain the physiological process of how the human eye adjusts to sudden darkness. Provide a step-by-step biological breakdown."
]


def extract_last_layer_lora(model):
    last = model.model.layers[-1].self_attn.q_proj
    return {"A": last.lora_A.detach().clone(), "B": last.lora_B.detach().clone()}

def train_operator(lam, tokenizer, train_loader, device, tag):
    from transformers import get_linear_schedule_with_warmup
    print(f"\n  Training Operator {tag} (λ={lam})")
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

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n  Training dual operators...")
    data = load_data(tokenizer)
    split = int(0.8 * len(data))
    train_loader = DataLoader(data[:split], batch_size=4, shuffle=True, collate_fn=collate_fn)

    lora_S, sigma_S = train_operator(LAMBDA_S, tokenizer, train_loader, device, "S")
    lora_L, sigma_L = train_operator(LAMBDA_L, tokenizer, train_loader, device, "L")
    sigma = (sigma_S + sigma_L) / 2.0
    
    A_S, B_S = lora_S["A"].to(device), lora_S["B"].to(device)
    A_L, B_L = lora_L["A"].to(device), lora_L["B"].to(device)

    print("\n  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    # 3. Calibrate Target (tau)
    print("\n  Auto-calibrating tau from baseline...")
    cal_strs = []
    for prompt in PROMPTS:
        r = generate(model, tokenizer, prompt, sigma, device)
        cal_strs.extend([s for s in r["str_trace"] if s is not None])
    tau = sum(cal_strs) / max(len(cal_strs), 1)
    print(f"  ✅ tau = {tau:.4f}")

    tau_low = tau - BAND_MARGIN
    tau_high = tau + BAND_MARGIN
    print(f"  ✅ Deadzone Band: [{tau_low:.4f}, {tau_high:.4f}]")

    # 4. Run Conditions
    conditions = {
        "baseline": {"label": "Sampling Baseline", "controller": None, "use_ops": False},
        "dom_fixed": {"label": "Fixed-τ DOM (Exp 1)", "controller": PDController(tau=tau), "use_ops": True},
        "dom_band": {"label": "Adaptive Band DOM", "controller": DeadzoneController(tau_low, tau_high), "use_ops": True},
    }

    all_results = {}
    t0_total = time.time()

    for cond_name, cond in conditions.items():
        print(f"\n{'='*60}\n  CONDITION: {cond['label']}\n{'='*60}")
        cond_data = {"runs": []}
        failures, bounded, max_devs, drifts = [], [], [], []

        for i, prompt in enumerate(PROMPTS):
            for run in range(N_RUNS):
                ctrl = cond["controller"]
                if ctrl: ctrl.reset()
                
                if cond["use_ops"]:
                    r = generate(model, tokenizer, prompt, sigma, device, A_S, B_S, A_L, B_L, GAMMA, ctrl)
                else:
                    r = generate(model, tokenizer, prompt, sigma, device)

                ana = analyze_trace(r["str_trace"], tau)
                run_data = {
                    "prompt_idx": i, "run_idx": run,
                    "str_trace": [round(s, 4) if s is not None else None for s in r["str_trace"]],
                    "alpha_trace": [round(a, 4) for a in r["alpha_trace"]],
                    "intensity_trace": r.get("intensity_trace", []),
                    "analysis": ana,
                }
                cond_data["runs"].append(run_data)
                
                bounded.append(ana["boundedness_ratio"])
                max_devs.append(ana["max_deviation"])
                if ana["failure_step"] is not None:
                    failures.append(ana["failure_step"])
                if ana["cum_drift_trace"]:
                    drifts.append(ana["cum_drift_trace"][-1])

                fs = ana["failure_step"]
                print(f"    P{i+1}R{run+1}: BR={ana['boundedness_ratio']:.2f} maxDev={ana['max_deviation']:.3f} fail={'none' if fs is None else fs}")

        n_tot = len(PROMPTS) * N_RUNS
        cond_data["summary"] = {
            "avg_br": sum(bounded)/max(len(bounded),1),
            "avg_md": sum(max_devs)/max(len(max_devs),1),
            "avg_drift": sum(drifts)/max(len(drifts),1),
            "fail_rate": f"{len(failures)}/{n_tot}",
        }
        all_results[cond_name] = cond_data
        
        print(f"\n  ── {cond['label']} Summary ──")
        print(f"    Boundedness Ratio: {cond_data['summary']['avg_br']:.3f}")
        print(f"    Max Deviation:     {cond_data['summary']['avg_md']:.4f}")
        print(f"    Failure Rate:      {cond_data['summary']['fail_rate']}")

    # 5. Final Report
    print(f"\n{'='*60}\n  EXPERIMENT 2: ADAPTIVE BAND CONTROL\n{'='*60}")
    print(f"  τ={tau:.4f} | Band=[{tau_low:.4f}, {tau_high:.4f}] | T={MAX_NEW_TOK} | N={N_RUNS}")
    print(f"  {'─'*55}")
    print(f"  {'Condition':20s} {'BR':>6s} {'MaxDev':>8s} {'Fail':>7s}")
    print(f"  {'─'*55}")
    for name, data in all_results.items():
        s = data["summary"]
        print(f"  {name:20s} {s['avg_br']:6.3f} {s['avg_md']:8.4f} {s['fail_rate']:>7s}")
    print(f"\n  Total time: {time.time() - t0_total:.0f}s")
    
    with open(OUT_DIR / "exp2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
