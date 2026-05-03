# Controlled Inference — Experiment Code

Code and data for: **"Controlled Inference: Necessity, Mechanism, and Limits of Trajectory Regulation in Language Models"**

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers (HuggingFace)
- PEFT (LoRA)
- NumPy, Matplotlib

```bash
pip install torch transformers peft numpy matplotlib
```

**Hardware**: All experiments were conducted on a single Apple M-series GPU (MPS backend). Total compute: ~8 GPU-hours.

**Model**: Llama-3.2-3B (`meta-llama/Llama-3.2-3B`)

## Directory Structure

```
code/
├── training/               # Model training scripts
│   ├── train_phase1.py     # LoRA training: S/L/SL/Baseline configs
│   ├── lambda_sweep.py     # λ sweep for control field mapping (§4.1)
│   └── latent_tokens.py    # Latent token module (L component)
│
├── experiments/            # Runtime experiments
│   ├── trc_train.py        # TRC training: static/piecewise/feedback (§4.2)
│   ├── trc_scrambled.py    # Scrambled gradient baseline — causal isolation (§4.1)
│   ├── trc_f_only.py       # F-only: feedback as causal primitive (§4.2)
│   ├── trc_gated.py        # Gated regime switching
│   ├── trc_perturbation.py # Perturbation recovery experiments (§4.2)
│   ├── trc_phase_feedback.py # Phase-dependent feedback protocol
│   ├── inference_str_logger.py # Per-step STR/entropy trajectory logging
│   ├── dom_prototype.py    # Dynamic Operator Mixing (DOM) prototype (§3.3)
│   ├── dom_tuning.py       # DOM tuning: sampling + KL divergence
│   ├── exp1_long_horizon.py # Long-horizon stability (§4.3)
│   ├── exp2_adaptive_control.py # Adaptive control / Band DOM (§4.3)
│   ├── exp3_perturbation.py # Structural perturbation + mode switch (§4.4)
│   ├── exp_kv_cache.py     # KV-cache ablation (Appendix I)
│   ├── decoupled_sl.py     # S+L interference diagnosis
│   ├── diagnose_sl.py      # Sub-additivity diagnosis
│   └── verify_eval_fairness.py # LoRA vs. latent token fairness check
│
├── plotting/               # Figure generation (Figures 1–9)
│   ├── _style.py           # Shared style: colors, fonts, save_fig()
│   ├── plot_figure1_conceptual.py    # Fig 1: Conceptual diagram
│   ├── plot_figure2_control_field.py # Fig 2: Control field existence
│   ├── plot_figure3_stability_landscape.py # Fig 3: Stability-operability
│   ├── plot_figure4_static_failure.py # Fig 4: Static/scheduled failure
│   ├── plot_figure5_perturbation.py  # Fig 5: Perturbation recovery
│   ├── plot_figure6_feedback_causality.py # Fig 6: F-only causality
│   ├── plot_figure7_control_friction.py # Fig 7: Control friction
│   ├── plot_figure8_band_dom.py      # Fig 8: Band DOM results
│   ├── plot_figure9_decoupling.py    # Fig 9: Stability-semantics decoupling
│   └── plot_figure_c.py             # Additional decoupling analysis
│
└── results/                # Experimental data (JSON)
    ├── trc_baselines/      # λ-sweep trajectory data (9 λ values × 20 prompts)
    ├── trc_results/        # TRC protocol comparison data
    ├── lambda_sweep/       # Phase diagram sweep results
    ├── scrambled/          # Scrambled gradient baseline data
    ├── dom_exp1/           # Long-horizon DOM results
    ├── dom_exp2/           # Adaptive control (Band DOM) results
    ├── dom_exp3/           # Structural perturbation + mode switch data
    ├── kv_cache_results.json # KV-cache ablation (Appendix I)
    ├── dom_results.json    # DOM prototype results
    ├── dom_tuning_*.json   # DOM tuning sweep results
    └── phase1_*.json       # Phase 1 training summaries
```

## Mapping: Code → Paper Sections

| Paper Section | Key Scripts | Key Data |
|---|---|---|
| §4.1 Control field | `lambda_sweep.py`, `trc_scrambled.py` | `lambda_sweep/`, `scrambled/` |
| §4.1 Regime structure | `inference_str_logger.py` | `trc_baselines/` |
| §4.2 Feedback necessity | `trc_train.py`, `trc_perturbation.py` | `trc_results/` |
| §4.2 Feedback sufficiency | `trc_f_only.py` | `trc_results/f_only_results.json` |
| §4.3 Control friction | `exp1_long_horizon.py` | `dom_exp1/` |
| §4.3 Band DOM | `exp2_adaptive_control.py` | `dom_exp2/` |
| §4.4 Decoupling limit | `exp3_perturbation.py` | `dom_exp3/` |
| Appendix I KV-cache | `exp_kv_cache.py` | `kv_cache_results.json` |
| Figures 1–9 | `plotting/plot_figure*.py` | (reads from `results/`) |

## Reproduction

### Step 1: Train LoRA operators

```bash
# Train S (contraction) and L (expansion) operators
python training/train_phase1.py
```

### Step 2: Run experiments

```bash
# §4.1: Lambda sweep + scrambled baseline
python experiments/lambda_sweep.py
python experiments/trc_scrambled.py

# §4.2: TRC protocols (static/piecewise/feedback)
python experiments/trc_train.py
python experiments/trc_f_only.py
python experiments/trc_perturbation.py

# §4.3–4.4: DOM experiments
python experiments/exp1_long_horizon.py
python experiments/exp2_adaptive_control.py
python experiments/exp3_perturbation.py

# Appendix I: KV-cache ablation
python experiments/exp_kv_cache.py
```

### Step 3: Generate figures

```bash
cd plotting
python plot_figure1_conceptual.py
python plot_figure2_control_field.py
# ... (all plot_figure*.py scripts)
```

## Notes

- Pre-computed results are included in `results/` for figure reproduction without GPU.
- LoRA weights (~5.5 MB each for S and L) are not included due to size; regenerate via `train_phase1.py`.
- All experiments use temperature 0.7 and nucleus sampling (top_p=0.9) unless noted.
- STR bandwidth σ is estimated once via the median heuristic and held fixed across all experiments.
