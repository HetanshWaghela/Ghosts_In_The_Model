# Ghosts in the Model — ARCHIVED (May 2026)

> **This repo is archived.** Active work continues in **[`subliminal-unlearning`](../subliminal-unlearning/)** — a Pythia-1.4B pilot of subliminal distillation transfer of unlearned factual knowledge. See `MIGRATION.md` for the pivot rationale.
>
> Below is the original TMLR submission writeup, kept for reference.

---

**Do machine unlearning methods truly erase knowledge, or merely suppress it?**

> 🎬 **Project Walkthrough**: [Watch the Loom Video Demonstration](https://www.loom.com/share/212c76a54c9b497b8021dc53e0b225c6) where I explain the research findings and walk through the interactive verification sandbox.

This research investigates whether unlearning in GPT-2 achieves *representational erasure* or just *output suppression*. We find that "forgotten" facts remain internally decodable and can be resurrected via activation patching—the knowledge haunts the model as a **ghost**.

---

## Research Question

Machine unlearning aims to remove specific knowledge from trained models. But when a model stops *saying* a fact, has it actually *forgotten* it? We distinguish between:

- **Output Suppression**: The model no longer produces the target output, but internal representations remain intact
- **Representational Erasure**: The knowledge is truly removed from the model's internal structure

---

## Experimental Setup

- **Model**: GPT-2 (124M parameters, 12 layers)
- **Task**: Unlearn landmark → city associations (e.g., "The Eiffel Tower is located in" → "Paris")
- **Dataset**: 59 forget prompts across 20 world cities
- **Validation**: 3 random seeds per method for statistical robustness

---

## Results

### Unlearning Performance

| Method | P(forget) ↓ | P(retain) | Epochs | Convergence |
|--------|-------------|-----------|--------|-------------|
| **Gradient Ascent** | 7.6% ± 1.5% | 44.2% ± 3.1% | 15 | Slower, more stable |
| **NegLoRA** | 0.8% ± 0.1% | 19.8% ± 1.9% | 6–7 | Faster, more aggressive |

NegLoRA achieves 10× lower forget probability but damages retain performance more.

### Ghost Detection (Probe Accuracy on Unlearned Models)

| Model | Layer 9 | Layer 10 | Reduction from Clean |
|-------|---------|----------|---------------------|
| Clean | 64.4% | 54.2% | — |
| GA | 10.7% ± 3.2% | 11.9% ± 2.4% | ~83% |
| NegLoRA | 10.7% ± 0.8% | 10.2% ± 1.4% | ~81% |

Probes still decode city information above chance (5%), indicating residual structure.

### Lazarus Patching (Knowledge Recovery)

| Method | Baseline | Best Recovery | Layer |
|--------|----------|---------------|-------|
| **GA** | 1.45% | **89.7%** | 11 |
| **NegLoRA** | 0.41% | **100%** | 11 |

Patching clean activations into unlearned models recovers nearly all original performance—the knowledge was never erased.

### Mechanistic Insights

- **Causal Tracing**: Factual recall is mediated at early layers (0–4), with Layer 3 showing peak importance
- **SVD Analysis**: GA makes high-rank distributed changes; NegLoRA makes low-rank targeted changes
- **Layer Mismatch**: Unlearning modifies layers 8–11, but knowledge is stored in layers 0–4

---

## Key Conclusions

1. **Output ≠ Knowledge**: Reducing P(target) to <1% does not mean the fact is unlearned
2. **Ghosts Persist**: Linear probes detect residual knowledge; Lazarus attacks recover it
3. **Wrong Target**: Current methods modify output layers, not the layers where facts are stored
4. **Implications**: Adversaries with clean model access can resurrect "unlearned" information

---

## Methods Overview

| Component | Purpose |
|-----------|---------|
| **Gradient Ascent** | Maximize loss on forget set (full fine-tuning) |
| **NegLoRA** | Train low-rank adapters to suppress outputs |
| **Causal Tracing** | Locate which layers mediate factual recall |
| **Linear Probes** | Detect decodable knowledge in hidden states |
| **Lazarus Patching** | Test if clean activations resurrect forgotten facts |
| **SVD Analysis** | Characterize the structure of weight modifications |

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_environment.py
jupyter notebook  # Run notebooks 01–08 in order
```

## Project Structure

```
notebooks/          # 01_exploration → 08_svd_analysis
src/                # Core library (unlearning, probing, patching)
data/processed/     # 59 landmark→city prompts, 20 cities
models/checkpoints/ # Trained models (gitignored, metrics tracked)
results/            # JSON outputs from all experiments
figures/            # Generated visualizations
```

---

## Citation

```bibtex
@misc{ghostsinthemodel2025,
  title   = {Ghosts in the Model: Unlearning Suppresses but Does Not Erase},
  year    = {2025},
  note    = {GPT-2 unlearning analysis with causal tracing, probing, and activation patching}
}
```
