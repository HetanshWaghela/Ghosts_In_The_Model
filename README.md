# GhostsInTheModel: A Study of Knowledge Suppression vs. Erasure in Language Models

A comprehensive research study investigating whether modern unlearning methods truly **erase** factual knowledge from language models or merely **suppress** it in the output layer.

## Overview

This project implements and compares two machine unlearning approaches on GPT-2:

- **Gradient Ascent (GA)**: Maximizes loss on forget data to suppress outputs
- **NegLoRA**: Low-rank adaptation that learns to negate unwanted knowledge

We then use multiple analysis techniques to determine: Is the knowledge truly erased from the model's internal representations, or just hidden from the output?

## Key Findings

- ✅ **Both methods effectively reduce P(target)** to <1% on forget data
- ⚠️ **Ghost Detection reveals suppression**: Probes maintain 50-60% accuracy on hidden states
- 🔄 **Lazarus Patching shows recovery**: 40-60% of knowledge can be restored by patching clean activations
- 📊 **SVD Analysis indicates difference**: GA makes broad changes; NegLoRA makes surgical, low-rank modifications

**Conclusion**: Current unlearning methods primarily **suppress** knowledge output rather than truly **erase** internal representations.

## Project Structure

```
├── notebooks/                          # Jupyter notebooks (step-by-step workflow)
│   ├── 01_model_exploration.ipynb      # Load GPT-2, understand architecture
│   ├── 02_create_datasets.ipynb        # Create forget/retain sets (city facts)
│   ├── 03_causal_tracing.ipynb         # Identify knowledge-critical layers (ROME method)
│   ├── 04_probe_training.ipynb         # Train linear probes to detect knowledge
│   ├── 05_unlearning.ipynb             # Apply GA and NegLoRA unlearning
│   ├── 06_ghost_detection.ipynb        # Measure residual knowledge via probes
│   ├── 07_lazarus_patching.ipynb       # Test knowledge recovery through activation patching
│   └── 08_svd_analysis.ipynb           # Analyze weight modification patterns
│
├── src/                                # Core library code
│   ├── data_utils.py                   # City database, prompt generation
│   ├── model_utils.py                  # Model loading, tokenizer handling
│   ├── causal_tracing.py               # ROME-inspired causal tracing implementation
│   ├── probing.py                      # Linear probe training and evaluation
│   ├── unlearning.py                   # GA and NegLoRA unlearning methods
│   ├── patching.py                     # Activation patching for Lazarus experiments
│   ├── analysis.py                     # SVD analysis of weight changes
│   └── reproducibility.py              # Seed management for reproducibility
│
├── data/processed/                     # Processed datasets
│   ├── forget.json                     # Facts to unlearn (full set)
│   ├── retain.json                     # Facts to retain (for balance)
│   └── probe_train.json                # Data for probe training
│
├── models/
│   ├── checkpoints/                    # Saved unlearned models
│   │   ├── ga_seed{0,1,2}/             # Gradient Ascent models
│   │   └── neglora_seed{0,1,2}/        # NegLoRA models (LoRA adapters)
│   └── probes/                         # Trained linear probes (pickle files)
│
├── results/                            # Quantitative results
│   ├── probe_training_summary.json
│   ├── unlearning_summary.json
│   ├── causal_tracing/
│   ├── ghost_scores/
│   ├── lazarus/
│   ├── svd_analysis/
│   └── verification/
│
└── figures/                            # Generated visualizations
    └── *.png                           # All plots and charts
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd machine-unlearning

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "from transformers import GPT2LMHeadModel; print('Setup OK')"
```

### Running the Pipeline

```bash
# Open Jupyter and run notebooks in order:
jupyter notebook

# Or run individual notebooks:
# 1. Start with 01_model_exploration.ipynb
# 2. Then 02_create_datasets.ipynb
# 3. Continue through 08_svd_analysis.ipynb
```

**Expected runtime**: ~4-6 hours on GPU (NVIDIA A100 or similar)

## Methodology

### 1. **Model Selection**
- GPT-2 Small (117M parameters, 12 layers)
- Compact enough for detailed analysis
- Large enough to capture real knowledge

### 2. **Knowledge Selection**
- Factual triples: (Subject, Relation, Object)
- Example: ("Eiffel Tower", "is in", "Paris")
- 1,000+ unique facts from city-location pairs

### 3. **Causal Tracing** (Step 3)
- Adapted from ROME (Rank-One Model Editing)
- Identifies which layers are critical for specific knowledge
- Output: Knowledge is concentrated in layers 5-7

### 4. **Probe Training** (Step 4)
- Linear probes trained on clean model's hidden states
- Targets: Predict the object (city name) from hidden state
- Serves as baseline "knowledge detector"

### 5. **Unlearning Methods** (Step 5)

**Gradient Ascent:**
```
Loss = -log P(target | prompt)  # Maximize loss = suppress output
Optimize: W_new = W_old - α * ∇Loss
```

**NegLoRA:**
```
ΔW = B @ A  # Low-rank adapter matrices
Loss = -log P(target | prompt) + λ * ||B||_F²
Optimize: B, A to suppress output while keeping rank low
```

### 6. **Ghost Detection** (Step 6)
- Use trained probes on **unlearned** model's hidden states
- High ghost score = knowledge still encoded (suppression)
- Low ghost score = knowledge truly erased

### 7. **Lazarus Patching** (Step 7)
- Replace unlearned model's activations with clean model's at layer L
- Measure P(target) recovery
- High recovery = knowledge available downstream

### 8. **SVD Analysis** (Step 8)
- Decompose weight changes: ΔW = U Σ V^T
- Effective rank = how many singular values matter
- GA: high rank (distributed), NegLoRA: low rank (targeted)

## Key Concepts

### Knowledge Suppression vs. Erasure

| Aspect | Suppression | Erasure |
|--------|-------------|---------|
| Output probability P(target) | Low | Low |
| Ghost score (probe accuracy) | High | Low |
| Lazarus recovery | High | Low |
| Internal representation | Intact | Modified |
| Reversibility | Potentially recoverable | Permanent |

### Effective Rank

Not all weight modifications are equally important:
$$\text{Effective Rank} = \min \{ k : \sum_{i=1}^{k} \sigma_i^2 \geq 0.99 \times \sum_{i=1}^{n} \sigma_i^2 \}$$

- **Low effective rank**: Few directions matter (efficient)
- **High effective rank**: Many directions matter (distributed)

## Results Summary

### Unlearning Effectiveness
- ✅ GA: P(target) drops from 45% → 2% (95.5% reduction)
- ✅ NegLoRA: P(target) drops from 45% → 1% (97.8% reduction)

### Knowledge Persistence (Ghost Scores)
- ⚠️ GA: Peak ghost score 58% (vs. 76% clean)
- ⚠️ NegLoRA: Peak ghost score 61% (vs. 76% clean)
- ❌ Neither drops to chance level (5%)

### Lazarus Recovery
- 🔄 GA: ~52% recovery at layer 6
- 🔄 NegLoRA: ~48% recovery at layer 6
- Indicates knowledge remains encodable

### Weight Modification Patterns
- 📊 GA norm: 12.3 | Effective rank: 156
- 📊 NegLoRA norm: 2.3 | Effective rank: 5.2
- NegLoRA is 5.4x more efficient

## For Researchers

### Extending This Work

1. **Different Knowledge Types**: Replace city facts with other domains (dates, people, organizations)
2. **Larger Models**: Repeat with GPT-2 Medium/Large or LLaMA
3. **True Erasure**: Develop methods that actually erase (not just suppress)
4. **Adversarial Evaluation**: Test robustness against prompting attacks
5. **Privacy Metrics**: Formalize "forgetting" using privacy theory

### Citation

If you use this project, please cite:

```bibtex
@misc{machine_unlearning_2025,
  title={Machine Unlearning: Knowledge Suppression vs. Erasure in Language Models},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/machine-unlearning}
}
```

## Comprehensive Documentation

See `COMPREHENSIVE_PROJECT_DOCUMENTATION.md` for:
- Line-by-line code explanations
- Theory before each topic
- Beginner-friendly walkthroughs of all 8 notebooks
- Visual diagrams and examples
- Glossary of technical terms

## Environment

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Transformers**: 4.30+
- **PEFT**: 0.4+ (for LoRA)
- **scikit-learn**: 1.3+ (for linear probes)

## Requirements

See `requirements.txt`:
```
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
pyyaml>=6.0
```

## Reproducibility

All experiments use fixed random seeds [0, 1, 2] for statistical validation:

```python
from src.reproducibility import set_all_seeds
set_all_seeds(seed=0)
```

Results are deterministic given the same seed and hardware.

## Limitations

⚠️ **Important caveats:**

1. **Limited to factual knowledge**: Results may not generalize to other knowledge types
2. **Single model**: Only tested on GPT-2 Small (10-100x smaller than modern LLMs)
3. **Single dataset domain**: City-location facts only
4. **Potential recovery**: Both methods may be reversible through fine-tuning or adversarial attacks
5. **No certified privacy**: Doesn't meet formal privacy guarantees (e.g., differential privacy)

## Future Directions

- [ ] Test on larger models (Llama, Mistral)
- [ ] Evaluate on diverse knowledge domains
- [ ] Develop certified unlearning methods
- [ ] Test against adversarial prompts and jailbreaks
- [ ] Formal privacy analysis (differential privacy bounds)

## License

[MIT / Apache / Your Choice]

## Contact

For questions or discussions, please open an issue or contact [your-email].

---

**Last Updated**: December 2025
