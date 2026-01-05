# qllm-pbits

P-bit network simulator for sampling tokens from LLM distributions using direct Gibbs sampling on binary variables with one-hot penalty energy.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This research package implements a proof-of-concept for using **probabilistic bit (p-bit) networks** to sample tokens from LLM output distributions. The core innovation is using **direct Gibbs sampling on binary variables** y ∈ {0,1}^V with a one-hot penalty energy function, avoiding incorrect spin mappings and ensuring mathematical rigor.

### Key Features

- **Direct binary Gibbs sampler** (y ∈ {0,1}^V) with mathematically correct conditional probabilities
- **Softmax-matching token sampler** with explicit one-hot constraint enforcement
- **Resample fallback strategy**: 3 resample attempts → argmax → invalid-rate tracking
- **GPT-2 integration** for real-world text generation experiments
- **Comprehensive validation**: TV, KL, invalid-rate metrics (ESS/autocorr utilities included, not reported in current results)
- **Reproducible experiments** with strict pass criteria

### ✅ Validation Results (V=32, n=10,000, steps=100)

| Status | Metric | Result |
|--------|--------|--------|
| ✅ | **Tests Passed** | 19/19 (100%) |
| ✅ | **Invalid Rate** | 0–0.4% (low in tested configs) |
| ✅ | **Mathematical Correctness** | Verified |
| ⚠️ | **TV Distance** | 0.158-0.397 (high) |
| ⚠️ | **Mixing Quality** | Poor in some settings |
| ❌ | **ESS/Autocorr** | Not measured |

**Best TV for this setup**: λ=5.0 → TV=0.158, 99.6% valid  
**Note**: This is a research PoC demonstrating the approach, not a production-ready sampler.

## Tech Stack

- **Python**: 3.11+
- **Core**: PyTorch 2.x, NumPy 1.24+
- **LLM Integration**: Transformers 4.30+
- **Visualization**: Matplotlib 3.7+
- **Development**: pytest 7.3+, Jupyter 1.0+

## How It Works

### 1. Energy Function

The sampler uses a penalty-based energy on binary variables:

```
E(y) = -Σ_k z_k y_k + λ(Σ_k y_k - 1)²
```

where:
- `z_k` are logits (unnormalized scores, NOT log-probabilities)
- `λ` is the penalty strength (encourages one-hot)
- `y ∈ {0,1}^V` are binary variables

### 2. Direct Gibbs Sampling on {0,1}

The conditional probability for bit `i` is derived as:

```
P(y_i=1 | y_{-i}) = σ(β(z_i - λ(2S - 1)))
```

where `S = Σ_{k≠i} y_k` and `σ` is the logistic sigmoid.

**No spin mapping** to {-1,+1} is used — this ensures correctness.

### 3. Top-k/Top-p via Vocabulary Reduction

Filtering is done by **reducing the vocabulary** to a subset (not masking with `-inf`), which is critical for energy-based samplers:

```python
filtered_logits, keep_indices = apply_vocab_filter(logits, top_k=50)
# filtered_logits contains NO -inf values
```

### 4. Resample Strategy with Fallback

1. Run Gibbs for `n_gibbs_steps`
2. Check if result is valid one-hot
3. If not, resample (up to 3 attempts)
4. After failures, use `argmax(y)` as fallback
5. Track invalid rate for diagnostics

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Farx1/qllm-pbits.git
cd qllm-pbits

# Install with pinned versions (recommended)
pip install -r requirements-lock.txt
pip install -e .

# Or install with ranges
pip install -e .
```

### Running Tests

```bash
# Run fast tests only
pytest -m "not slow"

# Run all tests (including HuggingFace model loading)
RUN_SLOW=1 pytest

# With coverage
pytest --cov=qllm_pbits
```

### Quick Example

```python
from qllm_pbits.token_sampler import PBitOneHotSampler
import torch

# Create sampler with recommended configuration
sampler = PBitOneHotSampler(lam=20.0, n_gibbs_steps=100)

# Sample from logits
logits = torch.randn(50)  # Vocabulary size 50
token_id = sampler.sample(logits, temperature=1.0, top_k=10)

print(f"Sampled token: {token_id}")
print(f"Invalid rate: {sampler.get_invalid_rate()}")  # Expected: 0%

# For best distribution matching (research use)
sampler_research = PBitOneHotSampler(lam=5.0, n_gibbs_steps=100)
# Expected: TV~0.16, ~0.4% invalid rate
```

### CLI Usage

```bash
# Run softmax matching experiment (reproduces paper results)
qllm-pbits experiment --type softmax --vocab-size 32 --device cpu --seed 42

# Run calibration to find optimal lambda
qllm-pbits calibrate --vocab-size 32 --seed 42
# Output: Best lambda: 5.0 (TV=0.158)

# Generate text with P-bit sampler
qllm-pbits generate \
  --prompt "The future of AI" \
  --sampler pbit \
  --max-tokens 50 \
  --temperature 1.0 \
  --top-k 50 \
  --lambda 20 \
  --device cpu \
  --seed 42

# Run demonstration script
python demo_sampler.py
# Generates docs/assets/pbit_sampler_demo.png with comparison plots
```

## Experiments & Results

### 1. Test Suite: ✅ 19/19 Tests Passed

All unit tests pass successfully, validating:
- Direct binary Gibbs sampler correctness
- Vocabulary filtering (no -inf values)
- One-hot constraint enforcement
- Invalid rate tracking
- Token sampling functionality

### 2. Softmax Matching Experiment (V=32)

**Configuration**: 10,000 samples per λ, 100 Gibbs steps

| Lambda | TV Distance | KL (nats) | Invalid Rate | Time/Sample (ms) |
|--------|-------------|-----------|--------------|------------------|
| **5.0** | **0.158** | 0.079 | 0.4% | 16.1 |
| 10.0 | 0.365 | 0.388 | 0.0% | 13.9 |
| 20.0 | 0.370 | 0.390 | 0.0% | 15.3 |
| 50.0 | 0.378 | 0.415 | 0.0% | 15.1 |
| 100.0 | 0.397 | 0.443 | 0.0% | 14.6 |

**Key Findings**:
- ✅ **Performance**: 14–16ms per sample (observed end-to-end call time in this setup)
- ✅ **Constraint Satisfaction**: 99.6-100% valid one-hot samples
- 📊 **Trade-off**: Lower λ gives better distribution matching (TV=0.16) but ~0.4% invalid samples; higher λ (≥10) gives 0% invalid but TV~0.37

### 3. Calibration Results

Best lambda values for this setup (V=32, steps=100, β=1.0):
- **Best TV distance**: λ=5.0 (TV=0.158, 99.6% valid)
- **Best validity**: λ≥10 (TV~0.37, 100% valid)  
- ⚠️ **Setup-dependent**: Different vocabularies or step counts require recalibration

### 4. Demonstration (V=16, n=2000)

Stress test showing distribution shift:
- Configuration: λ=20, 100 Gibbs steps
- Invalid rate: 0% (perfect constraint)
- TV distance: 0.41 (poor mixing - top tokens differ from baseline)
- See `docs/assets/pbit_sampler_demo.png` for visualization
- **Interpretation**: Shows mixing limitations, not successful matching

## Tested Versions

This package has been validated on **Windows 11** with the exact versions in `requirements-lock.txt`:

```
torch==2.1.0
transformers==4.35.2
numpy==1.26.2
matplotlib==3.8.2
tqdm==4.66.1
```

To regenerate the lock file:

```bash
pip-compile pyproject.toml -o requirements-lock.txt
# or
uv pip compile pyproject.toml -o requirements-lock.txt
```

## Project Structure

```
qllm_pbits/
├── pbits/                  # Core p-bit implementations
│   ├── gibbs_binary.py    # Direct {0,1} Gibbs sampler
│   ├── ising.py           # Ising validation (spins)
│   └── metrics.py         # TV, KL, ESS, invalid-rate
├── token_sampler/         # Token sampling layer
│   ├── base.py            # Abstract interface
│   ├── vocab_filter.py    # Top-k/p reduction (NO -inf)
│   ├── softmax_baseline.py
│   ├── pbit_onehot.py     # P-bit sampler with fallback
│   └── calibration.py     # Lambda tuning tools
├── llm/                   # LLM integration
│   ├── hf_wrapper.py      # HuggingFace model wrapper
│   └── generate.py        # Text generation loop
├── experiments/           # Validation experiments
│   ├── exp_ising_convergence.py
│   ├── exp_softmax_match.py
│   └── exp_text_generation.py
└── utils/
    ├── cli.py             # Unified CLI
    └── seeding.py         # Reproducibility helpers

tests/                     # Comprehensive test suite
notebooks/                 # Interactive Jupyter notebooks
```

## Research Findings & Insights

### Mathematical Correctness Validated
- ✅ Direct Gibbs implementation verified with 19/19 tests passing
- ✅ Conditional probability formula confirmed numerically
- ✅ No spin mapping errors (avoided common pitfall)
- ✅ Vocabulary filtering without -inf (critical for energy-based samplers)

### Performance Characteristics
- **End-to-end call time**: 14-16ms (includes Python overhead, filtering, potential resamples)
- **Constraint enforcement**: 0% invalid rate achievable with λ≥10  
- **Tested vocabulary**: V=32 primarily; V=16 showed poor mixing
- ⚠️ **Timing caveat**: No controlled micro-benchmark; not directly comparable to raw softmax+multinomial

### Key Trade-off Discovery
The experiments reveal a **fidelity-constraint trade-off** and **mixing challenges**:
- **Low λ (5.0)**: Lower TV to baseline (0.158) with 99.6% valid samples
- **High λ (≥10)**: Perfect constraint (0% invalid) but higher TV (~0.37)
- **V=16 stress test**: Severe distribution shift (TV=0.41) indicating poor mixing

**Important**: The sampler targets a penalized energy distribution, NOT exact softmax. For finite λ and limited Gibbs steps, significant approximation error is observed. The TV=0.16-0.40 indicates **mixing limitations** that would need to be addressed for practical use.

### What I Learned
- **Direct Gibbs correctness**: Why spin mappings {-1,+1} can introduce errors, and how to derive conditionals directly on {0,1}
- **Vocabulary filtering**: The importance of reducing vocabulary (not using `-inf` masking) for stable energy-based sampling
- **MCMC convergence diagnostics**: Practical use of TV, KL, ESS, and invalid-rate metrics for validation
- **Trade-offs**: Balancing λ (constraint strength) vs mixing time vs invalid-rate in real applications
- **Research methodology**: How to validate novel sampling approaches with rigorous testing

## Possible Improvements

### Short-term
- **Annealing schedules**: Adaptive λ and temperature during generation
- **Batch sampling**: Parallel chains for higher throughput
- **Hardware simulation**: Model actual p-bit circuit constraints

### Long-term
- **Larger models**: Test on Llama 2, GPT-J (requires more optimization)
- **Multi-token lookahead**: Sample multiple tokens jointly
- **Continuous relaxation**: Differentiable version for gradient-based tuning
- **Deployment**: Optimize for production (compiled kernels, quantization)

## Key Limitations & Characteristics

### Expected Trade-offs & Issues (Research Findings)
- **Distribution Approximation**: TV distance 0.16-0.40 due to finite λ and mixing limitations
  - λ=5: TV=0.158 (best measured, 99.6% valid)
  - λ≥10: TV~0.37 (perfect validity, 0% invalid)
  - TV increases sharply λ=5→10, then saturates
- **Mixing Problems**: Poor convergence observed in V=16 test (TV=0.41, wrong top tokens)
- **Sample Correlation**: Autocorrelated (not measured, expected high for MCMC)
- **Parameter Sensitivity**: λ, steps require per-vocabulary calibration
- **No ESS Analysis**: Effective sample size not computed

### Implementation Constraints
- **Not hardware**: Software PoC, no claims about physical p-bit speed/energy
- **Mixing Time**: Gibbs requires O(V·steps) operations per sample vs multinomial O(1)
- **Scalability**: Current implementation is CPU-bound for V>1000, single-threaded
- **Fallback Rate**: With practical λ/steps, 0-0.4% samples use argmax fallback

### How to Frame These
These are **characteristics of the approach**, not bugs or failures. They represent the trade-off space of this novel energy-based sampling method and can be tuned based on application requirements.

## About Me

I'm **Jules Barth**, an M2 **Data & AI Engineering** student at **ESILV (Paris, France)**, specializing in **LLMs, agentic AI, privacy-preserving ML, and quantum computing**.

- 🎓 Currently: M2 Data & IA, Quantum track
- 💼 Looking for: **6-month end-of-studies internship (Data / ML / LLM)** starting **February 2026**
- 🌐 Portfolio: [julesbarth-myportfolio.fr](https://julesbarth-myportfolio.fr)
- 💼 LinkedIn: [linkedin.com/in/jules-barth](https://www.linkedin.com/in/jules-barth)
- 📧 Email: julesbarth13@gmail.com

Feel free to reach out if this project resonates with what you're building! 🚀

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{barth2026qllmpbits,
  author = {Barth, Jules},
  title = {qllm-pbits: P-bit Networks for LLM Token Sampling},
  year = {2026},
  url = {https://github.com/Farx1/qllm-pbits}
}
```

## Acknowledgments

This project was developed as part of research into probabilistic computing and LLM sampling methods. The mathematical framework is based on Gibbs sampling theory and energy-based models.

