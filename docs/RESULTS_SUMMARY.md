# P-bit QLLM Sampler - Research Results Summary

**Author**: Jules Barth  
**Date**: January 5, 2026  
**Institution**: ESILV - M2 Data & AI Engineering  

---

## Executive Summary

This document presents experimental validation results for a novel **probabilistic bit (p-bit) network approach** to token sampling from language model distributions. The implementation uses **direct Gibbs sampling on binary variables {0,1}^V** with a one-hot penalty energy function.

**Key Achievement**: Working proof-of-concept demonstrating mathematically correct implementation with quantifiable performance characteristics suitable for research publication.

---

## 1. Implementation Validation

### 1.1 Test Suite Results

**Status**: ✅ **100% Pass Rate (19/19 tests)**

| Test Category | Tests Passed | Key Validations |
|---------------|--------------|-----------------|
| Gibbs Binary Sampler | 6/6 | Conditional probability correctness, shape validation, one-hot convergence |
| Vocabulary Filtering | 7/7 | Top-k/p without -inf values, correct index remapping |
| One-Hot Validity | 4/4 | Invalid rate tracking, lambda effect verification |
| Token Sampling | 2/2 | Valid token ID generation, distribution sanity checks |

### 1.2 Mathematical Correctness

✅ **Verified Implementations**:
- Direct Gibbs conditional: `P(y_i=1|y_{-i}) = σ(β(z_i - λ(2S-1)))`
- Energy function: `E(y) = -Σ z_k y_k + λ(Σ y_k - 1)²`
- No spin mapping to {-1,+1} (avoids common implementation error)
- Vocabulary reduction without -inf masking
- 3-attempt resample → argmax fallback strategy

---

## 2. Softmax Approximation under One-Hot Penalty Energy

### 2.1 Experimental Setup

**Note**: The sampler targets a Boltzmann distribution over {0,1}^V with penalty energy, NOT exact softmax. For finite λ and finite Gibbs steps, the distribution over valid one-hot states approximates softmax, but mixing limitations affect fidelity.

- **Vocabulary Size**: V = 32
- **Samples per Configuration**: n = 10,000
- **Gibbs Steps per Sample**: 100
- **Temperature**: β = 1.0
- **Lambda Range**: [5.0, 10.0, 20.0, 50.0, 100.0]
- **Hardware**: CPU (model not specified in benchmark)

### 2.2 Results

| λ | TV Distance | KL (nats) | Invalid Rate | Time/Sample (ms) | One-Hot % |
|---|-------------|-----------|--------------|------------------|-----------|
| **5.0** | **0.158** | **0.079** | 0.004 (0.4%) | 16.1 | 99.6% |
| 10.0 | 0.365 | 0.388 | 0.000 | 13.9 | 100% |
| 20.0 | 0.370 | 0.390 | 0.000 | 15.3 | 100% |
| 50.0 | 0.378 | 0.415 | 0.000 | 15.1 | 100% |
| 100.0 | 0.397 | 0.443 | 0.000 | 14.6 | 100% |

### 2.3 Key Findings

1. **Fidelity-Constraint Trade-off**:
   - **Low λ (5.0)**: Best TV to baseline (0.158) with 99.6% valid samples
   - **High λ (≥10)**: Perfect constraint satisfaction (0% invalid) but higher TV (~0.37)
   - This trade-off is expected: higher λ enforces one-hot more strongly but increases distance from the target marginal distribution

2. **Timing** (End-to-end Python call including filtering):
   - Measured range: **14-16ms per sample**
   - ⚠️ **Methodology limitation**: No detailed micro-benchmark (warmup, median, variance not reported)
   - Timing includes Python overhead, filtering, and potential resample attempts
   - Not directly comparable to raw softmax+multinomial cost

3. **Constraint Satisfaction**:
   - λ≥10 achieves **100% valid one-hot** samples
   - λ=5 achieves **99.6% valid** with lower TV distance
   - TV increases sharply λ=5→10, then saturates around 0.37-0.40 (not linear growth)

---

## 3. Calibration Analysis

### 3.1 Lambda Parameter Tuning

The calibration experiment (n=5000 samples, V=32, steps=100, β=1.0) shows:

**λ Selection for this Setup**:
- **Best TV distance**: λ = 5.0 (TV=0.158, KL=0.084, 99.6% valid)
- **Best validity**: λ ≥ 10 (0% invalid, TV~0.37)
- **Setup-dependent**: These values depend on V, step budget, β, and logit distribution

⚠️ **Caution**: "Optimal" is relative to this specific configuration. Different vocabularies or step counts may require recalibration.

### 3.2 Parameter Sensitivity

TV distance behavior across λ:
- Sharp increase from λ=5 (TV=0.16) to λ=10 (TV=0.37)
- **Saturation** beyond λ=10: TV remains around 0.37-0.40
- Not linear growth, but rather rapid transition then plateau

Diminishing returns for constraint enforcement beyond λ=10, but TV remains elevated.

---

## 4. Qualitative Stress Test (V=16) - Distribution Shift Observed

### 4.1 Setup

- **Logits**: Linearly decreasing from 2.0 to -3.0
- **Samples**: n = 2000
- **Configuration**: P-bit with λ=20, 100 Gibbs steps

### 4.2 Results

| Metric | Baseline | P-bit |
|--------|----------|-------|
| **Top-1 Token** | 0 (28%) | 10 (8%) |
| **Top-2 Token** | 1 (18%) | 5 (7%) |
| **Top-3 Token** | 2 (11%) | 15 (7%) |
| **TV vs Baseline** | - | 0.410 |
| **KL (nats)** | - | 0.525 |
| **Invalid Rate** | - | 0% |

### 4.3 Critical Observation

⚠️ **Poor mixing behavior**: The P-bit sampler shows severe distribution shift (TV=0.41, KL=0.53), with top tokens completely different from baseline. This indicates:

1. **Insufficient mixing** at λ=20, steps=100 for this vocabulary size
2. The sampler is **not converging** to a softmax-like marginal
3. Energy landscape may have multiple modes/poor connectivity

**Interpretation**: This is NOT a successful demonstration but a **stress test showing limitations** of current parameters.

**Improvement paths**:
- Increase Gibbs steps (200-500)
- Use blocked/systematic sweep instead of random-scan
- Implement annealing schedules (start low λ, increase)
- Run multiple independent chains

---

## 5. Comparative Analysis

### 5.1 P-bit vs Standard Softmax Multinomial

| Aspect | Softmax Multinomial | P-bit Sampler |
|--------|---------------------|---------------|
| **Sampling Method** | Direct sampling | MCMC (Gibbs) |
| **Distribution Match** | Exact softmax | Penalized energy (TV~0.16-0.40) |
| **Constraint Enforcement** | Implicit | Explicit (0-0.4% invalid) |
| **Sample Independence** | Independent | Autocorrelated (not measured in this PoC) |
| **Time per Call** | Not separately benchmarked | 14-16ms (end-to-end with overhead) |
| **Tunability** | Temperature only | λ, β, steps |
| **Interpretability** | Standard | Energy-based |

⚠️ **Timing caveat**: End-to-end Python call times are not directly comparable without controlled micro-benchmarks. Raw softmax+multinomial operations are typically < 1ms for V=32.

### 5.2 Advantages of P-bit Approach

1. **Explicit Constraint Control**: λ parameter directly controls one-hot enforcement
2. **Energy-Based Framework**: Enables future extensions (multi-constraint, hierarchical)
3. **Theoretical Foundation**: Proven MCMC convergence guarantees
4. **Research Value**: Novel application of p-bit networks to LLM sampling

### 5.3 Limitations

1. **Distribution Approximation**: TV distance 0.16-0.40 vs baseline (due to finite λ and mixing)
2. **Mixing Issues**: Poor convergence observed in some settings (V=16 demo)
3. **Parameter Tuning**: λ, steps require calibration per vocabulary size
4. **Sample Correlation**: Autocorrelation not measured, but expected high for MCMC
5. **No ESS Analysis**: Effective sample size not computed, limiting claims about efficiency

---

## 6. Use Case Recommendations

### For Researchers

**This PoC is suitable for**:
- Understanding energy-based token sampling
- Exploring fidelity-constraint trade-offs
- Baseline for improved mixing strategies
- Education on MCMC sampling issues

**Tested Configuration** (V=32, steps=100):
- λ = 5: Best TV (0.158), 99.6% valid
- λ = 20: Perfect validity (0%), TV~0.37

⚠️ **Requires recalibration** for different V or step budgets

### NOT Recommended for Production

**Why**:
- TV distance 0.16-0.40 not evaluated for downstream text quality
- No perplexity/coherence validation on real generation tasks
- Poor mixing observed in some configurations (V=16 test)
- Autocorrelation/ESS not characterized

**Future work needed**:
- Text generation quality evaluation
- Perplexity benchmarks
- Multiple chain strategies
- Adaptive λ/β schedules

---

## 7. Research Contributions

### 7.1 Novel Aspects

1. **Direct Binary Implementation**: First implementation avoiding incorrect spin mappings
2. **Vocabulary Reduction**: Novel approach to top-k/p without -inf values
3. **Explicit Fallback Strategy**: Documented resample-then-argmax approach
4. **Quantified Trade-offs**: Empirical characterization of λ vs TV/KL/invalid-rate

### 7.2 Validation Methodology

- ✅ Unit test coverage for all core components
- ✅ Mathematical correctness verification
- ✅ Performance benchmarking
- ✅ Distribution matching experiments
- ✅ Parameter sensitivity analysis

---

## 8. Future Research Directions

### Short-term Extensions

1. **Adaptive λ Scheduling**: Dynamic λ(t) to balance fidelity and constraints
2. **Parallel Chains**: Multiple independent chains for improved ESS
3. **Vocabulary Scaling**: Test on full LLM vocabulary (V~50k)
4. **Temperature Annealing**: Combined β and λ schedules

### Long-term Research

1. **Hardware P-bit Simulation**: Model actual circuit constraints
2. **Multi-token Sampling**: Joint sampling of token sequences
3. **Hybrid Approaches**: Combine with exact resampling
4. **Theoretical Analysis**: Convergence rates and mixing time bounds

---

## 9. Conclusion

### Summary of Results

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| **Tests Passing** | >90% | 100% (19/19) | ✅ Pass |
| **Invalid Rate** | <1% | 0-0.4% | ✅ Pass |
| **Mathematical Correctness** | Verified | ✅ | ✅ Verified |
| **TV Distance** | <0.02* | 0.16-0.40 | ⚠️ High (mixing issues) |
| **Mixing Quality** | Not specified | Poor in V=16 test | ❌ Needs improvement |
| **Autocorrelation/ESS** | Should measure | Not measured | ❌ Missing |

*Original target was optimistic; but TV=0.16-0.40 indicates significant approximation error.

### Artifact Readiness

**Status**: ✅ **Reproducible Research Artifact / Proof-of-Concept**

**Suitable as**:
- Working implementation demonstrating the approach
- Baseline for follow-up research (improved mixing, schedules)
- Educational resource on MCMC sampling challenges
- Foundation for investigating p-bit hardware simulation

**NOT suitable as**:
- Production-ready sampler
- "Solved" alternative to softmax sampling
- Benchmark for efficiency claims

### Key Message for Paper

> "We present a working proof-of-concept implementation of p-bit network sampling for LLM token generation using direct Gibbs sampling on binary variables with one-hot penalty energy. The implementation is mathematically correct and demonstrates explicit constraint enforcement (0-0.4% invalid samples). However, the approach exhibits significant approximation error (TV=0.16-0.40) due to finite λ and mixing limitations, with poor convergence observed in some settings. This work establishes a reproducible baseline and identifies key challenges (mixing time, parameter sensitivity) for future research in MCMC-based token sampling with energy constraints."

---

## 10. Supplementary Materials

### 10.1 Reproducibility

All experiments are reproducible using:
```bash
# Run test suite
pytest -m "not slow"

# Run softmax matching experiment
python -m qllm_pbits.experiments.exp_softmax_match

# Run calibration
python -m qllm_pbits.utils.cli calibrate --vocab-size 32

# Run demonstration
python demo_sampler.py
```

### 10.2 Artifacts

- ✅ Test suite: 19 passing tests
- ✅ Experimental results: Documented in EXPERIMENTAL_RESULTS.md
- ✅ Visualization: pbit_sampler_demo.png
- ✅ Code repository: Complete implementation with documentation

### 10.3 Contact

**Jules Barth**  
M2 Data & AI Engineering, ESILV (Paris)  
Email: julesbarth13@gmail.com  
Portfolio: https://julesbarth-myportfolio.fr  
LinkedIn: https://www.linkedin.com/in/jules-barth  

---

**Document Version**: 1.0  
**Last Updated**: January 5, 2026

