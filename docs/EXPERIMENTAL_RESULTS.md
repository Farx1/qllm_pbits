# Experimental Results - P-bit QLLM Sampler

## Test Suite Results

**Date**: January 5, 2026  
**Platform**: Windows 11, Python 3.11.9  
**Status**: ✅ **19/19 tests passed**  
**Classification**: Proof-of-Concept / Research Artifact

### Core Functionality Tests

All unit tests pass successfully:

- **Gibbs Binary Sampler**: 6/6 tests passed
  - Initialization, shape validation, one-hot convergence
  - Conditional probability correctness
  - Chain generation
  
- **Vocabulary Filtering**: 7/7 tests passed
  - Top-k filtering (no -inf values introduced ✓)
  - Top-p (nucleus) sampling
  - Correct index remapping
  
- **One-Hot Validity**: 4/4 tests passed
  - Invalid rate tracking
  - Stats reset functionality
  - Lambda effect on constraint satisfaction
  
- **Token Sampling**: 2/2 tests passed
  - Produces valid token IDs
  - Distribution quality checks

## Experiment 1: Softmax Approximation under Penalty Energy (V=32)

**Configuration**:
- Vocabulary size: V=32
- Samples per λ: n=10,000
- Gibbs steps per sample: 100
- Temperature: β=1.0
- Lambda range: [5.0, 10.0, 20.0, 50.0]

**Important Note**: The sampler targets a Boltzmann distribution with penalty energy E(y) = -Σ z_k y_k + λ(Σ y_k - 1)², NOT exact softmax. Comparison is to empirical softmax baseline samples.

### Results Table

| Lambda | TV Distance | KL Divergence (nats) | Invalid Rate | Time/Sample (ms) |
|--------|-------------|----------------------|--------------|------------------|
| 5.0    | 0.1585      | 0.0791              | 0.59%        | 17.66           |
| 10.0   | 0.3772      | 0.3940              | 0.00%        | 14.60           |
| 20.0   | 0.3806      | 0.4010              | 0.00%        | 14.39           |
| 50.0   | 0.3778      | 0.3901              | 0.00%        | 14.75           |

### Key Findings

1. **Timing**: End-to-end Python call time 14-18ms (includes overhead, filtering, resamples)
   - ⚠️ **Caveat**: Not a controlled micro-benchmark; no warmup/median/variance reported
   - Not directly comparable to raw softmax+multinomial (typically <1ms for V=32)
   
2. **Constraint Satisfaction**: Invalid rate decreases with λ (0.59% at λ=5, 0% at λ≥10)
   
3. **Distribution Approximation**: 
   - Best TV distance: 0.1585 at λ=5.0
   - TV increases sharply to ~0.37 at λ=10, then saturates
   - Higher λ enforces constraint but increases approximation error

### Research Insights

The results reveal **fidelity-constraint trade-off and mixing limitations**:

- **Low λ (5.0)**: Lower TV to baseline (0.16) with 99.6% valid samples
- **High λ (≥10)**: Perfect constraint (0% invalid) but higher TV (~0.38)
- **Approximation error**: TV=0.16-0.40 indicates significant distance from target distribution
- **Parameter dependence**: These values are specific to V=32, steps=100, β=1.0

This reflects **finite λ and limited mixing**, not just an inherent MCMC characteristic.

## Mathematical Correctness Validation

✅ **Direct Gibbs Conditional**: The sampler correctly implements:
```
P(y_i=1 | y_{-i}) = σ(β(z_i - λ(2S - 1)))
```

✅ **No Spin Mapping**: Implementation uses direct {0,1} variables without incorrect Ising conversions

✅ **Vocabulary Reduction**: Top-k/p filtering avoids -inf values in energy calculations

✅ **Resample Strategy**: 3-attempt resample → argmax fallback is correctly implemented

## Performance Metrics Summary

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Invalid Rate** | <1% | 0-0.59% | ✅ Pass |
| **Constraint Enforcement** | One-hot | 99.4-100% | ✅ Pass |
| **Mathematical Correctness** | Verified | ✅ Tests pass | ✅ Pass |
| **TV Distance** | <0.02* | 0.16-0.38 | ⚠️ High approximation error |
| **Mixing Quality** | Not specified | Poor in V=16 | ❌ Needs improvement |
| **ESS/Autocorrelation** | Should measure | Not measured | ❌ Missing |

*Note: TV<0.02 was too optimistic. Achieved TV=0.16-0.38 indicates significant approximation error due to finite λ and mixing limitations, not just "reasonable" MCMC behavior.

## Use Case Recommendations

Based on experimental results:

### For Research & Experimentation Only
- **Lambda**: 5.0 for best TV (0.158), 99.6% valid
- **Lambda**: 20.0 for perfect validity (0%), TV~0.37
- **Setup-specific**: These values tested only for V=32, steps=100, β=1.0
- **Use case**: Understanding energy-based sampling, baseline for improved methods

### NOT Recommended for Production
**Reasons**:
- TV=0.16-0.38 indicates significant approximation error
- No evaluation of downstream text quality/perplexity
- Poor mixing observed in some settings (V=16 test)
- Autocorrelation/ESS not characterized
- No controlled performance benchmarks

### Future Work Needed
- Text generation quality evaluation
- Perplexity/coherence metrics
- Proper micro-benchmarks
- ESS and autocorrelation analysis
- Multiple chain strategies
- Adaptive schedules

## Comparison to Baseline (Softmax Multinomial)

The P-bit sampler demonstrates:
- **Constraint enforcement**: 0% invalid rate achievable (advantage)
- **Interpretable parameters**: λ directly controls constraint strength
- **Approximation error**: TV=0.16-0.38 vs exact baseline
- **MCMC characteristics**: Autocorrelated samples, mixing issues
- **Timing**: End-to-end call time similar, but not directly comparable (lacks proper benchmarking)

⚠️ **Timing Note**: Reported 14-18ms includes Python overhead and is NOT a controlled comparison to raw softmax+multinomial operations.

## Limitations & Future Work

### Current Limitations
1. **Distribution approximation**: TV distance of 0.16-0.38 vs exact softmax
2. **Sampling correlation**: Adjacent samples are not independent (MCMC property)
3. **Vocabulary size**: Not yet tested on full LLM vocabulary (V~50k)

### Proposed Improvements
1. **Adaptive λ scheduling**: Anneal λ during sampling to balance fidelity/constraints
2. **Parallel chains**: Run multiple independent chains for better ESS
3. **Hardware acceleration**: GPU kernels for large-vocabulary sampling
4. **Hybrid approaches**: Combine P-bit with occasional exact resampling

## Conclusion

The P-bit token sampler is a **working proof-of-concept** that:
- ✅ Implements mathematically correct direct binary Gibbs sampling
- ✅ Achieves strong constraint satisfaction (0% invalid at λ≥10)
- ⚠️ Shows significant approximation error (TV=0.16-0.40)
- ⚠️ Exhibits mixing issues in some settings (V=16 poor convergence)
- ❌ Lacks ESS/autocorrelation characterization
- ❌ Lacks controlled performance benchmarks

**For your research paper**: This is a **reproducible research artifact** demonstrating the approach and identifying key challenges (mixing, parameter sensitivity). Suitable for publication as:
- Proof-of-concept with honest limitations
- Baseline for improved mixing strategies
- Educational resource on MCMC sampling challenges

**NOT suitable as**: Production-ready sampler or efficiency benchmark.

