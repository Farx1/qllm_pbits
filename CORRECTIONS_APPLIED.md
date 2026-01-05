# Corrections Applied - Scientific Rigor

**Date**: January 5, 2026  
**Purpose**: Address factual errors and overstated claims in results presentation

---

## ✅ All Corrections Applied

### 1. Fixed Factual Errors

#### A) Logits terminology
- ❌ **Before**: "logits (log-probabilities)"
- ✅ **After**: "logits (unnormalized scores, NOT log-probabilities)"
- **Files updated**: README.md

#### B) Timing claims
- ❌ **Before**: "~15ms comparable to softmax" (misleading)
- ✅ **After**: "14-16ms end-to-end Python call time (includes overhead, filtering, resamples)"
- **Added**: Warning that this is NOT a controlled micro-benchmark
- **Added**: Note that raw softmax+multinomial is typically <1ms for V=32
- **Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md, README.md

#### C) Publication readiness
- ❌ **Before**: "Ready for Research Publication"
- ✅ **After**: "Reproducible Research Artifact / Proof-of-Concept"
- **Added**: "Baseline for follow-up work", "NOT production-ready"
- **Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md

### 2. Fixed Distribution Target Mismatch

#### Problem
Sampler targets penalized Boltzmann distribution, NOT exact softmax

#### Solution
- **Title changed**: "Softmax Distribution Matching" → "Softmax Approximation under One-Hot Penalty Energy"
- **Added explanation**: "For finite λ and finite Gibbs steps, the distribution over valid one-hot states approximates softmax, but mixing limitations affect fidelity"
- **Contextualized results**: TV distance reflects finite λ AND mixing issues, not just "inherent MCMC"
- **Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md

### 3. Fixed Overstated Calibration Claims

#### A) "Optimal λ" language
- ❌ **Before**: "Optimal λ for fidelity: 5.0", "For production: 20.0"
- ✅ **After**: "Best TV for this setup (V=32, steps=100, β=1.0): λ=5.0"
- **Added**: "Setup-dependent: Different V or step counts require recalibration"
- **Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md

#### B) TV growth pattern
- ❌ **Before**: "TV grows approximately linearly with λ"
- ✅ **After**: "TV increases sharply from λ=5 to λ=10, then saturates around 0.37-0.40"
- **Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md

### 4. Reframed V=16 Demonstration

#### Problem
Demo showed poor mixing (TV=0.41, wrong top tokens) but was presented as success

#### Solution
- **Section renamed**: "Demonstration Results" → "Qualitative Stress Test - Distribution Shift Observed"
- **Added critical analysis**: 
  - "Poor mixing behavior"
  - "NOT a successful demonstration but a stress test showing limitations"
  - "Improvement paths: more steps, blocked updates, annealing, multiple chains"
- **Files updated**: RESULTS_SUMMARY.md

### 5. Added Missing Limitations

#### A) Autocorrelation / ESS
- **Added to limitations**: "Autocorrelation not measured, but expected high for MCMC"
- **Added to summary table**: "ESS/Autocorr: Not measured ❌"
- **Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md, README.md

#### B) Mixing quality
- **Added new limitation category**: "Mixing Problems"
- **Evidence**: "Poor convergence observed in V=16 test (TV=0.41)"
- **Files updated**: All results documents

### 6. Removed/Attenuated Production Claims

#### Changes
- ❌ **Removed**: "For Production" section with configurations
- ❌ **Removed**: "Production/Reliability" recommendations
- ✅ **Added**: "NOT Recommended for Production" section with clear reasons:
  - No downstream quality evaluation
  - No perplexity metrics
  - Poor mixing in some settings
  - No ESS characterization

- **Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md

### 7. Fixed Comparison Table

#### Before
| Time per Sample | ~15ms | ~15ms (comparable) |

#### After
| Time per Call | Not separately benchmarked | 14-16ms (end-to-end with overhead) |

**Added note**: "End-to-end Python call times are not directly comparable without controlled micro-benchmarks. Raw softmax+multinomial operations are typically < 1ms for V=32."

**Files updated**: RESULTS_SUMMARY.md

### 8. Updated Key Messages

#### Old message:
> "We present the first working implementation... achieves competitive performance (15ms/sample)... establishing a foundation..."

#### New message:
> "We present a working proof-of-concept implementation... exhibits significant approximation error (TV=0.16-0.40) due to finite λ and mixing limitations, with poor convergence observed in some settings. This work establishes a reproducible baseline and identifies key challenges (mixing time, parameter sensitivity) for future research."

**Files updated**: RESULTS_SUMMARY.md

### 9. Updated Validation Tables

#### Added honest status indicators:
- ✅ Tests Passing: 100%
- ✅ Invalid Rate: 0-0.4%
- ✅ Mathematical Correctness: Verified
- ⚠️ TV Distance: 0.16-0.40 (high approximation error)
- ❌ Mixing Quality: Poor in some settings
- ❌ ESS/Autocorr: Not measured

**Files updated**: RESULTS_SUMMARY.md, EXPERIMENTAL_RESULTS.md, README.md

---

## Summary of Corrections by Category

### Factual Corrections
- [x] Logits terminology fixed
- [x] Timing methodology clarified
- [x] Publication status downgraded appropriately

### Methodological Corrections
- [x] Distribution target clarified (penalized energy, not exact softmax)
- [x] Calibration claims scoped to specific setup
- [x] TV growth pattern corrected (not linear)

### Honest Reporting
- [x] V=16 demo reframed as stress test showing limitations
- [x] Missing metrics acknowledged (ESS, autocorr)
- [x] Mixing problems explicitly stated
- [x] Production claims removed

### Status Updates
- [x] "Ready for publication" → "Reproducible research artifact"
- [x] Added "baseline for future work" framing
- [x] Clear "NOT production-ready" statements

---

## Files Updated

1. **RESULTS_SUMMARY.md** - Comprehensive corrections (8 major sections)
2. **EXPERIMENTAL_RESULTS.md** - All sections updated for rigor
3. **README.md** - Key features, results table, limitations
4. **CORRECTIONS_APPLIED.md** - This document (new)

---

## Remaining As-Is (Correct)

- ✅ Code implementation (mathematically correct)
- ✅ Test suite (19/19 passing)
- ✅ Experimental data (TV, invalid rates are accurate)
- ✅ General approach description

---

## For Your Research Paper

### Recommended Framing

**What to emphasize**:
- First working implementation with correct math
- Reproducible artifact with comprehensive testing
- Identifies key challenges for MCMC token sampling
- Baseline for future improvements

**What to acknowledge honestly**:
- TV=0.16-0.40 indicates significant approximation error
- Mixing issues observed (V=16 poor convergence)
- Parameter sensitivity (setup-dependent)
- No ESS/autocorrelation analysis
- No downstream text quality evaluation

**What NOT to claim**:
- "Competitive performance" with softmax
- "Production-ready" or "suitable for deployment"
- "Optimal" parameters (setup-dependent only)
- "Solved" alternative to softmax sampling

### Honest Key Message

> "Working proof-of-concept demonstrating p-bit network sampling for tokens. Implementation is mathematically correct but exhibits significant approximation error (TV=0.16-0.40) and mixing challenges. Suitable as a reproducible baseline for future research on MCMC-based token sampling with energy constraints."

---

## Conclusion

All major scientific rigor issues have been addressed:
- Factual errors corrected
- Overstated claims downgraded
- Missing measurements acknowledged
- Honest limitations added
- Appropriate framing as research PoC

The documents now present an honest, rigorous assessment suitable for peer review.

