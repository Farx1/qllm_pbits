# Changelog

## [0.1.0] - 2026-01-05

### Added
- Initial proof-of-concept implementation
- Direct binary Gibbs sampler on {0,1}^V (no spin mapping)
- P-bit token sampler with one-hot penalty energy
- Softmax baseline for comparison
- Vocabulary filtering without -inf masking
- Resample-argmax fallback strategy
- Comprehensive test suite (19 tests)
- CLI tools for experiments (experiment, calibrate, generate)
- Experimental results documentation
- HuggingFace GPT-2 integration

### Validated
- Mathematical correctness (19/19 tests pass)
- Constraint enforcement (0-0.4% invalid rate)
- TV distance 0.158-0.397 (setup-dependent)
- End-to-end call time: 14-16ms

### Known Issues
- High TV distance indicates mixing limitations  
- Poor convergence in V=16 stress test (TV=0.41)
- ESS/autocorrelation not measured in experiments
- No downstream text quality evaluation (perplexity, coherence)
- Parameter sensitivity (λ, steps require per-vocabulary calibration)

### Dependencies
- Python ≥3.11
- PyTorch ≥2.0.0
- Transformers ≥4.30.0
- NumPy, Matplotlib, tqdm

