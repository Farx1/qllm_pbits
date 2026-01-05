# Reproduction Guide

This document provides exact commands to reproduce all results reported in the research documentation.

## Prerequisites

- Python 3.11 or 3.12
- Git
- ~2GB disk space (for PyTorch + Transformers)

## Installation

### Windows

```powershell
# Clone repository
git clone https://github.com/Farx1/qllm-pbits.git
cd qllm-pbits

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install with tested versions
pip install -r requirements-tested.txt
pip install -e ".[dev]"

# Verify installation
python -c "import qllm_pbits; print(qllm_pbits.__version__)"
```

### Linux/macOS

```bash
# Clone repository
git clone https://github.com/Farx1/qllm-pbits.git
cd qllm-pbits

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with tested versions
pip install -r requirements-tested.txt
pip install -e ".[dev]"

# Verify installation
python -c "import qllm_pbits; print(qllm_pbits.__version__)"
```

## Run Tests

### Fast Tests Only (No Model Downloads)

```bash
# Using pytest directly
pytest -m "not slow" -v

# Using Makefile (Linux/macOS)
make test

# Windows PowerShell
python -m pytest -m "not slow" -v
```

**Expected**: 19/19 tests pass in ~15-20 seconds

### All Tests (Including HuggingFace Integration)

```bash
# Linux/macOS
RUN_SLOW=1 pytest -v

# Windows PowerShell
$env:RUN_SLOW="1"; python -m pytest -v

# Using Makefile
make test-all
```

**Expected**: 25/25 tests pass (downloads distilgpt2 model, ~2-3 minutes first time)

## Reproduce Experimental Results

### Experiment 1: Softmax Approximation (V=32)

```bash
python -m qllm_pbits.experiments.exp_softmax_match
```

**Expected output**:
```
=== Softmax Match Experiment (V=32) ===

Testing lambda=5.0...
  TV: 0.1585 (target: < 0.02)
  KL: 0.0791 nats (target: < 0.05)
  Invalid rate: 0.0059 (target: < 0.01)
  Time/sample: ~17ms
  PASSED: False

[... results for lambda=10, 20, 50 ...]

Best lambda: 5.0 (TV=0.1585)
```

**Runtime**: ~2-3 minutes (10,000 samples × 4 lambda values)

### Experiment 2: Lambda Calibration

```bash
python -m qllm_pbits.utils.cli calibrate --vocab-size 32 --seed 42
```

**Expected output**:
```
Best lambda: 5.0
Best TV: 0.1576

Results table:
 lambda     TV       KL  invalid_rate   time_ms
    5.0 0.1576 0.083924         0.004 16.114932
   10.0 0.3646 0.387677         0.000 13.921114
   ...
```

**Runtime**: ~1-2 minutes (5,000 samples × 5 lambda values)

### Experiment 3: Demonstration Plot

```bash
python demo_sampler.py
```

**Expected output**:
```
P-bit Token Sampler Demonstration
...
TV distance (baseline vs P-bit): 0.4095
KL divergence (baseline||P-bit): 0.5252 nats

[OK] Plot saved as 'docs/assets/pbit_sampler_demo.png'
```

**Runtime**: ~30 seconds (2,000 samples)

## Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open and run:
# - notebooks/01_binary_gibbs_validation.ipynb
# - notebooks/02_softmax_vs_pbit_sampler.ipynb
```

**Expected**: Both notebooks run without errors, showing validation results

## CLI Usage Examples

### Generate Text

```bash
python -m qllm_pbits.utils.cli generate \
  --prompt "The future of AI" \
  --sampler pbit \
  --max-tokens 20 \
  --lambda 20 \
  --seed 42
```

**Note**: First run downloads distilgpt2 model (~250MB)

### Run Ising Validation (Small System)

```bash
python -m qllm_pbits.utils.cli experiment --type ising --seed 42
```

**Expected**: Validates Gibbs convergence on N=8 Ising model

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'qllm_pbits'`:
```bash
pip install -e .
```

### CUDA/GPU Issues

All experiments default to CPU. For GPU:
```bash
python -m qllm_pbits.experiments.exp_softmax_match --device cuda:0
```

### Slow Tests Timeout

If HuggingFace tests timeout, skip them:
```bash
pytest -m "not slow"
```

## Verification Checklist

- [ ] Tests pass: `pytest -m "not slow"` → 19/19 pass
- [ ] CLI works: `python -m qllm_pbits.utils.cli --help`
- [ ] Softmax experiment: TV=0.158 at λ=5.0
- [ ] Calibration: Best lambda=5.0
- [ ] Demo plot: Generated in `docs/assets/`
- [ ] Notebooks: Both run without errors

## Hardware/Software Used for Reported Results

- **OS**: Windows 11
- **Python**: 3.11.9
- **CPU**: Not specified (end-to-end timing includes Python overhead)
- **Date**: January 5, 2026

**Timing Note**: Reported 14-16ms is end-to-end Python call time, NOT a controlled micro-benchmark. Not directly comparable to raw softmax+multinomial operations.

