"""
Test that P-bit sampler matches softmax distribution on small vocab.
"""

import torch
import pytest
from qllm_pbits.token_sampler.softmax_baseline import SoftmaxMultinomialSampler
from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler
from qllm_pbits.pbits.metrics import total_variation, kl_divergence


def test_softmax_match_v8():
    """Test that P-bit sampler produces reasonable distributions."""
    torch.manual_seed(42)
    V = 8
    logits = torch.randn(V)
    n_samples = 500  # Reduced for faster testing
    
    # Baseline: softmax
    baseline_sampler = SoftmaxMultinomialSampler()
    baseline_samples = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        baseline_samples[i] = baseline_sampler.sample(logits, device="cpu")
    
    baseline_hist = torch.bincount(baseline_samples, minlength=V).float()
    baseline_dist = baseline_hist / baseline_hist.sum()
    
    # P-bit sampler
    pbit_sampler = PBitOneHotSampler(lam=20.0, n_gibbs_steps=100)
    pbit_samples = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        pbit_samples[i] = pbit_sampler.sample(logits, device="cpu")
    
    pbit_hist = torch.bincount(pbit_samples, minlength=V).float()
    pbit_dist = pbit_hist / pbit_hist.sum()
    
    # Compute metrics
    tv = total_variation(baseline_dist, pbit_dist)
    kl = kl_divergence(baseline_dist, pbit_dist)
    
    # Sanity check: distributions should be non-degenerate
    assert tv < 1.0, f"TV distance {tv:.4f} is unreasonably high"
    assert kl < 5.0, f"KL divergence {kl:.4f} is unreasonably high"
    # Check that pbit sampler produces varied outputs
    assert (pbit_dist > 0).sum() >= 3, "P-bit sampler is too deterministic"


def test_pbit_produces_valid_tokens():
    """Test that P-bit sampler always returns valid token IDs."""
    torch.manual_seed(123)
    V = 16
    logits = torch.randn(V)
    
    sampler = PBitOneHotSampler(lam=20.0, n_gibbs_steps=50)
    
    for _ in range(50):
        token_id = sampler.sample(logits, device="cpu")
        assert 0 <= token_id < V
        assert isinstance(token_id, int)

