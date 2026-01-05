"""
Tests for one-hot constraint validity.
"""

import torch
import pytest
from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler


def test_invalid_rate_tracking():
    """Test that invalid rate is tracked correctly."""
    torch.manual_seed(42)
    sampler = PBitOneHotSampler(lam=50.0, n_gibbs_steps=100)
    
    logits = torch.randn(10)
    
    # Sample multiple times
    for _ in range(100):
        sampler.sample(logits, device="cpu")
    
    invalid_rate = sampler.get_invalid_rate()
    
    # With high lambda, invalid rate should be low
    assert invalid_rate < 0.1, f"Invalid rate {invalid_rate:.3f} too high"


def test_reset_stats():
    """Test that reset_stats clears counters."""
    sampler = PBitOneHotSampler(lam=10.0, n_gibbs_steps=50)
    
    logits = torch.randn(8)
    
    # Sample some
    for _ in range(10):
        sampler.sample(logits, device="cpu")
    
    assert sampler.total_count == 10
    
    # Reset
    sampler.reset_stats()
    
    assert sampler.total_count == 0
    assert sampler.invalid_count == 0
    assert sampler.get_invalid_rate() == 0.0


def test_high_lambda_reduces_invalid_rate():
    """Test that higher lambda leads to lower invalid rate."""
    torch.manual_seed(123)
    logits = torch.randn(12)
    
    # Low lambda
    sampler_low = PBitOneHotSampler(lam=5.0, n_gibbs_steps=50)
    for _ in range(100):
        sampler_low.sample(logits, device="cpu")
    rate_low = sampler_low.get_invalid_rate()
    
    # High lambda
    sampler_high = PBitOneHotSampler(lam=50.0, n_gibbs_steps=50)
    for _ in range(100):
        sampler_high.sample(logits, device="cpu")
    rate_high = sampler_high.get_invalid_rate()
    
    # Higher lambda should have lower invalid rate
    assert rate_high <= rate_low


def test_get_invalid_rate_zero_samples():
    """Test that invalid rate is 0 when no samples taken."""
    sampler = PBitOneHotSampler(lam=20.0, n_gibbs_steps=100)
    assert sampler.get_invalid_rate() == 0.0

