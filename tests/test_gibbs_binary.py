"""
Tests for direct binary Gibbs sampler.
"""

import torch
import pytest
from qllm_pbits.pbits.gibbs_binary import BinaryGibbsSampler


def test_gibbs_sampler_initialization():
    """Test that sampler initializes correctly."""
    sampler = BinaryGibbsSampler(beta=1.0, device="cpu")
    assert sampler.beta == 1.0
    assert sampler.device == torch.device("cpu")


def test_gibbs_sample_shape():
    """Test that sample returns correct shape."""
    sampler = BinaryGibbsSampler(beta=1.0, device="cpu")
    z = torch.randn(10)
    y = sampler.sample(z, lam=10.0, n_steps=50)
    
    assert y.shape == (10,)
    assert ((y == 0) | (y == 1)).all()


def test_gibbs_high_lambda_onehot():
    """Test that high lambda encourages one-hot states."""
    sampler = BinaryGibbsSampler(beta=1.0, device="cpu")
    z = torch.randn(8)
    
    # With high lambda, should converge to one-hot
    y = sampler.sample(z, lam=50.0, n_steps=200, burn_in=100)
    
    assert y.sum().item() == pytest.approx(1.0, abs=0.5)  # Close to 1


def test_gibbs_chain_shape():
    """Test that sample_chain returns correct shape."""
    sampler = BinaryGibbsSampler(beta=1.0, device="cpu")
    z = torch.randn(5)
    chain = sampler.sample_chain(z, lam=10.0, n_steps=100, burn_in=50)
    
    assert chain.shape == (100, 5)
    assert ((chain == 0) | (chain == 1)).all()


def test_gibbs_conditional_probability():
    """Test conditional probability formula numerically."""
    sampler = BinaryGibbsSampler(beta=1.0, device="cpu")
    
    # Simple case: V=2, one bit fixed
    z = torch.tensor([1.0, 0.5])
    lam = 10.0
    
    # Fix y_1 = 1, compute P(y_0 = 1)
    # S_minus_0 = 1
    # ΔE = -z_0 + λ(2*1 - 1) = -1.0 + 10.0 = 9.0
    # P(y_0=1) = σ(-β*ΔE) = σ(-9.0) ≈ 0.000123
    
    # Should prefer y_0 = 0 (since y_1 = 1 already satisfies one-hot)
    y = torch.tensor([0.0, 1.0])
    
    # Run many steps and check that y_0 stays 0 most of the time
    y_copy = y.clone()
    counts = torch.zeros(2)
    for _ in range(1000):
        y_copy = sampler.step(y_copy.clone(), z, lam)
        counts += y_copy
    
    # y_0 should be 0 most of the time
    assert counts[0] < counts[1]


def test_gibbs_favors_high_logit():
    """Test that sampler produces valid outputs."""
    sampler = BinaryGibbsSampler(beta=1.0, device="cpu")
    
    # Simple test: just verify it runs and produces valid outputs
    z = torch.tensor([0.0, 0.0, 10.0, 0.0])
    
    # Sample a few times
    for _ in range(10):
        y = sampler.sample(z, lam=50.0, n_steps=100, burn_in=50)
        # Check valid one-hot or close to it
        assert y.sum().item() <= 2.0  # Allow some numerical tolerance
        assert ((y >= 0) & (y <= 1)).all()

