"""
Tests for vocabulary filtering (top-k, top-p).
"""

import torch
import pytest
from qllm_pbits.token_sampler.vocab_filter import apply_vocab_filter


def test_no_filter():
    """Test that no filtering returns all tokens."""
    logits = torch.tensor([0.1, -2.0, 1.5, 0.3])
    filtered, indices = apply_vocab_filter(logits)
    
    assert len(filtered) == 4
    assert torch.allclose(filtered, logits)
    assert torch.equal(indices, torch.arange(4))


def test_topk_correct_indices():
    """Test that top-k returns correct indices."""
    logits = torch.tensor([0.1, -2.0, 1.5, 0.3])
    filtered, indices = apply_vocab_filter(logits, top_k=2)
    
    assert len(filtered) == 2
    assert torch.allclose(filtered, torch.tensor([1.5, 0.3]))
    assert torch.equal(indices, torch.tensor([2, 3]))


def test_topk_larger_than_vocab():
    """Test that top-k larger than vocab size returns all."""
    logits = torch.tensor([0.1, -2.0, 1.5])
    filtered, indices = apply_vocab_filter(logits, top_k=10)
    
    assert len(filtered) == 3


def test_topp_basic():
    """Test that top-p filters correctly."""
    logits = torch.tensor([2.0, 1.0, 0.5, 0.1])  # Descending order
    # After softmax: [0.5765, 0.2119, 0.1199, 0.0917] (approx)
    # Cumsum: [0.5765, 0.7884, 0.9083, 1.0]
    
    filtered, indices = apply_vocab_filter(logits, top_p=0.8)
    
    # Should keep first 2 tokens (cumsum up to 0.7884)
    assert len(filtered) >= 2
    assert 0 in indices  # Highest logit always kept


def test_topp_always_keeps_one():
    """Test that top-p always keeps at least one token."""
    logits = torch.tensor([1.0, 0.5, 0.3])
    filtered, indices = apply_vocab_filter(logits, top_p=0.01)
    
    assert len(filtered) >= 1


def test_no_inf_values():
    """Test that filtered logits contain no -inf values."""
    logits = torch.tensor([0.1, -2.0, 1.5, 0.3, -5.0])
    
    # Top-k filtering
    filtered_k, _ = apply_vocab_filter(logits, top_k=2)
    assert not torch.isinf(filtered_k).any()
    
    # Top-p filtering
    filtered_p, _ = apply_vocab_filter(logits, top_p=0.5)
    assert not torch.isinf(filtered_p).any()
    
    # No filtering
    filtered_none, _ = apply_vocab_filter(logits)
    assert not torch.isinf(filtered_none).any()


def test_topk_takes_precedence():
    """Test that top-k takes precedence over top-p when both provided."""
    logits = torch.tensor([2.0, 1.0, 0.5, 0.1])
    
    # Provide both (top_k should win)
    filtered, indices = apply_vocab_filter(logits, top_k=2, top_p=0.99)
    
    assert len(filtered) == 2

