"""
End-to-end pipeline tests (marked as slow).
"""

import torch
import pytest
from qllm_pbits.llm.hf_wrapper import HFModel
from qllm_pbits.llm.generate import generate_text
from qllm_pbits.token_sampler.softmax_baseline import SoftmaxMultinomialSampler
from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler


@pytest.mark.slow
def test_generate_text_softmax():
    """Test text generation with softmax sampler."""
    model = HFModel("distilgpt2", device="cpu")
    sampler = SoftmaxMultinomialSampler()
    
    result = generate_text(
        model, 
        "Hello", 
        sampler, 
        max_tokens=10, 
        temperature=1.0,
        device="cpu"
    )
    
    assert "text" in result
    assert "tokens" in result
    assert "time_per_token" in result
    assert "total_time" in result
    assert len(result["tokens"]) <= 10
    assert isinstance(result["text"], str)


@pytest.mark.slow
def test_generate_text_pbit():
    """Test text generation with P-bit sampler."""
    model = HFModel("distilgpt2", device="cpu")
    sampler = PBitOneHotSampler(lam=20.0, n_gibbs_steps=50)
    
    result = generate_text(
        model,
        "The future",
        sampler,
        max_tokens=10,
        temperature=1.0,
        device="cpu",
    )
    
    assert "text" in result
    assert "tokens" in result
    assert "invalid_rate" in result  # P-bit specific
    assert len(result["tokens"]) <= 10


@pytest.mark.slow
def test_generate_with_topk():
    """Test generation with top-k filtering."""
    model = HFModel("distilgpt2", device="cpu")
    sampler = SoftmaxMultinomialSampler()
    
    result = generate_text(
        model,
        "Once upon",
        sampler,
        max_tokens=5,
        top_k=10,
        device="cpu",
    )
    
    assert len(result["tokens"]) <= 5

