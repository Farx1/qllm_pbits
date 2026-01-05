"""
Tests for HuggingFace integration (marked as slow).
"""

import torch
import pytest
from qllm_pbits.llm.hf_wrapper import HFModel


@pytest.mark.slow
def test_hf_model_loading():
    """Test that model loads correctly."""
    model = HFModel("distilgpt2", device="cpu")
    
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.model_name == "distilgpt2"


@pytest.mark.slow
def test_hf_get_logits_shape():
    """Test that logits have correct shape."""
    model = HFModel("distilgpt2", device="cpu")
    
    input_ids = torch.tensor([[1, 2, 3]])
    logits, past_kv = model.get_logits(input_ids)
    
    # Should return logits for next token
    assert logits.shape[0] == 1  # batch size
    assert logits.shape[1] == len(model.tokenizer)  # vocab size
    assert past_kv is not None


@pytest.mark.slow
def test_hf_kv_caching():
    """Test that KV caching works."""
    model = HFModel("distilgpt2", device="cpu")
    
    input_ids = torch.tensor([[1, 2, 3]])
    logits1, past_kv = model.get_logits(input_ids)
    
    # Use cached KV for next token
    next_id = torch.tensor([[4]])
    logits2, past_kv2 = model.get_logits(next_id, past_kv)
    
    assert logits2.shape[0] == 1
    assert past_kv2 is not None

