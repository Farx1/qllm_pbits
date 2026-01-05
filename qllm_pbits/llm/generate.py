"""
Text generation utilities with custom samplers.
"""

import torch
from torch import Tensor
import time
import numpy as np
from typing import Any

from qllm_pbits.llm.hf_wrapper import HFModel
from qllm_pbits.token_sampler.base import TokenSampler


def generate_text(
    model: HFModel,
    prompt: str,
    sampler: TokenSampler,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Autoregressive text generation with custom sampler.
    
    Args:
        model: HFModel instance
        prompt: Input text to start generation
        sampler: TokenSampler instance (e.g., SoftmaxMultinomialSampler, PBitOneHotSampler)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering (None = no filtering)
        top_p: Nucleus sampling (None = no filtering)
        device: torch device
    
    Returns:
        Dictionary with:
            - text: Generated text (decoded)
            - tokens: List of generated token IDs
            - time_per_token: Average time per token (seconds)
            - total_time: Total generation time (seconds)
            - invalid_rate: If sampler has get_invalid_rate() method
    
    Examples:
        >>> from qllm_pbits.llm import HFModel, generate_text
        >>> from qllm_pbits.token_sampler import SoftmaxMultinomialSampler
        >>> model = HFModel("distilgpt2")
        >>> sampler = SoftmaxMultinomialSampler()
        >>> result = generate_text(model, "Hello", sampler, max_tokens=10)
        >>> len(result['tokens']) <= 10
        True
    """
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_kv = None
    
    generated_ids = []
    times = []
    
    # Reset sampler stats if available
    if hasattr(sampler, "reset_stats"):
        sampler.reset_stats()
    
    for _ in range(max_tokens):
        start = time.time()
        
        # Get next-token logits
        logits, past_kv = model.get_logits(input_ids, past_kv)
        
        # Sample next token
        token_id = sampler.sample(logits[0], temperature, top_k, top_p, device)
        
        times.append(time.time() - start)
        generated_ids.append(token_id)
        
        # Update input for next iteration
        input_ids = torch.tensor([[token_id]], device=device)
        
        # Stop at EOS token
        if token_id == model.tokenizer.eos_token_id:
            break
    
    # Decode generated text
    text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    result = {
        "text": text,
        "tokens": generated_ids,
        "time_per_token": float(np.mean(times)),
        "total_time": float(sum(times)),
    }
    
    # Add invalid rate if available
    if hasattr(sampler, "get_invalid_rate"):
        result["invalid_rate"] = sampler.get_invalid_rate()
    
    return result

