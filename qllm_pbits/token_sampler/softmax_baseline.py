"""
Baseline token sampler using standard softmax + multinomial sampling.

This provides the ground truth distribution for validation.
"""

import torch
from torch import Tensor

from qllm_pbits.token_sampler.base import TokenSampler
from qllm_pbits.token_sampler.vocab_filter import apply_vocab_filter


class SoftmaxMultinomialSampler(TokenSampler):
    """Ground truth baseline using torch.multinomial.
    
    Implements standard softmax sampling with optional temperature scaling
    and top-k/top-p filtering.
    
    Examples:
        >>> sampler = SoftmaxMultinomialSampler()
        >>> logits = torch.tensor([1.0, 2.0, 0.5, 1.5])
        >>> token_id = sampler.sample(logits, temperature=1.0, top_k=3)
        >>> 0 <= token_id < 4
        True
    """
    
    def sample(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        device: str = "cpu",
    ) -> int:
        """Sample a token using softmax + multinomial.
        
        Args:
            logits: (V,) unnormalized log-probabilities
            temperature: Scaling factor (higher = more random)
            top_k: Keep only top-k tokens (None = no filtering)
            top_p: Nucleus sampling threshold (None = no filtering)
            device: torch device
        
        Returns:
            token_id: Integer in [0, V)
        """
        logits = logits.to(device)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Filter vocabulary (returns reduced set, NOT -inf masked)
        filtered_logits, keep_indices = apply_vocab_filter(
            scaled_logits, top_k, top_p
        )
        
        # Sample from filtered distribution
        probs = torch.softmax(filtered_logits, dim=0)
        local_idx = torch.multinomial(probs, 1).item()
        
        # Map back to original vocabulary
        return keep_indices[local_idx].item()

