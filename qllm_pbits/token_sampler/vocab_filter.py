"""
Vocabulary filtering utilities for top-k and top-p (nucleus) sampling.

CRITICAL: This module implements vocabulary reduction WITHOUT using -inf
masking, which is essential for energy-based samplers.
"""

import torch
from torch import Tensor


def apply_vocab_filter(
    logits: Tensor, top_k: int | None = None, top_p: float | None = None
) -> tuple[Tensor, Tensor]:
    """Reduce vocabulary to filtered subset.
    
    Returns ONLY the logits and indices of kept tokens (no -inf masking).
    This is critical for energy-based samplers that cannot handle infinite
    energies.
    
    Args:
        logits: (V,) unnormalized logits
        top_k: Keep top-k highest logits (None = no filtering)
        top_p: Nucleus sampling - keep smallest set with cumulative
            probability > p (None = no filtering)
    
    Returns:
        filtered_logits: (V',) logits for kept tokens (V' ≤ V)
        keep_indices: (V',) original indices of kept tokens
    
    Notes:
        - If both top_k and top_p are provided, top_k takes precedence
        - At least one token is always kept
        - No -inf values are introduced
    
    Examples:
        >>> logits = torch.tensor([0.1, -2.0, 1.5, 0.3])
        >>> filtered, indices = apply_vocab_filter(logits, top_k=2)
        >>> filtered
        tensor([1.5, 0.3])
        >>> indices
        tensor([2, 3])
        
        >>> # No -inf values in output
        >>> assert not torch.isinf(filtered).any()
    """
    # Top-k filtering
    if top_k is not None:
        top_k = min(top_k, logits.shape[0])
        values, indices = torch.topk(logits, top_k)
        return values, indices
    
    # Top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=0)
        cumsum = torch.cumsum(probs, dim=0)
        
        # Keep tokens until cumulative probability exceeds top_p
        mask = cumsum <= top_p
        mask[0] = True  # Always keep at least one token
        
        return sorted_logits[mask], sorted_indices[mask]
    
    # No filtering: return all tokens
    return logits, torch.arange(logits.shape[0], device=logits.device)

