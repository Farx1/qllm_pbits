"""
Abstract base class for token samplers.
"""

from abc import ABC, abstractmethod
from torch import Tensor


class TokenSampler(ABC):
    """Abstract interface for sampling tokens from logits.
    
    All token samplers must implement the sample() method, which takes
    unnormalized logits and returns a single token index.
    """
    
    @abstractmethod
    def sample(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        device: str = "cpu",
    ) -> int:
        """Sample a token index from logits.
        
        Args:
            logits: (V,) unnormalized log-probabilities
            temperature: Scaling factor (higher = more random).
                Logits are divided by temperature before sampling.
            top_k: Keep only top-k tokens (None = no filtering)
            top_p: Nucleus sampling threshold (None = no filtering)
            device: torch device for computation
        
        Returns:
            token_id: Integer in [0, V) representing the sampled token
        
        Notes:
            - If both top_k and top_p are provided, top_k takes precedence
            - Temperature should be positive; values < 1 make sampling
              more deterministic, values > 1 make it more random
        """
        pass

