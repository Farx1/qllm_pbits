"""
P-bit token sampler using direct binary Gibbs with one-hot penalty.

Implements explicit resample fallback strategy:
1. Run Gibbs for n_gibbs_steps
2. If result is not valid one-hot, resample (up to max_resample times)
3. After max_resample failures, use argmax(y) as fallback
4. Track invalid-rate for diagnostics
"""

import torch
from torch import Tensor

from qllm_pbits.token_sampler.base import TokenSampler
from qllm_pbits.token_sampler.vocab_filter import apply_vocab_filter
from qllm_pbits.pbits.gibbs_binary import BinaryGibbsSampler


class PBitOneHotSampler(TokenSampler):
    """P-bit token sampler using direct binary Gibbs with one-hot penalty.
    
    This sampler uses Gibbs sampling on binary variables with a penalty
    energy that encourages one-hot states. If the sampler fails to produce
    a valid one-hot vector after multiple attempts, it falls back to argmax.
    
    Attributes:
        lam (float): One-hot penalty strength λ
        n_gibbs_steps (int): Number of Gibbs iterations per sample
        beta (float): Inverse temperature (1/T)
        max_resample (int): Max resample attempts before fallback
        invalid_count (int): Count of samples requiring fallback
        total_count (int): Total number of samples
    
    Examples:
        >>> sampler = PBitOneHotSampler(lam=20.0, n_gibbs_steps=100)
        >>> logits = torch.randn(50)
        >>> token_id = sampler.sample(logits, temperature=1.0, top_k=10)
        >>> sampler.get_invalid_rate()  # Should be low with high λ
        0.0
    """
    
    def __init__(
        self,
        lam: float = 10.0,
        n_gibbs_steps: int = 100,
        beta: float = 1.0,
        max_resample: int = 3,
    ):
        """Initialize P-bit sampler.
        
        Args:
            lam: One-hot penalty strength λ (higher = stronger constraint)
            n_gibbs_steps: Gibbs iterations per sample
            beta: Inverse temperature (1/T)
            max_resample: Max resample attempts before argmax fallback
        """
        self.lam = lam
        self.n_gibbs_steps = n_gibbs_steps
        self.beta = beta
        self.max_resample = max_resample
        self.invalid_count = 0
        self.total_count = 0
    
    def sample(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        device: str = "cpu",
    ) -> int:
        """Sample a token using P-bit Gibbs sampler.
        
        Strategy:
        1. Apply temperature and filter vocabulary
        2. Run Gibbs sampling on reduced vocabulary
        3. Check if result is valid one-hot
        4. If not, resample up to max_resample times
        5. If still invalid, use argmax as fallback
        
        Args:
            logits: (V,) unnormalized log-probabilities
            temperature: Scaling factor
            top_k: Keep only top-k tokens
            top_p: Nucleus sampling threshold
            device: torch device
        
        Returns:
            token_id: Integer in [0, V)
        """
        logits = logits.to(device)
        
        # Apply temperature to logits
        scaled_logits = logits / temperature
        
        # Filter vocabulary (NO -inf, just reduction)
        filtered_logits, keep_indices = apply_vocab_filter(
            scaled_logits, top_k, top_p
        )
        
        # Gibbs sampling on reduced vocabulary V'
        sampler = BinaryGibbsSampler(beta=self.beta, device=device)
        
        # Resample strategy: try max_resample times
        for attempt in range(self.max_resample):
            y = sampler.sample(
                filtered_logits,
                self.lam,
                self.n_gibbs_steps,
                burn_in=self.n_gibbs_steps // 2,
            )
            
            # Check if valid one-hot
            is_binary = ((y == 0) | (y == 1)).all().item()
            sum_is_one = y.sum().item() == 1
            
            if is_binary and sum_is_one:
                # Valid one-hot: return token
                self.total_count += 1
                local_idx = y.argmax().item()
                return keep_indices[local_idx].item()
        
        # All resamples failed: use argmax fallback
        self.invalid_count += 1
        self.total_count += 1
        
        local_idx = filtered_logits.argmax().item()
        return keep_indices[local_idx].item()
    
    def get_invalid_rate(self) -> float:
        """Return fraction of samples that required argmax fallback.
        
        Returns:
            Invalid rate in [0, 1] (0 = all samples were valid one-hot)
        """
        if self.total_count == 0:
            return 0.0
        return self.invalid_count / self.total_count
    
    def reset_stats(self):
        """Reset invalid-rate counters."""
        self.invalid_count = 0
        self.total_count = 0

