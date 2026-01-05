"""
Metrics for evaluating sampler quality.

Includes Total Variation, KL divergence, Effective Sample Size,
and one-hot constraint violation rate.
"""

import torch
from torch import Tensor
import numpy as np


def total_variation(p: Tensor, q: Tensor) -> float:
    """Compute Total Variation distance TV(p, q) = 0.5 * Σ |p_i - q_i|.
    
    Measures the maximum difference in probability that the two distributions
    assign to the same event. Range: [0, 1].
    
    Args:
        p: First probability distribution (V,)
        q: Second probability distribution (V,)
    
    Returns:
        TV distance in [0, 1]
    
    Examples:
        >>> p = torch.tensor([0.5, 0.3, 0.2])
        >>> q = torch.tensor([0.4, 0.4, 0.2])
        >>> total_variation(p, q)
        0.1
    """
    return 0.5 * torch.abs(p - q).sum().item()


def kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-10) -> float:
    """Compute KL divergence KL(p||q) = Σ p_i log(p_i/q_i).
    
    Measures how much information is lost when q is used to approximate p.
    Always non-negative, zero iff p = q almost everywhere.
    
    Args:
        p: True distribution (V,)
        q: Approximate distribution (V,)
        eps: Small constant to avoid log(0)
    
    Returns:
        KL divergence in nats (use / np.log(2) for bits)
    
    Examples:
        >>> p = torch.tensor([0.5, 0.3, 0.2])
        >>> q = torch.tensor([0.4, 0.4, 0.2])
        >>> kl = kl_divergence(p, q)
        >>> kl > 0
        True
    """
    p = p + eps
    q = q + eps
    # Renormalize
    p = p / p.sum()
    q = q / q.sum()
    return (p * torch.log(p / q)).sum().item()


def effective_sample_size(chain: Tensor, max_lag: int = 100) -> float:
    """Compute Effective Sample Size (ESS) using autocorrelation.
    
    ESS measures the number of independent samples equivalent to the
    correlated chain. Uses the formula:
        ESS = N / (1 + 2 Σ_k ρ_k)
    where ρ_k is the autocorrelation at lag k.
    
    Args:
        chain: MCMC chain (n_steps, V) or (n_steps,) for single variable
        max_lag: Maximum lag to compute autocorrelation
    
    Returns:
        Effective sample size (higher is better)
    
    Examples:
        >>> # Independent samples: ESS ≈ N
        >>> chain = torch.randn(1000, 10)
        >>> ess = effective_sample_size(chain)
        >>> ess > 500  # Should be close to 1000
        True
    """
    if chain.ndim == 1:
        chain = chain.unsqueeze(1)
    
    n_steps, n_vars = chain.shape
    
    # Compute autocorrelation for each variable
    ess_per_var = []
    for v in range(n_vars):
        x = chain[:, v].cpu().numpy()
        x = x - x.mean()
        
        # Compute autocorrelation using FFT
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Truncate at max_lag
        autocorr = autocorr[:min(max_lag, len(autocorr))]
        
        # Sum autocorrelations (stop if they become negative)
        tau = 1.0
        for k in range(1, len(autocorr)):
            if autocorr[k] < 0:
                break
            tau += 2 * autocorr[k]
        
        ess_per_var.append(n_steps / tau)
    
    return float(np.mean(ess_per_var))


def invalid_onehot_rate(samples: Tensor) -> float:
    """Compute fraction of samples that are not valid one-hot vectors.
    
    A valid one-hot vector has exactly one element equal to 1 and all
    others equal to 0.
    
    Args:
        samples: (N, V) binary samples
    
    Returns:
        Fraction in [0, 1] of invalid samples (0 = all valid)
    
    Examples:
        >>> # All valid one-hot
        >>> samples = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> invalid_onehot_rate(samples)
        0.0
        
        >>> # One invalid
        >>> samples = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        >>> invalid_onehot_rate(samples)
        0.333...
    """
    is_binary = ((samples == 0) | (samples == 1)).all(dim=1)
    sum_is_one = (samples.sum(dim=1) == 1)
    valid = is_binary & sum_is_one
    return 1 - valid.float().mean().item()


def autocorrelation(chain: Tensor, max_lag: int = 100) -> Tensor:
    """Compute autocorrelation function of a chain.
    
    Args:
        chain: MCMC chain (n_steps,) for single variable
        max_lag: Maximum lag to compute
    
    Returns:
        Autocorrelation (max_lag,)
    """
    x = chain.cpu().numpy()
    x = x - x.mean()
    
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]
    
    return torch.tensor(autocorr[:max_lag])

