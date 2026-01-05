"""
Tools for calibrating lambda parameter and analyzing convergence.
"""

import torch
from torch import Tensor
import numpy as np
import pandas as pd
from typing import Any
import time

from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler
from qllm_pbits.token_sampler.softmax_baseline import SoftmaxMultinomialSampler
from qllm_pbits.pbits.metrics import total_variation, kl_divergence


def calibrate_lambda(
    logits: Tensor,
    lambda_range: list[float],
    n_samples: int = 10000,
    target_tv: float = 0.01,
    n_gibbs_steps: int = 100,
    beta: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Grid search over λ to minimize TV distance to softmax.
    
    For each λ value, samples tokens using both softmax and p-bit samplers,
    then computes TV distance, KL divergence, invalid rate, and timing.
    
    Args:
        logits: (V,) unnormalized logits to sample from
        lambda_range: List of λ values to test
        n_samples: Number of samples per λ
        target_tv: Target TV distance (for reference)
        n_gibbs_steps: Gibbs steps per sample
        beta: Inverse temperature
        device: torch device
    
    Returns:
        Dictionary with:
            - best_lambda: λ value with lowest TV
            - best_tv: Lowest TV achieved
            - results: DataFrame with columns (lambda, TV, KL, invalid_rate, time_ms)
    
    Examples:
        >>> logits = torch.randn(32)
        >>> result = calibrate_lambda(logits, [5.0, 10.0, 20.0])
        >>> result['best_lambda']
        20.0
    """
    logits = logits.to(device)
    V = logits.shape[0]
    
    # Ground truth: softmax distribution
    baseline_sampler = SoftmaxMultinomialSampler()
    baseline_samples = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        baseline_samples[i] = baseline_sampler.sample(logits, device=device)
    
    # Compute empirical distribution
    baseline_hist = torch.bincount(baseline_samples, minlength=V).float()
    baseline_dist = baseline_hist / baseline_hist.sum()
    
    results = []
    best_tv = float("inf")
    best_lambda = lambda_range[0]
    
    for lam in lambda_range:
        # P-bit sampler
        pbit_sampler = PBitOneHotSampler(
            lam=lam, n_gibbs_steps=n_gibbs_steps, beta=beta
        )
        
        pbit_samples = torch.zeros(n_samples, dtype=torch.long)
        start_time = time.time()
        
        for i in range(n_samples):
            pbit_samples[i] = pbit_sampler.sample(logits, device=device)
        
        elapsed = (time.time() - start_time) * 1000 / n_samples  # ms per sample
        
        # Compute empirical distribution
        pbit_hist = torch.bincount(pbit_samples, minlength=V).float()
        pbit_dist = pbit_hist / pbit_hist.sum()
        
        # Metrics
        tv = total_variation(baseline_dist, pbit_dist)
        kl = kl_divergence(baseline_dist, pbit_dist)
        invalid_rate = pbit_sampler.get_invalid_rate()
        
        results.append({
            "lambda": lam,
            "TV": tv,
            "KL": kl,
            "invalid_rate": invalid_rate,
            "time_ms": elapsed,
        })
        
        if tv < best_tv:
            best_tv = tv
            best_lambda = lam
    
    df = pd.DataFrame(results)
    
    return {
        "best_lambda": best_lambda,
        "best_tv": best_tv,
        "target_tv": target_tv,
        "results": df,
    }


def convergence_analysis(
    logits: Tensor,
    lam: float,
    step_range: list[int],
    n_samples: int = 1000,
    beta: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Analyze TV/KL vs number of Gibbs steps.
    
    Args:
        logits: (V,) logits to sample from
        lam: Penalty strength
        step_range: List of n_gibbs_steps values to test
        n_samples: Number of samples per step count
        beta: Inverse temperature
        device: torch device
    
    Returns:
        Dictionary with:
            - results: DataFrame with (n_steps, TV, KL, invalid_rate, time_ms)
            - recommended_steps: Minimum steps to achieve TV < 0.02
    """
    logits = logits.to(device)
    V = logits.shape[0]
    
    # Ground truth
    baseline_sampler = SoftmaxMultinomialSampler()
    baseline_samples = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        baseline_samples[i] = baseline_sampler.sample(logits, device=device)
    
    baseline_hist = torch.bincount(baseline_samples, minlength=V).float()
    baseline_dist = baseline_hist / baseline_hist.sum()
    
    results = []
    recommended_steps = step_range[-1]  # Default to max
    
    for n_steps in step_range:
        pbit_sampler = PBitOneHotSampler(
            lam=lam, n_gibbs_steps=n_steps, beta=beta
        )
        
        pbit_samples = torch.zeros(n_samples, dtype=torch.long)
        start_time = time.time()
        
        for i in range(n_samples):
            pbit_samples[i] = pbit_sampler.sample(logits, device=device)
        
        elapsed = (time.time() - start_time) * 1000 / n_samples
        
        pbit_hist = torch.bincount(pbit_samples, minlength=V).float()
        pbit_dist = pbit_hist / pbit_hist.sum()
        
        tv = total_variation(baseline_dist, pbit_dist)
        kl = kl_divergence(baseline_dist, pbit_dist)
        invalid_rate = pbit_sampler.get_invalid_rate()
        
        results.append({
            "n_steps": n_steps,
            "TV": tv,
            "KL": kl,
            "invalid_rate": invalid_rate,
            "time_ms": elapsed,
        })
        
        # Find minimum steps for TV < 0.02
        if tv < 0.02 and n_steps < recommended_steps:
            recommended_steps = n_steps
    
    df = pd.DataFrame(results)
    
    return {
        "results": df,
        "recommended_steps": recommended_steps,
    }

