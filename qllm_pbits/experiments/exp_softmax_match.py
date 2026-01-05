"""
Core validation experiment: P-bit sampler matches softmax distribution.

Tests on V=16, 32, 64 with strict pass criteria.
"""

import torch
import numpy as np
import pandas as pd
from qllm_pbits.token_sampler.softmax_baseline import SoftmaxMultinomialSampler
from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler
from qllm_pbits.pbits.metrics import total_variation, kl_divergence
import time


def run_softmax_match_experiment(
    V: int = 32,
    lambda_values: list[float] = [5.0, 10.0, 20.0, 50.0],
    n_samples: int = 10000,
    n_gibbs_steps: int = 100,
    device: str = "cpu",
) -> dict:
    """Test P-bit sampler vs softmax baseline.
    
    Args:
        V: Vocabulary size
        lambda_values: List of λ values to test
        n_samples: Number of samples per λ
        n_gibbs_steps: Gibbs steps per sample
        device: torch device
    
    Returns:
        Dictionary with:
            - results: DataFrame with (lambda, TV, KL, invalid_rate, time_ms)
            - best_lambda: λ with lowest TV
            - passed: Whether best λ meets all criteria
    """
    print(f"\n=== Softmax Match Experiment (V={V}) ===\n")
    
    # Generate random logits
    torch.manual_seed(42)
    logits = torch.randn(V)
    
    # Ground truth: softmax
    print("Collecting baseline samples (softmax)...")
    baseline_sampler = SoftmaxMultinomialSampler()
    baseline_samples = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        baseline_samples[i] = baseline_sampler.sample(logits, device=device)
    
    baseline_hist = torch.bincount(baseline_samples, minlength=V).float()
    baseline_dist = baseline_hist / baseline_hist.sum()
    
    results = []
    best_tv = float("inf")
    best_lambda = lambda_values[0]
    
    for lam in lambda_values:
        print(f"\nTesting lambda={lam}...")
        
        pbit_sampler = PBitOneHotSampler(lam=lam, n_gibbs_steps=n_gibbs_steps)
        pbit_samples = torch.zeros(n_samples, dtype=torch.long)
        
        start_time = time.time()
        for i in range(n_samples):
            pbit_samples[i] = pbit_sampler.sample(logits, device=device)
        elapsed_ms = (time.time() - start_time) * 1000 / n_samples
        
        pbit_hist = torch.bincount(pbit_samples, minlength=V).float()
        pbit_dist = pbit_hist / pbit_hist.sum()
        
        tv = total_variation(baseline_dist, pbit_dist)
        kl = kl_divergence(baseline_dist, pbit_dist)
        invalid_rate = pbit_sampler.get_invalid_rate()
        
        print(f"  TV: {tv:.4f} (target: < 0.02)")
        print(f"  KL: {kl:.4f} nats (target: < 0.05)")
        print(f"  Invalid rate: {invalid_rate:.4f} (target: < 0.01)")
        print(f"  Time/sample: {elapsed_ms:.2f} ms (target: < 50 ms)")
        
        passed = tv < 0.02 and kl < 0.05 and invalid_rate < 0.01 and elapsed_ms < 50
        print(f"  PASSED: {passed}")
        
        results.append({
            "lambda": lam,
            "TV": tv,
            "KL": kl,
            "invalid_rate": invalid_rate,
            "time_ms": elapsed_ms,
            "passed": passed,
        })
        
        if tv < best_tv:
            best_tv = tv
            best_lambda = lam
    
    df = pd.DataFrame(results)
    
    print(f"\n=== Summary ===")
    print(f"Best lambda: {best_lambda} (TV={best_tv:.4f})")
    print("\nResults table:")
    print(df.to_string(index=False))
    
    # Check if best lambda passes all criteria
    best_result = df[df["lambda"] == best_lambda].iloc[0]
    overall_passed = best_result["passed"]
    
    return {
        "V": V,
        "results": df,
        "best_lambda": best_lambda,
        "best_tv": best_tv,
        "passed": overall_passed,
    }


if __name__ == "__main__":
    # Run for V=32
    result_32 = run_softmax_match_experiment(V=32)
    
    # Run for V=64
    # result_64 = run_softmax_match_experiment(V=64)

