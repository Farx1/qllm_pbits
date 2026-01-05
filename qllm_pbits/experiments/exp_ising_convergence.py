"""
Validation experiment: Ising model exact convergence test.

Uses exact enumeration for small systems (N=8, 12) to verify that
Gibbs sampling converges to the true Boltzmann distribution.
"""

import torch
import numpy as np
from qllm_pbits.pbits.ising import IsingModel
from qllm_pbits.pbits.metrics import total_variation, kl_divergence, effective_sample_size


def run_ising_convergence_experiment(
    N: int = 8,
    beta: float = 1.0,
    n_samples: int = 10000,
    burn_in: int = 5000,
    device: str = "cpu",
) -> dict:
    """Test Gibbs convergence on Ising model with exact distribution.
    
    Args:
        N: Number of spins (max 16 for exact enumeration)
        beta: Inverse temperature
        n_samples: Number of samples to collect
        burn_in: Burn-in steps
        device: torch device
    
    Returns:
        Dictionary with:
            - tv: Total variation distance
            - kl: KL divergence
            - ess: Effective sample size
            - passed: Whether all criteria are met
    """
    print(f"\n=== Ising Convergence Test (N={N}) ===\n")
    
    # Create random Ising model
    torch.manual_seed(42)
    J = torch.randn(N, N) * 0.5
    J = (J + J.T) / 2  # Make symmetric
    J.fill_diagonal_(0)  # No self-interaction
    h = torch.randn(N) * 0.5
    
    model = IsingModel(N=N, J=J, h=h, beta=beta, device=device)
    
    # Compute exact distribution
    print("Computing exact distribution...")
    exact_probs, exact_states = model.exact_distribution()
    
    # Sample using Gibbs
    print(f"Running Gibbs sampling ({n_samples} samples, {burn_in} burn-in)...")
    chain = model.sample_chain(n_steps=n_samples, burn_in=burn_in)
    
    # Compute empirical distribution
    # Convert states to integers for bincount
    def state_to_int(state):
        """Convert spin state to integer index."""
        binary = ((state + 1) / 2).long()  # Convert {-1,+1} to {0,1}
        return sum(b * (2**i) for i, b in enumerate(binary))
    
    sample_indices = torch.tensor([state_to_int(chain[i]) for i in range(n_samples)])
    empirical_hist = torch.bincount(sample_indices, minlength=2**N).float()
    empirical_dist = empirical_hist / empirical_hist.sum()
    
    # Compute metrics
    tv = total_variation(exact_probs, empirical_dist)
    kl = kl_divergence(exact_probs, empirical_dist)
    
    # Compute ESS for first spin
    first_spin_chain = chain[:, 0]
    ess = effective_sample_size(first_spin_chain)
    
    print(f"\nResults:")
    print(f"  TV distance: {tv:.4f} (target: < 0.01)")
    print(f"  KL divergence: {kl:.4f} nats (target: < 0.05)")
    print(f"  ESS (first spin): {ess:.1f} (target: > 500)")
    
    # Pass criteria
    passed = tv < 0.01 and kl < 0.05 and ess > 500
    print(f"\n  PASSED: {passed}")
    
    return {
        "N": N,
        "tv": tv,
        "kl": kl,
        "ess": ess,
        "passed": passed,
    }


if __name__ == "__main__":
    # Run for N=8
    result_8 = run_ising_convergence_experiment(N=8)
    
    # Run for N=12 (takes longer)
    # result_12 = run_ising_convergence_experiment(N=12)

