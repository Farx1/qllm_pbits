"""
Ising model implementation for validation experiments only.

This module implements standard Ising spin systems on {-1,+1} for testing
Gibbs convergence via exact enumeration on small systems.
"""

import torch
from torch import Tensor
import itertools


class IsingModel:
    """Ising model with spins m ∈ {-1,+1}^N.
    
    Energy: E(m) = -Σ_{ij} J_ij m_i m_j - Σ_i h_i m_i
    
    Used only for validation: can compute exact distribution for N ≤ 12
    by enumerating all 2^N states.
    
    Attributes:
        N (int): Number of spins
        J (Tensor): Coupling matrix (N, N)
        h (Tensor): External field (N,)
        beta (float): Inverse temperature
        device (torch.device): Device for computation
    """
    
    def __init__(
        self,
        N: int,
        J: Tensor | None = None,
        h: Tensor | None = None,
        beta: float = 1.0,
        device: str = "cpu",
    ):
        """Initialize Ising model.
        
        Args:
            N: Number of spins
            J: Coupling matrix (N, N). Default: zeros
            h: External field (N,). Default: zeros
            beta: Inverse temperature
            device: torch device
        """
        self.N = N
        self.beta = beta
        self.device = torch.device(device)
        
        if J is None:
            self.J = torch.zeros(N, N, device=self.device)
        else:
            self.J = J.to(self.device)
        
        if h is None:
            self.h = torch.zeros(N, device=self.device)
        else:
            self.h = h.to(self.device)
    
    def energy(self, m: Tensor) -> Tensor:
        """Compute energy E(m) = -Σ_{ij} J_ij m_i m_j - Σ_i h_i m_i.
        
        Args:
            m: Spin configuration (N,) in {-1, +1}
        
        Returns:
            Energy scalar
        """
        interaction = -torch.sum(self.J * torch.outer(m, m))
        field = -torch.sum(self.h * m)
        return interaction + field
    
    def exact_distribution(self) -> tuple[Tensor, Tensor]:
        """Compute exact Boltzmann distribution by enumerating all states.
        
        Only feasible for N ≤ 12. Returns normalized probabilities and
        corresponding spin configurations.
        
        Returns:
            probs: (2^N,) probabilities
            states: (2^N, N) spin configurations
        
        Raises:
            ValueError: If N > 16 (too large for exact enumeration)
        """
        if self.N > 16:
            raise ValueError(f"N={self.N} too large for exact enumeration (max 16)")
        
        # Generate all 2^N binary strings
        all_configs = list(itertools.product([-1, 1], repeat=self.N))
        states = torch.tensor(all_configs, dtype=torch.float32, device=self.device)
        
        # Compute energies
        energies = torch.tensor(
            [self.energy(state) for state in states], device=self.device
        )
        
        # Boltzmann distribution
        log_probs = -self.beta * energies
        log_probs = log_probs - torch.logsumexp(log_probs, dim=0)
        probs = torch.exp(log_probs)
        
        return probs, states
    
    def gibbs_step(self, m: Tensor) -> Tensor:
        """Single random-scan Gibbs update for spins.
        
        Args:
            m: Current spin configuration (N,) in {-1, +1}
        
        Returns:
            Updated m with one spin flipped
        """
        i = torch.randint(0, self.N, (1,), device=self.device).item()
        
        # Local field at site i
        h_local = self.h[i] + torch.sum(self.J[i] * m) - self.J[i, i] * m[i]
        
        # Conditional probability P(m_i=+1 | m_{-i})
        p_plus = torch.sigmoid(2 * self.beta * h_local)
        
        # Sample
        if torch.rand(1, device=self.device).item() < p_plus:
            m[i] = 1.0
        else:
            m[i] = -1.0
        
        return m
    
    def sample_chain(self, n_steps: int, burn_in: int = 0) -> Tensor:
        """Run Gibbs chain and return trajectory.
        
        Args:
            n_steps: Number of steps to collect
            burn_in: Number of burn-in steps
        
        Returns:
            chain: (n_steps, N) spin configurations
        """
        m = torch.ones(self.N, device=self.device)  # Start at all +1
        
        # Burn-in
        for _ in range(burn_in):
            m = self.gibbs_step(m)
        
        # Collect chain
        chain = torch.zeros(n_steps, self.N, device=self.device)
        for t in range(n_steps):
            m = self.gibbs_step(m)
            chain[t] = m.clone()
        
        return chain

