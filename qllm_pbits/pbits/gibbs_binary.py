"""
Direct Gibbs sampling on binary variables y ∈ {0,1}^V with one-hot penalty.

This module implements the mathematically correct Gibbs sampler without
any spin mapping to {-1,+1}.
"""

import torch
from torch import Tensor


class BinaryGibbsSampler:
    """Direct Gibbs sampler on {0,1}^V with one-hot penalty.
    
    Samples from π(y) ∝ exp(-β E(y)) where:
        E(y) = -Σ_k z_k y_k + λ(Σ_k y_k - 1)²
    
    The conditional probability for bit i is derived as:
        P(y_i=1 | y_{-i}) = σ(β(z_i - λ(2S - 1)))
    
    where S = Σ_{k≠i} y_k and σ is the logistic sigmoid.
    
    Uses random-scan Gibbs: at each step, pick a random index i and
    update y_i from its conditional distribution.
    
    Attributes:
        beta (float): Inverse temperature (1/T). Higher values make sampling
            more deterministic.
        device (torch.device): Device for computation.
    
    Examples:
        >>> sampler = BinaryGibbsSampler(beta=1.0, device="cpu")
        >>> logits = torch.randn(10)
        >>> y = sampler.sample(logits, lam=20.0, n_steps=100)
        >>> assert y.shape == (10,)
        >>> assert ((y == 0) | (y == 1)).all()
    """
    
    def __init__(self, beta: float = 1.0, device: str = "cpu"):
        """Initialize the binary Gibbs sampler.
        
        Args:
            beta: Inverse temperature (1/T). Default is 1.0.
            device: torch device ("cpu", "cuda", "cuda:0", etc.)
        """
        self.beta = beta
        self.device = torch.device(device)
    
    def step(self, y: Tensor, z: Tensor, lam: float) -> Tensor:
        """Perform a single random-scan Gibbs update.
        
        Picks a random index i and samples y_i from its conditional
        distribution P(y_i | y_{-i}).
        
        The conditional is derived from the energy:
            ΔE = E(y_i=1) - E(y_i=0) = -z_i + λ(2S - 1)
        where S = Σ_{k≠i} y_k.
        
        Then:
            P(y_i=1 | y_{-i}) = σ(-β ΔE) = σ(β(z_i - λ(2S - 1)))
        
        Args:
            y: Current binary state (V,) in {0,1}
            z: Logits (V,)
            lam: Penalty strength λ for one-hot constraint
        
        Returns:
            Updated y with one bit resampled (modifies in place)
        """
        V = y.shape[0]
        
        # Pick random index
        i = torch.randint(0, V, (1,), device=self.device).item()
        
        # Compute S = Σ_{k≠i} y_k
        S_total = y.sum()
        S_minus_i = S_total - y[i]
        
        # Conditional probability P(y_i=1 | y_{-i})
        # ΔE = -z_i + λ(2S - 1)
        # P(y_i=1) = σ(β(z_i - λ(2S - 1)))
        delta_E = -z[i] + lam * (2 * S_minus_i - 1)
        p_one = torch.sigmoid(self.beta * (-delta_E))
        
        # Sample new y_i
        y[i] = torch.bernoulli(p_one).item()
        return y
    
    def sample(
        self,
        z: Tensor,
        lam: float,
        n_steps: int,
        burn_in: int = 0,
        initial_state: Tensor | None = None,
    ) -> Tensor:
        """Run Gibbs chain and return final state.
        
        Args:
            z: Logits (V,)
            lam: Penalty strength λ
            n_steps: Number of sampling steps (after burn-in)
            burn_in: Number of burn-in steps to discard
            initial_state: Starting state (default: random one-hot)
        
        Returns:
            Final binary state (V,) after burn_in + n_steps iterations
        
        Examples:
            >>> sampler = BinaryGibbsSampler(beta=1.0)
            >>> z = torch.tensor([1.0, 0.5, 2.0, 0.1])
            >>> y = sampler.sample(z, lam=10.0, n_steps=100, burn_in=50)
            >>> y.sum()  # Should be close to 1 with high λ
            tensor(1.)
        """
        V = z.shape[0]
        
        # Initialize
        if initial_state is not None:
            y = initial_state.clone().to(self.device)
        else:
            y = torch.zeros(V, device=self.device)
            y[torch.randint(0, V, (1,))] = 1  # Start with random one-hot
        
        # Burn-in + sampling
        for step in range(burn_in + n_steps):
            y = self.step(y, z, lam)
        
        return y
    
    def sample_chain(
        self, z: Tensor, lam: float, n_steps: int, burn_in: int = 0
    ) -> Tensor:
        """Run Gibbs chain and return full trajectory for diagnostics.
        
        Useful for computing autocorrelation, ESS, and visualizing
        the convergence behavior.
        
        Args:
            z: Logits (V,)
            lam: Penalty strength
            n_steps: Number of steps to collect (after burn-in)
            burn_in: Number of burn-in steps to discard
        
        Returns:
            chain: (n_steps, V) binary states
        
        Examples:
            >>> sampler = BinaryGibbsSampler(beta=1.0)
            >>> z = torch.randn(8)
            >>> chain = sampler.sample_chain(z, lam=20.0, n_steps=1000, burn_in=100)
            >>> chain.shape
            torch.Size([1000, 8])
        """
        V = z.shape[0]
        y = torch.zeros(V, device=self.device)
        y[torch.randint(0, V, (1,))] = 1
        
        # Burn-in
        for _ in range(burn_in):
            y = self.step(y, z, lam)
        
        # Collect chain
        chain = torch.zeros(n_steps, V, device=self.device)
        for t in range(n_steps):
            y = self.step(y, z, lam)
            chain[t] = y.clone()
        
        return chain

