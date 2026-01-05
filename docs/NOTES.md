# Research Notes (handwritten-style)

This is a lightweight devlog of what I tried, what worked, and what surprised me. It's not meant to be polished.

## Iteration 1 — get the math right first
- Wrote the energy on y ∈ {0,1}^V with a one-hot penalty.
- Derived the single-site Gibbs conditional directly (no spin mapping).
- Added unit tests to validate:
  - conditional probability behavior
  - chain shapes / binary states
  - one-hot tendency when λ is large

**Takeaway**: correctness is the easiest part to get wrong if you do a sloppy spin conversion.

## Iteration 2 — make top-k/top-p compatible with energy-based sampling
- Usual LLM sampling masks logits with -∞. That breaks energy-based methods.
- Implemented *vocabulary reduction* instead: keep a subset of indices and sample in that reduced space.

**Takeaway**: the filtering method is part of the model when you move to EBMs.

## Iteration 3 — measure the trade-off (V=32, fixed step budget)
With a fixed number of Gibbs steps, I observed a consistent pattern:
- Increasing λ drives the invalid-rate toward 0%.
- But TV distance to the softmax baseline can worsen, likely due to harder mixing.

**Takeaway**: "more constraint" is not free — it changes the geometry of the chain.

## Iteration 4 — stress test (V=16) exposed mixing issues
A small V=16 qualitative test produced a clear distribution shift (wrong top tokens), even though one-hot validity was perfect.

**Takeaway**: validity is not enough. You need mixing diagnostics (ESS/autocorr) and better update strategies.

## What I would try next
- Blocked Gibbs or systematic sweeps (instead of random-scan)
- Annealing schedules (λ(t), β(t))
- Multiple chains + ESS reporting
- Compare downstream text quality (perplexity/coherence), not only distribution metrics

