# Engineering Decisions (and Why)

This project is intentionally small and explicit. The goal is to make every modeling and engineering choice easy to audit.

## 1) Binary state space: y ∈ {0,1}^V (no Ising spin mapping)

**Decision**: sample directly over binary variables y instead of mapping to spins s∈{-1,+1}.

**Why**:
- The token constraint is naturally **one-hot** in {0,1}^V.
- Spin mappings add extra bookkeeping and are a common source of sign/scale bugs.
- Deriving the Gibbs conditional directly in {0,1}^V keeps the sampler mathematically transparent.

## 2) Energy function: logits + one-hot penalty

We use a penalty energy over binary vectors:

```
E(y) = -Σ_k z_k y_k + λ(Σ_k y_k - 1)²
```

**Why**:
- The first term couples the state to the model's logits (scores).
- The quadratic penalty lets us keep the sampler on {0,1}^V while encouraging one-hot states.

**Trade-off**:
- Increasing λ reduces invalid one-hot samples, but makes the energy landscape stiffer and can hurt mixing for a fixed step budget.

## 3) Direct Gibbs conditional (single-site, random-scan)

**Decision**: implement **single-site** random-scan Gibbs.

**Why**:
- Minimal implementation surface.
- Easy to test and reason about.
- Matches the "p-bit" mental model (each bit fluctuates stochastically).

**Known limitation**:
- Mixing can be poor in stiff regimes (large λ) unless you increase steps or use blocked/tempered variants.

## 4) Top-k / Top-p as vocabulary reduction (no `-inf` masking)

**Decision**: apply top-k/top-p by **reducing the vocabulary** to a subset of indices.

**Why**:
- Energy-based samplers don't behave well with infinite energies.
- `-inf` logits (common in standard decoding) are a perfect fit for softmax, but a bad fit for a Boltzmann energy formulation.

## 5) One-hot enforcement policy: resample ×3 then argmax

**Decision**:
1. Run Gibbs for `n_gibbs_steps`.
2. If the final state is not valid one-hot, resample (up to `max_resample` times).
3. If still invalid, fall back to `argmax` over filtered logits.

**Why**:
- Keeps the interface identical to standard token sampling (always returns a token id).
- Prevents rare invalid states from crashing a text generation loop.

**What the invalid-rate means in this repo**:
- `invalid_rate` counts how often we had to use the final argmax fallback.
- It is **not** the same as "raw invalid Gibbs states" (which would require logging every attempt).

## 6) Results reporting philosophy

**Decision**: report what the implementation actually does, even when it looks bad.

- TV/KL are reported as an empirical comparison against a softmax baseline **under a fixed step budget**.
- We do not claim speedups vs softmax, and we explicitly call out benchmarking caveats.
- A V=16 run is presented as a **stress test** showing distribution shift / poor mixing under certain settings.

## 7) Why this is published

This repository is meant to be a **reproducible baseline**:
- Clear math → clear code → clear tests.
- Honest results → clear next steps (blocked Gibbs, annealing, multi-chain, ESS).

