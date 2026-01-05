"""
qllm-pbits: P-bit networks for LLM token sampling

A research package implementing probabilistic bit networks using direct Gibbs
sampling on binary variables to sample tokens from LLM distributions.
"""

__version__ = "0.1.0"

from qllm_pbits.pbits.gibbs_binary import BinaryGibbsSampler
from qllm_pbits.pbits.metrics import (
    total_variation,
    kl_divergence,
    effective_sample_size,
    invalid_onehot_rate,
)
from qllm_pbits.token_sampler.base import TokenSampler
from qllm_pbits.token_sampler.softmax_baseline import SoftmaxMultinomialSampler
from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler
from qllm_pbits.token_sampler.vocab_filter import apply_vocab_filter

__all__ = [
    "BinaryGibbsSampler",
    "total_variation",
    "kl_divergence",
    "effective_sample_size",
    "invalid_onehot_rate",
    "TokenSampler",
    "SoftmaxMultinomialSampler",
    "PBitOneHotSampler",
    "apply_vocab_filter",
]

