"""LLM integration utilities."""

from qllm_pbits.llm.hf_wrapper import HFModel
from qllm_pbits.llm.generate import generate_text

__all__ = ["HFModel", "generate_text"]

