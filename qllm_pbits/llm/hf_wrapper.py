"""
HuggingFace model wrapper for LLM integration.
"""

import torch
from torch import Tensor
from typing import Any


class HFModel:
    """Wrapper for HuggingFace causal language models.
    
    Provides a simple interface for loading models and getting next-token
    logits with KV caching for efficient autoregressive generation.
    
    Attributes:
        model_name (str): HuggingFace model identifier
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device (str): Device for computation
    
    Examples:
        >>> model = HFModel("distilgpt2", device="cpu")
        >>> input_ids = model.tokenizer.encode("Hello", return_tensors="pt")
        >>> logits, past_kv = model.get_logits(input_ids)
        >>> logits.shape
        torch.Size([1, vocab_size])
    """
    
    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        """Load HuggingFace model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "distilgpt2", "gpt2")
            device: torch device ("cpu", "cuda", "cuda:0")
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.model_name = model_name
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_logits(
        self, input_ids: Tensor, past_key_values: Any = None
    ) -> tuple[Tensor, Any]:
        """Forward pass returning next-token logits and KV cache.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            past_key_values: Cached key-value states from previous forward pass
        
        Returns:
            logits: Next-token logits (batch_size, vocab_size)
            past_key_values: Updated KV cache
        
        Examples:
            >>> model = HFModel("distilgpt2")
            >>> ids = torch.tensor([[1, 2, 3]])
            >>> logits, kv = model.get_logits(ids)
            >>> logits.shape[1] == len(model.tokenizer)
            True
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids, past_key_values=past_key_values, use_cache=True
            )
        return outputs.logits[:, -1, :], outputs.past_key_values

