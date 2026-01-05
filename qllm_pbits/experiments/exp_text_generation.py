"""
GPT-2 text generation experiment with invalid-rate logging.

Compares baseline softmax sampler with P-bit sampler on text generation.
"""

import torch
from qllm_pbits.llm.hf_wrapper import HFModel
from qllm_pbits.llm.generate import generate_text
from qllm_pbits.token_sampler.softmax_baseline import SoftmaxMultinomialSampler
from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler


TEST_PROMPTS = [
    "Once upon a time in a distant",
    "The future of artificial intelligence will",
    "In the beginning, there was",
    "Scientists have discovered that",
    "The most important thing to remember is",
]


def run_text_generation_experiment(
    model_name: str = "distilgpt2",
    lam: float = 20.0,
    n_gibbs_steps: int = 100,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = "cpu",
) -> dict:
    """Compare text generation with baseline and P-bit samplers.
    
    Args:
        model_name: HuggingFace model name
        lam: Lambda for P-bit sampler
        n_gibbs_steps: Gibbs steps per token
        max_tokens: Max tokens to generate per prompt
        temperature: Sampling temperature
        top_k: Top-k filtering
        device: torch device
    
    Returns:
        Dictionary with generation results and metrics
    """
    print(f"\n=== Text Generation Experiment ===")
    print(f"Model: {model_name}")
    print(f"P-bit config: λ={lam}, steps={n_gibbs_steps}\n")
    
    # Load model
    print("Loading model...")
    model = HFModel(model_name, device=device)
    
    # Create samplers
    baseline_sampler = SoftmaxMultinomialSampler()
    pbit_sampler = PBitOneHotSampler(lam=lam, n_gibbs_steps=n_gibbs_steps)
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(TEST_PROMPTS)} ---")
        print(f"Prompt: '{prompt}'")
        
        # Baseline generation
        print("\nBaseline (softmax):")
        baseline_result = generate_text(
            model, prompt, baseline_sampler,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        print(f"  Text: {baseline_result['text'][:100]}...")
        print(f"  Time/token: {baseline_result['time_per_token']*1000:.2f} ms")
        
        # P-bit generation
        print("\nP-bit sampler:")
        pbit_result = generate_text(
            model, prompt, pbit_sampler,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        print(f"  Text: {pbit_result['text'][:100]}...")
        print(f"  Time/token: {pbit_result['time_per_token']*1000:.2f} ms")
        print(f"  Invalid rate: {pbit_result['invalid_rate']:.4f}")
        
        results.append({
            "prompt": prompt,
            "baseline_text": baseline_result["text"],
            "baseline_time_ms": baseline_result["time_per_token"] * 1000,
            "pbit_text": pbit_result["text"],
            "pbit_time_ms": pbit_result["time_per_token"] * 1000,
            "pbit_invalid_rate": pbit_result["invalid_rate"],
        })
    
    # Summary statistics
    avg_baseline_time = sum(r["baseline_time_ms"] for r in results) / len(results)
    avg_pbit_time = sum(r["pbit_time_ms"] for r in results) / len(results)
    avg_invalid_rate = sum(r["pbit_invalid_rate"] for r in results) / len(results)
    
    print(f"\n=== Summary ===")
    print(f"Average time/token (baseline): {avg_baseline_time:.2f} ms")
    print(f"Average time/token (P-bit): {avg_pbit_time:.2f} ms")
    print(f"Average invalid rate (P-bit): {avg_invalid_rate:.4f}")
    
    # Pass criteria
    passed_time = avg_pbit_time < 100  # < 100ms on CPU
    passed_invalid = avg_invalid_rate < 0.01  # < 1%
    passed = passed_time and passed_invalid
    
    print(f"\nPassed time criterion (<100ms): {passed_time}")
    print(f"Passed invalid rate criterion (<1%): {passed_invalid}")
    print(f"OVERALL PASSED: {passed}")
    
    return {
        "results": results,
        "avg_baseline_time_ms": avg_baseline_time,
        "avg_pbit_time_ms": avg_pbit_time,
        "avg_invalid_rate": avg_invalid_rate,
        "passed": passed,
    }


if __name__ == "__main__":
    result = run_text_generation_experiment()

