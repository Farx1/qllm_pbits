"""
Unified command-line interface for qllm-pbits.
"""

import argparse
import torch
from qllm_pbits.utils.seeding import set_seed


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="qllm-pbits: P-bit networks for LLM token sampling"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ===== Experiment command =====
    exp_parser = subparsers.add_parser("experiment", help="Run validation experiments")
    exp_parser.add_argument(
        "--type",
        choices=["ising", "softmax", "generation"],
        required=True,
        help="Experiment type",
    )
    exp_parser.add_argument("--vocab-size", type=int, default=32, help="Vocabulary size (for softmax)")
    exp_parser.add_argument("--model", type=str, default="distilgpt2", help="HF model name (for generation)")
    exp_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, cuda:0)")
    exp_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # ===== Calibrate command =====
    calib_parser = subparsers.add_parser("calibrate", help="Calibrate lambda parameter")
    calib_parser.add_argument("--vocab-size", type=int, default=32, help="Vocabulary size")
    calib_parser.add_argument("--target-tv", type=float, default=0.01, help="Target TV distance")
    calib_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    calib_parser.add_argument("--device", type=str, default="cpu", help="Device")
    
    # ===== Generate command =====
    gen_parser = subparsers.add_parser("generate", help="Generate text with P-bit sampler")
    gen_parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    gen_parser.add_argument(
        "--sampler",
        choices=["softmax", "pbit"],
        default="pbit",
        help="Sampler type",
    )
    gen_parser.add_argument("--model", type=str, default="distilgpt2", help="HF model name")
    gen_parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=None, help="Top-k filtering")
    gen_parser.add_argument("--top-p", type=float, default=None, help="Top-p filtering")
    gen_parser.add_argument("--lambda", dest="lam", type=float, default=20.0, help="Lambda (for pbit)")
    gen_parser.add_argument("--gibbs-steps", type=int, default=100, help="Gibbs steps (for pbit)")
    gen_parser.add_argument("--device", type=str, default="cpu", help="Device")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Set seed
    if hasattr(args, "seed"):
        set_seed(args.seed)
    
    # Execute command
    if args.command == "experiment":
        run_experiment(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    elif args.command == "generate":
        run_generate(args)


def run_experiment(args):
    """Run validation experiment."""
    if args.type == "ising":
        from qllm_pbits.experiments.exp_ising_convergence import run_ising_convergence_experiment
        result = run_ising_convergence_experiment(device=args.device)
        
    elif args.type == "softmax":
        from qllm_pbits.experiments.exp_softmax_match import run_softmax_match_experiment
        result = run_softmax_match_experiment(V=args.vocab_size, device=args.device)
        
    elif args.type == "generation":
        from qllm_pbits.experiments.exp_text_generation import run_text_generation_experiment
        result = run_text_generation_experiment(
            model_name=args.model, device=args.device
        )


def run_calibrate(args):
    """Run lambda calibration."""
    from qllm_pbits.token_sampler.calibration import calibrate_lambda
    
    print(f"Calibrating lambda for V={args.vocab_size}...")
    
    torch.manual_seed(args.seed)
    logits = torch.randn(args.vocab_size)
    
    result = calibrate_lambda(
        logits,
        lambda_range=[5.0, 10.0, 20.0, 50.0, 100.0],
        n_samples=5000,
        target_tv=args.target_tv,
        device=args.device,
    )
    
    print(f"\n=== Calibration Results ===")
    print(f"Best lambda: {result['best_lambda']}")
    print(f"Best TV: {result['best_tv']:.4f}")
    print(f"\nResults table:")
    print(result["results"].to_string(index=False))


def run_generate(args):
    """Run text generation."""
    from qllm_pbits.llm.hf_wrapper import HFModel
    from qllm_pbits.llm.generate import generate_text
    from qllm_pbits.token_sampler.softmax_baseline import SoftmaxMultinomialSampler
    from qllm_pbits.token_sampler.pbit_onehot import PBitOneHotSampler
    
    print(f"Loading model {args.model}...")
    model = HFModel(args.model, device=args.device)
    
    if args.sampler == "softmax":
        sampler = SoftmaxMultinomialSampler()
    else:
        sampler = PBitOneHotSampler(lam=args.lam, n_gibbs_steps=args.gibbs_steps)
    
    print(f"\nGenerating with {args.sampler} sampler...")
    print(f"Prompt: '{args.prompt}'")
    
    result = generate_text(
        model,
        args.prompt,
        sampler,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )
    
    print(f"\n=== Generated Text ===")
    print(result["text"])
    print(f"\n=== Statistics ===")
    print(f"Tokens generated: {len(result['tokens'])}")
    print(f"Time per token: {result['time_per_token']*1000:.2f} ms")
    print(f"Total time: {result['total_time']:.2f} s")
    
    if "invalid_rate" in result:
        print(f"Invalid rate: {result['invalid_rate']:.4f}")


if __name__ == "__main__":
    main()

