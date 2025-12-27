#!/usr/bin/env python3
"""
NAS-MuZero Experiment Runner

Runs the Neural Architecture Search via MuZero planning experiment.
"""

import argparse
from learning.nas_muzero import run_nas_experiment, NASConfig, NASMuZeroTrainer
from learning.fast_muzero import FastMuZeroTrainer


def compare_nas_vs_fixed(seq_length: int = 4, max_samples: int = 300000):
    """
    Compare NAS-MuZero (evolving architecture) vs fixed architecture.
    """
    print("\n" + "="*70)
    print("  COMPARISON: NAS-MuZero vs Fixed Architecture")
    print("="*70)
    
    results = {}
    
    # Run NAS-MuZero
    print("\n[1/2] NAS-MuZero (Architecture Search)")
    config = NASConfig(seq_length=seq_length, arch_action_freq=50)
    nas_trainer = NASMuZeroTrainer(config)
    results['nas'] = nas_trainer.train(max_samples, log_interval=15000, target_accuracy=0.90)
    results['nas']['eval_accuracy'] = nas_trainer.evaluate(5000)
    
    # Run fixed architecture (larger to match potential NAS outcome)
    print("\n[2/2] Fixed Architecture (Baseline)")
    from learning.fast_muzero import FastMuZeroTrainer
    fixed_trainer = FastMuZeroTrainer(seq_length=seq_length, num_parallel_envs=256, device="cuda")
    results['fixed'] = fixed_trainer.train(max_samples, log_interval=15000, target_accuracy=0.90)
    results['fixed']['eval_accuracy'] = fixed_trainer.evaluate(5000)
    
    # Compare
    print("\n" + "="*70)
    print("  COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n  NAS-MuZero:")
    print(f"    Final Accuracy:     {results['nas']['final_accuracy']:.2%}")
    print(f"    Eval Accuracy:      {results['nas']['eval_accuracy']:.2%}")
    print(f"    Final Architecture: {results['nas']['final_architecture']}")
    print(f"    Final Parameters:   {results['nas']['final_params']:,}")
    print(f"    Speed:              {results['nas']['samples_per_second']:,.0f} samples/sec")
    
    print(f"\n  Fixed Architecture:")
    print(f"    Final Accuracy:     {results['fixed']['final_accuracy']:.2%}")
    print(f"    Eval Accuracy:      {results['fixed']['eval_accuracy']:.2%}")
    print(f"    Speed:              {results['fixed']['samples_per_second']:,.0f} samples/sec")
    
    # Efficiency comparison
    if results['nas']['final_params'] > 0:
        nas_efficiency = results['nas']['eval_accuracy'] / results['nas']['final_params'] * 100000
        print(f"\n  NAS Efficiency Score: {nas_efficiency:.4f} (accuracy per 100k params)")
    
    print("\n" + "="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="NAS-MuZero Experiments")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="nas",
        choices=["nas", "compare"],
        help="Run mode: 'nas' for NAS-MuZero only, 'compare' for comparison with fixed"
    )
    
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4,
        help="Sequence length for task"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300000,
        help="Maximum training samples"
    )
    
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.90,
        help="Target accuracy"
    )
    
    args = parser.parse_args()
    
    if args.mode == "nas":
        run_nas_experiment(
            seq_length=args.seq_length,
            max_samples=args.max_samples,
            target_accuracy=args.target_accuracy
        )
    else:
        compare_nas_vs_fixed(
            seq_length=args.seq_length,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
