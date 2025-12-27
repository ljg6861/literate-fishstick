#!/usr/bin/env python3
"""
MuZero vs Supervised Learning Experiment

This script runs both MuZero and supervised learning on sequence reversal/sorting
and compares their sample efficiency.
"""

import argparse
import torch
import json
import time
from pathlib import Path
from datetime import datetime

from learning.muzero_trainer import MuZeroTrainer
from learning.supervised_baseline import SupervisedTrainer


def run_experiment(
    task: str = "reversal",
    seq_length: int = 8,
    max_samples: int = 10000,
    target_accuracy: float = 0.99,
    methods: list = None,
    device: str = None,
    output_dir: str = "results"
):
    """
    Run comparison experiment between MuZero and Supervised learning.
    
    Args:
        task: "reversal" or "sorting"
        seq_length: Length of sequences
        max_samples: Maximum training samples
        target_accuracy: Target accuracy to measure sample efficiency
        methods: List of methods to run ("muzero", "supervised", or both)
        device: Device to use (cuda/cpu)
        output_dir: Directory to save results
    """
    if methods is None:
        methods = ["muzero", "supervised"]
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"  MuZero vs Supervised Learning Experiment")
    print(f"{'='*70}")
    print(f"  Task:            {task}")
    print(f"  Sequence Length: {seq_length}")
    print(f"  Max Samples:     {max_samples}")
    print(f"  Target Accuracy: {target_accuracy:.1%}")
    print(f"  Device:          {device}")
    print(f"  Methods:         {methods}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Run MuZero
    if "muzero" in methods:
        print("\n" + "="*50)
        print("RUNNING MUZERO")
        print("="*50)
        
        trainer = MuZeroTrainer(
            task=task,
            seq_length=seq_length,
            device=device
        )
        
        muzero_results = trainer.train(
            max_samples=max_samples,
            eval_interval=50,
            target_accuracy=target_accuracy
        )
        
        # Final evaluation
        muzero_results['eval_accuracy'] = trainer.evaluate(100)
        results['muzero'] = muzero_results
    
    # Run Supervised
    if "supervised" in methods:
        print("\n" + "="*50)
        print("RUNNING SUPERVISED BASELINE")
        print("="*50)
        
        trainer = SupervisedTrainer(
            task=task,
            seq_length=seq_length,
            device=device
        )
        
        supervised_results = trainer.train(
            max_samples=max_samples,
            eval_interval=50,
            target_accuracy=target_accuracy
        )
        
        # Final evaluation
        supervised_results['eval_accuracy'] = trainer.evaluate(100)
        results['supervised'] = supervised_results
    
    # Compare results
    print("\n" + "="*70)
    print("  COMPARISON RESULTS")
    print("="*70)
    
    comparison_table = []
    
    for method, r in results.items():
        row = {
            'Method': method.upper(),
            'Final Accuracy': f"{r['final_accuracy']:.2%}",
            'Best Accuracy': f"{r['best_accuracy']:.2%}",
            'Samples to Target': r['samples_to_target'] or 'N/A',
            'Training Time': f"{r['elapsed_time']:.1f}s"
        }
        comparison_table.append(row)
        
        print(f"\n  {method.upper()}")
        print(f"    Final Accuracy:    {r['final_accuracy']:.2%}")
        print(f"    Best Accuracy:     {r['best_accuracy']:.2%}")
        print(f"    Samples to Target: {r['samples_to_target'] or 'Not reached'}")
        print(f"    Training Time:     {r['elapsed_time']:.1f}s")
    
    # Sample efficiency comparison
    if len(results) == 2 and all(r.get('samples_to_target') for r in results.values()):
        muzero_samples = results['muzero']['samples_to_target']
        supervised_samples = results['supervised']['samples_to_target']
        
        if muzero_samples < supervised_samples:
            efficiency = (supervised_samples - muzero_samples) / supervised_samples * 100
            print(f"\n  📊 MuZero is {efficiency:.1f}% more sample efficient!")
        elif supervised_samples < muzero_samples:
            efficiency = (muzero_samples - supervised_samples) / muzero_samples * 100
            print(f"\n  📊 Supervised is {efficiency:.1f}% more sample efficient!")
        else:
            print(f"\n  📊 Both methods reached target at the same time.")
    
    print("\n" + "="*70 + "\n")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"experiment_{task}_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'task': task,
                'seq_length': seq_length,
                'max_samples': max_samples,
                'target_accuracy': target_accuracy,
                'device': device
            },
            'results': {k: {kk: str(vv) if not isinstance(vv, (int, float, type(None))) else vv 
                           for kk, vv in v.items()} 
                       for k, v in results.items()}
        }, f, indent=2)
    
    print(f"Results saved to: {result_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare MuZero vs Supervised Learning on Seq2Seq tasks"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="reversal",
        choices=["reversal", "sorting"],
        help="Task to train on"
    )
    
    parser.add_argument(
        "--seq-length",
        type=int,
        default=8,
        help="Sequence length"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of training samples"
    )
    
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.99,
        help="Target accuracy to measure sample efficiency"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["muzero", "supervised", "both"],
        help="Which method(s) to run"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Determine methods
    if args.method == "both":
        methods = ["muzero", "supervised"]
    else:
        methods = [args.method]
    
    # Run experiment
    run_experiment(
        task=args.task,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        target_accuracy=args.target_accuracy,
        methods=methods,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
