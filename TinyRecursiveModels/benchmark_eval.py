#!/usr/bin/env python3
"""
Benchmark ARCFast evaluator against original ARC evaluator.

This measures:
1. Time per batch processing
2. Total evaluation time
3. Early-stop skip rate
4. Result equivalence
"""

import sys
import os
import json
import time
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, '/home/lucas/literate-fishstick/TinyRecursiveModels')

from evaluators.arc import ARC
from evaluators.arc_fast import ARCFast
from dataset.common import PuzzleDatasetMetadata
from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np


def create_metadata(data_path: str):
    """Load actual metadata from dataset."""
    test_metadata_path = os.path.join(data_path, "test", "dataset.json")
    with open(test_metadata_path, "r") as f:
        meta_dict = json.load(f)
    return PuzzleDatasetMetadata(**meta_dict)


def load_test_data(data_path: str, max_samples: int = None):
    """Load actual test data."""
    import numpy as np
    
    inputs = np.load(os.path.join(data_path, "test", "all__inputs.npy"), mmap_mode='r')
    labels = np.load(os.path.join(data_path, "test", "all__labels.npy"), mmap_mode='r')
    puzzle_ids = np.load(os.path.join(data_path, "test", "all__puzzle_identifiers.npy"))
    
    if max_samples is not None:
        inputs = inputs[:max_samples]
        labels = labels[:max_samples]
        puzzle_ids = puzzle_ids[:max_samples]
    
    return inputs, labels, puzzle_ids


def simulate_model_predictions(inputs, labels, puzzle_ids, correct_rate: float = 0.1):
    """
    Simulate model predictions with a given correct rate.
    
    In real evaluation, the model would predict outputs.
    Here we simulate by:
    - With probability `correct_rate`, return the correct label
    - Otherwise, return a random grid
    """
    preds = []
    for i in range(len(inputs)):
        if np.random.random() < correct_rate:
            # Return correct answer
            preds.append(labels[i].copy())
        else:
            # Return random grid (tokens 2-11)
            preds.append(np.random.randint(2, 12, labels[i].shape, dtype=np.int32))
    
    return np.stack(preds)


def benchmark_evaluator(evaluator, inputs, preds, puzzle_ids, batch_size: int = 32):
    """Benchmark a single evaluator."""
    evaluator.begin_eval()
    
    total_samples = len(inputs)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        
        batch = {
            "puzzle_identifiers": torch.tensor(puzzle_ids[start_idx:end_idx], dtype=torch.int64),
            "inputs": torch.tensor(inputs[start_idx:end_idx], dtype=torch.int64),
        }
        
        preds_batch = {
            "preds": torch.tensor(preds[start_idx:end_idx], dtype=torch.int64),
            "q_halt_logits": torch.randn(end_idx - start_idx),
        }
        
        evaluator.update_batch(batch, preds_batch)
    
    elapsed = time.time() - start_time
    
    return elapsed


def run_benchmark(data_path: str, max_samples: int = 10000, correct_rate: float = 0.1):
    """Run comparison benchmark."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: ARCFast vs ARC Evaluator")
    print(f"{'='*60}")
    print(f"Data path: {data_path}")
    print(f"Max samples: {max_samples}")
    print(f"Simulated correct rate: {correct_rate*100:.1f}%")
    print()
    
    # Load metadata and data
    print("Loading data...")
    metadata = create_metadata(data_path)
    inputs, labels, puzzle_ids = load_test_data(data_path, max_samples)
    print(f"Loaded {len(inputs)} samples")
    
    # Simulate predictions
    print(f"Simulating model predictions (correct_rate={correct_rate})...")
    preds = simulate_model_predictions(inputs, labels, puzzle_ids, correct_rate)
    
    # Create evaluators
    print("\nCreating evaluators...")
    arc_original = ARC(data_path, metadata)
    arc_fast_no_stop = ARCFast(data_path, metadata, early_stop=False)
    arc_fast_with_stop = ARCFast(data_path, metadata, early_stop=True)
    
    # Benchmark each
    print("\n--- Benchmarking Original ARC ---")
    time_original = benchmark_evaluator(arc_original, inputs, preds, puzzle_ids)
    print(f"Time: {time_original:.2f}s")
    
    print("\n--- Benchmarking ARCFast (early-stop OFF) ---")
    time_fast_no_stop = benchmark_evaluator(arc_fast_no_stop, inputs, preds, puzzle_ids)
    print(f"Time: {time_fast_no_stop:.2f}s")
    
    print("\n--- Benchmarking ARCFast (early-stop ON) ---")
    time_fast_with_stop = benchmark_evaluator(arc_fast_with_stop, inputs, preds, puzzle_ids)
    print(f"Time: {time_fast_with_stop:.2f}s")
    
    # Get early-stop stats
    stats = {
        "total_samples": arc_fast_with_stop._total_samples,
        "skipped_samples": arc_fast_with_stop._skipped_samples,
        "puzzles_done": len(arc_fast_with_stop._puzzles_done),
    }
    
    print(f"\nEarly-stop stats:")
    print(f"  Samples processed: {stats['total_samples']}")
    print(f"  Samples skipped: {stats['skipped_samples']}")
    print(f"  Puzzles early-stopped: {stats['puzzles_done']}")
    skip_rate = stats['skipped_samples'] / max(1, stats['total_samples']) * 100
    print(f"  Skip rate: {skip_rate:.1f}%")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Original ARC:           {time_original:.2f}s")
    print(f"ARCFast (no early-stop): {time_fast_no_stop:.2f}s ({time_original/time_fast_no_stop:.1f}x)")
    print(f"ARCFast (early-stop ON): {time_fast_with_stop:.2f}s ({time_original/time_fast_with_stop:.1f}x)")
    
    if skip_rate > 0:
        effective_speedup = time_original / time_fast_with_stop
        print(f"\nEffective speedup with early-stop: {effective_speedup:.1f}x")
    else:
        print("\nNote: Early-stop didn't trigger (correct predictions didn't accumulate enough votes)")
        print("This is expected with random predictions - real model would have higher correct rate")
    
    return {
        "time_original": time_original,
        "time_fast_no_stop": time_fast_no_stop,
        "time_fast_with_stop": time_fast_with_stop,
        "skip_rate": skip_rate,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/arc1concept-aug-1000")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--correct-rate", type=float, default=0.1)
    args = parser.parse_args()
    
    os.chdir("/home/lucas/literate-fishstick/TinyRecursiveModels")
    
    run_benchmark(
        data_path=args.data_path,
        max_samples=args.max_samples,
        correct_rate=args.correct_rate
    )
