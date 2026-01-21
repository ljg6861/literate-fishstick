#!/usr/bin/env python3
"""
Phase 0.1 Diagnostic: Measure prediction diversity and voting patterns.

This script runs a minimal evaluation to understand:
1. How many unique predictions per puzzle?
2. Does the correct answer typically win by votes?
3. What's the vote distribution like?
"""

import os
import sys
import json
from collections import Counter, defaultdict
import numpy as np
import torch

# Add TRM to path
sys.path.insert(0, '/home/lucas/literate-fishstick/TinyRecursiveModels')

from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np

def analyze_prediction_diversity():
    """Analyze saved predictions from a previous run if available."""
    
    # Check if we have any saved predictions
    checkpoint_dirs = [
        '/home/lucas/literate-fishstick/TinyRecursiveModels/checkpoints/trm_efficient',
        '/home/lucas/literate-fishstick/TinyRecursiveModels/outputs',
    ]
    
    print("Looking for saved predictions...")
    for d in checkpoint_dirs:
        if os.path.exists(d):
            print(f"  Found: {d}")
            files = os.listdir(d)
            print(f"  Contents: {files[:10]}...")
    
    # Load test puzzles for ground truth
    with open('/home/lucas/literate-fishstick/TinyRecursiveModels/data/arc1concept-aug-1000/test_puzzles.json', 'r') as f:
        test_puzzles = json.load(f)
    
    print(f"\nTest puzzles: {len(test_puzzles)}")
    
    # Load identifiers
    with open('/home/lucas/literate-fishstick/TinyRecursiveModels/data/arc1concept-aug-1000/identifiers.json', 'r') as f:
        identifiers = json.load(f)
    
    # Analyze augmentation structure
    print("\n=== AUGMENTATION ANALYSIS ===")
    
    # Group identifiers by base puzzle
    base_to_augs = defaultdict(list)
    for idx, name in enumerate(identifiers):
        if name == '<blank>':
            continue
        base_name, _ = inverse_aug(name)
        base_to_augs[base_name].append((idx, name))
    
    # Sample analysis
    print(f"Unique base puzzles: {len(base_to_augs)}")
    
    # Count augmentations per puzzle
    aug_counts = [len(augs) for augs in base_to_augs.values()]
    print(f"Augmentations per puzzle: mean={np.mean(aug_counts):.1f}, min={min(aug_counts)}, max={max(aug_counts)}")
    
    # Analyze transform distribution for one puzzle
    sample_puzzle = list(base_to_augs.keys())[0]
    sample_augs = base_to_augs[sample_puzzle]
    print(f"\nSample puzzle: {sample_puzzle}")
    print(f"  Total augmentations: {len(sample_augs)}")
    
    # Count transforms
    transform_counts = Counter()
    color_perm_counts = Counter()
    for idx, name in sample_augs:
        if '|||' in name:
            parts = name.split('|||')
            transform_counts[parts[1]] += 1  # t0-t7
            color_perm_counts[parts[2]] += 1  # color perm
    
    print(f"  Transform distribution: {dict(transform_counts)}")
    print(f"  Unique color permutations: {len(color_perm_counts)}")
    
    print("\n=== EVALUATION STRUCTURE ANALYSIS ===")
    
    # Check test set structure
    test_puzzle_names = set(test_puzzles.keys())
    base_names_in_test = set(base_to_augs.keys())
    
    # How many test puzzles have augmentations?
    test_with_augs = test_puzzle_names.intersection(base_names_in_test)
    print(f"Test puzzles with augmentations: {len(test_with_augs)} / {len(test_puzzle_names)}")
    
    # For a test puzzle, count augmented samples
    if test_with_augs:
        sample_test = list(test_with_augs)[0]
        num_augs = len(base_to_augs[sample_test])
        print(f"\nSample test puzzle '{sample_test}': {num_augs} augmented versions in eval set")
    
    print("\n=== KEY INSIGHT ===")
    print("""
The '1000 samples' in pass@1000 come from:
1. ~1000 data augmentations per puzzle (8 transforms × ~125 color permutations)
2. Each augmentation is a SEPARATE forward pass through the model
3. Model output is DETERMINISTIC (argmax) for each augmented input
4. Evaluator applies inverse_aug() to map predictions back to base puzzle
5. Then VOTES across all augmented predictions
6. pass@k = "is correct answer in top-k by votes?"

IMPLICATION: 
- All 1000 forward passes are NECESSARY to get diverse predictions
- The diversity comes from the INPUT, not model stochasticity  
- Early-stop is ONLY valid if we can verify correctness after inverse_aug()
- Since ground truth is known for dev/val, early-stop IS possible there
""")
    
    return True

if __name__ == "__main__":
    analyze_prediction_diversity()
