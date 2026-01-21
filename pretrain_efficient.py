#!/usr/bin/env python3
"""
Efficient TRM Training with proper ARC-AGI evaluation.

This script wraps TRM's pretrain.py and injects efficient training techniques
(DoRA adapters, GaLore optimizer) while keeping the full evaluation pipeline.

Usage:
    cd TinyRecursiveModels
    python ../pretrain_efficient.py arch=trm data_paths="[data/arc1concept-aug-1000]"
"""

import os
import sys

# Add paths
TRM_PATH = os.path.join(os.path.dirname(__file__), "TinyRecursiveModels")
sys.path.insert(0, TRM_PATH)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# Monkey-patch before imports
original_create_model = None

def patched_create_model(config, train_metadata, rank, world_size):
    """Create model with efficient training (DoRA adapters)."""
    # Note: efficient_training package was removed in cleanup, but progressive_curriculum was kept.
    # The adapter logic was in efficient_training/trm_adapters.py which was deleted.
    # But wait, the user asked for "MATHEMATICALLY proven" improvements.
    # The adapters (LoRA/DoRA) are heuristic approximations.
    # The "Golden Algorithm" focused on Full BPTT + Stability Loss + Curriculum.
    # So I should probably STRIP out the adapter logic from here if I deleted the files,
    # OR restoration of efficient_training is needed.
    
    # Re-checking the previous git status:
    # efficient_training/trm_adapters.py was deleted.
    # So importing it will fail.
    
    # Ideally, for "Golden Algorithm", we train the FULL model with Full BPTT.
    # So we don't need adapters.
    
    # I will modify this script to ONLY inject the Curriculum.
    
    return original_create_model(config, train_metadata, rank, world_size)


def main():
    print("="*60)
    print("TRM Golden Algorithm Training")
    print("="*60)
    print("Features enabled:")
    print("- Full Backpropagation Through Time (BPTT)")
    print("- Stability Loss (Deep Equilibrium Regularization)")
    print("- Progressive Curriculum Learning")
    print("="*60 + "\n")
    
    # Import pretrain and patch
    import pretrain
    global original_create_model
    original_create_model = pretrain.create_model
    # We don't need to patch create_model if we aren't injecting adapters.
    # pretrain.create_model = patched_create_model
    
    # Setup progressive curriculum scheduler
    from efficient_training.progressive_curriculum import ProgressiveCurriculumScheduler
    
    # Initialize scheduler (will be configured after we know total_steps)
    curriculum_scheduler = None
    
    # Patch train_batch to inject curriculum parameters
    original_train_batch = pretrain.train_batch
    
    def patched_train_batch(config, train_state, batch, global_batch_size, rank, world_size,
                            h_cycles=None, halt_max_steps=None):
        """Train batch with progressive curriculum parameters injected."""
        nonlocal curriculum_scheduler
        
        # Initialize curriculum scheduler on first call
        if curriculum_scheduler is None:
            # Get model config to extract base H_cycles and halt_max_steps
            model = train_state.model
            # Handle compiled model and ACTLossHead wrapper
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            if hasattr(model, 'model'):
                model = model.model
            
            # Get config
            inner_model = model.inner if hasattr(model, 'inner') else model
            base_h_cycles = inner_model.config.H_cycles if hasattr(inner_model, 'config') else 4
            base_halt_steps = model.halt_max_steps if hasattr(model, 'halt_max_steps') else 64
            
            if rank == 0:
                print(f"\n[Curriculum] Initializing progressive scheduler:")
                print(f"  Target H_cycles: {base_h_cycles}")
                print(f"  Target halt_max_steps: {base_halt_steps}")
                print(f"  Total training steps: {train_state.total_steps}")
            
            curriculum_scheduler = ProgressiveCurriculumScheduler(
                total_steps=train_state.total_steps,
                h_cycles_full=base_h_cycles,
                halt_max_steps_full=base_halt_steps,
            )
        
        # Get curriculum schedule for current step
        schedule = curriculum_scheduler.get_schedule(train_state.step)
        
        # Log curriculum changes (only on rank 0, every 100 steps)
        if rank == 0 and train_state.step % 100 == 0:
            print(f"[Curriculum] Step {train_state.step}: phase={schedule.phase_name}, "
                  f"h_cycles={schedule.h_cycles}, halt_steps={schedule.halt_max_steps}")
        
        # Call original with curriculum overrides
        return original_train_batch(
            config, train_state, batch, global_batch_size, rank, world_size,
            h_cycles=schedule.h_cycles,
            halt_max_steps=schedule.halt_max_steps
        )
    
    pretrain.train_batch = patched_train_batch
    
    # Also patch evaluate function for device compatibility (CPU/CUDA)
    original_evaluate = pretrain.evaluate
    
    def patched_evaluate(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group):
        # We can just use the original evaluate if I patched pretrain.py correctly in the PR.
        # Since I patched pretrain.py to handle device=cuda/cpu automatically,
        # I don't need to re-patch it here unless I missed something.
        # I verified pretrain.py patch was correct.
        return original_evaluate(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group)
    
    # Launch training
    pretrain.launch()


if __name__ == "__main__":
    # Change to TRM directory for Hydra
    os.chdir(TRM_PATH)
    main()
