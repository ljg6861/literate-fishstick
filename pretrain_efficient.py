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
    from efficient_training.trm_adapters import inject_trm_adapters, count_trm_parameters
    
    # Call original
    model, optimizers, optimizer_lrs = original_create_model(config, train_metadata, rank, world_size)
    
    # Get efficient config from arch config extras
    lora_rank = getattr(config.arch, 'lora_rank', 16)
    lora_alpha = getattr(config.arch, 'lora_alpha', 32)
    use_dora = getattr(config.arch, 'use_dora', True)
    
    # Inject DoRA adapters
    print(f"\n[Efficient] Applying {'DoRA' if use_dora else 'LoRA'} adapters (rank={lora_rank}, alpha={lora_alpha})...")
    adapted = inject_trm_adapters(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        use_dora=use_dora,
    )
    
    trainable, total = count_trm_parameters(model)
    print(f"[Efficient] Adapted {len(adapted)} modules")
    print(f"[Efficient] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Update optimizers to only include trainable params
    # The first optimizer is for the model weights
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Replace optimizer with one for trainable params only
    use_galore = getattr(config.arch, 'use_galore', False)
    
    if use_galore:
        from efficient_training.galore import GaLoreAdamW
        galore_rank = getattr(config.arch, 'galore_rank', 128)
        print(f"[Efficient] Using GaLore optimizer (rank={galore_rank})")
        
        new_optimizer = GaLoreAdamW(
            trainable_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            galore_rank=galore_rank,
        )
        optimizers = [new_optimizer] + list(optimizers)[1:]  # Keep puzzle embedding optimizer
    else:
        # Use AdamW for trainable params
        new_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        optimizers = [new_optimizer] + list(optimizers)[1:]
    
    return model, optimizers, optimizer_lrs


def main():
    print("="*60)
    print("TRM Efficient Training (with ARC-AGI Evaluation)")
    print("="*60)
    print("Note: This uses TRM's full training pipeline with DoRA adapters")
    print("      and early-stop evaluation for faster feedback")
    print("="*60 + "\n")
    
    # Import pretrain and patch
    import pretrain
    global original_create_model
    original_create_model = pretrain.create_model
    pretrain.create_model = patched_create_model
    
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
            
            print(f"\n[Efficient] Initializing progressive curriculum:")
            print(f"  Base H_cycles: {base_h_cycles}, Base halt_max_steps: {base_halt_steps}")
            print(f"  Total steps: {train_state.total_steps}")
            
            curriculum_scheduler = ProgressiveCurriculumScheduler(
                total_steps=train_state.total_steps,
                h_cycles_full=base_h_cycles,
                halt_max_steps_full=base_halt_steps,
            )
            print(f"  Expected compute savings: ~{curriculum_scheduler.get_expected_compute_savings():.1%}")
        
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
    
    # Also patch evaluate function for early-stop support
    original_evaluate = pretrain.evaluate
    
    def patched_evaluate(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group):
        """Evaluate with early-stop support - skip forward passes for solved puzzles."""
        import torch
        import torch.distributed as dist
        from typing import Any, List, Optional, Dict
        
        # Check if any evaluator supports early-stop
        has_early_stop = any(hasattr(e, 'get_skip_mask') for e in evaluators)
        
        if has_early_stop and rank == 0:
            print("[Efficient] Using early-stop evaluation")
        
        reduced_metrics = None
        
        with torch.inference_mode():
            return_keys = set(config.eval_save_outputs)
            for evaluator in evaluators:
                evaluator.begin_eval()
                return_keys.update(evaluator.required_outputs)
            
            set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
            save_preds = {}
            metric_keys = []
            metric_values = None
            
            carry = None
            processed_batches = 0
            skipped_samples_total = 0
            
            for set_name, batch, global_batch_size in eval_loader:
                processed_batches += 1
                
                # Check for samples to skip (early-stop)
                skip_mask = None
                if has_early_stop:
                    for evaluator in evaluators:
                        if hasattr(evaluator, 'get_skip_mask'):
                            skip_mask = evaluator.get_skip_mask(batch["puzzle_identifiers"])
                            break
                
                # Filter batch if needed
                if skip_mask is not None:
                    keep_mask = skip_mask
                    num_keep = keep_mask.sum().item()
                    num_skip = len(keep_mask) - num_keep
                    skipped_samples_total += num_skip
                    
                    if num_keep == 0:
                        # Skip entire batch
                        if rank == 0 and processed_batches % 100 == 0:
                            print(f"Batch {processed_batches}: SKIPPED (all {len(keep_mask)} samples from solved puzzles)")
                        continue
                    
                    if num_skip > 0:
                        batch = {k: v[keep_mask] for k, v in batch.items()}
                
                if rank == 0 and processed_batches % 100 == 0:
                    print(f"Processing batch {processed_batches}: {set_name} ({len(batch['inputs'])} samples)")
                
                # To device
                batch = {k: v.cuda() for k, v in batch.items()}
                with torch.device("cuda"):
                    carry = train_state.model.initial_carry(batch)
                
                # Forward (ACT loop)
                inference_steps = 0
                while True:
                    carry, loss, metrics, preds, all_finish = train_state.model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )
                    inference_steps += 1
                    if all_finish:
                        break
                
                # Save outputs if requested
                for collection in (batch, preds):
                    for k, v in collection.items():
                        if k in config.eval_save_outputs:
                            save_preds.setdefault(k, [])
                            save_preds[k].append(v.cpu())
                
                # Update evaluators
                for evaluator in evaluators:
                    evaluator.update_batch(batch, preds)
                
                del carry, loss, preds, batch, all_finish
                
                # Aggregate metrics
                set_id = set_ids[set_name]
                if metric_values is None and metrics:
                    metric_keys = list(sorted(metrics.keys()))
                    metric_values = torch.zeros(
                        (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                    )
                if metric_values is not None:
                    metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
                del metrics
            
            if rank == 0 and skipped_samples_total > 0:
                print(f"\n[Early-stop] Skipped {skipped_samples_total} samples from solved puzzles")
            
            # Concatenate save preds
            save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}
            
            # Save preds
            if config.checkpoint_path is not None and len(save_preds):
                import os
                os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
                torch.save(
                    save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
                )
            
            del save_preds
            
            # Reduce metrics
            if metric_values is not None:
                if world_size > 1:
                    dist.reduce(metric_values, dst=0)
                
                if rank == 0:
                    reduced_metrics = metric_values.cpu().numpy()
                    reduced_metrics = {
                        set_name: {
                            metric_name: reduced_metrics[set_id, metric_id]
                            for metric_id, metric_name in enumerate(metric_keys)
                        }
                        for set_id, set_name in enumerate(set_ids)
                    }
                    
                    # Post-process
                    for set_name, m in reduced_metrics.items():
                        count = max(m.get("count", 1), 1)
                        reduced_metrics[set_name] = {k: v / count for k, v in m.items()}
            
            # Run evaluator result functions
            if rank == 0:
                print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
            for i, evaluator in enumerate(evaluators):
                if rank == 0:
                    print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
                evaluator_save_path = None
                if config.checkpoint_path is not None:
                    import os
                    evaluator_save_path = os.path.join(
                        config.checkpoint_path,
                        f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                    )
                    os.makedirs(evaluator_save_path, exist_ok=True)
                
                metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
                if rank == 0 and metrics is not None:
                    if reduced_metrics is None:
                        reduced_metrics = {}
                    reduced_metrics.update(metrics)
                    print(f"  Completed {evaluator.__class__.__name__}")
            
            if rank == 0:
                print("All evaluators completed!")
        
        return reduced_metrics
    
    pretrain.evaluate = patched_evaluate
    
    # Launch training
    pretrain.launch()


if __name__ == "__main__":
    # Change to TRM directory for Hydra
    os.chdir(TRM_PATH)
    main()

