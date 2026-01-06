#!/usr/bin/env python3
"""
ThinkSort: Universal Transformer for Neural Sorting

Run this script to train and evaluate the model:
    python run.py

The model will:
1. Train on N=4, 8, 16 for 50,000 steps
2. Evaluate zero-shot on N=32, 64, 100, 500, 1000
"""

import os
# Fix memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# Enable TF32 for speed on Ampere GPUs (RTX 30xx)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gc
from model import Config
from trainer import Trainer


def hard_reset_gpu():
    """Complete GPU memory reset."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def main():
    # Configuration
    config = Config(
        dim=128,           # Model dimension
        heads=8,           # Attention heads
        ff=512,            # FFN dimension
        recurrent_steps=4, # Universal block iterations
        # vocab=10,        # REMOVED: Continuous inputs now!
        train_lengths=(4, 8, 16),  # Training sequence lengths
        samples_per_length=8192,  # Increased for Multi-GPU High Throughput
        lr=1e-4,           # Learning rate
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Device: {config.device}")
    
    # Create trainer
    trainer = Trainer(config)
    
    from muzero_nas import MuZeroNAS, ArchConfig
    from morpher import morph_model
    
    from tqdm import tqdm
    
    print("🧬 STARTING SELF-EVOLVING TRAINING")
    
    last_level = 0
    pbar = tqdm(total=100000, initial=trainer.step, desc="Training")
    
    # Evolution State
    last_evolution_step = 0
    best_ood_acc = 0.0
    ood_plateau_counter = 0
    EVOLUTION_COOLDOWN = 5000
    
    while trainer.step < 100000:
        loss, accs = trainer.train_step()
        trainer.check_curriculum()
        
        # Update styling
        pbar.update(1)
        
        # Format stats for progress bar
        stats = f"Loss: {loss:.4f} | Level: {trainer.active_level_idx} (N={trainer.active_lengths[-1]})"
        pbar.set_description(stats)
            
        # Log (less frequently to file/console history if needed, but pbar handles display)
         
        # === PERIODIC CHECKPOINT ===
        if trainer.step % 50 == 0:
            trainer.save("checkpoint_latest.pt")
        
        # === EVOLUTION TRIGGERS ===
        # 1. Level Up
        current_level = trainer.active_level_idx
        level_up_trigger = (current_level > last_level)
        
        # 2. OOD Plateau (Stagnation)
        # Check OOD accuracy every 2000 steps
        plateau_trigger = False
        if trainer.step % 2000 == 0:
            # Eval on a length slightly outside training (e.g. 32 or 2x current max)
            ood_len = max(trainer.active_lengths) * 2
            ood_acc = trainer.evaluate(ood_len, num_samples=100)
            
            if ood_acc > best_ood_acc + 0.01:
                best_ood_acc = ood_acc
                ood_plateau_counter = 0
            else:
                ood_plateau_counter += 1
                
            stats += f" | OOD(N={ood_len}): {ood_acc:.1%} (Counter: {ood_plateau_counter})"
            pbar.set_description(stats)
            
            if ood_plateau_counter >= 5: # 10k steps with no OOD improvement
                plateau_trigger = True
                print(f"\n⚠️ OOD PLATEAU DETECTED (Counter={ood_plateau_counter}). Force Evolution!")
        
        # Cooldown check
        steps_since_evo = trainer.step - last_evolution_step
        
        should_evolve = (level_up_trigger or plateau_trigger) and (steps_since_evo > EVOLUTION_COOLDOWN)
        
        if level_up_trigger:
             last_level = current_level # Update level tracking regardless of cooldown to avoid double trigger
        
        if should_evolve:
            trigger_reason = "Level Up" if level_up_trigger else "Plateau"
            print(f"\n⚡ DETECTED {trigger_reason}. TRIGGERING EVOLUTION!")
            last_evolution_step = trainer.step
            ood_plateau_counter = 0 # Reset plateau counter after evolution
            
            # === CHECKPOINT-BASED EVOLUTION ===
            # Step 1: Save current state to disk
            temp_checkpoint = "/tmp/thinksort_evolution_checkpoint.pt"
            trainer.save(temp_checkpoint)
            current_step = trainer.step
            current_lengths = list(trainer.active_lengths)
            current_level_idx = trainer.active_level_idx
            
            # Extract current architecture and model
            current_arch = ArchConfig(
                dim=trainer.config.dim,
                heads=trainer.config.heads,
                ff_mult=trainer.config.ff // trainer.config.dim,
                recurrent=trainer.config.recurrent_steps,
                lr=trainer.config.lr,
                batch_scale=1.0 # Baseline
            )
            # KEEP THE SEED MODEL (Move to CPU to free VRAM for NAS)
            seed_model = trainer.model
            seed_model.to('cpu')
            
            # Step 2: DELETE trainer to free GPU
            del trainer
            hard_reset_gpu()
            print(f"   💾 Saved checkpoint. Model moved to CPU. GPU memory cleared.")
            
            # Step 3: Run NAS with fresh GPU
            nas = MuZeroNAS(
                initial_config=current_arch,
                device=config.device,
                burst_steps=200,
                train_lengths=tuple(current_lengths),
                seed_model=seed_model  # PASS THE SEED!
            )
            
            print("   Running MuZero Search (with Network Morphism)...")
            best_config = nas.search(num_iterations=5)
            
            # Cleanup NAS
            del nas
            hard_reset_gpu()
            print(f"   🔬 NAS complete. Best: dim={best_config.dim}, heads={best_config.heads}, LR={best_config.lr}")
            
            # Step 4: Create new trainer with evolved config
            # Calculate base batch size based on model size then scale
            # Base = 1024 for dim=128
            base_batch = max(64, int(1024 * (128 / best_config.dim) ** 1.0)) # Linear scaling with dim inverse
            final_batch = int(base_batch * best_config.batch_scale)
            
            new_config = Config(
                dim=best_config.dim,
                heads=best_config.heads,
                ff=best_config.ff_dim,
                recurrent_steps=best_config.recurrent,
                vocab=10,
                train_lengths=tuple(current_lengths),
                # Apply optimized batch scale
                samples_per_length=final_batch,
                lr=best_config.lr,
                device=config.device
            )
            
            print(f"   📉 New batch size: {new_config.samples_per_length} (Scale={best_config.batch_scale:.1f})")
            
            # Create fresh trainer (initializes random model)
            trainer = Trainer(new_config)
            
            # Morph the seed model to the new config
            print("   🧬 Morphing seed model to new architecture...")
            new_model = morph_model(seed_model, new_config)
            
            # Inject the morphed model into the trainer
            trainer.reload_model(new_model)
            
            # Restore training state
            trainer.step = current_step
            trainer.active_lengths = current_lengths
            trainer.active_level_idx = current_level_idx
            
            # We need to update the progress bar with the new trainer info if possible,
            # or just continue. The pbar wraps the loop, so it persists.
            
            print(f"   ✅ EVOLUTION COMPLETE: {seed_model.config.dim}d -> {new_model.config.dim}d")
            print(f"   Resuming from step {trainer.step} with lengths {trainer.active_lengths}")
            
            # Cleanup seed
            del seed_model
            hard_reset_gpu()
            
            last_level = current_level
            
    # Save checkpoint
    trainer.save("checkpoint_evolved.pt")
    
    # Final evaluation
    trainer.zero_shot_eval()
    
    return trainer


if __name__ == "__main__":
    main()
