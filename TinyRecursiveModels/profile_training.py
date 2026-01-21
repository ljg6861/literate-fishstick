#!/usr/bin/env python3
"""
Training Profiling Script for TRM.

This script measures:
1. GPU utilization and bottleneck identification
2. Memory usage at different batch sizes
3. Backward pass precision
4. Data loading overhead
"""

import os
import sys
import time
import gc

# Disable wandb and compile for clean profiling
os.environ['WANDB_MODE'] = 'disabled'
os.environ['DISABLE_COMPILE'] = '1'

sys.path.insert(0, '/home/lucas/literate-fishstick/TinyRecursiveModels')
sys.path.insert(0, '/home/lucas/literate-fishstick')

import torch
import torch.distributed as dist
import numpy as np

def check_gpu_memory():
    """Check available GPU memory."""
    print("\n" + "="*60)
    print("GPU MEMORY CHECK")
    print("="*60)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1e9
        print(f"GPU {i}: {props.name}")
        print(f"  Total memory: {total:.1f} GB")
        
        # Check current usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  Currently allocated: {allocated:.2f} GB")
        print(f"  Currently reserved: {reserved:.2f} GB")
        print(f"  Available: ~{total - reserved:.1f} GB")


def profile_data_loading(num_batches=20):
    """Profile data loading speed."""
    print("\n" + "="*60)
    print("DATA LOADING PROFILE")
    print("="*60)
    
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from torch.utils.data import DataLoader
    
    # NOTE: Dataset only supports num_workers=1
    print("NOTE: PuzzleDataset only supports num_workers=1 (assertion in code)")
    
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_paths=['data/arc1concept-aug-1000'],
        rank=0,
        num_replicas=1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=64,
    ), split="train")
    
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Time data loading
    start = time.time()
    batch_count = 0
    for batch in loader:
        batch_count += 1
        if batch_count >= num_batches:
            break
    elapsed = time.time() - start
    
    batches_per_sec = num_batches / elapsed
    samples_per_sec = num_batches * 64 / elapsed
    
    print(f"num_workers=1: {batches_per_sec:.1f} batches/s, {samples_per_sec:.0f} samples/s")
    print(f"Data loading is {'LIKELY NOT' if batches_per_sec > 100 else 'POSSIBLY'} a bottleneck")
    
    del loader, dataset
    gc.collect()


def profile_forward_backward(batch_sizes=[32, 64, 128, 256], num_steps=5):
    """Profile forward/backward at different batch sizes."""
    print("\n" + "="*60)
    print("FORWARD/BACKWARD PROFILE")
    print("="*60)
    
    import json
    from utils.functions import load_model_class
    
    # Load metadata
    with open('data/arc1concept-aug-1000/train/dataset.json', 'r') as f:
        meta = json.load(f)
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Create model config
            model_cfg = dict(
                batch_size=batch_size,
                vocab_size=meta['vocab_size'],
                seq_len=meta['seq_len'],
                num_puzzle_identifiers=meta['num_puzzle_identifiers'],
                causal=False,
                # TRM arch defaults
                H_cycles=3,
                L_cycles=6,
                H_layers=0,
                L_layers=2,
                hidden_size=512,
                num_heads=8,
                expansion=4,
                puzzle_emb_ndim=512,
                puzzle_emb_len=16,
                pos_encodings='rope',
                forward_dtype='bfloat16',
                mlp_t=False,
                no_ACT_continue=True,
                halt_max_steps=16,
                halt_exploration_prob=0.1,
            )
            
            # Create model
            with torch.device("cuda"):
                model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
                loss_cls = load_model_class("losses@ACTLossHead")
                
                model = model_cls(model_cfg)
                model = loss_cls(model, loss_type='stablemax_cross_entropy')
                model.train()
            
            # Create dummy batch
            batch = {
                'inputs': torch.randint(0, 12, (batch_size, 900), device='cuda'),
                'labels': torch.randint(0, 12, (batch_size, 900), device='cuda'),
                'puzzle_identifiers': torch.randint(1, 1000, (batch_size,), device='cuda'),
            }
            
            # Warmup
            with torch.device("cuda"):
                carry = model.model.initial_carry(batch)
            
            for _ in range(2):
                carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
                loss.backward()
                model.zero_grad()
            
            torch.cuda.synchronize()
            
            # Profile
            forward_times = []
            backward_times = []
            
            for step in range(num_steps):
                carry = model.model.initial_carry(batch)
                
                torch.cuda.synchronize()
                t0 = time.time()
                
                carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
                
                torch.cuda.synchronize()
                t1 = time.time()
                
                loss.backward()
                
                torch.cuda.synchronize()
                t2 = time.time()
                
                forward_times.append(t1 - t0)
                backward_times.append(t2 - t1)
                
                model.zero_grad()
            
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            
            avg_forward = np.mean(forward_times) * 1000
            avg_backward = np.mean(backward_times) * 1000
            avg_total = avg_forward + avg_backward
            
            print(f"\nbatch_size={batch_size}:")
            print(f"  Forward:  {avg_forward:.1f} ms")
            print(f"  Backward: {avg_backward:.1f} ms")
            print(f"  Total:    {avg_total:.1f} ms")
            print(f"  Throughput: {batch_size / (avg_total/1000):.0f} samples/s")
            print(f"  Peak memory: {peak_memory:.2f} GB")
            
            # Check backward precision
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"  Gradient dtype: {param.grad.dtype}")
                    break
            
            del model, batch, carry
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nbatch_size={batch_size}: OUT OF MEMORY")
            else:
                print(f"\nbatch_size={batch_size}: ERROR - {e}")
            
        torch.cuda.empty_cache()
        gc.collect()


def profile_gpu_utilization(duration=10):
    """Check if GPU is fully utilized during training."""
    print("\n" + "="*60)
    print("GPU UTILIZATION ANALYSIS")
    print("="*60)
    print("(Check nvidia-smi during training for real utilization)")
    print("Key metrics to look for:")
    print("  - GPU Util %: Should be >90% if GPU-bound")
    print("  - Memory %: Shows how much headroom we have")
    print("  - If GPU Util is low, we're likely data-bound")


def main():
    print("="*60)
    print("TRM TRAINING PROFILER")
    print("="*60)
    
    os.chdir('/home/lucas/literate-fishstick/TinyRecursiveModels')
    
    # Run checks
    check_gpu_memory()
    profile_data_loading(num_batches=10)
    profile_forward_backward(batch_sizes=[32, 64, 128, 256, 384], num_steps=3)
    profile_gpu_utilization()
    
    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
