#!/usr/bin/env python3
"""
Integration test: Efficient Training with TRM

Tests that efficient training techniques (DoRA, GaLore) integrate correctly 
with the actual TRM model architecture.
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TinyRecursiveModels"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# Import efficient training
from efficient_training import (
    GaLoreAdamW,
    print_memory_summary,
    memory_tracker,
)
from efficient_training.trm_adapters import inject_trm_adapters, count_trm_parameters

# Import TRM model
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def create_test_model():
    """Create a TRM model for testing."""
    config = {
        "batch_size": 4,
        "seq_len": 64,
        "puzzle_emb_ndim": 256,
        "num_puzzle_identifiers": 100,
        "vocab_size": 11,
        "H_cycles": 2,
        "L_cycles": 3,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 256,
        "expansion": 4,
        "num_heads": 4,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 8,
        "no_ACT_continue": True,
    }
    return TinyRecursiveReasoningModel_ACTV1(config)


def create_batch_and_carry(model, batch_size, seq_len, vocab_size, device):
    """Create batch and properly initialized carry on device."""
    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device),
    }
    carry = model.initial_carry(batch)
    
    # Move carry tensors to device
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.steps = carry.steps.to(device)
    carry.halted = carry.halted.to(device)
    carry.current_data = {k: v.to(device) for k, v in carry.current_data.items()}
    
    return batch, carry


def test_dora_integration():
    """Test DoRA adapter integration with TRM."""
    print("\n" + "="*60)
    print("TEST: DoRA + TRM Integration")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = create_test_model().to(device)
    
    # Count params before
    total_before = sum(p.numel() for p in model.parameters())
    print(f"\nOriginal parameters: {total_before:,}")
    
    # Inject DoRA adapters
    adapted = inject_trm_adapters(model, rank=8, alpha=16, use_dora=True)
    
    print(f"Adapted modules: {len(adapted)}")
    for name in list(adapted.keys())[:3]:
        print(f"  - {name}")
    
    # Count params after
    trainable, total = count_trm_parameters(model)
    print(f"After DoRA: {trainable:,} trainable / {total:,} total")
    print(f"Trainable ratio: {100*trainable/total:.2f}%")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch, carry = create_batch_and_carry(model, 4, 64, 11, device)
    
    with memory_tracker("Forward"):
        carry, outputs = model(carry, batch)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    # Test backward pass
    print("Testing backward pass...")
    loss = outputs["logits"].sum()
    
    with memory_tracker("Backward"):
        loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"Parameters with gradients: {grad_count}")
    
    print("\n✓ DoRA integration test PASSED")
    return True


def test_galore_optimizer():
    """Test GaLore optimizer with TRM + DoRA."""
    print("\n" + "="*60)
    print("TEST: GaLore Optimizer with TRM + DoRA")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with DoRA
    model = create_test_model().to(device)
    inject_trm_adapters(model, rank=8, alpha=16, use_dora=True)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    
    # Create GaLore optimizer
    optimizer = GaLoreAdamW(
        trainable_params,
        lr=1e-3,
        galore_rank=32,
        galore_update_freq=50,
        min_dim_for_galore=64,
    )
    print("GaLore optimizer created")
    
    # Training loop
    print("\nRunning mini training loop (5 steps)...")
    model.train()
    
    losses = []
    for step in range(5):
        batch, carry = create_batch_and_carry(model, 4, 64, 11, device)
        
        optimizer.zero_grad()
        carry, outputs = model(carry, batch)
        
        loss = outputs["logits"].sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        print(f"  Step {step}: loss = {loss.item():.2f}")
    
    print(f"\nLoss: {losses[0]:.2f} -> {losses[-1]:.2f}")
    print("\n✓ GaLore optimizer test PASSED")
    return True


def test_lora_training_loop():
    """Test complete training loop with LoRA."""
    print("\n" + "="*60)
    print("TEST: Complete LoRA Training Loop")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with LoRA (not DoRA)
    model = create_test_model().to(device)
    adapted = inject_trm_adapters(model, rank=8, alpha=16, use_dora=False)
    
    print(f"Adapted {len(adapted)} modules")
    
    # Standard AdamW optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    
    print("\nRunning training loop (5 steps)...")
    model.train()
    
    losses = []
    for step in range(5):
        batch, carry = create_batch_and_carry(model, 4, 64, 11, device)
        
        optimizer.zero_grad()
        carry, outputs = model(carry, batch)
        
        loss = outputs["logits"].sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        print(f"  Step {step}: loss = {loss.item():.2f}")
    
    # Check that gradients flowed
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable_params)
    print(f"\nGradients flowing: {has_grads}")
    
    print("\n✓ LoRA training loop test PASSED")
    return True


def main():
    print("="*60)
    print("TRM Efficient Training Integration Tests")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print_memory_summary(prefix="Initial ")
    else:
        print("CUDA not available, running on CPU")
    
    results = []
    
    # Run tests
    try:
        results.append(("DoRA Integration", test_dora_integration()))
    except Exception as e:
        print(f"\n✗ DoRA Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DoRA Integration", False))
    
    try:
        results.append(("GaLore Optimizer", test_galore_optimizer()))
    except Exception as e:
        print(f"\n✗ GaLore Optimizer FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("GaLore Optimizer", False))
    
    try:
        results.append(("LoRA Training", test_lora_training_loop()))
    except Exception as e:
        print(f"\n✗ LoRA Training FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("LoRA Training", False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if torch.cuda.is_available():
        print_memory_summary(prefix="Final ")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
