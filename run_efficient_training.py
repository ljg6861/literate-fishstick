#!/usr/bin/env python3
"""
Run TRM training with efficient techniques (DoRA + GaLore)

This script runs actual training on ARC data with the efficient training stack.
"""

import os
import sys
import argparse
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TinyRecursiveModels"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import efficient training
from efficient_training import (
    GaLoreAdamW,
    print_memory_summary,
    memory_tracker,
)
from efficient_training.trm_adapters import inject_trm_adapters, count_trm_parameters

# Import TRM components
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def parse_args():
    parser = argparse.ArgumentParser(description="TRM Efficient Training")
    parser.add_argument("--data-path", type=str, default="TinyRecursiveModels/data/arc1concept-aug-1000",
                        help="Path to ARC dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--use-dora", action="store_true", default=True, help="Use DoRA")
    parser.add_argument("--use-galore", action="store_true", default=True, help="Use GaLore")
    parser.add_argument("--galore-rank", type=int, default=64, help="GaLore rank")
    parser.add_argument("--eval-interval", type=int, default=10, help="Eval every N epochs")
    parser.add_argument("--save-path", type=str, default="checkpoints/efficient", help="Save path")
    return parser.parse_args()


def create_model(metadata, device, batch_size):
    """Create TRM model."""
    config = {
        "batch_size": batch_size,
        "seq_len": metadata.seq_len,
        "puzzle_emb_ndim": 512,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "vocab_size": metadata.vocab_size,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
    }
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    return model.to(device)


def move_carry_to_device(carry, device):
    """Move carry state to device."""
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.steps = carry.steps.to(device)
    carry.halted = carry.halted.to(device)
    carry.current_data = {k: v.to(device) for k, v in carry.current_data.items()}
    return carry


def compute_loss(outputs, batch, vocab_size):
    """Compute cross-entropy loss."""
    logits = outputs["logits"]  # [B, seq, vocab]
    targets = batch.get("targets", batch["inputs"])  # Use inputs as targets for now
    
    # Flatten - use reshape to handle non-contiguous tensors
    logits_flat = logits.contiguous().view(-1, vocab_size).float()  # Cast to float32 for loss
    targets_flat = targets.contiguous().view(-1).long()  # Cast to long for cross_entropy
    
    # Cross entropy
    loss = nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
    return loss


def train_epoch(model, dataloader, optimizer, device, vocab_size, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (set_name, batch, global_batch_size) in enumerate(dataloader):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        carry = model.initial_carry(batch)
        carry = move_carry_to_device(carry, device)
        
        optimizer.zero_grad()
        
        # Forward
        carry, outputs = model(carry, batch)
        
        # Compute loss
        loss = compute_loss(outputs, batch, vocab_size)
        
        # Backward
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device, vocab_size):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for set_name, batch, global_batch_size in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)
            carry = move_carry_to_device(carry, device)
            
            carry, outputs = model(carry, batch)
            
            loss = compute_loss(outputs, batch, vocab_size)
            total_loss += loss.item()
            
            # Accuracy
            preds = outputs["logits"].argmax(dim=-1)
            targets = batch.get("targets", batch["inputs"])
            mask = targets != -100
            total_correct += ((preds == targets) & mask).sum().item()
            total_tokens += mask.sum().item()
    
    accuracy = total_correct / max(total_tokens, 1)
    return total_loss, accuracy


def main():
    args = parse_args()
    
    print("="*60)
    print("TRM Efficient Training")
    print("="*60)
    print(f"Data: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"DoRA: {args.use_dora}")
    print(f"GaLore: {args.use_galore} (rank={args.galore_rank})")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print_memory_summary(prefix="Initial ")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset_config = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[args.data_path],
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        global_batch_size=args.batch_size,
        test_set_mode=False,
    )
    
    train_dataset = PuzzleDataset(dataset_config, split="train")
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    
    metadata = train_dataset.metadata
    print(f"Dataset: {metadata.total_groups} groups, vocab={metadata.vocab_size}, seq_len={metadata.seq_len}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(metadata, device, args.batch_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {total_params:,}")
    
    # Apply efficient training
    print("\nApplying efficient training techniques...")
    adapted = inject_trm_adapters(
        model,
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,
        use_dora=args.use_dora,
    )
    print(f"Adapted {len(adapted)} modules with {'DoRA' if args.use_dora else 'LoRA'}")
    
    trainable, total = count_trm_parameters(model)
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if args.use_galore:
        print(f"\nUsing GaLore optimizer (rank={args.galore_rank})")
        optimizer = GaLoreAdamW(
            trainable_params,
            lr=args.lr,
            galore_rank=args.galore_rank,
            galore_update_freq=100,
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    print_memory_summary(prefix="After setup ")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        epoch_start = datetime.now()
        
        avg_loss = train_epoch(model, train_loader, optimizer, device, metadata.vocab_size, epoch)
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\nEpoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            print("\nEvaluating...")
            eval_loss, accuracy = evaluate(model, train_loader, device, metadata.vocab_size)
            print(f"Eval Loss: {eval_loss:.4f} | Accuracy: {100*accuracy:.2f}%")
            
            if eval_loss < best_loss:
                best_loss = eval_loss
                os.makedirs(args.save_path, exist_ok=True)
                save_file = os.path.join(args.save_path, f"best_model.pt")
                torch.save(model.state_dict(), save_file)
                print(f"Saved best model to {save_file}")
        
        print_memory_summary(prefix="")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
