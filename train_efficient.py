#!/usr/bin/env python3
"""
Efficient Training Script for TRM

Combines PEFT adapters (LoRA/DoRA), quantization (QLoRA), and memory optimization
(GaLore) for training TRM on consumer GPUs.

Usage:
    # Dry run to check memory
    python train_efficient.py --dry-run --config efficient_4090
    
    # Train with QLoRA + DoRA
    python train_efficient.py --use-qlora --use-dora arch=trm data_paths="[data/arc1concept-aug-1000]"
    
    # Train with GaLore (no quantization)
    python train_efficient.py --use-galore arch=trm data_paths="[data/arc1concept-aug-1000]"
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW

# Add TinyRecursiveModels to path
TRM_PATH = os.path.join(os.path.dirname(__file__), "TinyRecursiveModels")
sys.path.insert(0, TRM_PATH)

# Try to import efficient training modules
try:
    from efficient_training import (
        inject_lora_adapters,
        merge_lora_weights,
        get_lora_param_groups,
        count_lora_parameters,
        prepare_for_qlora,
        GaLoreAdamW,
        estimate_memory_savings,
        estimate_galore_memory_savings,
        get_gpu_memory_info,
        print_memory_summary,
        estimate_training_memory,
    )
    HAS_EFFICIENT_TRAINING = True
except ImportError as e:
    print(f"Warning: Could not import efficient_training: {e}")
    HAS_EFFICIENT_TRAINING = False


@dataclass
class EfficientTrainingConfig:
    """Configuration for efficient training techniques."""
    
    # PEFT settings
    use_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    use_pissa: bool = False
    use_rslora: bool = False
    
    # LoRA+ settings
    lora_plus: bool = False
    lora_lr_A: float = 1e-4
    lora_lr_B: float = 2e-4  # LoRA+ typically uses higher LR for B
    
    # Quantization settings
    use_qlora: bool = False
    quant_bits: int = 4
    use_loftq: bool = False
    compute_dtype: str = "bfloat16"
    
    # GaLore settings
    use_galore: bool = False
    galore_rank: int = 128
    galore_update_freq: int = 200
    galore_scale: float = 1.0
    
    # Target modules (None = all linear layers)
    target_modules: Optional[List[str]] = None
    
    # General
    dry_run: bool = False
    profile: bool = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Efficient TRM Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Efficient training flags
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters")
    parser.add_argument("--use-dora", action="store_true", help="Use DoRA instead of LoRA")
    parser.add_argument("--use-qlora", action="store_true", help="Enable 4-bit QLoRA")
    parser.add_argument("--use-galore", action="store_true", help="Enable GaLore optimizer")
    parser.add_argument("--use-pissa", action="store_true", help="Use PiSSA initialization")
    parser.add_argument("--use-loftq", action="store_true", help="Use LoftQ initialization")
    parser.add_argument("--lora-plus", action="store_true", help="Enable LoRA+ (different LRs)")
    
    # Hyperparameters
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--galore-rank", type=int, default=128, help="GaLore rank")
    
    # Utility flags
    parser.add_argument("--dry-run", action="store_true", help="Estimate memory without training")
    parser.add_argument("--profile", action="store_true", help="Profile memory usage")
    parser.add_argument("--config", type=str, default=None, help="Efficient config name")
    
    # Pass-through for Hydra
    args, remaining = parser.parse_known_args()
    args.hydra_args = remaining
    
    return args


def create_efficient_config(args: argparse.Namespace) -> EfficientTrainingConfig:
    """Create efficient training config from args."""
    return EfficientTrainingConfig(
        use_lora=args.use_lora or args.use_dora or args.use_qlora,
        use_dora=args.use_dora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_pissa=args.use_pissa,
        lora_plus=args.lora_plus,
        use_qlora=args.use_qlora,
        use_loftq=args.use_loftq,
        use_galore=args.use_galore,
        galore_rank=args.galore_rank,
        dry_run=args.dry_run,
        profile=args.profile,
    )


def apply_efficient_training(
    model: nn.Module,
    config: EfficientTrainingConfig,
) -> Dict[str, Any]:
    """
    Apply efficient training techniques to a model.
    
    Returns:
        Dictionary with adaptation info
    """
    if not HAS_EFFICIENT_TRAINING:
        raise ImportError("efficient_training module not available")
    
    info = {
        "techniques": [],
        "trainable_params": 0,
        "total_params": 0,
        "adapted_modules": [],
    }
    
    # Compute dtype
    compute_dtype = getattr(torch, config.compute_dtype)
    
    # Apply quantization if requested
    if config.use_qlora:
        print(f"Applying QLoRA (4-bit quantization + LoRA rank={config.lora_rank})")
        adapted = prepare_for_qlora(
            model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            compute_dtype=compute_dtype,
            use_loftq=config.use_loftq,
            use_dora=config.use_dora,
        )
        info["techniques"].append("qlora")
        info["adapted_modules"] = list(adapted.keys())
        
        if config.use_loftq:
            info["techniques"].append("loftq")
    
    # Apply LoRA/DoRA if requested (without quantization)
    elif config.use_lora or config.use_dora:
        technique = "dora" if config.use_dora else "lora"
        print(f"Applying {technique.upper()} (rank={config.lora_rank}, alpha={config.lora_alpha})")
        
        adapted = inject_lora_adapters(
            model,
            target_modules=set(config.target_modules) if config.target_modules else None,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            use_dora=config.use_dora,
            use_pissa=config.use_pissa,
            use_rslora=config.use_rslora,
        )
        info["techniques"].append(technique)
        info["adapted_modules"] = list(adapted.keys())
        
        if config.use_pissa:
            info["techniques"].append("pissa")
    
    # Count parameters
    trainable, total = count_lora_parameters(model)
    info["trainable_params"] = trainable
    info["total_params"] = total
    info["trainable_ratio"] = trainable / total if total > 0 else 0
    
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*info['trainable_ratio']:.2f}%)")
    
    return info


def create_efficient_optimizer(
    model: nn.Module,
    config: EfficientTrainingConfig,
    base_lr: float = 1e-4,
    weight_decay: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Create optimizer with efficient training techniques.
    
    Returns:
        Configured optimizer
    """
    if config.use_galore:
        print(f"Using GaLore optimizer (rank={config.galore_rank})")
        return GaLoreAdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
            galore_rank=config.galore_rank,
            galore_update_freq=config.galore_update_freq,
            galore_scale=config.galore_scale,
        )
    
    if config.lora_plus and (config.use_lora or config.use_dora):
        print(f"Using LoRA+ (lr_A={config.lora_lr_A}, lr_B={config.lora_lr_B})")
        param_groups = get_lora_param_groups(
            model,
            lr_A=config.lora_lr_A,
            lr_B=config.lora_lr_B,
        )
        return AdamW(param_groups, weight_decay=weight_decay)
    
    # Standard optimizer for trainable params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)


def run_dry_run(config: EfficientTrainingConfig):
    """Run memory estimation without actual training."""
    print("\n" + "="*60)
    print("DRY RUN: Memory Estimation")
    print("="*60 + "\n")
    
    # Dummy model stats for TRM
    print("Estimated for TRM (7M params, hidden=512, seq=900):")
    print()
    
    # Base memory
    base_params = 7_000_000
    param_bytes = base_params * 2  # bfloat16
    print(f"Model parameters: {param_bytes / (1024**2):.1f} MB")
    
    if config.use_qlora:
        # 4-bit: ~0.5 bytes per param
        quant_bytes = int(base_params * 0.5 * 1.1)
        print(f"After QLoRA quantization: {quant_bytes / (1024**2):.1f} MB ({100*(1-quant_bytes/param_bytes):.0f}% savings)")
        
        # LoRA params
        lora_params = config.lora_rank * 512 * 2 * 4  # rough: rank * hidden * 2 matrices * num_layers
        print(f"LoRA adapter parameters: {lora_params * 2 / (1024**2):.3f} MB")
    
    if config.use_galore:
        # GaLore optimizer savings
        standard_opt = base_params * 2 * 2 / (1024**2)  # 2 states * 2 bytes
        galore_opt = standard_opt * (config.galore_rank / 512)
        print(f"Optimizer states (standard): {standard_opt:.1f} MB")
        print(f"Optimizer states (GaLore): {galore_opt:.1f} MB ({100*(1-galore_opt/standard_opt):.0f}% savings)")
    
    print()
    print_memory_summary(prefix="Current ")


def main():
    """Main entry point for efficient training."""
    args = parse_args()
    config = create_efficient_config(args)
    
    print("\n" + "="*60)
    print("TRM Efficient Training")
    print("="*60)
    print(f"Techniques: ", end="")
    techniques = []
    if config.use_dora:
        techniques.append("DoRA")
    elif config.use_lora:
        techniques.append("LoRA")
    if config.use_qlora:
        techniques.append("QLoRA (4-bit)")
    if config.use_pissa:
        techniques.append("PiSSA init")
    if config.use_loftq:
        techniques.append("LoftQ init")
    if config.use_galore:
        techniques.append("GaLore")
    if config.lora_plus:
        techniques.append("LoRA+")
    print(", ".join(techniques) if techniques else "None (full fine-tuning)")
    print("="*60 + "\n")
    
    if config.dry_run:
        run_dry_run(config)
        return
    
    # Import TRM pretrain for actual training
    try:
        os.chdir(TRM_PATH)
        from pretrain import launch
    except ImportError as e:
        print(f"Error: Could not import TRM pretrain module: {e}")
        print("Make sure you're running from the correct directory.")
        sys.exit(1)
    
    # TODO: Hook into TRM training loop
    # For now, print instructions
    print("To train with efficient techniques, modify pretrain.py to:")
    print("1. Call apply_efficient_training(model, config) after model creation")
    print("2. Use create_efficient_optimizer() instead of standard optimizer")
    print()
    print("Or run the standard pretrain.py with the model modifications applied.")
    
    # Pass through to hydra launch if args provided
    if args.hydra_args:
        sys.argv = [sys.argv[0]] + args.hydra_args
        launch()


if __name__ == "__main__":
    main()
