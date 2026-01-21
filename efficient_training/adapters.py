"""
PEFT Adapters for TRM: LoRA, DoRA, LoRA+, and PiSSA

Low-Rank Adaptation (LoRA): Freeze base weights, train low-rank A·B decomposition
DoRA: Weight-Decomposed LoRA - splits into magnitude and direction components
LoRA+: Different learning rates for A and B matrices
PiSSA: Principal Singular Values and Singular Vectors Adaptation (SVD-based init)
"""

from typing import Optional, Tuple, Dict, List, Set
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer for Linear layers.
    
    Instead of updating W directly, we train W + BA where:
    - B: [out_features, rank] 
    - A: [rank, in_features]
    
    This reduces trainable parameters from out*in to rank*(out+in)
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_rslora: bool = False,  # Rank-stabilized LoRA scaling
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA matrices: W' = W + (alpha/rank) * B @ A
        # A initialized with Kaiming, B initialized to zero (start at identity)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        if use_rslora:
            # Rank-stabilized scaling: alpha / sqrt(rank)
            self.scaling = alpha / math.sqrt(rank)
        else:
            # Standard LoRA scaling: alpha / rank
            self.scaling = alpha / rank
        
        self._init_weights()
    
    def _init_weights(self):
        # Kaiming uniform for A, zero for B (paper recommendation)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already zero-initialized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward
        base_out = self.base_layer(x)
        
        # LoRA forward: dropout(x) @ A^T @ B^T * scaling
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # [*, rank]
        lora_out = F.linear(lora_out, self.lora_B)  # [*, out_features]
        
        return base_out + lora_out * self.scaling
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base layer for inference."""
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None
        )
        
        with torch.no_grad():
            # W' = W + scaling * B @ A
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            merged.weight.copy_(self.base_layer.weight + delta_w)
            if self.base_layer.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
        
        return merged


class DoRALinear(nn.Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).
    
    Decomposes weight updates into:
    - Magnitude component (m): scalar per output dimension
    - Direction component (LoRA): low-rank update to direction
    
    W' = m * (W + BA) / ||W + BA||
    
    This often achieves full fine-tuning quality with adapter efficiency.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / rank
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA matrices for direction
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude vector (initialized from base weight norms)
        with torch.no_grad():
            weight_norm = base_layer.weight.norm(dim=1, keepdim=True)
        self.magnitude = nn.Parameter(weight_norm.squeeze().clone())
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute adapted weight: W + scaling * B @ A
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        adapted_weight = self.base_layer.weight + delta_w
        
        # Normalize to get direction
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        direction = adapted_weight / (weight_norm + 1e-8)
        
        # Apply magnitude
        final_weight = self.magnitude.unsqueeze(1) * direction
        
        # Forward pass
        out = F.linear(self.dropout(x), final_weight, self.base_layer.bias)
        return out
    
    def merge_weights(self) -> nn.Linear:
        """Merge DoRA weights into base layer for inference."""
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None
        )
        
        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            adapted_weight = self.base_layer.weight + delta_w
            weight_norm = adapted_weight.norm(dim=1, keepdim=True)
            direction = adapted_weight / (weight_norm + 1e-8)
            final_weight = self.magnitude.unsqueeze(1) * direction
            
            merged.weight.copy_(final_weight)
            if self.base_layer.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
        
        return merged


def pissa_init(
    base_weight: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Principal Singular Values and Singular Vectors Adaptation (PiSSA) initialization.
    
    Instead of random init, use SVD of the base weight to initialize LoRA matrices.
    This captures the principal components and leads to faster convergence.
    
    Returns:
        residual: The low-rank approximation residual (new base weight)
        A: [rank, in_features] initialized from V^T
        B: [out_features, rank] initialized from U * S
    """
    # Perform SVD
    U, S, Vh = torch.linalg.svd(base_weight.float(), full_matrices=False)
    
    # Take top-r components
    U_r = U[:, :rank]  # [out, rank]
    S_r = S[:rank]     # [rank]
    Vh_r = Vh[:rank, :]  # [rank, in]
    
    # Initialize A from V^T, B from U * sqrt(S)
    # Using sqrt(S) on both sides for balanced gradients
    sqrt_S = torch.sqrt(S_r)
    A = (sqrt_S.unsqueeze(1) * Vh_r).to(base_weight.dtype)  # [rank, in]
    B = (U_r * sqrt_S.unsqueeze(0)).to(base_weight.dtype)   # [out, rank]
    
    # Compute residual (base weight minus the rank-r approximation)
    low_rank_approx = B @ A
    residual = base_weight - low_rank_approx
    
    return residual, A, B


def apply_pissa_init(lora_layer: LoRALinear) -> None:
    """Apply PiSSA initialization to an existing LoRA layer."""
    with torch.no_grad():
        residual, A, B = pissa_init(
            lora_layer.base_layer.weight,
            lora_layer.rank
        )
        lora_layer.lora_A.copy_(A)
        lora_layer.lora_B.copy_(B)
        # Update base layer to residual
        lora_layer.base_layer.weight.copy_(residual)


def inject_lora_adapters(
    model: nn.Module,
    target_modules: Optional[Set[str]] = None,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    use_dora: bool = False,
    use_pissa: bool = False,
    use_rslora: bool = False,
) -> Dict[str, nn.Module]:
    """
    Inject LoRA/DoRA adapters into a model's Linear layers.
    
    Args:
        model: The model to modify
        target_modules: Set of module names to adapt. If None, adapts all Linear layers.
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA path
        use_dora: Use DoRA instead of LoRA
        use_pissa: Apply PiSSA initialization
        use_rslora: Use rank-stabilized scaling
        
    Returns:
        Dictionary of original module names to their LoRA-adapted versions
    """
    adapted_modules = {}
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
            
        if target_modules is not None and name not in target_modules:
            continue
        
        # Create adapter
        if use_dora:
            adapted = DoRALinear(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        else:
            adapted = LoRALinear(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_rslora=use_rslora,
            )
        
        # Apply PiSSA if requested
        if use_pissa and not use_dora:  # PiSSA only for standard LoRA
            apply_pissa_init(adapted)
        
        # Replace module in parent
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else dict(model.named_modules())[parent_name]
        setattr(parent, child_name, adapted)
        
        adapted_modules[name] = adapted
    
    return adapted_modules


def merge_lora_weights(model: nn.Module) -> None:
    """
    Merge all LoRA/DoRA adapters back into base weights.
    
    This is useful for inference to remove adapter overhead.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, DoRALinear)):
            merged = module.merge_weights()
            
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if not parent_name else dict(model.named_modules())[parent_name]
            setattr(parent, child_name, merged)


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get only the trainable LoRA parameters (for optimizer)."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A, module.lora_B])
        elif isinstance(module, DoRALinear):
            lora_params.extend([module.lora_A, module.lora_B, module.magnitude])
    return lora_params


def get_lora_param_groups(
    model: nn.Module,
    lr_A: float = 1e-4,
    lr_B: float = 1e-4,
    lr_magnitude: float = 1e-4,
) -> List[Dict]:
    """
    Get parameter groups for LoRA+ optimization.
    
    LoRA+ uses different learning rates for A and B matrices.
    Typically lr_B > lr_A works well.
    """
    param_groups = []
    
    params_A = []
    params_B = []
    params_magnitude = []
    
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params_A.append(module.lora_A)
            params_B.append(module.lora_B)
        elif isinstance(module, DoRALinear):
            params_A.append(module.lora_A)
            params_B.append(module.lora_B)
            params_magnitude.append(module.magnitude)
    
    if params_A:
        param_groups.append({"params": params_A, "lr": lr_A, "name": "lora_A"})
    if params_B:
        param_groups.append({"params": params_B, "lr": lr_B, "name": "lora_B"})
    if params_magnitude:
        param_groups.append({"params": params_magnitude, "lr": lr_magnitude, "name": "magnitude"})
    
    return param_groups


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable vs total parameters in a LoRA-adapted model.
    
    Returns:
        (trainable_params, total_params)
    """
    trainable = 0
    total = 0
    
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    
    return trainable, total
