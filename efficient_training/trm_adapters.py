"""
TRM-specific adapter integration.

Provides adapters compatible with TRM's CastedLinear layers.
"""

from typing import Optional, Set, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import TRM layers
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "TinyRecursiveModels"))
    from models.layers import CastedLinear
    HAS_TRM = True
except ImportError:
    HAS_TRM = False
    CastedLinear = None


class LoRACastedLinear(nn.Module):
    """
    LoRA adapter for TRM's CastedLinear layers.
    
    CastedLinear stores weight directly as nn.Parameter, not as a sub-module.
    """
    
    def __init__(
        self,
        base_layer,  # CastedLinear
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = base_layer.weight.shape[1]
        self.out_features = base_layer.weight.shape[0]
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Get device from base layer
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        
        # Store frozen base weight (not as parameter to avoid double-counting)
        self.register_buffer("base_weight", base_layer.weight.data.clone())
        self.has_bias = base_layer.bias is not None
        if self.has_bias:
            self.register_buffer("base_bias", base_layer.bias.data.clone())
        
        # LoRA parameters - initialize on same device as base
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward with proper dtype casting
        dtype = x.dtype
        base_out = F.linear(x, self.base_weight.to(dtype), 
                           self.base_bias.to(dtype) if self.has_bias else None)
        
        # LoRA forward
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A.to(dtype))
        lora_out = F.linear(lora_out, self.lora_B.to(dtype))
        
        return base_out + lora_out * self.scaling


class DoRACastedLinear(nn.Module):
    """
    DoRA adapter for TRM's CastedLinear layers.
    """
    
    def __init__(
        self,
        base_layer,  # CastedLinear
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = base_layer.weight.shape[1]
        self.out_features = base_layer.weight.shape[0]
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Get device from base layer
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        
        # Store frozen base weight
        self.register_buffer("base_weight", base_layer.weight.data.clone())
        self.has_bias = base_layer.bias is not None
        if self.has_bias:
            self.register_buffer("base_bias", base_layer.bias.data.clone())
        
        # LoRA parameters - on same device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        
        # Magnitude parameter (initialized from base weight norms)
        with torch.no_grad():
            weight_norm = base_layer.weight.norm(dim=1)
        self.magnitude = nn.Parameter(weight_norm.clone())
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        
        # Compute adapted weight
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        adapted_weight = self.base_weight + delta_w.to(self.base_weight.dtype)
        
        # Normalize and apply magnitude
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        direction = adapted_weight / (weight_norm + 1e-8)
        final_weight = self.magnitude.unsqueeze(1) * direction
        
        # Forward
        return F.linear(self.dropout(x), final_weight.to(dtype),
                       self.base_bias.to(dtype) if self.has_bias else None)


def inject_trm_adapters(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    use_dora: bool = False,
    target_modules: Optional[Set[str]] = None,
) -> Dict[str, nn.Module]:
    """
    Inject LoRA/DoRA adapters into TRM's CastedLinear layers.
    
    Args:
        model: TRM model
        rank: LoRA rank
        alpha: LoRA scaling
        dropout: Dropout probability
        use_dora: Use DoRA instead of LoRA
        target_modules: Specific modules to adapt (None = all)
        
    Returns:
        Dictionary of adapted modules
    """
    if not HAS_TRM:
        raise ImportError("TRM models not found. Make sure TinyRecursiveModels is available.")
    
    adapted_modules = {}
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, CastedLinear):
            continue
        
        if target_modules is not None and name not in target_modules:
            continue
        
        # Create adapter
        if use_dora:
            adapted = DoRACastedLinear(module, rank=rank, alpha=alpha, dropout=dropout)
        else:
            adapted = LoRACastedLinear(module, rank=rank, alpha=alpha, dropout=dropout)
        
        # Replace module
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else dict(model.named_modules())[parent_name]
        setattr(parent, child_name, adapted)
        
        adapted_modules[name] = adapted
    
    return adapted_modules


def count_trm_parameters(model: nn.Module):
    """Count trainable vs total parameters in a TRM model."""
    trainable = 0
    total = 0
    
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    
    # Also count buffers (frozen weights)
    for buf in model.buffers():
        total += buf.numel()
    
    return trainable, total
