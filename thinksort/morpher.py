"""
Network Morphism Utilities for ThinkSort.

Enables "evolving" the architecture by transferring weights from a smaller model 
to a larger one, attempting to preserve functionality (approximate Net2Net).
"""

import torch
import torch.nn as nn
from copy import deepcopy
from model import UniversalPointerNet, Config

def pad_tensor(old_tensor: torch.Tensor, new_shape: tuple, value: float = 0.0, device: torch.device = None) -> torch.Tensor:
    """
    Pad a tensor to new_shape with value, placing old_tensor in the top-left corner.
    Handles cross-device copying if device is specified.
    """
    if device is None:
        device = old_tensor.device
        
    if old_tensor.shape == new_shape and old_tensor.device == device:
        return old_tensor
    
    new_tensor = torch.full(new_shape, value, device=device, dtype=old_tensor.dtype)
    
    # Calculate slice indices
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_tensor.shape, new_shape))
    
    # Copy data, handling device transfer
    new_tensor[slices] = old_tensor[slices].to(device).detach().clone()
    
    return new_tensor

def morph_linear(old_layer: nn.Linear, new_layer: nn.Linear):
    """Transfer weights for Linear layer."""
    with torch.no_grad():
        new_layer.weight.data = pad_tensor(
            old_layer.weight.data, 
            new_layer.weight.shape, 
            0.0,
            device=new_layer.weight.device
        )
        if old_layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.data = pad_tensor(
                old_layer.bias.data, 
                new_layer.bias.shape, 
                0.0,
                device=new_layer.bias.device
            )

def morph_ln(old_ln: nn.LayerNorm, new_ln: nn.LayerNorm):
    """Transfer weights for LayerNorm."""
    with torch.no_grad():
        new_ln.weight.data = pad_tensor(
            old_ln.weight.data, 
            new_ln.weight.shape, 
            1.0, # Scale=1
            device=new_ln.weight.device
        ) 
        new_ln.bias.data = pad_tensor(
            old_ln.bias.data, 
            new_ln.bias.shape, 
            0.0,
            device=new_ln.bias.device
        )

def morph_embedding(old_emb: nn.Embedding, new_emb: nn.Embedding):
    """Transfer weights for Embedding."""
    with torch.no_grad():
        new_emb.weight.data = pad_tensor(
            old_emb.weight.data, 
            new_emb.weight.shape, 
            0.0,
            device=new_emb.weight.device
        )

def morph_model(old_model: UniversalPointerNet, new_config: Config) -> UniversalPointerNet:
    """
    Create a new model with new_config and transfer weights from old_model.
    """
    # Create new model on the device specified in new_config
    new_model = UniversalPointerNet(new_config).to(new_config.device)
    
    print(f"🧬 Morphing model: {old_model.config.dim} -> {new_config.dim} dim")
    
    # helper for safe transfer
    def transfer(name, old_mod, new_mod):
        if isinstance(old_mod, nn.Linear):
            morph_linear(old_mod, new_mod)
        elif isinstance(old_mod, nn.LayerNorm):
            morph_ln(old_mod, new_mod)
        elif isinstance(old_mod, nn.Embedding):
            morph_embedding(old_mod, new_mod)
    
    # Iterate named modules and transfer if they match
    # We do a rigorous match by name
    old_modules = dict(old_model.named_modules())
    for name, new_mod in new_model.named_modules():
        if name in old_modules:
            old_mod = old_modules[name]
            # Verify types match
            if type(old_mod) == type(new_mod):
                transfer(name, old_mod, new_mod)
    
    # Special handling for Parameters that aren't Modules
    with torch.no_grad():
        new_model.selected_emb.data = pad_tensor(
            old_model.selected_emb.data, 
            new_model.selected_emb.shape,
            0.0,
            device=new_model.selected_emb.device
        )

    return new_model
