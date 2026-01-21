"""
Memory utilities for efficient training.

Provides profiling, estimation, and optimization tools for GPU memory management.
"""

from typing import Dict, Any, Optional, List, Callable
import gc
import torch
import torch.nn as nn
from contextlib import contextmanager


def get_gpu_memory_info(device: int = 0) -> Dict[str, int]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory info in bytes
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    free = total - reserved
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "total": total,
        "free": free,
        "allocated_mb": allocated / (1024 ** 2),
        "reserved_mb": reserved / (1024 ** 2),
        "total_mb": total / (1024 ** 2),
        "free_mb": free / (1024 ** 2),
    }


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def memory_tracker(label: str = ""):
    """
    Context manager to track memory usage of a code block.
    
    Usage:
        with memory_tracker("Forward pass"):
            output = model(input)
    """
    if not torch.cuda.is_available():
        yield {}
        return
    
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()
    
    result = {"label": label, "start_mb": start_mem / (1024 ** 2)}
    
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated()
        result["end_mb"] = end_mem / (1024 ** 2)
        result["delta_mb"] = (end_mem - start_mem) / (1024 ** 2)
        print(f"[Memory] {label}: {result['delta_mb']:.2f} MB (total: {result['end_mb']:.2f} MB)")


def profile_model_memory(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    include_backward: bool = True,
) -> Dict[str, Any]:
    """
    Profile memory usage for a model.
    
    Args:
        model: Model to profile
        input_shape: Shape of input (without batch dimension)
        batch_size: Batch size for profiling
        dtype: Data type
        device: Device to profile on
        include_backward: Include backward pass memory
        
    Returns:
        Dictionary with memory breakdown
    """
    if not torch.cuda.is_available() or device != "cuda":
        return {"error": "CUDA not available"}
    
    clear_gpu_memory()
    
    model = model.to(device).to(dtype)
    
    result = {
        "batch_size": batch_size,
        "input_shape": input_shape,
        "dtype": str(dtype),
    }
    
    # Model parameters memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    result["parameters_mb"] = param_bytes / (1024 ** 2)
    
    # Buffers memory
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    result["buffers_mb"] = buffer_bytes / (1024 ** 2)
    
    clear_gpu_memory()
    baseline = torch.cuda.memory_allocated()
    
    # Forward pass
    full_shape = (batch_size,) + tuple(input_shape)
    x = torch.randn(full_shape, dtype=dtype, device=device)
    
    torch.cuda.synchronize()
    pre_forward = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        output = model(x) if not include_backward else None
    
    if include_backward:
        output = model(x)
        torch.cuda.synchronize()
        post_forward = torch.cuda.memory_allocated()
        result["forward_activations_mb"] = (post_forward - pre_forward) / (1024 ** 2)
        
        # Backward pass
        if isinstance(output, torch.Tensor):
            loss = output.sum()
        elif isinstance(output, (tuple, list)):
            loss = output[0].sum() if isinstance(output[0], torch.Tensor) else 0
        elif isinstance(output, dict):
            loss = sum(v.sum() for v in output.values() if isinstance(v, torch.Tensor))
        else:
            loss = torch.tensor(0.0, device=device)
        
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            torch.cuda.synchronize()
            pre_backward = torch.cuda.memory_allocated()
            loss.backward()
            torch.cuda.synchronize()
            post_backward = torch.cuda.memory_allocated()
            result["backward_peak_mb"] = (post_backward - pre_backward) / (1024 ** 2)
            
            # Gradient memory
            grad_bytes = sum(
                p.grad.numel() * p.grad.element_size() 
                for p in model.parameters() 
                if p.grad is not None
            )
            result["gradients_mb"] = grad_bytes / (1024 ** 2)
    else:
        torch.cuda.synchronize()
        post_forward = torch.cuda.memory_allocated()
        result["forward_activations_mb"] = (post_forward - pre_forward) / (1024 ** 2)
    
    # Peak memory
    result["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    result["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
    
    clear_gpu_memory()
    torch.cuda.reset_peak_memory_stats()
    
    return result


def estimate_training_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    optimizer: str = "adam",
    use_gradient_checkpointing: bool = False,
    use_galore: bool = False,
    galore_rank: int = 128,
) -> Dict[str, float]:
    """
    Estimate total training memory requirements.
    
    Returns:
        Dictionary with memory estimates in MB
    """
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model parameters
    param_mem = num_params * bytes_per_param / (1024 ** 2)
    
    # Gradients (only for trainable params)
    grad_mem = trainable_params * bytes_per_param / (1024 ** 2)
    
    # Optimizer states
    if optimizer.lower() == "adam":
        if use_galore:
            # GaLore: reduced optimizer states
            # Rough estimate: rank / dim reduction
            avg_reduction = galore_rank / hidden_size
            optimizer_mem = 2 * trainable_params * bytes_per_param * avg_reduction / (1024 ** 2)
        else:
            # Standard Adam: 2 states per param
            optimizer_mem = 2 * trainable_params * bytes_per_param / (1024 ** 2)
    elif optimizer.lower() == "sgd":
        optimizer_mem = trainable_params * bytes_per_param / (1024 ** 2)  # Only momentum
    else:
        optimizer_mem = 2 * trainable_params * bytes_per_param / (1024 ** 2)  # Default to Adam
    
    # Activations (rough estimate based on batch size and sequence length)
    # This is highly model-dependent
    if use_gradient_checkpointing:
        # Checkpointing reduces activation memory by sqrt(layers)
        num_layers = sum(1 for _ in model.modules() if isinstance(_, (nn.Linear, nn.MultiheadAttention)))
        activation_reduction = 1 / max(1, num_layers ** 0.5)
    else:
        activation_reduction = 1.0
    
    # Rough activation estimate: batch * seq * hidden * layers * 2 (for transformer)
    num_layers = max(1, sum(1 for _ in model.modules() if isinstance(_, nn.Linear)) // 4)
    activation_mem = (batch_size * seq_len * hidden_size * num_layers * 2 * bytes_per_param * activation_reduction) / (1024 ** 2)
    
    total = param_mem + grad_mem + optimizer_mem + activation_mem
    
    return {
        "parameters_mb": param_mem,
        "gradients_mb": grad_mem,
        "optimizer_states_mb": optimizer_mem,
        "activations_mb": activation_mem,
        "total_estimated_mb": total,
        "total_estimated_gb": total / 1024,
        "trainable_params": trainable_params,
        "total_params": num_params,
    }


class GradientCheckpointWrapper(nn.Module):
    """
    Wrapper to apply gradient checkpointing to a module.
    
    Trades compute for memory by recomputing activations during backward pass.
    """
    
    def __init__(self, module: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant
    
    def forward(self, *args, **kwargs):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module,
                *args,
                use_reentrant=self.use_reentrant,
                **kwargs
            )
        else:
            return self.module(*args, **kwargs)


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_modules: Optional[List[type]] = None,
) -> None:
    """
    Apply gradient checkpointing to specific module types.
    
    Args:
        model: Model to modify
        checkpoint_modules: List of module types to checkpoint (default: large layers)
    """
    if checkpoint_modules is None:
        # Default: checkpoint attention and large linear layers
        checkpoint_modules = []
    
    for name, module in model.named_children():
        if any(isinstance(module, t) for t in checkpoint_modules):
            wrapped = GradientCheckpointWrapper(module)
            setattr(model, name, wrapped)
        elif len(list(module.children())) > 0:
            # Recurse into children
            apply_gradient_checkpointing(module, checkpoint_modules)


def print_memory_summary(prefix: str = ""):
    """Print a summary of current GPU memory usage."""
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available")
        return
    
    info = get_gpu_memory_info()
    print(f"{prefix}GPU Memory: {info['allocated_mb']:.1f} MB allocated / "
          f"{info['reserved_mb']:.1f} MB reserved / "
          f"{info['total_mb']:.1f} MB total / "
          f"{info['free_mb']:.1f} MB free")
