"""
Quantization support for TRM: QLoRA, NF4, and LoftQ

QLoRA: 4-bit quantization of base model with trainable LoRA adapters
NF4: Normal Float 4-bit quantization (better for neural network weights)
LoftQ: Quantization-aware LoRA initialization
"""

from typing import Optional, Dict, Set, Tuple
import torch
import torch.nn as nn

# Try to import bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    bnb = None

from .adapters import LoRALinear, DoRALinear, inject_lora_adapters


class Linear4bit(nn.Module):
    """
    4-bit quantized linear layer wrapper.
    
    If bitsandbytes is available, uses NF4 quantization.
    Otherwise, falls back to simulated quantization for testing.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        compute_dtype: torch.dtype = torch.bfloat16,
        quant_type: str = "nf4",
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        
        if HAS_BNB:
            # Use real bitsandbytes 4-bit quantization
            self.weight = bnb.nn.Params4bit(
                base_layer.weight.data,
                requires_grad=False,
                compress_statistics=True,
                quant_type=quant_type,
            )
        else:
            # Fallback: store quantized representation manually
            self.register_buffer("weight_quantized", self._simulate_quantize(base_layer.weight))
            self.register_buffer("weight_scale", base_layer.weight.abs().max() / 7.0)
        
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
    
    def _simulate_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """Simulate 4-bit quantization for testing without bitsandbytes."""
        # Simple uniform quantization to 4-bit range [-8, 7]
        scale = weight.abs().max() / 7.0
        quantized = torch.clamp(torch.round(weight / scale), -8, 7).to(torch.int8)
        return quantized
    
    def _simulate_dequantize(self) -> torch.Tensor:
        """Dequantize simulated weights."""
        return self.weight_quantized.float() * self.weight_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_BNB:
            # bitsandbytes handles dequantization internally
            return bnb.matmul_4bit(
                x, 
                self.weight.t(),
                bias=self.bias,
                quant_state=self.weight.quant_state,
            )
        else:
            # Fallback: dequantize and compute
            weight = self._simulate_dequantize().to(x.dtype)
            return nn.functional.linear(x, weight, self.bias)


def quantize_model_nf4(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    skip_modules: Optional[Set[str]] = None,
) -> Dict[str, Linear4bit]:
    """
    Quantize all Linear layers in a model to 4-bit NF4.
    
    Args:
        model: Model to quantize
        compute_dtype: Dtype for computations (bfloat16 recommended)
        skip_modules: Set of module names to skip
        
    Returns:
        Dictionary of module names to their quantized versions
    """
    skip_modules = skip_modules or set()
    quantized_modules = {}
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name in skip_modules:
            continue
        
        # Create quantized layer
        quantized = Linear4bit(module, compute_dtype=compute_dtype)
        
        # Replace in model
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else dict(model.named_modules())[parent_name]
        setattr(parent, child_name, quantized)
        
        quantized_modules[name] = quantized
    
    return quantized_modules


def loftq_init(
    base_weight: torch.Tensor,
    rank: int,
    num_iters: int = 5,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LoftQ: Quantization-aware LoRA initialization.
    
    Iteratively optimizes the quantized weight and LoRA matrices
    to minimize ||W - Q(W - BA) + BA||.
    
    This reduces the "quantization gap" compared to naive post-quantization LoRA.
    
    Args:
        base_weight: Original weight matrix [out, in]
        rank: LoRA rank
        num_iters: Number of alternating optimization iterations
        compute_dtype: Dtype for computation
        
    Returns:
        quantized_residual: Quantized (W - BA)
        A: LoRA A matrix [rank, in]
        B: LoRA B matrix [out, rank]
    """
    weight = base_weight.to(compute_dtype)
    out_features, in_features = weight.shape
    
    # Initialize LoRA matrices to zero
    A = torch.zeros(rank, in_features, dtype=compute_dtype, device=weight.device)
    B = torch.zeros(out_features, rank, dtype=compute_dtype, device=weight.device)
    
    for iteration in range(num_iters):
        # Step 1: Compute residual R = W - BA
        residual = weight - B @ A
        
        # Step 2: Quantize residual
        scale = residual.abs().max() / 7.0
        quantized = torch.clamp(torch.round(residual / scale), -8, 7)
        dequantized = quantized * scale
        
        # Step 3: Update LoRA to minimize ||W - Q(R)||
        # Error = W - dequantized
        error = weight - dequantized
        
        # SVD of error to get optimal low-rank approximation
        U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
        
        # Update A and B
        sqrt_S = torch.sqrt(S[:rank])
        B = (U[:, :rank] * sqrt_S.unsqueeze(0)).to(compute_dtype)
        A = (sqrt_S.unsqueeze(1) * Vh[:rank, :]).to(compute_dtype)
    
    # Final residual (to be quantized)
    final_residual = weight - B @ A
    
    return final_residual.to(base_weight.dtype), A.to(base_weight.dtype), B.to(base_weight.dtype)


class QLoRALinear(nn.Module):
    """
    Combined 4-bit quantized base + LoRA adapter layer.
    
    The base weights are stored in 4-bit, while LoRA adapters
    are kept in higher precision (bfloat16/float16).
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        compute_dtype: torch.dtype = torch.bfloat16,
        use_loftq: bool = False,
        loftq_iters: int = 5,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.compute_dtype = compute_dtype
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        if use_loftq:
            # Use LoftQ initialization
            residual, A, B = loftq_init(
                base_layer.weight, 
                rank, 
                num_iters=loftq_iters,
                compute_dtype=torch.float32,
            )
            # Create quantized layer from residual
            temp_linear = nn.Linear(in_features, out_features, bias=base_layer.bias is not None)
            temp_linear.weight.data = residual
            if base_layer.bias is not None:
                temp_linear.bias.data = base_layer.bias
            self.quantized_base = Linear4bit(temp_linear, compute_dtype=compute_dtype)
            
            # Initialize LoRA with LoftQ values
            self.lora_A = nn.Parameter(A.to(compute_dtype))
            self.lora_B = nn.Parameter(B.to(compute_dtype))
        else:
            # Standard quantization + zero-init LoRA
            self.quantized_base = Linear4bit(base_layer, compute_dtype=compute_dtype)
            self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=compute_dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=compute_dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantized base forward
        base_out = self.quantized_base(x.to(self.compute_dtype))
        
        # LoRA forward
        lora_out = x.to(self.compute_dtype) @ self.lora_A.t() @ self.lora_B.t()
        
        return base_out + lora_out * self.scaling


def prepare_for_qlora(
    model: nn.Module,
    target_modules: Optional[Set[str]] = None,
    rank: int = 16,
    alpha: float = 32.0,
    compute_dtype: torch.dtype = torch.bfloat16,
    use_loftq: bool = False,
    use_dora: bool = False,
) -> Dict[str, nn.Module]:
    """
    Prepare a model for QLoRA training.
    
    This combines 4-bit quantization with LoRA adapters.
    
    Args:
        model: Model to prepare
        target_modules: Specific modules to adapt (None = all Linear)
        rank: LoRA rank
        alpha: LoRA scaling
        compute_dtype: Dtype for LoRA computations
        use_loftq: Use LoftQ initialization
        use_dora: Use DoRA instead of LoRA (not compatible with QLoRA directly)
        
    Returns:
        Dictionary of adapted modules
    """
    if use_dora:
        # For DoRA with quantization, we first quantize then add DoRA
        quantize_model_nf4(model, compute_dtype=compute_dtype, skip_modules=target_modules)
        # Then inject DoRA on non-quantized modules
        # (DoRA on quantized layers is more complex and not fully supported here)
        return inject_lora_adapters(
            model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            use_dora=True,
        )
    
    adapted_modules = {}
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None and name not in target_modules:
            continue
        
        # Create QLoRA layer
        qlora = QLoRALinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            compute_dtype=compute_dtype,
            use_loftq=use_loftq,
        )
        
        # Replace in model
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else dict(model.named_modules())[parent_name]
        setattr(parent, child_name, qlora)
        
        adapted_modules[name] = qlora
    
    return adapted_modules


def estimate_memory_savings(
    model: nn.Module,
    original_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, int]:
    """
    Estimate memory savings from 4-bit quantization.
    
    Returns:
        Dictionary with memory estimates in bytes
    """
    linear_params = 0
    other_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_params += module.weight.numel()
            if module.bias is not None:
                linear_params += module.bias.numel()
        elif isinstance(module, (nn.Embedding,)):
            other_params += sum(p.numel() for p in module.parameters())
    
    # Calculate bytes
    if original_dtype == torch.bfloat16 or original_dtype == torch.float16:
        bytes_per_param = 2
    elif original_dtype == torch.float32:
        bytes_per_param = 4
    else:
        bytes_per_param = 2  # Default assumption
    
    original_bytes = (linear_params + other_params) * bytes_per_param
    
    # 4-bit = 0.5 bytes per param (plus some overhead for scales)
    quantized_linear_bytes = int(linear_params * 0.5 * 1.1)  # 10% overhead for scales
    other_bytes = other_params * bytes_per_param
    
    quantized_bytes = quantized_linear_bytes + other_bytes
    
    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "savings_bytes": original_bytes - quantized_bytes,
        "savings_ratio": 1 - (quantized_bytes / original_bytes) if original_bytes > 0 else 0,
    }
