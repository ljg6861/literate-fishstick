"""
GaLore: Gradient Low-Rank Projection for memory-efficient training.

GaLore projects gradients into a low-rank subspace, dramatically reducing
optimizer state memory. This enables training larger models on consumer GPUs.

Reference: "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
"""

from typing import Optional, Dict, Any, Tuple, List
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer


class GaLoreProjector:
    """
    Low-rank gradient projector for a single parameter.
    
    Projects gradients into a rank-r subspace using SVD.
    The projection basis is updated periodically.
    """
    
    def __init__(
        self,
        rank: int,
        update_freq: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",  # "std" or "reverse_std"
    ):
        self.rank = rank
        self.update_freq = update_freq
        self.scale = scale
        self.proj_type = proj_type
        
        self.ortho_matrix: Optional[torch.Tensor] = None
        self.step = 0
    
    def project(self, grad: torch.Tensor, update_proj: bool = True) -> torch.Tensor:
        """
        Project gradient to low-rank subspace.
        
        For a gradient G of shape [m, n] with m >= n:
        - Compute projection matrix P = U[:, :r] from SVD(G)
        - Return projected gradient P^T @ G (shape [r, n])
        
        This reduces optimizer state from m*n to r*n.
        """
        original_shape = grad.shape
        
        # Reshape to 2D if needed
        if grad.dim() == 1:
            grad = grad.unsqueeze(0)
        
        # Ensure we project along the larger dimension for efficiency
        if grad.shape[0] < grad.shape[1]:
            grad = grad.T
            transposed = True
        else:
            transposed = False
        
        m, n = grad.shape
        
        # Update projection basis periodically
        if update_proj and (self.ortho_matrix is None or self.step % self.update_freq == 0):
            self.ortho_matrix = self._get_orthogonal_matrix(grad)
        
        self.step += 1
        
        # Project: P^T @ G -> [rank, n]
        if self.proj_type == "std":
            projected = self.ortho_matrix.T @ grad
        else:  # reverse_std: from right side
            projected = grad @ self.ortho_matrix
        
        if transposed:
            projected = projected.T
        
        return projected * self.scale
    
    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        """
        Project low-rank gradient back to full space.
        
        G_full = P @ G_low_rank
        """
        if self.ortho_matrix is None:
            raise RuntimeError("Must call project() before project_back()")
        
        if low_rank_grad.dim() == 1:
            low_rank_grad = low_rank_grad.unsqueeze(0)
        
        if self.proj_type == "std":
            full = self.ortho_matrix @ low_rank_grad
        else:
            full = low_rank_grad @ self.ortho_matrix.T
        
        return full / self.scale
    
    def _get_orthogonal_matrix(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute orthogonal basis from gradient via SVD."""
        # Use randomized SVD for efficiency on large matrices
        if min(grad.shape) > 2 * self.rank:
            # Power iteration for approximate SVD
            U = self._randomized_svd(grad.float(), self.rank)
        else:
            U, _, _ = torch.linalg.svd(grad.float(), full_matrices=False)
            U = U[:, :self.rank]
        
        return U.to(grad.dtype)
    
    def _randomized_svd(self, A: torch.Tensor, rank: int, n_iter: int = 2) -> torch.Tensor:
        """Randomized SVD for large matrices."""
        m, n = A.shape
        
        # Random projection
        Omega = torch.randn(n, rank + 10, device=A.device, dtype=A.dtype)
        
        # Power iteration for better approximation
        Y = A @ Omega
        for _ in range(n_iter):
            Y = A @ (A.T @ Y)
        
        # QR decomposition
        Q, _ = torch.linalg.qr(Y)
        
        return Q[:, :rank]


class GaLoreAdamW(Optimizer):
    """
    AdamW optimizer with GaLore gradient projection.
    
    For each parameter, maintains optimizer states in the projected
    low-rank subspace, reducing memory by factor of (dim / rank).
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        galore_rank: int = 128,
        galore_update_freq: int = 200,
        galore_scale: float = 1.0,
        galore_proj_type: str = "std",
        min_dim_for_galore: int = 256,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Adam beta parameters
            eps: Adam epsilon
            weight_decay: Weight decay coefficient
            galore_rank: Rank for gradient projection
            galore_update_freq: How often to update projection basis
            galore_scale: Scaling factor for projected gradients
            galore_proj_type: Projection type ("std" or "reverse_std")
            min_dim_for_galore: Minimum dimension to apply GaLore
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            galore_rank=galore_rank,
            galore_update_freq=galore_update_freq,
            galore_scale=galore_scale,
            galore_proj_type=galore_proj_type,
            min_dim_for_galore=min_dim_for_galore,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError("GaLore does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    
                    # Determine if we should use GaLore for this param
                    use_galore = (
                        p.dim() >= 2 and 
                        max(p.shape) >= group["min_dim_for_galore"]
                    )
                    state["use_galore"] = use_galore
                    
                    if use_galore:
                        # Create projector
                        state["projector"] = GaLoreProjector(
                            rank=min(group["galore_rank"], min(p.shape)),
                            update_freq=group["galore_update_freq"],
                            scale=group["galore_scale"],
                            proj_type=group["galore_proj_type"],
                        )
                        # Low-rank optimizer states
                        proj_shape = self._get_projected_shape(p.shape, state["projector"].rank)
                        state["exp_avg"] = torch.zeros(proj_shape, device=p.device, dtype=p.dtype)
                        state["exp_avg_sq"] = torch.zeros(proj_shape, device=p.device, dtype=p.dtype)
                    else:
                        # Standard Adam states
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                
                state["step"] += 1
                
                # Get hyperparameters
                beta1, beta2 = group["betas"]
                
                if state["use_galore"]:
                    # Project gradient to low-rank subspace
                    projector = state["projector"]
                    grad_projected = projector.project(grad, update_proj=True)
                    
                    # Adam update in low-rank space
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    
                    # Flatten for Adam if needed
                    grad_flat = grad_projected.reshape(exp_avg.shape)
                    
                    exp_avg.mul_(beta1).add_(grad_flat, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_flat, grad_flat, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    
                    step_size = group["lr"] / bias_correction1
                    
                    # Compute update in low-rank space
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                    update_low_rank = exp_avg / denom
                    
                    # Project back to full space
                    update_full = projector.project_back(update_low_rank.reshape(grad_projected.shape))
                    update_full = update_full.reshape(p.shape)
                    
                    # Apply update
                    p.add_(update_full, alpha=-step_size)
                else:
                    # Standard AdamW update
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    
                    step_size = group["lr"] / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                    
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Weight decay (decoupled)
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
        
        return loss
    
    def _get_projected_shape(self, original_shape: torch.Size, rank: int) -> torch.Size:
        """Get the shape of projected optimizer states."""
        if len(original_shape) == 1:
            return torch.Size([rank])
        elif len(original_shape) == 2:
            # Project along larger dimension
            if original_shape[0] >= original_shape[1]:
                return torch.Size([rank, original_shape[1]])
            else:
                return torch.Size([original_shape[0], rank])
        else:
            # Flatten to 2D, project, keep same approach
            total = 1
            for s in original_shape:
                total *= s
            return torch.Size([rank, total // max(original_shape)])


def estimate_galore_memory_savings(
    model: nn.Module,
    galore_rank: int = 128,
    min_dim: int = 256,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    Estimate memory savings from using GaLore.
    
    Returns:
        Dictionary with memory estimates
    """
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    
    standard_optimizer_bytes = 0
    galore_optimizer_bytes = 0
    
    galore_applied = 0
    galore_skipped = 0
    
    for name, param in model.named_parameters():
        numel = param.numel()
        
        # Standard Adam: 2 states per param (exp_avg, exp_avg_sq)
        standard_optimizer_bytes += 2 * numel * bytes_per_param
        
        # Check if GaLore applies
        if param.dim() >= 2 and max(param.shape) >= min_dim:
            # GaLore: states are rank * smaller_dim
            smaller_dim = min(param.shape)
            larger_dim = max(param.shape)
            rank = min(galore_rank, smaller_dim)
            
            galore_state_size = rank * smaller_dim
            galore_optimizer_bytes += 2 * galore_state_size * bytes_per_param
            galore_applied += 1
        else:
            # No GaLore: standard states
            galore_optimizer_bytes += 2 * numel * bytes_per_param
            galore_skipped += 1
    
    return {
        "standard_optimizer_bytes": standard_optimizer_bytes,
        "galore_optimizer_bytes": galore_optimizer_bytes,
        "savings_bytes": standard_optimizer_bytes - galore_optimizer_bytes,
        "savings_ratio": 1 - (galore_optimizer_bytes / standard_optimizer_bytes),
        "params_with_galore": galore_applied,
        "params_without_galore": galore_skipped,
    }
