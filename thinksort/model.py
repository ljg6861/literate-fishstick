"""
Universal Transformer Pointer Network for Neural Sorting

Architecture:
- RoPE (Rotary Position Embeddings) for length-invariant positions
- Universal Transformer: Single weight-shared block run N times
- Pointer Network: Outputs attention over remaining positions
- Hard Masking: Already-selected positions masked with -∞
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """
    Rotary Position Embeddings - scale invariant.
    
    Key: Position is encoded in the rotation of query/key vectors,
    making it purely relative (works for ANY sequence length).
    """
    
    def __init__(self, dim: int, max_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len: int):
        """Pre-compute cos/sin for positions up to seq_len."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos()[None, None], persistent=False)
        self.register_buffer('sin', emb.sin()[None, None], persistent=False)
        self.max_len = seq_len
    
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        Apply rotary embeddings to query and key.
        
        Args:
            q: (B, H, L, D) query
            k: (B, H, L, D) key
        """
        L = q.size(2)
        if L > self.max_len:
            self._build_cache(L)
        cos, sin = self.cos[:, :, :L], self.sin[:, :, :L]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU activation: x * SiLU(gate). Requires 2x input dim."""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class UniversalBlock(nn.Module):
    """
    Weight-shared Transformer block - the "Universal Comparator".
    
    This single block is run multiple times (recurrently),
    forcing the model to learn a general comparison operation.
    """
    
    def __init__(self, dim: int, heads: int, ff_dim: int, activation: str = 'gelu'):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        
        # Self-attention
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.rope = RoPE(self.head_dim)
        
        # Feed-forward with configurable activation
        if activation == 'swiglu':
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),  # ff_dim should be 2x for SwiGLU
                SwiGLU(),
                nn.Linear(ff_dim // 2, dim)
            )
        elif activation == 'relu':
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, dim)
            )
        else:  # gelu (default)
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, dim)
            )
        
        # Pre-LayerNorm
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input
            mask: (B, L) - 1 for positions to mask out (already selected)
        """
        B, L, D = x.shape
        
        # Pre-LN Self-Attention with RoPE
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        
        # Attention scores using Flash Attention
        # F.scaled_dot_product_attention handles scaling (1/sqrt(d)) and masking efficiently
        # We need to construct the attention mask properly for SDPA
        
        if mask is not None:
             # Mask is 1 for selected positions (to hide). 
             # Use float mask: 0 for keep, -inf for mask
             # shape: (B, 1, 1, L) broadcastable to (B, H, L_q, L_k)
             mask_bool = mask.unsqueeze(1).unsqueeze(2).bool()
             attn_mask = torch.zeros(
                 (B, 1, 1, L), 
                 device=q.device, 
                 dtype=q.dtype
             )
             attn_mask = attn_mask.masked_fill(mask_bool, float('-inf'))
        else:
             attn_mask = None

        # F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        # Note: input shapes are (B, H, L, D_head) which matches q, k, v
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape: (B, H, L, D_head) -> (B, L, H, D_head) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, D)
        
        x = x + self.out(out)
        
        # Pre-LN FFN
        x = x + self.ff(self.ln2(x))
        
        return x


@dataclass
class Config:
    """Model configuration."""
    dim: int = 128
    heads: int = 8
    ff: int = 512
    recurrent_steps: int = 4  # Universal Transformer: same layer N times
    # vocab: int = 10  # REMOVED
    train_lengths: tuple = (4, 8, 16)
    samples_per_length: int = 128
    lr: float = 1e-4
    device: str = "cuda"


class UniversalPointerNet(nn.Module):
    """
    Universal Transformer Pointer Network
    
    Key Design:
    - ONE shared block run multiple times (recurrent/universal)
    - Output: ONE pointer per forward pass (finds next minimum)
    - RoPE: Length-invariant positional encoding
    - Hard Mask: Physically mask selected positions
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Scalar embedding: Project 1D scalar -> dim
        self.embed = nn.Linear(1, config.dim)
        
        # SINGLE Universal Block (weight-shared)
        self.universal_block = UniversalBlock(config.dim, config.heads, config.ff)
        
        # Pointer head
        self.query_proj = nn.Linear(config.dim, config.dim)
        self.key_proj = nn.Linear(config.dim, config.dim)
        
        # Value head: predicts future sorting accuracy
        self.value_head = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, 1),
            nn.Sigmoid()  # Output in [0, 1] = predicted accuracy
        )
        
        # "Already selected" embedding
        self.selected_emb = nn.Parameter(torch.zeros(config.dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def encode(self, x: torch.Tensor, selected_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode with Universal Transformer (weight-shared recurrent).
        
        Args:
            x: (B, L) input scalars
            selected_mask: (B, L) - 1 for already selected positions
        """
        # (B, L) -> (B, L, 1) -> (B, L, D)
        h = self.embed(x.unsqueeze(-1))
        
        # Add selected embedding to already-chosen positions
        if selected_mask is not None:
            h = h + selected_mask.unsqueeze(-1) * self.selected_emb
        
        # Universal Transformer: same block multiple times!
        for _ in range(self.config.recurrent_steps):
            h = self.universal_block(h, selected_mask)
        
        return h
    
    def pointer(self, encoded: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        """
        Output ONE pointer to the next smallest element.
        
        Args:
            encoded: (B, L, D)
            selected_mask: (B, L) - 1 for already selected
        
        Returns:
            logits: (B, L) - pointer attention over remaining positions
        """
        B, L, D = encoded.shape
        
        # Context: mean of remaining (unselected) elements
        remaining = 1 - selected_mask
        remaining_count = remaining.sum(dim=1, keepdim=True).clamp(min=1)
        context = (encoded * remaining.unsqueeze(-1)).sum(dim=1) / remaining_count
        
        # Pointer attention
        q = self.query_proj(context)  # (B, D)
        k = self.key_proj(encoded)     # (B, L, D)
        
        logits = (q.unsqueeze(1) @ k.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        
        # HARD MASK: selected positions = -inf
        logits = logits.masked_fill(selected_mask.bool(), float('-inf'))
        
        return logits
    
    def value(self, encoded: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        """
        Predict future sorting accuracy from current state.
        
        Args:
            encoded: (B, L, D) encoded sequence
            selected_mask: (B, L) - 1 for already selected positions
        
        Returns:
            value: (B,) predicted accuracy in [0, 1]
        """
        # Context: mean of remaining (unselected) positions
        remaining = 1 - selected_mask
        remaining_count = remaining.sum(dim=1, keepdim=True).clamp(min=1)
        context = (encoded * remaining.unsqueeze(-1)).sum(dim=1) / remaining_count
        
        return self.value_head(context).squeeze(-1)

    def forward(self, x: torch.Tensor, target_full: torch.Tensor = None):
        """
        Forward pass.
        
        If target_full is provided, runs the full training loop and returns (loss, correct_count).
        If target_full is None, runs inference (greedy decoding) and returns preds.
        """
        if target_full is None:
            # Inference Mode (Greedy)
            B, L = x.shape
            selected_mask = torch.zeros(B, L, device=x.device)
            preds = []
            for _ in range(L):
                encoded = self.encode(x, selected_mask)
                logits = self.pointer(encoded, selected_mask)
                p = logits.argmax(dim=-1)
                preds.append(p)
                selected_mask[torch.arange(B, device=x.device), p.clamp(0, L-1)] = 1
            return torch.stack(preds, dim=1)
            
        # Training Mode
        B, L = x.shape
        selected_mask = torch.zeros(B, L, device=x.device)
        
        pointer_loss = 0.0
        value_loss = 0.0
        correct = 0
        
        all_preds = []
        
        # Pass 1: Pointer & Multi-target Loss
        temp_mask = selected_mask.clone()
        
        for t in range(L):
            encoded = self.encode(x, temp_mask)
            logits = self.pointer(encoded, temp_mask)
            
            # --- Multi-Target Logic ---
            x_masked = x.float().clone()
            x_masked.masked_fill_(temp_mask.bool(), float('inf'))
            min_vals, _ = x_masked.min(dim=1, keepdim=True)
            valid_moves = (x_masked == min_vals) & (temp_mask == 0)
            
            target_probs = valid_moves.float()
            target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Mask out 0 targets to avoid 0*-inf=NaN
            term = target_probs * log_probs
            term = term.masked_fill(target_probs == 0, 0.0)
            step_loss = -term.sum(dim=1).mean()
            
            pointer_loss = pointer_loss + step_loss
            
            preds = logits.argmax(dim=-1)
            is_valid = valid_moves.gather(1, preds.unsqueeze(1)).squeeze(1)
            correct += is_valid.sum().item()
            
            all_preds.append(preds)
            
            # Teacher Forcing Update
            forced_next = target_full[:, t]
            temp_mask = temp_mask.clone()
            temp_mask[torch.arange(B, device=x.device), forced_next] = 1
            
        # Pass 2: Value Head
        # (We could optimize this by combining loops, but keep separate for clarity/correctness)
        selected_mask = torch.zeros(B, L, device=x.device)
        all_preds_stacked = torch.stack(all_preds, dim=1)
        
        for t in range(L - 1): 
            encoded = self.encode(x, selected_mask)
            value_pred = self.value(encoded, selected_mask)
            
            future_correct = (all_preds_stacked[:, t+1:] == target_full[:, t+1:]).float().mean(dim=1)
            value_loss = value_loss + F.mse_loss(value_pred, future_correct)
            
            selected_mask = selected_mask.clone()
            selected_mask[torch.arange(B, device=x.device), target_full[:, t]] = 1
            
        total_loss = pointer_loss / L + 0.1 * value_loss / max(1, L - 1)
        
        # Return loss as tensor (1D if unreduced, but here we reduced to scalar already)
        # DataParallel will stack these scalars
        # We need to return tensors for DataParallel to gather them
        return total_loss, torch.tensor(correct, device=x.device)

