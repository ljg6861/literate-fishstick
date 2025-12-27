"""
ThinkSort v8: The 90%+ Neural Sorting Algorithm

Fixed issues from v7:
1. NaN loss - proper handling of -inf in cross_entropy
2. Simpler architecture - back to basics that work
3. Progressive curriculum - start easy, scale up
4. More training capacity

Key changes:
- Use label smoothing to handle edge cases
- Larger model (but still efficient)
- Single-length training per batch (no padding complexity)
- Longer training with save/resume
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import time


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        self.max_seq_len_cached = seq_len
    
    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:, :, :seq_len].to(device), self.sin_cached[:, :, :seq_len].to(device)


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        cos, sin = self.rotary(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.d_model)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.dropout(self.attn(x)))
        x = self.ln2(x + self.ffn(x))
        return x


@dataclass
class SortNetConfig:
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.0
    vocab_size: int = 10
    max_seq_len: int = 512
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 512
    label_smoothing: float = 0.1  # Helps with numerical stability
    
    # Curriculum
    train_lengths: Tuple[int, ...] = (4, 8, 16, 32)
    accuracy_threshold: float = 0.92
    
    device: str = "cuda"


class SortNet(nn.Module):
    """
    SortNet: Clean Pointer Network for Sorting
    
    Architecture:
    - RoPE-based Transformer encoder
    - Pointer mechanism with hard masking
    - No length-specific components
    """
    
    def __init__(self, config: SortNetConfig):
        super().__init__()
        self.config = config
        
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        self.encoder = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.ln = nn.LayerNorm(config.d_model)
        
        # Pointer mechanism
        self.query_proj = nn.Linear(config.d_model, config.d_model)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        
        # Selected position embedding
        self.selected_bias = nn.Parameter(torch.zeros(config.d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        for block in self.encoder:
            h = block(h)
        return self.ln(h)
    
    def pointer_logits(self, encoded: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute pointer logits over remaining positions.
        
        Args:
            encoded: (B, L, D) - encoded sequence
            selected_mask: (B, L) - 1 where already selected
        """
        B, L, D = encoded.shape
        
        # Add bias to selected positions
        modified = encoded + selected_mask.unsqueeze(-1) * self.selected_bias
        
        # Context = mean of remaining positions
        remaining = 1 - selected_mask
        remaining_count = remaining.sum(dim=1, keepdim=True).clamp(min=1)
        context = (modified * remaining.unsqueeze(-1)).sum(dim=1) / remaining_count
        
        # Pointer attention
        query = self.query_proj(context)  # (B, D)
        keys = self.key_proj(modified)     # (B, L, D)
        
        logits = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)  # (B, L)
        logits = logits / math.sqrt(D)
        
        # Hard mask selected positions
        logits = logits.masked_fill(selected_mask.bool(), -1e9)  # Use -1e9 instead of -inf
        
        return logits


class SortNetTrainer:
    def __init__(self, config: SortNetConfig = None):
        if config is None:
            config = SortNetConfig()
        
        self.config = config
        self.device = config.device
        
        print(f"🎯 SortNet v8: Push to 90%+")
        print(f"   Model: {config.n_layers}L x {config.d_model}D x {config.n_heads}H")
        print(f"   Label smoothing: {config.label_smoothing}")
        
        self.model = SortNet(config).to(config.device)
        print(f"   Parameters: {self.model.count_parameters():,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50000, T_mult=2
        )
        
        # Curriculum
        self.current_length_idx = 0
        self.current_length = config.train_lengths[0]
        
        self.total_samples = 0
        self.recent_acc = deque(maxlen=200)
        self.best_accs = {n: 0.0 for n in config.train_lengths}
    
    def generate_batch(self, seq_len: int):
        """Generate a batch of sorting problems."""
        B = self.config.batch_size
        inputs = torch.randint(0, self.config.vocab_size, (B, seq_len), 
                              device=self.device, dtype=torch.long)
        targets = torch.argsort(inputs, dim=1)
        return inputs, targets
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """Single training step. Returns (loss, accuracy)."""
        self.model.train()
        
        B, L = inputs.shape
        encoded = self.model.encode(inputs)
        
        total_loss = 0.0
        total_correct = 0
        selected_mask = torch.zeros(B, L, device=self.device)
        
        for step in range(L):
            logits = self.model.pointer_logits(encoded, selected_mask)
            target = targets[:, step]
            
            # Cross entropy with label smoothing
            loss = F.cross_entropy(logits, target, label_smoothing=self.config.label_smoothing)
            total_loss = total_loss + loss
            
            # Track accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == target).sum().item()
            
            # Teacher forcing: mark correct position as selected
            batch_idx = torch.arange(B, device=self.device)
            selected_mask = selected_mask.clone()
            selected_mask[batch_idx, target] = 1
        
        avg_loss = total_loss / L
        accuracy = total_correct / (B * L)
        
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return avg_loss.item(), accuracy
    
    @torch.no_grad()
    def evaluate(self, seq_len: int, num_samples: int = 2000) -> float:
        """Evaluate accuracy on given sequence length."""
        self.model.eval()
        
        # Process in batches
        total_correct = 0
        total_elements = 0
        batch_size = min(500, num_samples)
        
        for _ in range(num_samples // batch_size):
            inputs = torch.randint(0, self.config.vocab_size, (batch_size, seq_len),
                                  device=self.device, dtype=torch.long)
            targets = torch.argsort(inputs, dim=1)
            
            encoded = self.model.encode(inputs)
            selected_mask = torch.zeros(batch_size, seq_len, device=self.device)
            
            for step in range(seq_len):
                logits = self.model.pointer_logits(encoded, selected_mask)
                preds = logits.argmax(dim=-1)
                
                total_correct += (preds == targets[:, step]).sum().item()
                
                batch_idx = torch.arange(batch_size, device=self.device)
                selected_mask[batch_idx, preds.clamp(0, seq_len-1)] = 1
            
            total_elements += batch_size * seq_len
        
        return total_correct / total_elements
    
    def train(self, max_samples: int = 10000000, log_interval: int = 50000):
        print(f"\n{'='*75}")
        print(f"Training SortNet v8")
        print(f"Target: 90%+ on all trained lengths")
        print(f"{'='*75}\n")
        
        start_time = time.time()
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            # Generate batch at current curriculum length
            inputs, targets = self.generate_batch(self.current_length)
            loss, acc = self.train_step(inputs, targets)
            
            recent_losses.append(loss)
            self.recent_acc.append(acc)
            self.total_samples += self.config.batch_size * self.current_length
            
            # Check curriculum advancement
            avg_acc = np.mean(self.recent_acc)
            if avg_acc >= self.config.accuracy_threshold:
                # Update best
                self.best_accs[self.current_length] = max(self.best_accs[self.current_length], avg_acc)
                
                # Advance curriculum
                if self.current_length_idx < len(self.config.train_lengths) - 1:
                    self.current_length_idx += 1
                    old_len = self.current_length
                    self.current_length = self.config.train_lengths[self.current_length_idx]
                    self.recent_acc.clear()
                    
                    print(f"\n🎯 Accuracy {avg_acc:.1%} >= {self.config.accuracy_threshold:.0%}")
                    print(f"   ⬆️ CURRICULUM: {old_len} → {self.current_length}\n")
            
            # Logging
            if self.total_samples % log_interval < self.config.batch_size * self.current_length:
                avg_loss = np.mean(recent_losses)
                elapsed = time.time() - start_time
                speed = self.total_samples / elapsed
                
                # Evaluate all lengths
                eval_str = " | ".join([
                    f"N{n}:{self.evaluate(n, 1000):.0%}" 
                    for n in self.config.train_lengths
                ])
                
                print(f"Samples: {self.total_samples:8,} | Curr: N={self.current_length} "
                      f"({avg_acc:.0%}) | {eval_str} | {speed:,.0f}/s")
                
                # Check if we hit 90% on all
                all_90 = all(self.evaluate(n, 1000) >= 0.90 for n in self.config.train_lengths)
                if all_90:
                    print(f"\n🌟 ACHIEVED 90%+ ON ALL TRAINED LENGTHS!")
                    break
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*75}")
        print(f"Training Complete! Time: {elapsed:.1f}s")
        print(f"{'='*75}")
        
        return self.best_accs
    
    def full_evaluation(self):
        print("\n" + "="*75)
        print("FINAL EVALUATION")
        print("="*75)
        
        print("\n📊 TRAINED LENGTHS:")
        for n in self.config.train_lengths:
            acc = self.evaluate(n, 2000)
            status = '✅' if acc >= 0.90 else ('🔶' if acc >= 0.80 else '❌')
            print(f"  Sort {n:3} numbers: {acc:.2%} {status}")
        
        print("\n🌟 ZERO-SHOT OOD:")
        for n in [48, 64, 96, 128, 192, 256]:
            acc = self.evaluate(n, 1000)
            if acc >= 0.90:
                status = '🌟 NEURAL ALGORITHM!'
            elif acc >= 0.70:
                status = '✅ Generalizing'
            elif acc >= 0.50:
                status = '🔶 Partial'
            else:
                status = '❌'
            print(f"  Sort {n:3} numbers: {acc:.2%} {status}")


def run_sortnet_v8(max_samples: int = 10000000):
    """Run SortNet v8."""
    print("\n" + "="*75)
    print("  🎯 SortNet v8: The 90%+ Neural Sorting Algorithm")
    print("="*75)
    
    config = SortNetConfig(
        d_model=128,
        n_heads=8,
        d_ff=512,
        n_layers=4,
        vocab_size=10,
        batch_size=512,
        label_smoothing=0.1,
        learning_rate=3e-4,
        accuracy_threshold=0.92,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = SortNetTrainer(config)
    trainer.train(max_samples, log_interval=100000)
    trainer.full_evaluation()
    
    return trainer


if __name__ == "__main__":
    run_sortnet_v8()
