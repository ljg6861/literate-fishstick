"""
ThinkSort v6: RoPE Pointer Network for Length Generalization

Key Innovation: Replace absolute position embeddings with Rotary Position Embeddings (RoPE)
- RoPE encodes relative positions, enabling generalization to unseen lengths
- Train on N=4,8,16,32 → Test on N=64,128,256

This is the "Neural Algorithm" formulation:
1. Correct inductive bias (Pointer Network)
2. Length-invariant encoding (RoPE)
3. Should generalize like a true sorting algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for length generalization."""
    
    def __init__(self, dim: int, max_seq_len: int = 256, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute for max_seq_len
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))  # (1, 1, seq, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=256)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out(out)


class RoPEEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE."""
    
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
class RoPEPointerConfig:
    """Configuration for RoPE Pointer Network."""
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    dropout: float = 0.0
    
    vocab_size: int = 10
    max_seq_len: int = 256  # Can generalize beyond this!
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_parallel_envs: int = 256
    
    device: str = "cuda"


class RoPEPointerNetwork(nn.Module):
    """
    Pointer Network with RoPE for length generalization.
    
    Key: No absolute position embeddings!
    - Token embedding only (no position)
    - RoPE in attention handles relative positions
    - Should generalize to any sequence length
    """
    
    def __init__(self, config: RoPEPointerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding ONLY (no position embedding!)
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Encoder with RoPE
        self.encoder_layers = nn.ModuleList([
            RoPEEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.encoder_ln = nn.LayerNorm(config.d_model)
        
        # Pointer mechanism (position-invariant)
        # Use relative step encoding instead of absolute
        self.step_token = nn.Parameter(torch.randn(config.d_model) * 0.02)
        
        # Pointer attention
        self.pointer_query = nn.Linear(config.d_model, config.d_model)
        self.pointer_key = nn.Linear(config.d_model, config.d_model)
        
        # Selected/available embedding
        self.selected_embed = nn.Embedding(2, config.d_model)
        
        # Output heads
        self.value_head = nn.Linear(config.d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input (no position embeddings - RoPE handles it)."""
        h = self.token_embed(x)
        
        for layer in self.encoder_layers:
            h = layer(h)
        
        return self.encoder_ln(h)
    
    def pointer_step(self, encoder_output: torch.Tensor, 
                    selected_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Single pointer step - position-invariant!
        
        Returns: pointer_logits (batch, seq_len)
        """
        batch_size, seq_len, _ = encoder_output.shape
        
        # Add selected information
        if selected_mask is not None:
            selected_emb = self.selected_embed(selected_mask.long())
            encoder_with_mask = encoder_output + selected_emb
        else:
            encoder_with_mask = encoder_output
        
        # Query: global step token (same for all steps - no step encoding!)
        query = self.step_token.unsqueeze(0).expand(batch_size, -1)
        
        # Also incorporate context from remaining elements
        available_mask = (1 - selected_mask.float()) if selected_mask is not None else torch.ones(batch_size, seq_len, device=encoder_output.device)
        available_count = available_mask.sum(dim=1, keepdim=True).clamp(min=1)
        context = (encoder_with_mask * available_mask.unsqueeze(-1)).sum(dim=1) / available_count
        
        query = query + context  # Combine step token with context
        
        # Pointer attention
        q = self.pointer_query(query)  # (batch, d_model)
        k = self.pointer_key(encoder_with_mask)  # (batch, seq_len, d_model)
        
        pointer_logits = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        pointer_logits = pointer_logits / (self.config.d_model ** 0.5)
        
        # Mask already selected
        if selected_mask is not None:
            pointer_logits = pointer_logits.masked_fill(selected_mask.bool(), float('-inf'))
        
        return pointer_logits


class RoPEPointerEnv:
    """Environment for pointer-based sorting."""
    
    def __init__(self, num_envs: int, seq_length: int, vocab_size: int = 10, device: str = "cuda"):
        self.num_envs = num_envs
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.device = device
    
    def reset(self) -> torch.Tensor:
        self.input_seqs = torch.randint(0, self.vocab_size, (self.num_envs, self.seq_length),
                                        device=self.device, dtype=torch.long)
        self.target_positions = torch.argsort(self.input_seqs, dim=1)
        self.positions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.selected_mask = torch.zeros(self.num_envs, self.seq_length, device=self.device)
        return self.input_seqs
    
    def step(self, pointer_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_idx = torch.arange(self.num_envs, device=self.device)
        correct_positions = self.target_positions[batch_idx, self.positions]
        
        correct = (pointer_actions == correct_positions)
        rewards = torch.where(correct, torch.ones_like(pointer_actions, dtype=torch.float32),
                             -torch.ones_like(pointer_actions, dtype=torch.float32))
        
        self.selected_mask[batch_idx, pointer_actions.clamp(0, self.seq_length-1)] = 1
        self.positions = self.positions + 1
        dones = (self.positions >= self.seq_length)
        
        return rewards, dones, correct.float()


class RoPEPointerTrainer:
    """Trainer for RoPE Pointer Network."""
    
    def __init__(self, config: RoPEPointerConfig = None):
        if config is None:
            config = RoPEPointerConfig()
        
        self.config = config
        self.device = config.device
        
        print(f"🌟 RoPE Pointer Network: Length Generalization")
        print(f"   Rotary Position Embeddings: ENABLED")
        print(f"   Absolute Position Embeddings: REMOVED")
        print(f"   Goal: Train on N≤32, generalize to N=128+")
        
        self.model = RoPEPointerNetwork(config).to(config.device)
        print(f"   Parameters: {self.model.count_parameters():,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Curriculum
        self.min_seq_length = 4
        self.max_seq_length = 32
        self.current_seq_length = self.min_seq_length
        self.accuracy_threshold = 0.95
        
        self.envs = RoPEPointerEnv(
            config.num_parallel_envs, self.current_seq_length,
            config.vocab_size, config.device
        )
        
        # Buffer
        self.buffer = {'obs': [], 'target_positions': []}
        self.buffer_size = 30000
        
        self.total_samples = 0
        self.recent_accuracies = deque(maxlen=100)
    
    def _recreate_env(self, new_len: int):
        self.current_seq_length = new_len
        self.envs = RoPEPointerEnv(
            self.config.num_parallel_envs, new_len,
            self.config.vocab_size, self.device
        )
        self.buffer = {'obs': [], 'target_positions': []}
    
    @torch.no_grad()
    def collect_batch(self) -> float:
        self.model.eval()
        
        obs = self.envs.reset()
        target_positions = self.envs.target_positions
        seq_len = self.current_seq_length
        
        self.buffer['obs'].append(obs.cpu())
        self.buffer['target_positions'].append(target_positions.cpu())
        
        max_entries = self.buffer_size // self.config.num_parallel_envs
        for k in self.buffer:
            if len(self.buffer[k]) > max_entries:
                self.buffer[k] = self.buffer[k][-max_entries:]
        
        encoder_output = self.model.encode(obs)
        selected_mask = torch.zeros(self.config.num_parallel_envs, seq_len, device=self.device)
        
        total_correct = 0
        for step in range(seq_len):
            logits = self.model.pointer_step(encoder_output, selected_mask)
            predictions = logits.argmax(dim=-1)
            
            correct_pos = target_positions[:, step]
            correct = (predictions == correct_pos).float()
            total_correct += correct.sum().item()
            
            batch_idx = torch.arange(self.config.num_parallel_envs, device=self.device)
            selected_mask[batch_idx, predictions.clamp(0, seq_len-1)] = 1
        
        self.total_samples += self.config.num_parallel_envs * seq_len
        accuracy = total_correct / (self.config.num_parallel_envs * seq_len)
        self.recent_accuracies.append(accuracy)
        
        return accuracy
    
    def train_step(self) -> float:
        if len(self.buffer['obs']) < 2:
            return 0.0
        
        self.model.train()
        
        idx = np.random.randint(0, len(self.buffer['obs']))
        obs = self.buffer['obs'][idx].to(self.device)
        target_positions = self.buffer['target_positions'][idx].to(self.device)
        
        batch_size, seq_len = obs.shape
        
        encoder_output = self.model.encode(obs)
        
        total_loss = 0.0
        selected_mask = torch.zeros(batch_size, seq_len, device=self.device)
        
        for step in range(seq_len):
            logits = self.model.pointer_step(encoder_output, selected_mask)
            target = target_positions[:, step]
            
            loss = F.cross_entropy(logits, target)
            total_loss = total_loss + loss
            
            batch_idx = torch.arange(batch_size, device=self.device)
            selected_mask[batch_idx, target] = 1
        
        total_loss = total_loss / seq_len
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, max_samples: int = 1000000, log_interval: int = 50000) -> Dict:
        print(f"\n{'='*70}")
        print(f"Starting RoPE Pointer Training")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            accuracy = self.collect_batch()
            
            if len(self.buffer['obs']) >= 2:
                for _ in range(4):
                    loss = self.train_step()
                    recent_losses.append(loss)
            
            avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
            if avg_acc >= self.accuracy_threshold and self.current_seq_length < self.max_seq_length:
                old_len = self.current_seq_length
                new_len = min(self.current_seq_length * 2, self.max_seq_length)
                
                print(f"\n🎯 Accuracy {avg_acc:.2%} >= {self.accuracy_threshold:.0%}")
                print(f"   ⬆️ CURRICULUM: {old_len} → {new_len} numbers")
                
                self._recreate_env(new_len)
                self.recent_accuracies.clear()
            
            if self.total_samples % log_interval < self.config.num_parallel_envs * self.current_seq_length:
                avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                speed = self.total_samples / elapsed
                
                print(f"Samples: {self.total_samples:8,} | N={self.current_seq_length:2} | "
                      f"Acc: {avg_acc:.2%} | Loss: {avg_loss:.4f} | {speed:,.0f}/s")
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(self.recent_accuracies)[-50:]) if self.recent_accuracies else 0
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Final seq_length: {self.current_seq_length}")
        print(f"Final accuracy: {final_acc:.2%}")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*70}")
        
        return {'final_seq_length': self.current_seq_length, 'final_accuracy': final_acc}
    
    @torch.no_grad()
    def evaluate(self, seq_length: int, num_envs: int = 1000) -> float:
        self.model.eval()
        
        env = RoPEPointerEnv(num_envs, seq_length, self.config.vocab_size, self.device)
        obs = env.reset()
        target_positions = env.target_positions
        
        encoder_output = self.model.encode(obs)
        selected_mask = torch.zeros(num_envs, seq_length, device=self.device)
        
        total_correct = 0
        for step in range(seq_length):
            logits = self.model.pointer_step(encoder_output, selected_mask)
            predictions = logits.argmax(dim=-1).clamp(0, seq_length - 1)
            
            correct_pos = target_positions[:, step]
            correct = (predictions == correct_pos).float()
            total_correct += correct.sum().item()
            
            batch_idx = torch.arange(num_envs, device=self.device)
            selected_mask[batch_idx, predictions] = 1
        
        return total_correct / (num_envs * seq_length)


def run_rope_pointer(max_samples: int = 1500000):
    """Run RoPE Pointer experiment with OOD generalization test."""
    print("\n" + "="*75)
    print("  🌟 RoPE Pointer Network: The Neural Sorting Algorithm")
    print("  Train: N ≤ 32  |  Test: N = 64, 128, 256 (Zero-Shot!)")
    print("="*75)
    
    config = RoPEPointerConfig(
        d_model=96,
        n_heads=6,
        d_ff=384,
        n_layers=3,
        vocab_size=10,
        max_seq_len=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = RoPEPointerTrainer(config)
    results = trainer.train(max_samples, log_interval=50000)
    
    print("\n" + "="*75)
    print("EVALUATION: Trained Lengths vs Zero-Shot OOD")
    print("="*75)
    
    print("\n📊 TRAINED LENGTHS:")
    for n in [4, 8, 16, 32]:
        acc = trainer.evaluate(n, num_envs=1000)
        status = '✅' if acc > 0.95 else ('🔶' if acc > 0.8 else '❌')
        print(f"  Sort {n:3} numbers: {acc:.2%} {status}")
    
    print("\n🌟 ZERO-SHOT GENERALIZATION (Never Seen in Training!):")
    for n in [48, 64, 96, 128, 192, 256]:
        acc = trainer.evaluate(n, num_envs=500)
        if acc > 0.9:
            status = '🌟 NEURAL ALGORITHM!'
        elif acc > 0.7:
            status = '✅ Generalizing'
        elif acc > 0.5:
            status = '🔶 Partial'
        else:
            status = '❌'
        print(f"  Sort {n:3} numbers: {acc:.2%} {status}")
    
    print("="*75 + "\n")
    
    return results


if __name__ == "__main__":
    run_rope_pointer()
