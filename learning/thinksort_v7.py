"""
ThinkSort v7: The Neural Selection Sort Algorithm

Key Innovations:
1. Multi-Task Curriculum: Train on MIXED batches (4,8,16,32 simultaneously)
   - Prevents length-overfitting
   - Forces learning of universal "find minimum" logic

2. Recurrent Pointer with Hard Masking:
   - Each step: Find minimum among REMAINING elements
   - Hard mask: Selected positions = -inf (never seen again)
   - True Selection Sort loop

3. Logit Sharpening:
   - Temperature scaling based on sequence length
   - Sharper focus for longer sequences
   - Prevents attention blur

Goal: Train on N≤32, achieve 90%+ on N=64,128,256 (Zero-Shot!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
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
    """RoPE for length-invariant position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._update_cache(max_seq_len)
    
    def _update_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        self.max_seq_len_cached = seq_len
    
    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len)
        return self.cos_cached[:, :, :seq_len].to(device), self.sin_cached[:, :, :seq_len].to(device)


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=512)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        cos, sin = self.rotary_emb(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out(out)


class EncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
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
class AlgorithmicSortConfig:
    """Configuration for Neural Selection Sort."""
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    dropout: float = 0.0
    
    vocab_size: int = 10
    max_seq_len: int = 512
    
    # Multi-task curriculum
    train_lengths: Tuple[int, ...] = (4, 8, 16, 32)
    
    # Logit sharpening
    base_temperature: float = 1.0
    length_sharpening: bool = True  # Sharper for longer sequences
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_parallel_envs: int = 256
    
    device: str = "cuda"


class NeuralSelectionSort(nn.Module):
    """
    Neural Selection Sort: The Algorithmic Pointer Network
    
    True Selection Sort logic:
    1. Encode all elements
    2. Find minimum among REMAINING elements (hard mask selected)
    3. Output pointer to minimum
    4. Mask that position
    5. Repeat until all elements selected
    
    Key: No length-specific learning! Works on any N.
    """
    
    def __init__(self, config: AlgorithmicSortConfig):
        super().__init__()
        self.config = config
        
        # Token embedding (values only, no position - RoPE handles it)
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Encoder with RoPE
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.encoder_ln = nn.LayerNorm(config.d_model)
        
        # Pointer mechanism - context-aware query
        self.context_query = nn.Linear(config.d_model, config.d_model)
        self.pointer_key = nn.Linear(config.d_model, config.d_model)
        
        # Mask embedding for selected positions
        self.selected_embed = nn.Parameter(torch.randn(config.d_model) * 0.02)
        
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
        """Encode input tokens."""
        h = self.token_embed(x)
        for layer in self.encoder_layers:
            h = layer(h)
        return self.encoder_ln(h)
    
    def get_sharpness(self, seq_len: int) -> float:
        """Compute logit sharpening factor based on sequence length."""
        if not self.config.length_sharpening:
            return 1.0
        # Sharper attention for longer sequences
        # sqrt(seq_len/4) scaling: N=4 → 1.0, N=16 → 2.0, N=64 → 4.0
        return math.sqrt(seq_len / 4)
    
    def pointer_step(self, encoder_output: torch.Tensor,
                    selected_mask: torch.Tensor) -> torch.Tensor:
        """
        Single pointer step with hard masking.
        
        The "Selection Sort" logic:
        - Look only at REMAINING (unselected) positions
        - Point to the minimum among remaining
        
        Args:
            encoder_output: (batch, seq_len, d_model)
            selected_mask: (batch, seq_len) - 1 if already selected
        
        Returns:
            pointer_logits: (batch, seq_len)
        """
        batch_size, seq_len, d_model = encoder_output.shape
        
        # Add selected embedding to already-chosen positions
        encoder_modified = encoder_output.clone()
        selected_3d = selected_mask.unsqueeze(-1).expand_as(encoder_output)
        encoder_modified = encoder_modified + selected_3d * self.selected_embed
        
        # Context query: average of REMAINING elements
        remaining_mask = 1 - selected_mask.float()
        remaining_count = remaining_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Weighted average of remaining elements
        context = (encoder_modified * remaining_mask.unsqueeze(-1)).sum(dim=1) / remaining_count
        
        # Query from context
        query = self.context_query(context)  # (batch, d_model)
        
        # Keys from all positions (but masked ones will be ignored)
        keys = self.pointer_key(encoder_modified)  # (batch, seq_len, d_model)
        
        # Pointer attention
        logits = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)  # (batch, seq_len)
        
        # Apply length-based sharpening
        sharpness = self.get_sharpness(seq_len)
        logits = logits * sharpness / math.sqrt(d_model)
        
        # HARD MASK: Already selected positions get -inf
        logits = logits.masked_fill(selected_mask.bool(), float('-inf'))
        
        return logits


class MultiTaskPointerEnv:
    """
    Multi-Task Sorting Environment.
    
    Each batch contains MIXED sequence lengths (4, 8, 16, 32).
    This prevents length-overfitting!
    """
    
    def __init__(self, num_envs: int, lengths: Tuple[int, ...], 
                 vocab_size: int = 10, device: str = "cuda"):
        self.num_envs = num_envs
        self.lengths = lengths
        self.vocab_size = vocab_size
        self.device = device
        
        # Each env gets a random length
        self.env_lengths = None
        self.max_len = max(lengths)
    
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reset with mixed lengths.
        
        Returns:
            padded_input: (batch, max_len) - padded to max length
            target_positions: (batch, max_len) - sort order indices
            length_mask: (batch,) - actual length of each sequence
        """
        # Assign random length to each env
        self.env_lengths = torch.tensor(
            [self.lengths[np.random.randint(len(self.lengths))] for _ in range(self.num_envs)],
            device=self.device, dtype=torch.long
        )
        
        # Generate sequences with different lengths
        padded_input = torch.zeros(self.num_envs, self.max_len, device=self.device, dtype=torch.long)
        target_positions = torch.zeros(self.num_envs, self.max_len, device=self.device, dtype=torch.long)
        
        for i in range(self.num_envs):
            length = self.env_lengths[i].item()
            seq = torch.randint(0, self.vocab_size, (length,), device=self.device)
            padded_input[i, :length] = seq
            target_positions[i, :length] = torch.argsort(seq)
        
        return padded_input, target_positions, self.env_lengths


class NeuralSelectionSortTrainer:
    """
    Trainer for Neural Selection Sort.
    
    Key Features:
    - Multi-task curriculum (all lengths simultaneously)
    - Hard masking for true Selection Sort logic
    - Logit sharpening for OOD generalization
    """
    
    def __init__(self, config: AlgorithmicSortConfig = None):
        if config is None:
            config = AlgorithmicSortConfig()
        
        self.config = config
        self.device = config.device
        
        print(f"🧮 Neural Selection Sort Algorithm")
        print(f"   Multi-Task Curriculum: {config.train_lengths}")
        print(f"   Logit Sharpening: {config.length_sharpening}")
        print(f"   RoPE: ENABLED")
        
        self.model = NeuralSelectionSort(config).to(config.device)
        print(f"   Parameters: {self.model.count_parameters():,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.env = MultiTaskPointerEnv(
            config.num_parallel_envs, config.train_lengths,
            config.vocab_size, config.device
        )
        
        # Buffer
        self.buffer = {'padded': [], 'targets': [], 'lengths': []}
        self.buffer_size = 30000
        
        self.total_samples = 0
        self.accuracy_by_length = {n: deque(maxlen=100) for n in config.train_lengths}
    
    @torch.no_grad()
    def collect_batch(self) -> Dict[int, float]:
        """Collect multi-task batch and return per-length accuracy."""
        self.model.eval()
        
        padded, targets, lengths = self.env.reset()
        
        self.buffer['padded'].append(padded.cpu())
        self.buffer['targets'].append(targets.cpu())
        self.buffer['lengths'].append(lengths.cpu())
        
        max_entries = self.buffer_size // self.config.num_parallel_envs
        for k in self.buffer:
            if len(self.buffer[k]) > max_entries:
                self.buffer[k] = self.buffer[k][-max_entries:]
        
        # Evaluate per-length
        encoder_output = self.model.encode(padded)
        
        # For each unique length, compute accuracy
        accuracy_by_length = {}
        
        for length in self.config.train_lengths:
            mask = (lengths == length)
            if mask.sum() == 0:
                continue
            
            batch_enc = encoder_output[mask]
            batch_targets = targets[mask]
            batch_size = batch_enc.size(0)
            
            selected_mask = torch.zeros(batch_size, self.env.max_len, device=self.device)
            total_correct = 0
            
            for step in range(length):
                logits = self.model.pointer_step(batch_enc, selected_mask)
                # Only look at valid positions
                logits[:, length:] = float('-inf')
                
                preds = logits.argmax(dim=-1)
                correct = (preds == batch_targets[:, step]).float()
                total_correct += correct.sum().item()
                
                batch_idx = torch.arange(batch_size, device=self.device)
                selected_mask[batch_idx, preds.clamp(0, self.env.max_len-1)] = 1
            
            acc = total_correct / (batch_size * length)
            accuracy_by_length[length] = acc
            self.accuracy_by_length[length].append(acc)
            self.total_samples += batch_size * length
        
        return accuracy_by_length
    
    def train_step(self) -> float:
        if len(self.buffer['padded']) < 2:
            return 0.0
        
        self.model.train()
        
        idx = np.random.randint(0, len(self.buffer['padded']))
        padded = self.buffer['padded'][idx].to(self.device)
        targets = self.buffer['targets'][idx].to(self.device)
        lengths = self.buffer['lengths'][idx].to(self.device)
        
        batch_size = padded.size(0)
        max_len = self.env.max_len
        
        encoder_output = self.model.encode(padded)
        
        total_loss = 0.0
        total_steps = 0
        
        selected_mask = torch.zeros(batch_size, max_len, device=self.device)
        
        # Create length mask once (positions >= length are invalid)
        positions = torch.arange(max_len, device=self.device).unsqueeze(0)  # (1, max_len)
        length_mask = positions >= lengths.unsqueeze(1)  # (batch, max_len)
        
        # Process each step
        for step in range(max_len):
            # Which envs are still active at this step?
            active = (step < lengths).float()
            if active.sum() == 0:
                break
            
            logits = self.model.pointer_step(encoder_output, selected_mask)
            
            # Mask positions beyond actual length (using additive mask, not in-place!)
            logits = logits.masked_fill(length_mask, float('-inf'))
            
            target = targets[:, step]
            
            # Compute loss only for active envs
            loss = F.cross_entropy(logits, target, reduction='none')
            loss = (loss * active).sum() / active.sum().clamp(min=1)
            
            total_loss = total_loss + loss
            total_steps += 1
            
            # Update mask with correct positions (teacher forcing)
            batch_idx = torch.arange(batch_size, device=self.device)
            selected_mask = selected_mask.clone()
            selected_mask[batch_idx, target] = 1
        
        total_loss = total_loss / max(total_steps, 1)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, max_samples: int = 2000000, log_interval: int = 50000) -> Dict:
        print(f"\n{'='*75}")
        print(f"Starting Neural Selection Sort Training")
        print(f"Multi-Task Lengths: {self.config.train_lengths}")
        print(f"{'='*75}\n")
        
        start_time = time.time()
        recent_losses = deque(maxlen=100)
        
        last_log = 0
        while self.total_samples < max_samples:
            acc_by_len = self.collect_batch()
            
            if len(self.buffer['padded']) >= 2:
                for _ in range(4):
                    loss = self.train_step()
                    recent_losses.append(loss)
            
            if self.total_samples - last_log >= log_interval:
                last_log = self.total_samples
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                speed = self.total_samples / elapsed
                
                acc_str = " | ".join([
                    f"N{n}:{np.mean(self.accuracy_by_length[n]) if self.accuracy_by_length[n] else 0:.0%}"
                    for n in self.config.train_lengths
                ])
                
                print(f"Samples: {self.total_samples:8,} | {acc_str} | Loss: {avg_loss:.4f} | {speed:,.0f}/s")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*75}")
        print(f"Training Complete! Time: {elapsed:.1f}s")
        print(f"{'='*75}")
        
        return {'total_samples': self.total_samples}
    
    @torch.no_grad()
    def evaluate(self, seq_length: int, num_envs: int = 1000) -> float:
        """Evaluate on a specific length (can be OOD!)."""
        self.model.eval()
        
        # Generate test data
        input_seqs = torch.randint(0, self.config.vocab_size, (num_envs, seq_length),
                                   device=self.device, dtype=torch.long)
        target_positions = torch.argsort(input_seqs, dim=1)
        
        encoder_output = self.model.encode(input_seqs)
        selected_mask = torch.zeros(num_envs, seq_length, device=self.device)
        
        total_correct = 0
        for step in range(seq_length):
            logits = self.model.pointer_step(encoder_output, selected_mask)
            preds = logits.argmax(dim=-1)
            
            correct = (preds == target_positions[:, step]).float()
            total_correct += correct.sum().item()
            
            batch_idx = torch.arange(num_envs, device=self.device)
            selected_mask[batch_idx, preds.clamp(0, seq_length-1)] = 1
        
        return total_correct / (num_envs * seq_length)


def run_neural_selection_sort(max_samples: int = 3000000):
    """Run Neural Selection Sort experiment."""
    print("\n" + "="*75)
    print("  🧮 Neural Selection Sort: The Algorithmic Pointer")
    print("  Multi-Task Training | Hard Masking | Logit Sharpening")
    print("="*75)
    
    config = AlgorithmicSortConfig(
        d_model=96,
        n_heads=6,
        d_ff=384,
        n_layers=3,
        vocab_size=10,
        train_lengths=(4, 8, 16, 32),
        length_sharpening=True,
        max_seq_len=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = NeuralSelectionSortTrainer(config)
    trainer.train(max_samples, log_interval=100000)
    
    print("\n" + "="*75)
    print("EVALUATION: In-Distribution vs Zero-Shot OOD")
    print("="*75)
    
    print("\n📊 IN-DISTRIBUTION (Trained Lengths):")
    for n in [4, 8, 16, 32]:
        acc = trainer.evaluate(n, num_envs=1000)
        status = '✅' if acc > 0.95 else ('🔶' if acc > 0.8 else '❌')
        print(f"  Sort {n:3} numbers: {acc:.2%} {status}")
    
    print("\n🌟 ZERO-SHOT OOD (Never Seen in Training!):")
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
    
    return trainer


if __name__ == "__main__":
    run_neural_selection_sort()
