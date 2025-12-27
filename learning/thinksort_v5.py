"""
ThinkSort v5: Pointer Network

The CORRECT formulation for sorting:
- Don't predict VALUES (0-9) → that requires memorizing all numbers
- Predict POSITIONS (which input to select next) → directly learns sorting order

A Pointer Network outputs attention over input positions, not vocabulary.
This is the right inductive bias: "point to the minimum element, then the next..."

This is how all real sorting algorithms work:
- Selection Sort: Point to minimum, swap, repeat
- Our approach: Point to next-smallest element at each step

Architecture:
- Encoder: Process all input tokens
- Decoder: At each step, softmax over INPUT POSITIONS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class PointerConfig:
    """Configuration for Pointer Network ThinkSort."""
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2  # Small but deep enough
    dropout: float = 0.0
    
    vocab_size: int = 10
    max_seq_len: int = 64
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_parallel_envs: int = 256
    
    device: str = "cuda"


class TransformerLayer(nn.Module):
    """Standard Transformer layer for the encoder."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
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
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out(out)
        
        x = self.ln1(x + self.dropout(out))
        x = self.ln2(x + self.ffn(x))
        
        return x


class PointerThinkSort(nn.Module):
    """
    Pointer Network for Sorting.
    
    Instead of:
        output = softmax(W @ h) over vocab  # predicts value 0-9
    
    We do:
        output = softmax(h_query @ encoder_outputs.T)  # points to position 0-N
    
    This is the CORRECT formulation for sorting!
    """
    
    def __init__(self, config: PointerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding (values 0-9)
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Position embedding
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Encoder: process all input tokens
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.encoder_ln = nn.LayerNorm(config.d_model)
        
        # Decoder query: given step, produce query vector
        self.step_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Pointer attention: query attends to encoder outputs
        self.pointer_query = nn.Linear(config.d_model, config.d_model)
        self.pointer_key = nn.Linear(config.d_model, config.d_model)
        
        # Value head for MuZero
        self.value_head = nn.Linear(config.d_model, 1)
        
        # Mask embedding: marks which positions have been selected
        self.selected_embed = nn.Embedding(2, config.d_model)  # 0=available, 1=selected
        
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
        """Encode input sequence."""
        batch_size, seq_len = x.shape
        
        # Token + position embeddings
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = self.token_embed(x) + self.pos_embed(pos)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            h = layer(h)
        
        return self.encoder_ln(h)
    
    def pointer_step(self, encoder_output: torch.Tensor, step: int,
                    selected_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single pointer step: output distribution over input POSITIONS.
        
        Args:
            encoder_output: (batch, seq_len, d_model) - encoded input
            step: which output position (0, 1, 2, ...)
            selected_mask: (batch, seq_len) - 1 if position already selected
        
        Returns:
            pointer_logits: (batch, seq_len) - softmax over positions
            value: (batch, 1)
        """
        batch_size, seq_len, _ = encoder_output.shape
        
        # Query for this step
        step_tensor = torch.full((batch_size,), min(step, self.config.max_seq_len - 1),
                                device=encoder_output.device, dtype=torch.long)
        query = self.step_embed(step_tensor)  # (batch, d_model)
        
        # Add selected information to encoder output
        if selected_mask is not None:
            selected_emb = self.selected_embed(selected_mask.long())  # (batch, seq, d_model)
            encoder_with_mask = encoder_output + selected_emb
        else:
            encoder_with_mask = encoder_output
        
        # Pointer attention: query attends to all positions
        q = self.pointer_query(query)  # (batch, d_model)
        k = self.pointer_key(encoder_with_mask)  # (batch, seq_len, d_model)
        
        # Attention scores = Q @ K.T
        pointer_logits = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)  # (batch, seq_len)
        pointer_logits = pointer_logits / (self.config.d_model ** 0.5)
        
        # Mask already selected positions (set to -inf)
        if selected_mask is not None:
            pointer_logits = pointer_logits.masked_fill(selected_mask.bool(), float('-inf'))
        
        # Value prediction
        pooled = encoder_output.mean(dim=1)
        value = self.value_head(pooled)
        
        return pointer_logits, value
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Full forward pass: output pointer distributions for all steps.
        
        Returns:
            pointer_logits: list of (batch, seq_len) tensors
            encoder_output: (batch, seq_len, d_model)
        """
        encoder_output = self.encode(x)
        batch_size, seq_len = x.shape
        
        all_logits = []
        selected_mask = torch.zeros(batch_size, seq_len, device=x.device)
        
        for step in range(seq_len):
            logits, value = self.pointer_step(encoder_output, step, selected_mask)
            all_logits.append(logits)
            
            # Update mask (assuming greedy selection for now)
            with torch.no_grad():
                # Don't update mask during forward - let training handle it
                pass
        
        return all_logits, encoder_output


class PointerSortingEnv:
    """Environment that uses POSITIONS as actions (not values)."""
    
    def __init__(self, num_envs: int, seq_length: int, vocab_size: int = 10, device: str = "cuda"):
        self.num_envs = num_envs
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.device = device
    
    def reset(self) -> torch.Tensor:
        self.input_seqs = torch.randint(0, self.vocab_size, (self.num_envs, self.seq_length),
                                        device=self.device, dtype=torch.long)
        # Target: indices that would sort the input
        # argsort gives the indices that would sort the array
        self.target_positions = torch.argsort(self.input_seqs, dim=1)
        
        self.positions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.selected_mask = torch.zeros(self.num_envs, self.seq_length, device=self.device)
        
        return self.input_seqs
    
    def step(self, pointer_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pointer_actions: (batch,) indices into input sequence
        
        Returns:
            rewards, dones, correct
        """
        batch_idx = torch.arange(self.num_envs, device=self.device)
        correct_positions = self.target_positions[batch_idx, self.positions]
        
        correct = (pointer_actions == correct_positions)
        rewards = torch.where(correct, torch.ones_like(pointer_actions, dtype=torch.float32),
                             -torch.ones_like(pointer_actions, dtype=torch.float32))
        
        # Mark selected positions
        self.selected_mask[batch_idx, pointer_actions] = 1
        
        self.positions = self.positions + 1
        dones = (self.positions >= self.seq_length)
        
        return rewards, dones, correct.float()


class PointerThinkSortTrainer:
    """Trainer for Pointer Network ThinkSort."""
    
    def __init__(self, config: PointerConfig = None):
        if config is None:
            config = PointerConfig()
        
        self.config = config
        self.device = config.device
        
        print(f"🎯 Pointer ThinkSort: Position-based Sorting")
        print(f"   Output: Point to positions, not predict values!")
        print(f"   This is the CORRECT inductive bias for sorting")
        
        self.model = PointerThinkSort(config).to(config.device)
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
        self.accuracy_threshold = 0.95  # Higher threshold since this should work!
        
        self.envs = PointerSortingEnv(
            config.num_parallel_envs, self.current_seq_length,
            config.vocab_size, config.device
        )
        
        # Buffer
        self.buffer = {'obs': [], 'target_positions': []}
        self.buffer_size = 30000
        
        # Stats
        self.total_samples = 0
        self.recent_accuracies = deque(maxlen=100)
    
    def _recreate_env(self, new_len: int):
        self.current_seq_length = new_len
        self.envs = PointerSortingEnv(
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
        
        # Store for training
        self.buffer['obs'].append(obs.cpu())
        self.buffer['target_positions'].append(target_positions.cpu())
        
        max_entries = self.buffer_size // self.config.num_parallel_envs
        for k in self.buffer:
            if len(self.buffer[k]) > max_entries:
                self.buffer[k] = self.buffer[k][-max_entries:]
        
        # Evaluate current policy
        encoder_output = self.model.encode(obs)
        selected_mask = torch.zeros(self.config.num_parallel_envs, seq_len, device=self.device)
        
        total_correct = 0
        for step in range(seq_len):
            logits, _ = self.model.pointer_step(encoder_output, step, selected_mask)
            predictions = logits.argmax(dim=-1)
            
            correct_pos = target_positions[:, step]
            correct = (predictions == correct_pos).float()
            total_correct += correct.sum().item()
            
            # Update mask
            batch_idx = torch.arange(self.config.num_parallel_envs, device=self.device)
            selected_mask[batch_idx, predictions] = 1
        
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
        
        # Encode
        encoder_output = self.model.encode(obs)
        
        # Compute loss for each step
        total_loss = 0.0
        selected_mask = torch.zeros(batch_size, seq_len, device=self.device)
        
        for step in range(seq_len):
            logits, _ = self.model.pointer_step(encoder_output, step, selected_mask)
            target = target_positions[:, step]
            
            loss = F.cross_entropy(logits, target)
            total_loss = total_loss + loss
            
            # Update mask with CORRECT positions (teacher forcing)
            batch_idx = torch.arange(batch_size, device=self.device)
            selected_mask[batch_idx, target] = 1
        
        total_loss = total_loss / seq_len
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, max_samples: int = 500000, log_interval: int = 10000) -> Dict:
        print(f"\n{'='*70}")
        print(f"Starting Pointer ThinkSort Training")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Max samples: {max_samples:,}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            accuracy = self.collect_batch()
            
            if len(self.buffer['obs']) >= 2:
                for _ in range(4):
                    loss = self.train_step()
                    recent_losses.append(loss)
            
            # Curriculum
            avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
            if avg_acc >= self.accuracy_threshold and self.current_seq_length < self.max_seq_length:
                old_len = self.current_seq_length
                new_len = min(self.current_seq_length * 2, self.max_seq_length)
                
                print(f"\n🎯 Accuracy {avg_acc:.2%} >= {self.accuracy_threshold:.0%}")
                print(f"   ⬆️ CURRICULUM: {old_len} → {new_len} numbers")
                
                self._recreate_env(new_len)
                self.recent_accuracies.clear()
            
            # Logging
            if self.total_samples % log_interval < self.config.num_parallel_envs * self.current_seq_length:
                avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                speed = self.total_samples / elapsed
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"N={self.current_seq_length:2} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"{speed:,.0f}/s")
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(self.recent_accuracies)[-50:]) if self.recent_accuracies else 0
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Final seq_length: {self.current_seq_length}")
        print(f"Final accuracy: {final_acc:.2%}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*70}")
        
        return {
            'final_seq_length': self.current_seq_length,
            'final_accuracy': final_acc,
            'params': self.model.count_parameters(),
        }
    
    @torch.no_grad()
    def evaluate(self, seq_length: int, num_envs: int = 1000) -> float:
        self.model.eval()
        
        env = PointerSortingEnv(num_envs, seq_length, self.config.vocab_size, self.device)
        obs = env.reset()
        target_positions = env.target_positions
        
        encoder_output = self.model.encode(obs)
        selected_mask = torch.zeros(num_envs, seq_length, device=self.device)
        
        total_correct = 0
        for step in range(seq_length):
            logits, _ = self.model.pointer_step(encoder_output, step, selected_mask)
            predictions = logits.argmax(dim=-1).clamp(0, seq_length - 1)
            
            correct_pos = target_positions[:, step]
            correct = (predictions == correct_pos).float()
            total_correct += correct.sum().item()
            
            batch_idx = torch.arange(num_envs, device=self.device)
            selected_mask[batch_idx, predictions.clamp(0, seq_length - 1)] = 1
        
        return total_correct / (num_envs * seq_length)


def run_pointer_thinksort(max_samples: int = 500000):
    """Run Pointer ThinkSort experiment."""
    print("\n" + "="*70)
    print("  🎯 Pointer ThinkSort: Position-based Sorting")
    print("  Output = pointer to input position (not value prediction!)")
    print("="*70)
    
    config = PointerConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        vocab_size=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = PointerThinkSortTrainer(config)
    results = trainer.train(max_samples, log_interval=10000)
    
    print("\n" + "="*70)
    print("Evaluation: Pointer Network Generalization")
    print("="*70)
    
    for n in [4, 8, 16, 32]:
        acc = trainer.evaluate(n, num_envs=500)
        print(f"  Sort {n:2} numbers: {acc:.2%}")
    
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    run_pointer_thinksort()
