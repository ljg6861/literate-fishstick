"""
ThinkSort v4: Cross-Attention Anchored Universal Transformer

The "Source Anchor" architecture that breaks the representation drift problem:

1. Self-Attention: The hidden state attends to itself (internal reasoning)
2. Cross-Attention: The hidden state attends to ORIGINAL INPUT (source anchor)
3. Every thinking step can "look at the whiteboard"

This mimics a human programmer:
- Internal state = "what I've figured out so far"
- Cross-attention = "let me look at the problem again"

Key Insight:
- v3 problem: 20 steps of self-attention = memory of a memory of a memory
- v4 solution: Every step can directly read the original numbers

Goal: Confidence should now scale proportionally with thinking steps!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class ThinkSortV4Config:
    """Configuration for ThinkSort v4."""
    # Model - Efficient but with cross-attention
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    
    # Task
    vocab_size: int = 10
    max_seq_len: int = 64
    
    # Thinking
    min_think_steps: int = 2
    max_think_steps: int = 16
    confidence_threshold: float = 0.85
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_parallel_envs: int = 256
    think_penalty: float = 0.005
    
    # MuZero
    unroll_steps: int = 4
    discount: float = 1.0
    
    device: str = "cuda"


class SourceAnchoredBlock(nn.Module):
    """
    Transformer block with Cross-Attention to Source (original input).
    
    Architecture per thinking step:
    1. Self-Attention(hidden, hidden, hidden) - reason about current state
    2. Cross-Attention(hidden, source, source) - look at original problem
    3. FFN - combine information
    
    This ensures the model can ALWAYS see the original unsorted list.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_steps: int = 20, max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        
        # Step embedding
        self.step_embed = nn.Embedding(max_steps, d_model)
        
        # Dimensions
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Self-Attention (hidden attends to hidden)
        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)
        
        # Cross-Attention (hidden attends to SOURCE!)
        self.cross_q = nn.Linear(d_model, d_model)   # Query from hidden
        self.cross_kv = nn.Linear(d_model, 2 * d_model)  # Key, Value from source
        self.cross_out = nn.Linear(d_model, d_model)
        
        # Relative position bias for self-attention
        self.max_len = max_len
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_len - 1))
        nn.init.normal_(self.rel_pos_bias, std=0.02)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms for each sub-layer
        self.ln_self = nn.LayerNorm(d_model)
        self.ln_cross = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_rel_pos_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        rel_dist = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_dist = rel_dist + (self.max_len - 1)
        rel_dist = rel_dist.clamp(0, 2 * self.max_len - 2)
        return self.rel_pos_bias[rel_dist]
    
    def forward(self, hidden: torch.Tensor, source: torch.Tensor, 
                think_step: int) -> torch.Tensor:
        """
        Forward with cross-attention to source.
        
        Args:
            hidden: current reasoning state (batch, seq_len, d_model)
            source: ORIGINAL input embeddings (batch, seq_len, d_model)
            think_step: which thinking iteration
        """
        batch_size, seq_len, _ = hidden.shape
        
        # Add step embedding to hidden
        clamped_step = min(think_step, self.step_embed.num_embeddings - 1)
        step_emb = self.step_embed(
            torch.full((batch_size,), clamped_step, device=hidden.device, dtype=torch.long)
        ).unsqueeze(1)
        h = hidden + step_emb
        
        # ========== 1. SELF-ATTENTION ==========
        # Hidden state reasons about itself
        qkv = self.self_qkv(h).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        rel_bias = self._get_rel_pos_bias(seq_len, h.device)
        attn_scores = attn_scores + rel_bias.unsqueeze(0).unsqueeze(0)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        self_out = torch.matmul(attn_probs, v)
        self_out = self_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        self_out = self.self_out(self_out)
        
        h = self.ln_self(h + self.dropout(self_out))
        
        # ========== 2. CROSS-ATTENTION TO SOURCE (THE KEY INNOVATION!) ==========
        # Hidden state looks at the ORIGINAL input
        cross_q = self.cross_q(h).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        cross_q = cross_q.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        
        cross_kv = self.cross_kv(source).reshape(batch_size, seq_len, 2, self.n_heads, self.head_dim)
        cross_kv = cross_kv.permute(2, 0, 3, 1, 4)
        cross_k, cross_v = cross_kv[0], cross_kv[1]
        
        cross_scores = torch.matmul(cross_q, cross_k.transpose(-2, -1)) * self.scale
        cross_probs = F.softmax(cross_scores, dim=-1)
        cross_probs = self.dropout(cross_probs)
        
        cross_out = torch.matmul(cross_probs, cross_v)
        cross_out = cross_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        cross_out = self.cross_out(cross_out)
        
        h = self.ln_cross(h + self.dropout(cross_out))
        
        # ========== 3. FFN ==========
        h = self.ln_ffn(h + self.ffn(h))
        
        return h


class ThinkSortV4(nn.Module):
    """
    ThinkSort v4: Cross-Attention Anchored Universal Transformer
    
    Every thinking step has:
    - Self-attention (internal reasoning)
    - Cross-attention to SOURCE (original input)
    
    This prevents representation drift - the model can always
    "look at the whiteboard" where the original numbers are written.
    """
    
    def __init__(self, config: ThinkSortV4Config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Output position embedding (which sorted position we're predicting)
        self.out_pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # THE CORE: Single shared block with SOURCE ANCHOR
        self.think_block = SourceAnchoredBlock(
            config.d_model, config.n_heads, config.d_ff,
            config.max_think_steps, config.max_seq_len, config.dropout
        )
        
        # Output heads
        self.ln_out = nn.LayerNorm(config.d_model)
        self.policy_head = nn.Linear(config.d_model, config.vocab_size)
        self.value_head = nn.Linear(config.d_model, 1)
        
        # Confidence head - should now learn properly!
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
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
    
    def forward(self, x: torch.Tensor, out_pos: int = 0,
                adaptive_exit: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
        """
        Forward with cross-attention anchored thinking.
        
        Returns: (policy, value, hidden, think_steps, avg_confidence)
        """
        batch_size, seq_len = x.shape
        
        # Embed input - this is our SOURCE ANCHOR
        source = self.token_embed(x)
        
        # Initialize hidden state with source + output position
        out_pos_tensor = torch.full((batch_size,), min(out_pos, self.config.max_seq_len - 1),
                                    device=x.device, dtype=torch.long)
        h = source + self.out_pos_embed(out_pos_tensor).unsqueeze(1)
        
        # Thinking loop with cross-attention to source
        think_steps = 0
        confidences = []
        
        for step in range(self.config.max_think_steps):
            # Apply thinking block with CROSS-ATTENTION TO SOURCE
            h = self.think_block(h, source, step)
            think_steps += 1
            
            # Check confidence
            pooled = h.mean(dim=1)
            confidence = self.confidence_head(pooled).squeeze(-1)
            avg_conf = confidence.mean().item()
            confidences.append(avg_conf)
            
            # Early exit if confident enough
            if adaptive_exit and step >= self.config.min_think_steps - 1:
                if avg_conf >= self.config.confidence_threshold:
                    break
        
        h = self.ln_out(h)
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return policy, value, h, think_steps, avg_confidence
    
    def forward_fixed_steps(self, x: torch.Tensor, out_pos: int = 0,
                           num_steps: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with fixed thinking steps."""
        if num_steps is None:
            num_steps = self.config.min_think_steps
        
        batch_size, seq_len = x.shape
        
        source = self.token_embed(x)
        out_pos_tensor = torch.full((batch_size,), min(out_pos, self.config.max_seq_len - 1),
                                    device=x.device, dtype=torch.long)
        h = source + self.out_pos_embed(out_pos_tensor).unsqueeze(1)
        
        for step in range(num_steps):
            h = self.think_block(h, source, step)
        
        h = self.ln_out(h)
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy, value, h


class VectorizedSortingEnv:
    """Sorting environment."""
    
    def __init__(self, num_envs: int, seq_length: int, vocab_size: int = 10, device: str = "cuda"):
        self.num_envs = num_envs
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.device = device
    
    def reset(self) -> torch.Tensor:
        self.input_seqs = torch.randint(0, self.vocab_size, (self.num_envs, self.seq_length),
                                        device=self.device, dtype=torch.long)
        self.target_seqs, _ = torch.sort(self.input_seqs, dim=1)
        self.positions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        return self.input_seqs
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_idx = torch.arange(self.num_envs, device=self.device)
        correct_actions = self.target_seqs[batch_idx, self.positions]
        
        correct = (actions == correct_actions)
        rewards = torch.where(correct, torch.ones_like(actions, dtype=torch.float32),
                             -torch.ones_like(actions, dtype=torch.float32))
        
        self.positions = self.positions + 1
        dones = (self.positions >= self.seq_length)
        
        return rewards, dones, correct.float()


@dataclass
class ThinkingStats:
    """Track thinking behavior."""
    samples: List[int] = field(default_factory=list)
    seq_lengths: List[int] = field(default_factory=list)
    think_steps: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    
    def add(self, samples: int, seq_len: int, think: float, conf: float, acc: float):
        self.samples.append(samples)
        self.seq_lengths.append(seq_len)
        self.think_steps.append(think)
        self.confidences.append(conf)
        self.accuracies.append(acc)
    
    def print_summary(self):
        print("\n" + "="*75)
        print("ThinkSort v4 - Cross-Attention Anchored Thinking")
        print("="*75)
        print(f"{'Samples':>10} | {'SeqLen':>6} | {'ThinkSteps':>10} | {'Confidence':>10} | {'Accuracy':>8}")
        print("-"*75)
        for i in range(0, len(self.samples), max(1, len(self.samples) // 10)):
            print(f"{self.samples[i]:>10,} | {self.seq_lengths[i]:>6} | "
                  f"{self.think_steps[i]:>10.2f} | {self.confidences[i]:>10.3f} | "
                  f"{self.accuracies[i]:>7.2%}")
        print("="*75)


class ThinkSortV4Trainer:
    """Trainer for ThinkSort v4."""
    
    def __init__(self, config: ThinkSortV4Config = None):
        if config is None:
            config = ThinkSortV4Config()
        
        self.config = config
        self.device = config.device
        
        print(f"🧠 ThinkSort v4: Cross-Attention Anchored")
        print(f"   Source Anchor: ENABLED (cross-attention to original input)")
        print(f"   Adaptive Exit: threshold={config.confidence_threshold}")
        print(f"   Think steps: {config.min_think_steps} to {config.max_think_steps}")
        
        self.model = ThinkSortV4(config).to(config.device)
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
        self.accuracy_threshold = 0.90
        
        self.envs = VectorizedSortingEnv(
            config.num_parallel_envs, self.current_seq_length,
            config.vocab_size, config.device
        )
        
        # Buffer
        self.buffer = {'obs': [], 'actions': [], 'rewards': [], 'policies': []}
        self.buffer_size = 30000
        
        # Stats
        self.total_samples = 0
        self.recent_accuracies = deque(maxlen=100)
        self.recent_think_steps = deque(maxlen=100)
        self.recent_confidence = deque(maxlen=100)
        self.stats = ThinkingStats()
    
    def _recreate_env(self, new_len: int):
        self.current_seq_length = new_len
        self.envs = VectorizedSortingEnv(
            self.config.num_parallel_envs, new_len,
            self.config.vocab_size, self.device
        )
        self.buffer = {'obs': [], 'actions': [], 'rewards': [], 'policies': []}
    
    @torch.no_grad()
    def collect_batch(self) -> Tuple[float, float, float]:
        self.model.eval()
        
        obs = self.envs.reset()
        targets = self.envs.target_seqs
        seq_len = self.current_seq_length
        
        all_actions, all_rewards, all_policies = [], [], []
        total_correct = 0
        total_think = 0
        total_conf = 0
        
        for step in range(seq_len):
            target_action = targets[:, step]
            
            expert_policy = torch.zeros(self.config.num_parallel_envs, self.config.vocab_size, device=self.device)
            expert_policy.scatter_(1, target_action.unsqueeze(1), 1.0)
            
            policy_logits, value, hidden, think_steps, confidence = self.model(
                obs, out_pos=step, adaptive_exit=True
            )
            
            total_think += think_steps
            total_conf += confidence
            
            rand = np.random.random()
            if rand < 0.4:
                actions = target_action
            elif rand < 0.5:
                actions = torch.randint(0, self.config.vocab_size,
                                       (self.config.num_parallel_envs,), device=self.device)
            else:
                actions = policy_logits.argmax(dim=-1)
            
            all_actions.append(actions)
            all_policies.append(expert_policy)
            
            rewards, dones, correct = self.envs.step(actions)
            all_rewards.append(rewards)
            total_correct += correct.sum().item()
        
        self.buffer['obs'].append(obs.cpu())
        self.buffer['actions'].append(torch.stack(all_actions, dim=1).cpu())
        self.buffer['rewards'].append(torch.stack(all_rewards, dim=1).cpu())
        self.buffer['policies'].append(torch.stack(all_policies, dim=1).cpu())
        
        max_entries = self.buffer_size // self.config.num_parallel_envs
        for k in self.buffer:
            if len(self.buffer[k]) > max_entries:
                self.buffer[k] = self.buffer[k][-max_entries:]
        
        self.total_samples += self.config.num_parallel_envs * seq_len
        accuracy = total_correct / (self.config.num_parallel_envs * seq_len)
        avg_think = total_think / seq_len
        avg_conf = total_conf / seq_len
        
        self.recent_accuracies.append(accuracy)
        self.recent_think_steps.append(avg_think)
        self.recent_confidence.append(avg_conf)
        
        return accuracy, avg_think, avg_conf
    
    def train_step(self) -> float:
        if len(self.buffer['obs']) < 2:
            return 0.0
        
        self.model.train()
        
        idx = np.random.randint(0, len(self.buffer['obs']))
        obs = self.buffer['obs'][idx].to(self.device)
        actions = self.buffer['actions'][idx].to(self.device)
        rewards = self.buffer['rewards'][idx].to(self.device)
        policies = self.buffer['policies'][idx].to(self.device)
        
        seq_len = actions.size(1)
        batch_size = obs.size(0)
        
        think_steps = max(2, seq_len // 2)
        
        # Value targets
        value_targets = torch.zeros(batch_size, seq_len, device=self.device)
        running = torch.zeros(batch_size, device=self.device)
        for t in reversed(range(seq_len)):
            running = rewards[:, t] + self.config.discount * running
            value_targets[:, t] = running
        
        # Forward
        policy_logits, value_pred, hidden = self.model.forward_fixed_steps(obs, out_pos=0, num_steps=think_steps)
        
        target_idx = policies[:, 0].argmax(dim=-1)
        policy_loss = F.cross_entropy(policy_logits, target_idx)
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, 0])
        
        total_loss = policy_loss + value_loss
        
        # Unroll with cross-attention
        source = self.model.token_embed(obs)
        for k in range(min(self.config.unroll_steps, seq_len - 1)):
            action_emb = self.model.token_embed(actions[:, k]).unsqueeze(1)
            h = hidden + action_emb.expand(-1, hidden.size(1), -1)
            
            for step in range(2):
                h = self.model.think_block(h, source, step)
            
            h = self.model.ln_out(h)
            pooled = h.mean(dim=1)
            
            policy_logits = self.model.policy_head(pooled)
            value_pred = self.model.value_head(pooled)
            
            target_idx = policies[:, k+1].argmax(dim=-1)
            policy_loss = F.cross_entropy(policy_logits, target_idx)
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, k+1])
            
            total_loss = total_loss + policy_loss + value_loss
            hidden = h
        
        total_loss = total_loss + think_steps * self.config.think_penalty
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, max_samples: int = 500000, log_interval: int = 10000) -> Dict:
        print(f"\n{'='*70}")
        print(f"Starting ThinkSort v4 Training")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Max samples: {max_samples:,}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            accuracy, think_steps, confidence = self.collect_batch()
            
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
                avg_think = np.mean(self.recent_think_steps) if self.recent_think_steps else 0
                avg_conf = np.mean(self.recent_confidence) if self.recent_confidence else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                speed = self.total_samples / elapsed
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"N={self.current_seq_length:2} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Think: {avg_think:.1f} | "
                      f"Conf: {avg_conf:.3f} | "
                      f"Loss: {avg_loss:.3f} | "
                      f"{speed:,.0f}/s")
                
                self.stats.add(self.total_samples, self.current_seq_length,
                              avg_think, avg_conf, avg_acc)
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(self.recent_accuracies)[-50:]) if self.recent_accuracies else 0
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Final seq_length: {self.current_seq_length}")
        print(f"Final accuracy: {final_acc:.2%}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*70}")
        
        self.stats.print_summary()
        
        return {
            'final_seq_length': self.current_seq_length,
            'final_accuracy': final_acc,
            'params': self.model.count_parameters(),
        }
    
    @torch.no_grad()
    def evaluate(self, seq_length: int, num_envs: int = 1000) -> Tuple[float, float, float]:
        self.model.eval()
        
        env = VectorizedSortingEnv(num_envs, seq_length, self.config.vocab_size, self.device)
        obs = env.reset()
        
        total_correct = 0
        total_think = 0
        total_conf = 0
        
        for step in range(seq_length):
            policy_logits, _, _, think_steps, confidence = self.model(
                obs, out_pos=step, adaptive_exit=True
            )
            actions = policy_logits.argmax(dim=-1).clamp(0, self.config.vocab_size - 1)
            _, _, correct = env.step(actions)
            total_correct += correct.sum().item()
            total_think += think_steps
            total_conf += confidence
        
        return (
            total_correct / (num_envs * seq_length),
            total_think / seq_length,
            total_conf / seq_length
        )


def run_thinksort_v4(max_samples: int = 500000):
    """Run ThinkSort v4 experiment."""
    print("\n" + "="*75)
    print("  🧠 ThinkSort v4: Cross-Attention Anchored Universal Transformer")
    print("  Every thinking step can 'look at the whiteboard'")
    print("="*75)
    
    config = ThinkSortV4Config(
        d_model=64,
        n_heads=4,
        d_ff=256,
        vocab_size=10,
        min_think_steps=2,
        max_think_steps=16,
        confidence_threshold=0.85,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = ThinkSortV4Trainer(config)
    results = trainer.train(max_samples, log_interval=10000)
    
    print("\n" + "="*75)
    print("Evaluation: Cross-Attention in Action")
    print("="*75)
    
    for n in [4, 8, 16, 32]:
        acc, think, conf = trainer.evaluate(n, num_envs=500)
        print(f"  Sort {n:2} numbers: {acc:.2%} ({think:.1f} steps, {conf:.3f} confidence)")
    
    print("="*75 + "\n")
    
    return results


if __name__ == "__main__":
    run_thinksort_v4()
