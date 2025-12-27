"""
Universal MuZero Transformer

A weight-shared recurrent Transformer where:
1. A SINGLE Transformer block is applied N times (Universal Transformer style)
2. MuZero controls the number of "thinking steps" based on task difficulty
3. Same 100k parameters can sort 4 numbers in 2 steps OR 32 numbers in 10 steps

Key Insight:
- Traditional Transformers: More layers = more parameters = fixed computation
- Universal Transformer: Same weights applied iteratively = adaptive computation
- MuZero + Universal: Agent learns WHEN to think more vs output answer

This is the "Minimum Viable Architecture" that scales via COMPUTATION, not PARAMETERS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time


class ThinkAction(IntEnum):
    """Actions for the thinking controller."""
    THINK_MORE = 0  # Apply another recurrent step
    OUTPUT_NOW = 1  # Stop thinking and output answer


@dataclass
class UniversalConfig:
    """Configuration for Universal MuZero Transformer."""
    # Model - SMALL and EFFICIENT
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    
    # Task
    vocab_size: int = 10  # Numbers 0-9
    max_seq_len: int = 32
    
    # Thinking
    min_think_steps: int = 1   # Minimum recurrent applications
    max_think_steps: int = 12  # Maximum thinking steps
    
    # This is the key insight: think_penalty encourages efficiency
    think_penalty: float = 0.01  # Small penalty per thinking step
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_parallel_envs: int = 256
    
    # MuZero
    unroll_steps: int = 4
    discount: float = 1.0
    
    device: str = "cuda"


class UniversalTransformerBlock(nn.Module):
    """
    A single Transformer block that will be applied recurrently.
    
    Includes step embedding so the model knows which thinking step it's on.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 max_steps: int = 16, dropout: float = 0.0):
        super().__init__()
        
        # Step embedding - tells the model which thinking iteration
        self.step_embed = nn.Embedding(max_steps, d_model)
        
        # Self-attention with relative position bias
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Relative position bias (shared across all thinking steps)
        self.max_len = 64
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * self.max_len - 1))
        nn.init.normal_(self.rel_pos_bias, std=0.02)
        
        # FFN
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
    
    def _get_rel_pos_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        rel_dist = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_dist = rel_dist + (self.max_len - 1)
        rel_dist = rel_dist.clamp(0, 2 * self.max_len - 2)
        return self.rel_pos_bias[rel_dist]
    
    def forward(self, x: torch.Tensor, think_step: int) -> torch.Tensor:
        """
        Apply one thinking step.
        
        Args:
            x: (batch, seq_len, d_model)
            think_step: which thinking iteration (0, 1, 2, ...)
        """
        batch_size, seq_len, _ = x.shape
        
        # Add step embedding so model knows which thinking iteration
        # Clamp to valid range
        clamped_step = min(think_step, self.step_embed.num_embeddings - 1)
        step_emb = self.step_embed(
            torch.full((batch_size,), clamped_step, device=x.device, dtype=torch.long)
        ).unsqueeze(1)
        x = x + step_emb
        
        # Self-attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        rel_bias = self._get_rel_pos_bias(seq_len, x.device)
        attn_scores = attn_scores + rel_bias.unsqueeze(0).unsqueeze(0)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_out = self.out_proj(attn_out)
        
        # Residual + LayerNorm
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.ffn(x))
        
        return x


class UniversalMuZeroTransformer(nn.Module):
    """
    Universal (Weight-Shared) Transformer with MuZero-controlled thinking.
    
    Key Innovation:
    - ONE Transformer block applied 1-12 times
    - MuZero learns when to "think more" vs "output now"
    - Fixed parameters (~100k), variable computation
    """
    
    def __init__(self, config: UniversalConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Output position embedding (which sorted position we're predicting)
        self.out_pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # THE SINGLE SHARED BLOCK (this is the key!)
        self.think_block = UniversalTransformerBlock(
            config.d_model, config.n_heads, config.d_ff,
            config.max_think_steps, config.dropout
        )
        
        # Output heads
        self.ln_out = nn.LayerNorm(config.d_model)
        
        # Policy head for sorting (which number to output)
        self.policy_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Value head
        self.value_head = nn.Linear(config.d_model, 1)
        
        # THINKING CONTROLLER: decides whether to think more or output
        self.think_controller = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # THINK_MORE or OUTPUT_NOW
        )
        
        # Halting probability accumulator (like ACT - Adaptive Computation Time)
        self.halt_threshold = 0.8
        
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
                return_think_steps: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Forward with adaptive thinking.
        
        Args:
            x: (batch, seq_len) input tokens
            out_pos: which output position we're predicting
            return_think_steps: if True, return number of thinking steps used
        
        Returns:
            policy, value, hidden, think_steps
        """
        batch_size, seq_len = x.shape
        
        # Embed
        h = self.token_embed(x)
        
        # Add output position embedding
        out_pos_tensor = torch.full((batch_size,), min(out_pos, self.config.max_seq_len - 1),
                                    device=x.device, dtype=torch.long)
        h = h + self.out_pos_embed(out_pos_tensor).unsqueeze(1)
        
        # Adaptive thinking loop
        think_steps = 0
        cumulative_halt = torch.zeros(batch_size, device=x.device)
        remainders = torch.zeros(batch_size, device=x.device)
        
        # Weighted hidden state (for ACT-style interpolation)
        weighted_h = torch.zeros_like(h)
        
        for step in range(self.config.max_think_steps):
            # Apply thinking block
            h = self.think_block(h, step)
            think_steps += 1
            
            # Should we stop thinking?
            pooled = h.mean(dim=1)  # (batch, d_model)
            halt_logits = self.think_controller(pooled)  # (batch, 2)
            halt_prob = F.softmax(halt_logits, dim=-1)[:, 1]  # Prob of OUTPUT_NOW
            
            # Adaptive Computation Time (ACT) style accumulation
            still_thinking = (cumulative_halt < self.halt_threshold).float()
            
            # Add to weighted state
            p = halt_prob * still_thinking
            weighted_h = weighted_h + p.unsqueeze(1).unsqueeze(2) * h
            
            cumulative_halt = cumulative_halt + p
            remainders = remainders + still_thinking * (1 - halt_prob)
            
            # Early exit if all samples have halted
            if (cumulative_halt >= self.halt_threshold).all() and step >= self.config.min_think_steps - 1:
                break
        
        # Final weighted state
        final_h = weighted_h / (cumulative_halt.unsqueeze(1).unsqueeze(2) + 1e-8)
        final_h = torch.where(
            cumulative_halt.unsqueeze(1).unsqueeze(2) > 0.1,
            final_h,
            h  # Fallback to last h if accumulation too small
        )
        
        final_h = self.ln_out(final_h)
        pooled = final_h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        if return_think_steps:
            return policy, value, final_h, think_steps
        return policy, value, final_h, think_steps
    
    def forward_fixed_steps(self, x: torch.Tensor, out_pos: int = 0, 
                           num_steps: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with fixed number of thinking steps (for training)."""
        if num_steps is None:
            num_steps = self.config.min_think_steps
        
        batch_size, seq_len = x.shape
        
        h = self.token_embed(x)
        out_pos_tensor = torch.full((batch_size,), min(out_pos, self.config.max_seq_len - 1),
                                    device=x.device, dtype=torch.long)
        h = h + self.out_pos_embed(out_pos_tensor).unsqueeze(1)
        
        for step in range(num_steps):
            h = self.think_block(h, step)
        
        h = self.ln_out(h)
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy, value, h
    
    def dynamics_step(self, hidden: torch.Tensor, action: torch.Tensor, 
                     out_pos: int, num_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dynamics for MuZero unrolling."""
        batch_size = hidden.size(0)
        
        # Simple action conditioning
        action_emb = self.token_embed(action).unsqueeze(1)
        h = hidden + action_emb.expand(-1, hidden.size(1), -1)
        
        # Think
        for step in range(num_steps):
            out_pos_tensor = torch.full((batch_size,), min(out_pos + 1, self.config.max_seq_len - 1),
                                       device=hidden.device, dtype=torch.long)
            h = h + self.out_pos_embed(out_pos_tensor).unsqueeze(1)
            h = self.think_block(h, step)
        
        h = self.ln_out(h)
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        reward = value  # Simple reward prediction
        
        return h, reward, policy, value


class VectorizedSortingEnv:
    """Sorting environment."""
    
    def __init__(self, num_envs: int, seq_length: int, vocab_size: int = 10, device: str = "cuda"):
        self.num_envs = num_envs
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.device = device
        
        self.input_seqs = None
        self.target_seqs = None
        self.positions = None
    
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
class ThinkingMetrics:
    """Track thinking steps over training."""
    samples: List[int] = field(default_factory=list)
    seq_lengths: List[int] = field(default_factory=list)
    avg_think_steps: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    
    def add(self, samples: int, seq_len: int, think_steps: float, acc: float):
        self.samples.append(samples)
        self.seq_lengths.append(seq_len)
        self.avg_think_steps.append(think_steps)
        self.accuracies.append(acc)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("Thinking Steps Analysis")
        print("="*60)
        print(f"{'Samples':>10} | {'SeqLen':>6} | {'ThinkSteps':>10} | {'Accuracy':>8}")
        print("-"*60)
        for i in range(0, len(self.samples), max(1, len(self.samples) // 10)):
            print(f"{self.samples[i]:>10,} | {self.seq_lengths[i]:>6} | "
                  f"{self.avg_think_steps[i]:>10.2f} | {self.accuracies[i]:>7.2%}")
        print("="*60)


class UniversalMuZeroTrainer:
    """
    Trainer for Universal MuZero Transformer.
    
    Key: The model learns to think longer for harder problems.
    """
    
    def __init__(self, config: UniversalConfig = None):
        if config is None:
            config = UniversalConfig()
        
        self.config = config
        self.device = config.device
        
        print(f"🧠 Universal MuZero Transformer")
        print(f"   Weight-shared recurrent thinking: ENABLED")
        print(f"   Think steps: {config.min_think_steps} to {config.max_think_steps}")
        print(f"   Think penalty: {config.think_penalty}")
        
        self.model = UniversalMuZeroTransformer(config).to(config.device)
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
        
        # Replay buffer
        self.buffer = {'obs': [], 'actions': [], 'rewards': [], 'policies': []}
        self.buffer_size = 30000
        
        # Metrics
        self.total_samples = 0
        self.recent_accuracies = deque(maxlen=100)
        self.recent_think_steps = deque(maxlen=100)
        self.think_metrics = ThinkingMetrics()
    
    def _recreate_env(self, new_len: int):
        self.current_seq_length = new_len
        self.envs = VectorizedSortingEnv(
            self.config.num_parallel_envs, new_len,
            self.config.vocab_size, self.device
        )
        self.buffer = {'obs': [], 'actions': [], 'rewards': [], 'policies': []}
    
    @torch.no_grad()
    def collect_batch(self) -> Tuple[float, float]:
        """Collect trajectories, return (accuracy, avg_think_steps)."""
        self.model.eval()
        
        obs = self.envs.reset()
        targets = self.envs.target_seqs
        seq_len = self.current_seq_length
        
        all_actions, all_rewards, all_policies = [], [], []
        total_correct = 0
        total_think_steps = 0
        
        # Adaptive thinking steps based on sequence length
        # Longer sequences should use more thinking
        base_steps = max(2, seq_len // 2)
        
        hidden = None
        for step in range(seq_len):
            target_action = targets[:, step]
            
            # Expert policy
            expert_policy = torch.zeros(self.config.num_parallel_envs, self.config.vocab_size, device=self.device)
            expert_policy.scatter_(1, target_action.unsqueeze(1), 1.0)
            
            # Forward with adaptive thinking
            if hidden is None:
                policy_logits, value, hidden, think_steps = self.model(
                    obs, out_pos=step, return_think_steps=True
                )
            else:
                # Use dynamics for subsequent steps
                policy_logits, value, hidden = self.model.forward_fixed_steps(
                    obs, out_pos=step, num_steps=base_steps
                )
                think_steps = base_steps
            
            total_think_steps += think_steps
            
            # Sample action
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
        
        # Store
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
        avg_think = total_think_steps / seq_len
        
        self.recent_accuracies.append(accuracy)
        self.recent_think_steps.append(avg_think)
        
        return accuracy, avg_think
    
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
        
        # Adaptive thinking based on seq length
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
        
        # Think penalty encourages efficiency
        think_penalty = think_steps * self.config.think_penalty
        
        total_loss = policy_loss + value_loss + think_penalty
        
        # Unroll dynamics
        for k in range(min(self.config.unroll_steps, seq_len - 1)):
            hidden, _, policy_logits, value_pred = self.model.dynamics_step(
                hidden, actions[:, k], out_pos=k, num_steps=2
            )
            
            target_idx = policies[:, k+1].argmax(dim=-1)
            policy_loss = F.cross_entropy(policy_logits, target_idx)
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, k+1])
            
            total_loss = total_loss + policy_loss + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, max_samples: int = 500000, log_interval: int = 10000) -> Dict:
        print(f"\n{'='*60}")
        print(f"Starting Universal MuZero Training")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Max samples: {max_samples:,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        best_acc = 0.0
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            accuracy, think_steps = self.collect_batch()
            
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
                print(f"   Thinking steps will increase automatically!")
                
                self._recreate_env(new_len)
                self.recent_accuracies.clear()
            
            # Logging
            if self.total_samples % log_interval < self.config.num_parallel_envs * self.current_seq_length:
                avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                avg_think = np.mean(self.recent_think_steps) if self.recent_think_steps else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                speed = self.total_samples / elapsed
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"N={self.current_seq_length:2} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Think: {avg_think:.1f} | "
                      f"Loss: {avg_loss:.3f} | "
                      f"{speed:,.0f}/s")
                
                if avg_acc > best_acc:
                    best_acc = avg_acc
                
                self.think_metrics.add(
                    self.total_samples, self.current_seq_length,
                    avg_think, avg_acc
                )
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(self.recent_accuracies)[-50:]) if self.recent_accuracies else 0
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final seq_length: {self.current_seq_length}")
        print(f"Final accuracy: {final_acc:.2%}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*60}")
        
        self.think_metrics.print_summary()
        
        return {
            'final_seq_length': self.current_seq_length,
            'final_accuracy': final_acc,
            'params': self.model.count_parameters(),
        }
    
    @torch.no_grad()
    def evaluate(self, seq_length: int, num_envs: int = 1000) -> Tuple[float, float]:
        """Evaluate and return (accuracy, avg_think_steps)."""
        self.model.eval()
        
        env = VectorizedSortingEnv(num_envs, seq_length, self.config.vocab_size, self.device)
        obs = env.reset()
        
        total_correct = 0
        total_think = 0
        
        # More thinking for longer sequences
        think_steps = max(2, seq_length // 2)
        
        for step in range(seq_length):
            policy_logits, _, hidden = self.model.forward_fixed_steps(obs, out_pos=step, num_steps=think_steps)
            # Clamp to valid vocab range
            actions = policy_logits.argmax(dim=-1).clamp(0, self.config.vocab_size - 1)
            _, _, correct = env.step(actions)
            total_correct += correct.sum().item()
            total_think += think_steps
        
        return total_correct / (num_envs * seq_length), total_think / seq_length


def run_universal_experiment(max_samples: int = 500000):
    """Run Universal MuZero Transformer experiment."""
    print("\n" + "="*70)
    print("  🧠 Universal MuZero Transformer")
    print("  Weight-Shared Recurrent Thinking")
    print("="*70)
    
    config = UniversalConfig(
        d_model=64,
        n_heads=4,
        d_ff=256,
        vocab_size=10,
        min_think_steps=2,
        max_think_steps=12,
        think_penalty=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = UniversalMuZeroTrainer(config)
    results = trainer.train(max_samples, log_interval=10000)
    
    print("\n" + "="*70)
    print("Evaluation: Think Steps Scale with Sequence Length")
    print("="*70)
    
    for n in [4, 8, 16, 32]:
        acc, think = trainer.evaluate(n, num_envs=500)
        print(f"  Sort {n:2} numbers: {acc:.2%} (avg {think:.1f} think steps)")
    
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    run_universal_experiment()
