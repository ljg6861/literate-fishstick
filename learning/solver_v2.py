"""
Self-Optimizing Algorithmic Solver v2

Key Improvements:
1. Relative Positional Encoding - enables length generalization
2. Efficiency Frontier tracking - accuracy / params ratio
3. Value Loss spike detection for architecture decisions
4. Dynamic simulation budget consideration

The goal: 100% accuracy on 16-32 number sorting with autonomous scaling.
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

from .dynamic_transformer import ArchAction, ACTIVATIONS, DynamicConfig


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding for better length generalization.
    
    Instead of encoding absolute positions (1, 2, 3...), we encode
    relative distances between tokens. This allows patterns learned
    at length 4 to transfer to length 8, 16, 32.
    """
    
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Relative position embeddings (2 * max_len - 1 positions for all relative distances)
        self.rel_pos_embed = nn.Embedding(2 * max_len - 1, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with sinusoidal pattern for better starting point
        self._init_weights()
    
    def _init_weights(self):
        positions = torch.arange(-(self.max_len - 1), self.max_len).float()
        dim = torch.arange(0, self.d_model, 2).float()
        
        angles = positions.unsqueeze(1) / (10000 ** (dim / self.d_model))
        
        embeddings = torch.zeros(2 * self.max_len - 1, self.d_model)
        embeddings[:, 0::2] = torch.sin(angles)
        embeddings[:, 1::2] = torch.cos(angles)
        
        self.rel_pos_embed.weight.data.copy_(embeddings)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate relative position bias for attention.
        
        Returns: (seq_len, seq_len, d_model) tensor of relative position encodings
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        
        # Compute relative distances: positions[i] - positions[j]
        rel_distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        
        # Shift to positive indices
        rel_distances = rel_distances + (self.max_len - 1)
        
        # Clamp to valid range
        rel_distances = rel_distances.clamp(0, 2 * self.max_len - 2)
        
        # Get embeddings
        rel_pos = self.rel_pos_embed(rel_distances)  # (seq_len, seq_len, d_model)
        
        return self.dropout(rel_pos)


class RelativeAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    
    Uses a simpler additive relative position bias approach.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Relative position bias: learnable bias for each relative distance
        # Shape: (2 * max_len - 1,) - one bias value per relative position
        self.max_len = max_len
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_len - 1))
        nn.init.normal_(self.rel_pos_bias, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_rel_pos_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get relative position bias matrix."""
        positions = torch.arange(seq_len, device=device)
        rel_dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        rel_dist = rel_dist + (self.max_len - 1)  # Shift to positive
        rel_dist = rel_dist.clamp(0, 2 * self.max_len - 2)
        
        return self.rel_pos_bias[rel_dist]  # (seq_len, seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Content-based attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias (same for all heads)
        rel_bias = self._get_rel_pos_bias(seq_len, x.device)  # (seq_len, seq_len)
        attn_scores = attn_scores + rel_bias.unsqueeze(0).unsqueeze(0)
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class RelativeTransformerBlock(nn.Module):
    """Transformer block with relative positional attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 activation: str = 'gelu', dropout: float = 0.0, max_len: int = 64):
        super().__init__()
        
        self.attn = RelativeAttention(d_model, n_heads, max_len, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else (nn.ReLU() if activation == 'relu' else nn.SiLU()),
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


class SelfOptimizingSolver(nn.Module):
    """
    Self-Optimizing Algorithmic Solver with:
    - Relative positional encoding for length generalization
    - Dynamic architecture modification
    - Efficiency tracking
    """
    
    def __init__(self, config: DynamicConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Step embedding (which output position we're predicting)
        self.max_len = 64
        self.step_embed = nn.Embedding(self.max_len, config.d_model)
        
        # Transformer layers with relative attention
        self.layers = nn.ModuleList([
            RelativeTransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.activation, config.dropout, self.max_len
            )
            for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.ln_out = nn.LayerNorm(config.d_model)
        self.policy_head = nn.Linear(config.d_model, config.vocab_size)
        self.value_head = nn.Linear(config.d_model, 1)
        self.reward_head = nn.Linear(config.d_model, 1)
        
        # Action embedding for dynamics
        self.action_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, step: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for sorting.
        
        x: (batch, seq_len) - unsorted input sequence
        step: which sorted position we're predicting
        """
        batch_size, seq_len = x.shape
        
        # Token embedding only (relative pos is in attention)
        h = self.token_embed(x)
        
        # Add step embedding to indicate which output position
        step_tensor = torch.full((batch_size,), min(step, self.max_len - 1), 
                                 device=x.device, dtype=torch.long)
        step_emb = self.step_embed(step_tensor).unsqueeze(1)
        h = h + step_emb
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h)
        
        h = self.ln_out(h)
        
        # Pooled output
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy, value, h
    
    def dynamics_step(self, hidden: torch.Tensor, action: torch.Tensor, step: int = 0
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dynamics step with action conditioning."""
        batch_size = hidden.size(0)
        
        action_emb = self.action_embed(action).unsqueeze(1)
        
        step_tensor = torch.full((batch_size,), min(step + 1, self.max_len - 1),
                                 device=hidden.device, dtype=torch.long)
        step_emb = self.step_embed(step_tensor).unsqueeze(1)
        
        h = hidden + action_emb.expand(-1, hidden.size(1), -1) + step_emb
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.ln_out(h)
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        reward = self.reward_head(pooled)
        
        return h, reward, policy, value
    
    # ========== ARCHITECTURE MODIFICATION ==========
    
    def add_layer(self) -> bool:
        if self.config.n_layers >= self.config.max_layers:
            return False
        
        new_layer = RelativeTransformerBlock(
            self.config.d_model, self.config.n_heads, self.config.d_ff,
            self.config.activation, self.config.dropout, self.max_len
        ).to(self.config.device)
        
        # Initialize to near-identity
        for p in new_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        
        self.layers.append(new_layer)
        self.config.n_layers += 1
        return True
    
    def remove_layer(self) -> bool:
        if self.config.n_layers <= self.config.min_layers:
            return False
        self.layers = self.layers[:-1]
        self.config.n_layers -= 1
        return True
    
    def resize_heads(self, new_n_heads: int) -> bool:
        if new_n_heads < self.config.min_heads or new_n_heads > self.config.max_heads:
            return False
        if self.config.d_model % new_n_heads != 0:
            return False
        
        self.config.n_heads = new_n_heads
        
        new_layers = nn.ModuleList([
            RelativeTransformerBlock(
                self.config.d_model, new_n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout, self.max_len
            ).to(self.config.device)
            for _ in range(self.config.n_layers)
        ])
        
        self.layers = new_layers
        return True
    
    def resize_dim(self, new_d_model: int) -> bool:
        if new_d_model < self.config.min_d_model or new_d_model > self.config.max_d_model:
            return False
        if new_d_model % self.config.n_heads != 0:
            return False
        
        self.config.d_model = new_d_model
        self.config.d_ff = new_d_model * 4
        
        # Rebuild
        self.token_embed = nn.Embedding(self.config.vocab_size + 1, new_d_model).to(self.config.device)
        self.step_embed = nn.Embedding(self.max_len, new_d_model).to(self.config.device)
        self.action_embed = nn.Embedding(self.config.vocab_size, new_d_model).to(self.config.device)
        
        self.layers = nn.ModuleList([
            RelativeTransformerBlock(
                new_d_model, self.config.n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout, self.max_len
            ).to(self.config.device)
            for _ in range(self.config.n_layers)
        ])
        
        self.ln_out = nn.LayerNorm(new_d_model).to(self.config.device)
        self.policy_head = nn.Linear(new_d_model, self.config.vocab_size).to(self.config.device)
        self.value_head = nn.Linear(new_d_model, 1).to(self.config.device)
        self.reward_head = nn.Linear(new_d_model, 1).to(self.config.device)
        
        self._init_weights()
        return True
    
    def change_activation(self) -> bool:
        current_idx = ACTIVATIONS.index(self.config.activation)
        new_idx = (current_idx + 1) % len(ACTIVATIONS)
        self.config.activation = ACTIVATIONS[new_idx]
        
        # Rebuild layers
        new_layers = nn.ModuleList([
            RelativeTransformerBlock(
                self.config.d_model, self.config.n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout, self.max_len
            ).to(self.config.device)
            for _ in range(self.config.n_layers)
        ])
        self.layers = new_layers
        return True
    
    def apply_arch_action(self, action: ArchAction) -> bool:
        if action == ArchAction.NO_OP:
            return True
        elif action == ArchAction.ADD_LAYER:
            return self.add_layer()
        elif action == ArchAction.REMOVE_LAYER:
            return self.remove_layer()
        elif action == ArchAction.INCREASE_HEADS:
            new_heads = min(self.config.n_heads * 2, self.config.max_heads)
            return self.resize_heads(new_heads)
        elif action == ArchAction.DECREASE_HEADS:
            new_heads = max(self.config.n_heads // 2, self.config.min_heads)
            return self.resize_heads(new_heads)
        elif action == ArchAction.INCREASE_DIM:
            new_dim = min(self.config.d_model + 32, self.config.max_d_model)
            return self.resize_dim(new_dim)
        elif action == ArchAction.DECREASE_DIM:
            new_dim = max(self.config.d_model - 32, self.config.min_d_model)
            return self.resize_dim(new_dim)
        elif action == ArchAction.CHANGE_ACTIVATION:
            return self.change_activation()
        return False
    
    def get_architecture_string(self) -> str:
        return f"L{self.config.n_layers}_H{self.config.n_heads}_D{self.config.d_model}_{self.config.activation}"
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VectorizedSortingEnvV2:
    """Optimized sorting environment."""
    
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
class EfficiencyMetrics:
    """Track efficiency frontier over training."""
    samples: List[int] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    param_counts: List[int] = field(default_factory=list)
    efficiency_scores: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    seq_lengths: List[int] = field(default_factory=list)
    
    def add(self, samples: int, acc: float, params: int, value_loss: float, seq_len: int):
        self.samples.append(samples)
        self.accuracies.append(acc)
        self.param_counts.append(params)
        self.efficiency_scores.append(acc / np.sqrt(params) * 1000)
        self.value_losses.append(value_loss)
        self.seq_lengths.append(seq_len)
    
    def print_summary(self):
        print("\n" + "="*70)
        print("Efficiency Frontier Analysis")
        print("="*70)
        print(f"{'Samples':>10} | {'SeqLen':>6} | {'Acc':>7} | {'Params':>10} | {'Efficiency':>10} | {'VLoss':>8}")
        print("-"*70)
        for i in range(0, len(self.samples), max(1, len(self.samples) // 10)):
            print(f"{self.samples[i]:>10,} | {self.seq_lengths[i]:>6} | "
                  f"{self.accuracies[i]:>6.2%} | {self.param_counts[i]:>10,} | "
                  f"{self.efficiency_scores[i]:>10.4f} | {self.value_losses[i]:>8.4f}")
        print("="*70)


@dataclass
class SolverConfig:
    """Config for Self-Optimizing Solver."""
    # Task
    min_seq_length: int = 4
    max_seq_length: int = 32
    vocab_size: int = 10  # Small vocab for sorting
    accuracy_threshold: float = 0.90
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_parallel_envs: int = 256
    
    # Architecture
    min_d_model: int = 64
    max_d_model: int = 256
    min_heads: int = 2
    max_heads: int = 8
    min_layers: int = 2
    max_layers: int = 6
    
    # NAS
    arch_action_freq: int = 20  # More frequent
    arch_reward_scale: float = 10.0
    
    # MuZero
    unroll_steps: int = 4
    discount: float = 1.0
    
    device: str = "cuda"


class SelfOptimizingSolverTrainer:
    """
    Trainer for Self-Optimizing Algorithmic Solver.
    
    Features:
    - Relative positional encoding for length generalization
    - Automatic curriculum (4 → 8 → 16 → 32)
    - Efficiency frontier tracking
    - Value loss spike detection for architecture decisions
    """
    
    def __init__(self, config: SolverConfig = None):
        if config is None:
            config = SolverConfig()
        
        self.config = config
        self.device = config.device
        self.current_seq_length = config.min_seq_length
        
        print(f"🚀 Self-Optimizing Solver v2")
        print(f"   Relative Positional Encoding: ENABLED")
        print(f"   Sequence lengths: {config.min_seq_length} → {config.max_seq_length}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Accuracy threshold: {config.accuracy_threshold:.0%}")
        
        # Dynamic config
        self.dyn_config = DynamicConfig(
            d_model=config.min_d_model,
            n_heads=config.min_heads,
            n_layers=config.min_layers,
            d_ff=config.min_d_model * 4,
            activation='gelu',
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_length,
            min_d_model=config.min_d_model,
            max_d_model=config.max_d_model,
            min_heads=config.min_heads,
            max_heads=config.max_heads,
            min_layers=config.min_layers,
            max_layers=config.max_layers,
            device=config.device,
        )
        
        # Model with relative attention
        self.model = SelfOptimizingSolver(self.dyn_config).to(config.device)
        print(f"   Initial architecture: {self.model.get_architecture_string()}")
        print(f"   Initial params: {self.model.count_parameters():,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Architecture policy
        self.arch_policy = nn.Sequential(
            nn.Linear(6, 64),  # arch_vector + difficulty + accuracy
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ArchAction.num_actions())
        ).to(config.device)
        self.arch_optimizer = optim.Adam(self.arch_policy.parameters(), lr=1e-3)
        
        # Environment
        self.envs = VectorizedSortingEnvV2(
            config.num_parallel_envs, self.current_seq_length,
            config.vocab_size, config.device
        )
        
        # Replay buffer
        self.buffer = {'obs': [], 'actions': [], 'rewards': [], 'policies': []}
        self.buffer_size = 30000
        
        # Metrics
        self.total_samples = 0
        self.recent_accuracies = deque(maxlen=100)
        self.recent_value_losses = deque(maxlen=50)
        self.efficiency_metrics = EfficiencyMetrics()
        self.accuracy_before_arch = 0.0
    
    def _recreate_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _recreate_env(self, new_len: int):
        self.current_seq_length = new_len
        self.envs = VectorizedSortingEnvV2(
            self.config.num_parallel_envs, new_len,
            self.config.vocab_size, self.device
        )
        self.buffer = {'obs': [], 'actions': [], 'rewards': [], 'policies': []}
    
    @torch.no_grad()
    def collect_batch(self) -> float:
        self.model.eval()
        
        obs = self.envs.reset()
        targets = self.envs.target_seqs
        seq_len = self.current_seq_length
        
        all_actions, all_rewards, all_policies = [], [], []
        
        policy_logits, value, hidden = self.model(obs, step=0)
        total_correct = 0
        
        for step in range(seq_len):
            target_action = targets[:, step]
            
            # Expert policy
            expert_policy = torch.zeros(self.config.num_parallel_envs, self.config.vocab_size, device=self.device)
            expert_policy.scatter_(1, target_action.unsqueeze(1), 1.0)
            
            # Sample action
            rand = np.random.random()
            if rand < 0.4:  # More expert guidance
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
            
            if step < seq_len - 1:
                hidden, _, policy_logits, value = self.model.dynamics_step(hidden, actions, step)
        
        # Store
        self.buffer['obs'].append(obs.cpu())
        self.buffer['actions'].append(torch.stack(all_actions, dim=1).cpu())
        self.buffer['rewards'].append(torch.stack(all_rewards, dim=1).cpu())
        self.buffer['policies'].append(torch.stack(all_policies, dim=1).cpu())
        
        # Bound buffer
        max_entries = self.buffer_size // self.config.num_parallel_envs
        for k in self.buffer:
            if len(self.buffer[k]) > max_entries:
                self.buffer[k] = self.buffer[k][-max_entries:]
        
        self.total_samples += self.config.num_parallel_envs * seq_len
        accuracy = total_correct / (self.config.num_parallel_envs * seq_len)
        self.recent_accuracies.append(accuracy)
        
        return accuracy
    
    def train_step(self) -> Tuple[float, float]:
        """Returns (total_loss, value_loss)"""
        if len(self.buffer['obs']) < 2:
            return 0.0, 0.0
        
        self.model.train()
        
        idx = np.random.randint(0, len(self.buffer['obs']))
        obs = self.buffer['obs'][idx].to(self.device)
        actions = self.buffer['actions'][idx].to(self.device)
        rewards = self.buffer['rewards'][idx].to(self.device)
        policies = self.buffer['policies'][idx].to(self.device)
        
        seq_len = actions.size(1)
        batch_size = obs.size(0)
        
        # Value targets
        value_targets = torch.zeros(batch_size, seq_len, device=self.device)
        running = torch.zeros(batch_size, device=self.device)
        for t in reversed(range(seq_len)):
            running = rewards[:, t] + self.config.discount * running
            value_targets[:, t] = running
        
        # Forward
        policy_logits, value_pred, hidden = self.model(obs, step=0)
        
        target_idx = policies[:, 0].argmax(dim=-1)
        policy_loss = F.cross_entropy(policy_logits, target_idx)
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, 0])
        total_loss = policy_loss + value_loss
        
        # Unroll
        for k in range(min(self.config.unroll_steps, seq_len - 1)):
            hidden, reward_pred, policy_logits, value_pred = self.model.dynamics_step(hidden, actions[:, k], k)
            
            target_idx = policies[:, k+1].argmax(dim=-1)
            policy_loss = F.cross_entropy(policy_logits, target_idx)
            v_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, k+1])
            r_loss = F.mse_loss(reward_pred.squeeze(-1), rewards[:, k])
            
            value_loss = value_loss + v_loss
            total_loss = total_loss + policy_loss + v_loss + r_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.recent_value_losses.append(value_loss.item())
        
        return total_loss.item(), value_loss.item()
    
    def get_arch_state(self) -> torch.Tensor:
        arch = self.dyn_config.to_vector().float().to(self.device)
        diff = torch.tensor([(self.current_seq_length - self.config.min_seq_length) / 
                            (self.config.max_seq_length - self.config.min_seq_length)],
                           device=self.device, dtype=torch.float32)
        acc = torch.tensor([np.mean(self.recent_accuracies) if self.recent_accuracies else 0.5],
                          device=self.device, dtype=torch.float32)
        return torch.cat([arch, diff, acc])
    
    def select_arch_action(self) -> ArchAction:
        state = self.get_arch_state()
        
        with torch.no_grad():
            logits = self.arch_policy(state)
            
            mask = torch.ones_like(logits)
            if self.dyn_config.n_layers >= self.dyn_config.max_layers:
                mask[ArchAction.ADD_LAYER] = 0
            if self.dyn_config.n_layers <= self.dyn_config.min_layers:
                mask[ArchAction.REMOVE_LAYER] = 0
            if self.dyn_config.n_heads >= self.dyn_config.max_heads:
                mask[ArchAction.INCREASE_HEADS] = 0
            if self.dyn_config.n_heads <= self.dyn_config.min_heads:
                mask[ArchAction.DECREASE_HEADS] = 0
            if self.dyn_config.d_model >= self.dyn_config.max_d_model:
                mask[ArchAction.INCREASE_DIM] = 0
            if self.dyn_config.d_model <= self.dyn_config.min_d_model:
                mask[ArchAction.DECREASE_DIM] = 0
            
            logits = logits * mask + (1 - mask) * (-1e9)
            probs = F.softmax(logits / 0.3, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
        
        return ArchAction(action_idx)
    
    def apply_arch_action(self, action: ArchAction) -> Tuple[bool, str]:
        old_params = self.model.count_parameters()
        success = self.model.apply_arch_action(action)
        
        if success and action != ArchAction.NO_OP:
            self._recreate_optimizer()
            new_params = self.model.count_parameters()
            return True, f"🔧 {self.model.get_architecture_string()} ({old_params:,}→{new_params:,})"
        return False, ""
    
    def train(self, max_samples: int = 500000, log_interval: int = 5000) -> Dict:
        print(f"\n{'='*70}")
        print(f"Starting Self-Optimizing Solver Training")
        print(f"Max samples: {max_samples:,}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        best_acc = 0.0
        episodes_since_arch = 0
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            accuracy = self.collect_batch()
            episodes_since_arch += self.config.num_parallel_envs
            
            if len(self.buffer['obs']) >= 2:
                for _ in range(4):
                    loss, vloss = self.train_step()
                    recent_losses.append(loss)
            
            # Curriculum check
            avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
            if avg_acc >= self.config.accuracy_threshold and self.current_seq_length < self.config.max_seq_length:
                old_len = self.current_seq_length
                new_len = min(self.current_seq_length * 2, self.config.max_seq_length)
                
                print(f"\n🎯 Accuracy {avg_acc:.2%} >= {self.config.accuracy_threshold:.0%}")
                print(f"   ⬆️ CURRICULUM: {old_len} → {new_len} numbers")
                
                self._recreate_env(new_len)
                self.recent_accuracies.clear()
            
            # Architecture decision
            if episodes_since_arch >= self.config.arch_action_freq * self.config.num_parallel_envs:
                episodes_since_arch = 0
                
                action = self.select_arch_action()
                success, msg = self.apply_arch_action(action)
                if success and msg:
                    print(msg)
                
                self.accuracy_before_arch = avg_acc
            
            # Logging
            if self.total_samples % log_interval < self.config.num_parallel_envs * self.current_seq_length:
                avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                avg_vloss = np.mean(self.recent_value_losses) if self.recent_value_losses else 0
                elapsed = time.time() - start_time
                speed = self.total_samples / elapsed
                efficiency = avg_acc / np.sqrt(self.model.count_parameters()) * 1000
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"N={self.current_seq_length:2} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Eff: {efficiency:.2f} | "
                      f"VLoss: {avg_vloss:.3f} | "
                      f"Arch: {self.model.get_architecture_string()} | "
                      f"{speed:,.0f}/s")
                
                if avg_acc > best_acc:
                    best_acc = avg_acc
                
                self.efficiency_metrics.add(
                    self.total_samples, avg_acc, self.model.count_parameters(),
                    avg_vloss, self.current_seq_length
                )
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(self.recent_accuracies)[-50:]) if self.recent_accuracies else 0
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Final seq_length: {self.current_seq_length}")
        print(f"Final accuracy: {final_acc:.2%}")
        print(f"Final architecture: {self.model.get_architecture_string()}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*70}")
        
        self.efficiency_metrics.print_summary()
        
        return {
            'final_seq_length': self.current_seq_length,
            'final_accuracy': final_acc,
            'architecture': self.model.get_architecture_string(),
            'params': self.model.count_parameters(),
        }
    
    @torch.no_grad()
    def evaluate(self, seq_length: int, num_envs: int = 1000) -> float:
        self.model.eval()
        
        env = VectorizedSortingEnvV2(num_envs, seq_length, self.config.vocab_size, self.device)
        obs = env.reset()
        
        policy_logits, _, hidden = self.model(obs, step=0)
        
        total_correct = 0
        for step in range(seq_length):
            actions = policy_logits.argmax(dim=-1)
            _, _, correct = env.step(actions)
            total_correct += correct.sum().item()
            
            if step < seq_length - 1:
                hidden, _, policy_logits, _ = self.model.dynamics_step(hidden, actions, step)
        
        return total_correct / (num_envs * seq_length)


def run_solver_v2(max_samples: int = 500000):
    """Run the Self-Optimizing Solver v2."""
    print("\n" + "="*70)
    print("  🚀 Self-Optimizing Algorithmic Solver v2")
    print("  With Relative Positional Encoding")
    print("="*70)
    
    config = SolverConfig(
        min_seq_length=4,
        max_seq_length=32,
        vocab_size=10,
        accuracy_threshold=0.90,
        min_d_model=64,
        max_d_model=256,
        min_layers=2,
        max_layers=6,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = SelfOptimizingSolverTrainer(config)
    results = trainer.train(max_samples, log_interval=10000)
    
    print("\n" + "="*70)
    print("Evaluation on Different Sequence Lengths")
    print("="*70)
    
    for n in [4, 8, 16, 32]:
        if n <= config.max_seq_length:
            acc = trainer.evaluate(n, num_envs=1000)
            print(f"  Sort {n:2} numbers: {acc:.2%}")
    
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    run_solver_v2()
