"""
Sorting Evolution: NAS-MuZero with Variable-Length Sorting

This module implements a curriculum learning setup where:
1. The agent learns to sort lists of increasing length (4 → 8 → 16 → 32)
2. Architecture automatically scales via NAS-MuZero
3. Efficiency bonus rewards smaller architectures that solve harder tasks
4. "Predictive Morphing" - agent learns to add capacity BEFORE it needs it

Key Innovation:
- Multi-dimensional action space: Task actions + Meta (architecture) actions
- Automatic difficulty scaling when accuracy threshold is reached
- Efficiency bonus = accuracy / sqrt(params) to prevent "dense monster"
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

from .dynamic_transformer import (
    DynamicTransformer, DynamicConfig, ArchAction, ACTIVATIONS
)


@dataclass
class SortingEvolutionConfig:
    """Configuration for Sorting Evolution experiment."""
    # Task - Variable length!
    min_seq_length: int = 4
    max_seq_length: int = 32
    vocab_size: int = 100  # Numbers 0-99
    accuracy_threshold: float = 0.95  # Increase difficulty when reached
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_parallel_envs: int = 128
    
    # Architecture search
    arch_action_freq: int = 30  # More frequent arch decisions
    arch_reward_scale: float = 5.0
    efficiency_bonus_weight: float = 0.1  # Bonus for smaller architectures
    param_penalty: float = 0.00001  # Lower penalty, rely on efficiency bonus
    
    # Architecture constraints - more room to grow
    min_d_model: int = 32
    max_d_model: int = 256
    min_heads: int = 1
    max_heads: int = 8
    min_layers: int = 1
    max_layers: int = 6
    
    # MuZero
    unroll_steps: int = 3
    discount: float = 1.0
    
    device: str = "cuda"


class VectorizedSortingEnv:
    """
    Vectorized sorting environment for variable-length sequences.
    
    Task: Given an unordered list of integers, output them in sorted order.
    Reward: +1 for each correct element at each position, -1 for wrong.
    """
    
    def __init__(
        self, 
        num_envs: int, 
        seq_length: int, 
        vocab_size: int = 100,
        device: str = "cuda"
    ):
        self.num_envs = num_envs
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.device = device
        
        self.input_seqs = None
        self.target_seqs = None
        self.positions = None
        self.dones = None
    
    def reset(self) -> torch.Tensor:
        """Reset all environments with random unsorted sequences."""
        # Generate random sequences
        self.input_seqs = torch.randint(
            0, self.vocab_size, (self.num_envs, self.seq_length),
            device=self.device, dtype=torch.long
        )
        
        # Target is sorted sequence
        self.target_seqs, _ = torch.sort(self.input_seqs, dim=1)
        
        self.positions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        return self.input_seqs
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step environments with predicted next sorted element.
        
        Actions should be the predicted value at current position in sorted output.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)
        correct_actions = self.target_seqs[batch_indices, self.positions]
        
        # Reward for correct prediction
        correct = (actions == correct_actions)
        rewards = torch.where(
            correct,
            torch.ones_like(actions, dtype=torch.float32),
            -torch.ones_like(actions, dtype=torch.float32)
        )
        
        self.positions = self.positions + 1
        self.dones = (self.positions >= self.seq_length)
        
        return rewards, self.dones, correct.float()
    
    def get_target_actions(self) -> torch.Tensor:
        """Get correct actions for current positions."""
        batch_indices = torch.arange(self.num_envs, device=self.device)
        valid_positions = torch.clamp(self.positions, 0, self.seq_length - 1)
        return self.target_seqs[batch_indices, valid_positions]


@dataclass
class CurriculumHistory:
    """Track curriculum progression and architecture evolution."""
    samples: List[int] = field(default_factory=list)
    seq_lengths: List[int] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    architectures: List[str] = field(default_factory=list)
    param_counts: List[int] = field(default_factory=list)
    efficiency_scores: List[float] = field(default_factory=list)
    arch_actions: List[str] = field(default_factory=list)
    
    def add(self, samples: int, seq_len: int, acc: float, arch: str, 
            params: int, efficiency: float, action: str = ""):
        self.samples.append(samples)
        self.seq_lengths.append(seq_len)
        self.accuracies.append(acc)
        self.architectures.append(arch)
        self.param_counts.append(params)
        self.efficiency_scores.append(efficiency)
        self.arch_actions.append(action)
    
    def print_summary(self):
        print("\n" + "="*80)
        print("Curriculum Progression & Architecture Evolution")
        print("="*80)
        print(f"{'Samples':>10} | {'SeqLen':>6} | {'Accuracy':>8} | {'Architecture':>20} | "
              f"{'Params':>10} | {'Efficiency':>10} | Action")
        print("-"*80)
        for i in range(len(self.samples)):
            print(f"{self.samples[i]:>10,} | {self.seq_lengths[i]:>6} | "
                  f"{self.accuracies[i]:>7.2%} | {self.architectures[i]:>20} | "
                  f"{self.param_counts[i]:>10,} | {self.efficiency_scores[i]:>10.4f} | "
                  f"{self.arch_actions[i]}")
        print("="*80)


class SortingDynamicTransformer(nn.Module):
    """
    Extended Dynamic Transformer for variable-length sorting.
    
    Modifications:
    - Larger vocab (100 numbers instead of 2)
    - Variable sequence length handling
    - Position-aware output (knows which sorted position to predict)
    """
    
    def __init__(self, config: DynamicConfig):
        super().__init__()
        self.config = config
        
        # Token embedding for vocab_size numbers
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Positional embedding - support up to 64 positions
        self.max_positions = 64
        self.pos_embed = nn.Embedding(self.max_positions, config.d_model)
        
        # Step embedding - which output position we're predicting
        self.step_embed = nn.Embedding(self.max_positions, config.d_model)
        
        # Transformer layers
        from .dynamic_transformer import DynamicTransformerBlock
        self.layers = nn.ModuleList([
            DynamicTransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.activation, config.dropout
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, step: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len) token indices (unsorted input)
            step: which output position we're predicting
        """
        batch_size, seq_len = x.shape
        
        # Embeddings
        tok_emb = self.token_embed(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(pos_ids)
        
        # Add step encoding to indicate which output position
        step_tensor = torch.full((batch_size,), step, device=x.device, dtype=torch.long)
        step_emb = self.step_embed(step_tensor).unsqueeze(1)  # (batch, 1, d_model)
        
        h = tok_emb + pos_emb + step_emb
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h)
        
        h = self.ln_out(h)
        
        # Pooled output for heads
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy, value, h
    
    def dynamics_step(self, hidden: torch.Tensor, action: torch.Tensor, step: int = 0
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dynamics step with action conditioning."""
        action_emb = self.action_embed(action).unsqueeze(1)
        
        # Add step encoding
        batch_size = hidden.size(0)
        step_tensor = torch.full((batch_size,), step + 1, device=hidden.device, dtype=torch.long)
        step_tensor = torch.clamp(step_tensor, 0, self.max_positions - 1)
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
    
    # ========== ARCHITECTURE MODIFICATION (inherited from DynamicTransformer) ==========
    
    def add_layer(self) -> bool:
        if self.config.n_layers >= self.config.max_layers:
            return False
        
        from .dynamic_transformer import DynamicTransformerBlock
        new_layer = DynamicTransformerBlock(
            self.config.d_model, self.config.n_heads, self.config.d_ff,
            self.config.activation, self.config.dropout
        ).to(self.config.device)
        
        for p in new_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            else:
                nn.init.zeros_(p)
        
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
        
        from .dynamic_transformer import DynamicTransformerBlock
        new_layers = nn.ModuleList()
        for old_layer in self.layers:
            new_layer = DynamicTransformerBlock(
                self.config.d_model, new_n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout
            ).to(self.config.device)
            new_layer.ffn.load_state_dict(old_layer.ffn.state_dict())
            new_layer.ln1.load_state_dict(old_layer.ln1.state_dict())
            new_layer.ln2.load_state_dict(old_layer.ln2.state_dict())
            new_layers.append(new_layer)
        
        self.layers = new_layers
        return True
    
    def resize_dim(self, new_d_model: int) -> bool:
        if new_d_model < self.config.min_d_model or new_d_model > self.config.max_d_model:
            return False
        if new_d_model % self.config.n_heads != 0:
            return False
        
        old_d_model = self.config.d_model
        self.config.d_model = new_d_model
        self.config.d_ff = new_d_model * 4
        
        # Rebuild model components
        self.token_embed = nn.Embedding(self.config.vocab_size + 1, new_d_model).to(self.config.device)
        self.pos_embed = nn.Embedding(self.max_positions, new_d_model).to(self.config.device)
        self.step_embed = nn.Embedding(self.max_positions, new_d_model).to(self.config.device)
        self.action_embed = nn.Embedding(self.config.vocab_size, new_d_model).to(self.config.device)
        
        from .dynamic_transformer import DynamicTransformerBlock
        self.layers = nn.ModuleList([
            DynamicTransformerBlock(
                new_d_model, self.config.n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout
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
        
        from .dynamic_transformer import DynamicTransformerBlock
        for i, old_layer in enumerate(self.layers):
            new_layer = DynamicTransformerBlock(
                self.config.d_model, self.config.n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout
            ).to(self.config.device)
            
            new_layer.attn.load_state_dict(old_layer.attn.state_dict())
            new_layer.ln1.load_state_dict(old_layer.ln1.state_dict())
            new_layer.ln2.load_state_dict(old_layer.ln2.state_dict())
            new_layer.ffn[0].load_state_dict(old_layer.ffn[0].state_dict())
            new_layer.ffn[3].load_state_dict(old_layer.ffn[3].state_dict())
            
            self.layers[i] = new_layer
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
        return (f"L{self.config.n_layers}_H{self.config.n_heads}_"
                f"D{self.config.d_model}_{self.config.activation}")
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SortingEvolutionTrainer:
    """
    NAS-MuZero trainer for variable-length sorting with curriculum learning.
    
    Key Features:
    1. Automatic difficulty scaling (4 → 8 → 16 → 32)
    2. Architecture evolution via NAS-MuZero
    3. Efficiency bonus for smaller architectures
    4. Predictive morphing detection
    """
    
    def __init__(self, config: SortingEvolutionConfig = None):
        if config is None:
            config = SortingEvolutionConfig()
        
        self.config = config
        self.device = config.device
        
        # Current difficulty level
        self.current_seq_length = config.min_seq_length
        
        print(f"🧬 Sorting Evolution Trainer")
        print(f"   Device: {config.device}")
        print(f"   Sequence lengths: {config.min_seq_length} → {config.max_seq_length}")
        print(f"   Accuracy threshold: {config.accuracy_threshold:.1%}")
        print(f"   Efficiency bonus weight: {config.efficiency_bonus_weight}")
        
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
        
        # Create model
        self.model = SortingDynamicTransformer(self.dyn_config).to(config.device)
        print(f"   Initial architecture: {self.model.get_architecture_string()}")
        print(f"   Initial params: {self.model.count_parameters():,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Architecture policy
        self.arch_policy = nn.Sequential(
            nn.Linear(5 + 1, 64),  # arch_vector + difficulty_level + accuracy
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ArchAction.num_actions())
        ).to(config.device)
        
        self.arch_optimizer = optim.Adam(self.arch_policy.parameters(), lr=1e-3)
        
        # Create environment (will be recreated on difficulty change)
        self.envs = VectorizedSortingEnv(
            config.num_parallel_envs, self.current_seq_length, 
            config.vocab_size, config.device
        )
        
        # Replay buffer
        self.buffer_size = 20000
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_policies = []
        
        # Statistics
        self.total_samples = 0
        self.total_episodes = 0
        self.recent_accuracies = deque(maxlen=100)
        self.accuracy_before_arch_change = 0.0
        
        # Curriculum and architecture history
        self.curriculum_history = CurriculumHistory()
        
        # Track "predictive morphing" events
        self.arch_changes_before_difficulty_increase = []
    
    def _recreate_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _recreate_environment(self, new_seq_length: int):
        """Recreate environment with new sequence length."""
        self.current_seq_length = new_seq_length
        self.envs = VectorizedSortingEnv(
            self.config.num_parallel_envs, new_seq_length,
            self.config.vocab_size, self.device
        )
        # Clear buffer for new difficulty
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_policies = []
    
    def compute_efficiency_score(self, accuracy: float) -> float:
        """
        Efficiency = accuracy / sqrt(params)
        
        Rewards high accuracy with fewer parameters.
        """
        params = self.model.count_parameters()
        return accuracy / np.sqrt(params) * 1000  # Scale for readability
    
    @torch.no_grad()
    def collect_batch(self) -> float:
        """Collect training trajectories."""
        self.model.eval()
        
        observations = self.envs.reset()
        target_seqs = self.envs.target_seqs
        seq_len = self.current_seq_length
        
        all_actions = []
        all_rewards = []
        all_policies = []
        
        # Initial forward
        policy_logits, value, hidden = self.model(observations, step=0)
        
        total_correct = 0
        
        for step in range(seq_len):
            policy = F.softmax(policy_logits, dim=-1)
            target_action = target_seqs[:, step]
            
            # Create expert policy (sparse, just for the correct action)
            expert_policy = torch.zeros(self.config.num_parallel_envs, self.config.vocab_size,
                                       device=self.device)
            expert_policy.scatter_(1, target_action.unsqueeze(1), 1.0)
            
            # Sample actions with exploration
            rand_val = np.random.random()
            if rand_val < 0.3:  # Expert action
                actions = target_action
            elif rand_val < 0.4:  # Random
                actions = torch.randint(0, self.config.vocab_size, 
                                       (self.config.num_parallel_envs,), device=self.device)
            else:  # Policy action
                actions = policy_logits.argmax(dim=-1)
            
            all_actions.append(actions)
            all_policies.append(expert_policy)
            
            rewards, dones, correct = self.envs.step(actions)
            all_rewards.append(rewards)
            total_correct += correct.sum().item()
            
            if step < seq_len - 1:
                hidden, _, policy_logits, value = self.model.dynamics_step(hidden, actions, step)
        
        # Store in buffer
        self.buffer_obs.append(observations.cpu())
        self.buffer_actions.append(torch.stack(all_actions, dim=1).cpu())
        self.buffer_rewards.append(torch.stack(all_rewards, dim=1).cpu())
        self.buffer_policies.append(torch.stack(all_policies, dim=1).cpu())
        
        # Keep buffer bounded
        if len(self.buffer_obs) > self.buffer_size // self.config.num_parallel_envs:
            self.buffer_obs = self.buffer_obs[-self.buffer_size // self.config.num_parallel_envs:]
            self.buffer_actions = self.buffer_actions[-self.buffer_size // self.config.num_parallel_envs:]
            self.buffer_rewards = self.buffer_rewards[-self.buffer_size // self.config.num_parallel_envs:]
            self.buffer_policies = self.buffer_policies[-self.buffer_size // self.config.num_parallel_envs:]
        
        self.total_samples += self.config.num_parallel_envs * seq_len
        self.total_episodes += self.config.num_parallel_envs
        
        accuracy = total_correct / (self.config.num_parallel_envs * seq_len)
        self.recent_accuracies.append(accuracy)
        
        return accuracy
    
    def train_step(self) -> float:
        """One training step."""
        if len(self.buffer_obs) < 2:
            return 0.0
        
        self.model.train()
        
        # Sample from buffer
        idx = np.random.randint(0, len(self.buffer_obs))
        obs = self.buffer_obs[idx].to(self.device)
        actions = self.buffer_actions[idx].to(self.device)
        rewards = self.buffer_rewards[idx].to(self.device)
        policies = self.buffer_policies[idx].to(self.device)
        
        seq_len = obs.size(1) if obs.dim() > 1 else self.current_seq_length
        batch_size = obs.size(0)
        
        # Value targets
        value_targets = torch.zeros(batch_size, seq_len, device=self.device)
        running_return = torch.zeros(batch_size, device=self.device)
        for t in reversed(range(seq_len)):
            running_return = rewards[:, t] + self.config.discount * running_return
            value_targets[:, t] = running_return
        
        # Forward
        policy_logits, value_pred, hidden = self.model(obs, step=0)
        
        # Get target for first step
        target_indices = policies[:, 0].argmax(dim=-1)
        policy_loss = F.cross_entropy(policy_logits, target_indices)
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, 0])
        total_loss = policy_loss + value_loss
        
        # Unroll
        for k in range(min(self.config.unroll_steps, seq_len - 1)):
            hidden, reward_pred, policy_logits, value_pred = self.model.dynamics_step(
                hidden, actions[:, k], k
            )
            
            target_indices = policies[:, k+1].argmax(dim=-1)
            policy_loss = F.cross_entropy(policy_logits, target_indices)
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, k+1])
            reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards[:, k])
            
            total_loss = total_loss + policy_loss + value_loss + reward_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def get_arch_state(self) -> torch.Tensor:
        """Get state for architecture policy including difficulty level."""
        arch_vector = self.dyn_config.to_vector().float().to(self.device)
        difficulty = torch.tensor(
            [(self.current_seq_length - self.config.min_seq_length) / 
             (self.config.max_seq_length - self.config.min_seq_length)],
            device=self.device, dtype=torch.float32
        )
        accuracy = torch.tensor(
            [np.mean(self.recent_accuracies) if self.recent_accuracies else 0.5],
            device=self.device, dtype=torch.float32
        )
        return torch.cat([arch_vector, difficulty, accuracy])
    
    def select_arch_action(self) -> ArchAction:
        """Select architecture action based on current state."""
        state = self.get_arch_state()
        
        with torch.no_grad():
            logits = self.arch_policy(state)
            
            # Mask invalid actions
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
            
            probs = F.softmax(logits / 0.3, dim=-1)  # Lower temp for more exploitation
            action_idx = torch.multinomial(probs, 1).item()
        
        return ArchAction(action_idx)
    
    def update_arch_policy(self, action: ArchAction, reward: float):
        """Update architecture policy with reward."""
        state = self.get_arch_state()
        
        logits = self.arch_policy(state)
        log_prob = F.log_softmax(logits, dim=-1)[action]
        
        loss = -log_prob * reward
        
        self.arch_optimizer.zero_grad()
        loss.backward()
        self.arch_optimizer.step()
    
    def apply_arch_action(self, action: ArchAction) -> Tuple[bool, str]:
        """Apply architecture action."""
        old_params = self.model.count_parameters()
        success = self.model.apply_arch_action(action)
        
        if success and action != ArchAction.NO_OP:
            self._recreate_optimizer()
            new_params = self.model.count_parameters()
            msg = (f"🔧 Arch: {self.model.get_architecture_string()} "
                   f"({old_params:,} → {new_params:,})")
            return True, msg
        
        return False, ""
    
    def train(
        self,
        max_samples: int = 500000,
        log_interval: int = 5000
    ) -> Dict:
        """
        Main training loop with curriculum and architecture evolution.
        """
        print(f"\n{'='*70}")
        print(f"Starting Sorting Evolution Training")
        print(f"Max samples: {max_samples:,}")
        print(f"Starting difficulty: sort {self.current_seq_length} numbers")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        best_accuracy = 0.0
        recent_losses = deque(maxlen=100)
        episodes_since_arch = 0
        
        # Track difficulty increases
        difficulty_increases = []
        
        # Initial record
        self.curriculum_history.add(
            0, self.current_seq_length, 0.0,
            self.model.get_architecture_string(),
            self.model.count_parameters(),
            0.0, "INIT"
        )
        
        while self.total_samples < max_samples:
            # Collect batch
            accuracy = self.collect_batch()
            episodes_since_arch += self.config.num_parallel_envs
            
            # Train
            if len(self.buffer_obs) >= 2:
                for _ in range(2):
                    loss = self.train_step()
                    recent_losses.append(loss)
            
            # Check for difficulty increase
            avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
            if (avg_acc >= self.config.accuracy_threshold and 
                self.current_seq_length < self.config.max_seq_length):
                
                old_len = self.current_seq_length
                new_len = min(self.current_seq_length * 2, self.config.max_seq_length)
                
                print(f"\n🎯 Accuracy {avg_acc:.2%} >= {self.config.accuracy_threshold:.0%}")
                print(f"   ⬆️ Increasing difficulty: {old_len} → {new_len} numbers")
                
                difficulty_increases.append({
                    'samples': self.total_samples,
                    'from': old_len,
                    'to': new_len,
                    'architecture': self.model.get_architecture_string(),
                    'params': self.model.count_parameters()
                })
                
                self._recreate_environment(new_len)
                self.recent_accuracies.clear()
                
                # Record curriculum change
                self.curriculum_history.add(
                    self.total_samples, new_len, avg_acc,
                    self.model.get_architecture_string(),
                    self.model.count_parameters(),
                    self.compute_efficiency_score(avg_acc),
                    f"DIFFICULTY_{old_len}→{new_len}"
                )
            
            # Architecture decision
            if episodes_since_arch >= self.config.arch_action_freq * self.config.num_parallel_envs:
                episodes_since_arch = 0
                
                current_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                improvement = current_acc - self.accuracy_before_arch_change
                
                action = self.select_arch_action()
                success, msg = self.apply_arch_action(action)
                
                if success and action != ArchAction.NO_OP:
                    # Efficiency bonus: reward smaller architectures with high accuracy
                    efficiency = self.compute_efficiency_score(current_acc)
                    
                    # Combined reward
                    arch_reward = (
                        improvement * self.config.arch_reward_scale +
                        efficiency * self.config.efficiency_bonus_weight
                    )
                    
                    self.update_arch_policy(action, arch_reward)
                    print(msg)
                    
                    # Record
                    self.curriculum_history.add(
                        self.total_samples, self.current_seq_length, current_acc,
                        self.model.get_architecture_string(),
                        self.model.count_parameters(),
                        efficiency,
                        ArchAction(action).name
                    )
                
                self.accuracy_before_arch_change = current_acc
            
            # Logging
            if self.total_samples % log_interval < self.config.num_parallel_envs * self.current_seq_length:
                avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                samples_per_sec = self.total_samples / elapsed
                efficiency = self.compute_efficiency_score(avg_acc)
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"SeqLen: {self.current_seq_length:2} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Eff: {efficiency:.2f} | "
                      f"Arch: {self.model.get_architecture_string()} | "
                      f"Speed: {samples_per_sec:,.0f}/s")
                
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(self.recent_accuracies)[-50:]) if self.recent_accuracies else 0
        
        results = {
            'total_samples': self.total_samples,
            'best_accuracy': best_accuracy,
            'final_accuracy': final_acc,
            'final_seq_length': self.current_seq_length,
            'final_architecture': self.model.get_architecture_string(),
            'final_params': self.model.count_parameters(),
            'final_efficiency': self.compute_efficiency_score(final_acc),
            'elapsed_time': elapsed,
            'samples_per_second': self.total_samples / elapsed,
            'difficulty_increases': difficulty_increases,
        }
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Total samples: {results['total_samples']:,}")
        print(f"Final seq_length: {results['final_seq_length']}")
        print(f"Final accuracy: {results['final_accuracy']:.2%}")
        print(f"Final architecture: {results['final_architecture']}")
        print(f"Final parameters: {results['final_params']:,}")
        print(f"Efficiency score: {results['final_efficiency']:.4f}")
        print(f"Speed: {results['samples_per_second']:,.0f} samples/sec")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*70}")
        
        self.curriculum_history.print_summary()
        
        return results
    
    @torch.no_grad()
    def evaluate(self, seq_length: int = None, num_envs: int = 1000) -> float:
        """Evaluate on specific sequence length."""
        if seq_length is None:
            seq_length = self.current_seq_length
        
        self.model.eval()
        
        eval_env = VectorizedSortingEnv(num_envs, seq_length, self.config.vocab_size, self.device)
        obs = eval_env.reset()
        
        policy_logits, _, hidden = self.model(obs, step=0)
        
        total_correct = 0
        for step in range(seq_length):
            actions = policy_logits.argmax(dim=-1)
            rewards, _, correct = eval_env.step(actions)
            total_correct += correct.sum().item()
            
            if step < seq_length - 1:
                hidden, _, policy_logits, _ = self.model.dynamics_step(hidden, actions, step)
        
        return total_correct / (num_envs * seq_length)


def run_sorting_evolution(max_samples: int = 500000):
    """Run the Sorting Evolution experiment."""
    print("\n" + "="*70)
    print("  🧬 Sorting Evolution: Predictive Morphing Experiment")
    print("="*70)
    
    config = SortingEvolutionConfig(
        min_seq_length=4,
        max_seq_length=32,
        vocab_size=100,
        accuracy_threshold=0.95,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = SortingEvolutionTrainer(config)
    results = trainer.train(max_samples, log_interval=10000)
    
    # Evaluate on different sequence lengths
    print("\n" + "="*70)
    print("Evaluation on Different Sequence Lengths")
    print("="*70)
    
    for seq_len in [4, 8, 16, 32]:
        if seq_len <= config.max_seq_length:
            acc = trainer.evaluate(seq_len, num_envs=500)
            print(f"  Sort {seq_len:2} numbers: {acc:.2%}")
    
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    run_sorting_evolution()
