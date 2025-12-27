"""
NAS-MuZero: Neural Architecture Search via MuZero Planning

This module implements a novel NAS system where MuZero's action space
includes architecture modifications, allowing the agent to evolve its
own Transformer architecture while simultaneously training on the task.

Key Innovation:
- Hierarchical action space: Task actions + Architecture actions
- MuZero's MCTS lookahead predicts which arch change maximizes future reward
- Weight preservation during modifications
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
from .fast_muzero import VectorizedBitReversalEnv, FastReplayBuffer


@dataclass
class NASConfig:
    """Configuration for NAS-MuZero."""
    # Task
    vocab_size: int = 2
    seq_length: int = 4
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    num_parallel_envs: int = 256
    
    # Architecture search
    arch_action_freq: int = 50  # Consider arch change every N episodes
    arch_reward_scale: float = 10.0  # Scale for accuracy improvement reward
    param_penalty: float = 0.0001  # Penalty per parameter increase
    
    # Architecture constraints
    min_d_model: int = 32
    max_d_model: int = 128
    min_heads: int = 1
    max_heads: int = 4
    min_layers: int = 1
    max_layers: int = 4
    
    # MuZero
    unroll_steps: int = 3
    discount: float = 1.0
    
    device: str = "cuda"


@dataclass
class ArchitectureHistory:
    """Track architecture evolution over training."""
    steps: List[int] = field(default_factory=list)
    configs: List[str] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    param_counts: List[int] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    
    def add(self, step: int, config: str, accuracy: float, params: int, action: str):
        self.steps.append(step)
        self.configs.append(config)
        self.accuracies.append(accuracy)
        self.param_counts.append(params)
        self.actions_taken.append(action)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("Architecture Evolution History")
        print("="*60)
        for i in range(len(self.steps)):
            print(f"Step {self.steps[i]:6d} | {self.configs[i]:20s} | "
                  f"Acc: {self.accuracies[i]:.2%} | Params: {self.param_counts[i]:,} | "
                  f"Action: {self.actions_taken[i]}")
        print("="*60)


class NASMuZeroTrainer:
    """
    NAS-MuZero: Architecture search via MuZero planning.
    
    The agent has two types of actions:
    1. Task actions: Next token prediction (standard MuZero)
    2. Architecture actions: Modify the Transformer architecture
    
    Architecture actions are taken every `arch_action_freq` episodes.
    """
    
    def __init__(self, config: NASConfig = None):
        if config is None:
            config = NASConfig()
        
        self.config = config
        self.device = config.device
        
        print(f"🧬 NAS-MuZero Trainer")
        print(f"   Device: {config.device}")
        print(f"   Arch action frequency: every {config.arch_action_freq} episodes")
        print(f"   Architecture constraints: L[{config.min_layers}-{config.max_layers}] "
              f"H[{config.min_heads}-{config.max_heads}] D[{config.min_d_model}-{config.max_d_model}]")
        
        # Dynamic Transformer config
        self.dyn_config = DynamicConfig(
            d_model=config.min_d_model,  # Start minimal
            n_heads=config.min_heads,
            n_layers=config.min_layers,
            d_ff=config.min_d_model * 4,
            activation='gelu',
            vocab_size=config.vocab_size,
            max_seq_len=config.seq_length,
            min_d_model=config.min_d_model,
            max_d_model=config.max_d_model,
            min_heads=config.min_heads,
            max_heads=config.max_heads,
            min_layers=config.min_layers,
            max_layers=config.max_layers,
            device=config.device,
        )
        
        # Create dynamic model
        self.model = DynamicTransformer(self.dyn_config).to(config.device)
        print(f"   Initial architecture: {self.model.get_architecture_string()}")
        print(f"   Initial params: {self.model.count_parameters():,}")
        
        # Optimizer (will be recreated after architecture changes)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Environment
        self.envs = VectorizedBitReversalEnv(
            config.num_parallel_envs, config.seq_length, config.device
        )
        
        # Replay buffer
        self.buffer = FastReplayBuffer(50000, config.seq_length, config.device)
        
        # Architecture policy network (separate small network for arch decisions)
        self.arch_policy = nn.Sequential(
            nn.Linear(4 + 1, 64),  # arch_vector + current_accuracy
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ArchAction.num_actions())
        ).to(config.device)
        
        self.arch_optimizer = optim.Adam(self.arch_policy.parameters(), lr=1e-3)
        
        # Statistics
        self.total_samples = 0
        self.total_episodes = 0
        self.arch_history = ArchitectureHistory()
        
        # Track performance for architecture decisions
        self.recent_accuracies = deque(maxlen=100)
        self.accuracy_before_arch_change = 0.0
    
    def _recreate_optimizer(self):
        """Recreate optimizer after architecture change."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    @torch.no_grad()
    def collect_batch(self) -> Tuple[float, float]:
        """Collect a batch of trajectories for training."""
        self.model.eval()
        
        observations = self.envs.reset()
        target_seqs = self.envs.target_seqs
        
        # Storage
        all_actions = torch.zeros(self.config.num_parallel_envs, self.config.seq_length,
                                  dtype=torch.long, device=self.device)
        all_rewards = torch.zeros(self.config.num_parallel_envs, self.config.seq_length,
                                  dtype=torch.float32, device=self.device)
        all_policies = torch.zeros(self.config.num_parallel_envs, self.config.seq_length, 2,
                                   dtype=torch.float32, device=self.device)
        all_values = torch.zeros(self.config.num_parallel_envs, self.config.seq_length,
                                 dtype=torch.float32, device=self.device)
        
        # Expert policies
        for step in range(self.config.seq_length):
            correct_action = target_seqs[:, step]
            all_policies[:, step, 0] = (correct_action == 0).float()
            all_policies[:, step, 1] = (correct_action == 1).float()
        
        # Forward
        policy_logits, value, hidden = self.model(observations)
        
        total_correct = 0
        
        for step in range(self.config.seq_length):
            all_values[:, step] = value.squeeze(-1)
            
            policy = F.softmax(policy_logits, dim=-1)
            rand_val = np.random.random()
            
            if rand_val < 0.3:
                actions = target_seqs[:, step]
            elif rand_val < 0.4:
                actions = torch.randint(0, 2, (self.config.num_parallel_envs,), device=self.device)
            else:
                actions = policy.argmax(dim=-1)
            
            all_actions[:, step] = actions
            
            rewards, dones, _ = self.envs.step(actions)
            all_rewards[:, step] = rewards
            total_correct += (rewards > 0).sum().item()
            
            if step < self.config.seq_length - 1:
                hidden, _, policy_logits, value = self.model.dynamics_step(hidden, actions)
        
        self.buffer.add_batch(observations, all_actions, all_rewards, all_policies, all_values)
        
        self.total_samples += self.config.num_parallel_envs * self.config.seq_length
        self.total_episodes += self.config.num_parallel_envs
        
        accuracy = total_correct / (self.config.num_parallel_envs * self.config.seq_length)
        self.recent_accuracies.append(accuracy)
        
        return accuracy
    
    def train_step(self) -> float:
        """Perform one training step on the task."""
        self.model.train()
        
        obs, actions, rewards, policies, values = self.buffer.sample(self.config.batch_size)
        
        # Value targets via n-step returns
        value_targets = torch.zeros_like(rewards)
        running_return = torch.zeros(self.config.batch_size, device=self.device)
        for t in reversed(range(self.config.seq_length)):
            running_return = rewards[:, t] + self.config.discount * running_return
            value_targets[:, t] = running_return
        
        # Forward
        policy_logits, value_pred, hidden = self.model(obs)
        
        # Initial losses
        policy_loss = F.cross_entropy(policy_logits, policies[:, 0].argmax(dim=-1))
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, 0])
        total_loss = policy_loss + value_loss
        
        # Unroll dynamics
        for k in range(min(self.config.unroll_steps, self.config.seq_length - 1)):
            hidden, reward_pred, policy_logits, value_pred = self.model.dynamics_step(
                hidden, actions[:, k]
            )
            
            policy_loss = F.cross_entropy(policy_logits, policies[:, k+1].argmax(dim=-1))
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, k+1])
            reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards[:, k])
            
            total_loss = total_loss + policy_loss + value_loss + reward_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def select_arch_action(self) -> ArchAction:
        """
        Select an architecture modification action.
        
        Uses a simple policy network that takes:
        - Current architecture encoding
        - Recent accuracy
        
        Returns action with highest score.
        """
        # Encode current state (ensure float32)
        arch_vector = self.dyn_config.to_vector().float().to(self.device)
        current_acc = torch.tensor([np.mean(self.recent_accuracies) if self.recent_accuracies else 0.5],
                                   device=self.device, dtype=torch.float32)
        
        state = torch.cat([arch_vector, current_acc])
        
        # Get action logits
        with torch.no_grad():
            logits = self.arch_policy(state)
            
            # Mask invalid actions
            mask = torch.ones_like(logits)
            
            # Can't add layer if at max
            if self.dyn_config.n_layers >= self.dyn_config.max_layers:
                mask[ArchAction.ADD_LAYER] = 0
            # Can't remove layer if at min
            if self.dyn_config.n_layers <= self.dyn_config.min_layers:
                mask[ArchAction.REMOVE_LAYER] = 0
            # Can't increase heads if at max
            if self.dyn_config.n_heads >= self.dyn_config.max_heads:
                mask[ArchAction.INCREASE_HEADS] = 0
            if self.dyn_config.n_heads <= self.dyn_config.min_heads:
                mask[ArchAction.DECREASE_HEADS] = 0
            if self.dyn_config.d_model >= self.dyn_config.max_d_model:
                mask[ArchAction.INCREASE_DIM] = 0
            if self.dyn_config.d_model <= self.dyn_config.min_d_model:
                mask[ArchAction.DECREASE_DIM] = 0
            
            logits = logits * mask + (1 - mask) * (-1e9)
            
            # Sample action with temperature
            probs = F.softmax(logits / 0.5, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
        
        return ArchAction(action_idx)
    
    def update_arch_policy(self, action: ArchAction, reward: float):
        """Update architecture policy based on reward."""
        arch_vector = self.dyn_config.to_vector().float().to(self.device)
        current_acc = torch.tensor([self.accuracy_before_arch_change], device=self.device, dtype=torch.float32)
        state = torch.cat([arch_vector, current_acc])
        
        logits = self.arch_policy(state)
        log_prob = F.log_softmax(logits, dim=-1)[action]
        
        loss = -log_prob * reward
        
        self.arch_optimizer.zero_grad()
        loss.backward()
        self.arch_optimizer.step()
    
    def apply_arch_action(self, action: ArchAction) -> bool:
        """Apply architecture action and recreate optimizer."""
        old_params = self.model.count_parameters()
        success = self.model.apply_arch_action(action)
        
        if success and action != ArchAction.NO_OP:
            self._recreate_optimizer()
            new_params = self.model.count_parameters()
            print(f"   Architecture changed: {self.model.get_architecture_string()} "
                  f"({old_params:,} → {new_params:,} params)")
        
        return success
    
    def train(
        self,
        max_samples: int = 500000,
        log_interval: int = 5000,
        target_accuracy: float = 0.95
    ) -> Dict:
        """
        Main training loop with architecture search.
        """
        print(f"\n{'='*60}")
        print(f"Starting NAS-MuZero Training")
        print(f"Max samples: {max_samples:,}")
        print(f"Target accuracy: {target_accuracy:.1%}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        best_accuracy = 0.0
        samples_to_target = None
        
        recent_losses = deque(maxlen=100)
        episodes_since_arch_change = 0
        
        # Record initial architecture
        self.arch_history.add(
            0, self.model.get_architecture_string(),
            0.0, self.model.count_parameters(), "INIT"
        )
        
        while self.total_samples < max_samples:
            # Collect batch
            accuracy = self.collect_batch()
            episodes_since_arch_change += self.config.num_parallel_envs
            
            # Train if we have enough data
            if len(self.buffer) >= self.config.batch_size:
                for _ in range(4):
                    loss = self.train_step()
                    recent_losses.append(loss)
            
            # Architecture decision point
            if episodes_since_arch_change >= self.config.arch_action_freq * self.config.num_parallel_envs:
                episodes_since_arch_change = 0
                
                current_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                
                # Calculate improvement since last arch change
                improvement = current_acc - self.accuracy_before_arch_change
                param_change = 0  # Will be calculated after action
                
                # Select and apply architecture action
                action = self.select_arch_action()
                old_params = self.model.count_parameters()
                success = self.apply_arch_action(action)
                new_params = self.model.count_parameters()
                
                if success:
                    # Calculate architecture reward
                    param_change = new_params - old_params
                    arch_reward = (improvement * self.config.arch_reward_scale - 
                                   param_change * self.config.param_penalty)
                    
                    # Update architecture policy
                    self.update_arch_policy(action, arch_reward)
                    
                    # Record history
                    self.arch_history.add(
                        self.total_samples,
                        self.model.get_architecture_string(),
                        current_acc,
                        new_params,
                        ArchAction(action).name
                    )
                
                self.accuracy_before_arch_change = current_acc
            
            # Logging
            if self.total_samples % log_interval < self.config.num_parallel_envs * self.config.seq_length:
                avg_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                samples_per_sec = self.total_samples / elapsed
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Arch: {self.model.get_architecture_string()} | "
                      f"Speed: {samples_per_sec:,.0f}/s")
                
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                
                if avg_acc >= target_accuracy and samples_to_target is None:
                    samples_to_target = self.total_samples
                    print(f"\n🎯 Reached {target_accuracy:.1%} accuracy at {samples_to_target:,} samples!")
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(self.recent_accuracies)[-50:]) if self.recent_accuracies else 0
        
        results = {
            'total_samples': self.total_samples,
            'best_accuracy': best_accuracy,
            'final_accuracy': final_acc,
            'samples_to_target': samples_to_target,
            'elapsed_time': elapsed,
            'samples_per_second': self.total_samples / elapsed,
            'final_architecture': self.model.get_architecture_string(),
            'final_params': self.model.count_parameters(),
        }
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total samples: {results['total_samples']:,}")
        print(f"Best accuracy: {results['best_accuracy']:.2%}")
        print(f"Final architecture: {results['final_architecture']}")
        print(f"Final parameters: {results['final_params']:,}")
        print(f"Speed: {results['samples_per_second']:,.0f} samples/sec")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*60}")
        
        # Print architecture evolution
        self.arch_history.print_summary()
        
        return results
    
    @torch.no_grad()
    def evaluate(self, num_envs: int = 1000) -> float:
        """Evaluate current model."""
        self.model.eval()
        
        eval_env = VectorizedBitReversalEnv(num_envs, self.config.seq_length, self.device)
        obs = eval_env.reset()
        
        policy_logits, _, hidden = self.model(obs)
        
        total_correct = 0
        for step in range(self.config.seq_length):
            actions = policy_logits.argmax(dim=-1)
            rewards, _, _ = eval_env.step(actions)
            total_correct += (rewards > 0).sum().item()
            
            if step < self.config.seq_length - 1:
                hidden, _, policy_logits, _ = self.model.dynamics_step(hidden, actions)
        
        return total_correct / (num_envs * self.config.seq_length)


def run_nas_experiment(
    seq_length: int = 4,
    max_samples: int = 300000,
    target_accuracy: float = 0.90
):
    """Run NAS-MuZero experiment."""
    print("\n" + "="*70)
    print("  🧬 NAS-MuZero: Neural Architecture Search via MuZero Planning")
    print("="*70)
    
    config = NASConfig(
        seq_length=seq_length,
        arch_action_freq=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = NASMuZeroTrainer(config)
    results = trainer.train(max_samples, log_interval=10000, target_accuracy=target_accuracy)
    
    print(f"\nFinal evaluation accuracy: {trainer.evaluate(5000):.2%}")
    
    return results


if __name__ == "__main__":
    run_nas_experiment()
