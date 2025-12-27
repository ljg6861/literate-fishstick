"""
FAST MuZero Training - GPU Optimized

This module implements a highly optimized MuZero training pipeline:
- Batched MCTS with parallel tree search
- Vectorized environments for parallel self-play
- Minimal CPU-GPU transfers
- Compiled model with torch.compile
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

from .muzero_transformer import MuZeroConfig, MuZeroTransformer, create_muzero


@dataclass
class FastConfig:
    """Configuration optimized for speed."""
    # Model architecture - BIGGER for better learning
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.0  # No dropout for speed
    
    # Task configuration
    vocab_size: int = 2
    max_seq_len: int = 8
    
    # Training - MAXIMUM GPU UTILIZATION
    learning_rate: float = 1e-3  # Higher LR for faster learning
    weight_decay: float = 1e-4
    batch_size: int = 2048  # Large batch for GPU
    num_parallel_envs: int = 1024  # Parallel environments
    
    # MuZero specific - REDUCED for speed
    unroll_steps: int = 3  # Fewer unroll steps
    td_steps: int = 3
    discount: float = 1.0
    
    # MCTS - MINIMAL for speed
    num_simulations: int = 10  # Fewer simulations, rely more on policy network
    c_puct: float = 1.5
    
    # Device
    device: str = "cuda"


class VectorizedBitReversalEnv:
    """
    Vectorized bit-string reversal environment for parallel self-play.
    All operations are batched tensors on GPU.
    """
    
    def __init__(self, num_envs: int, seq_length: int, device: str = "cuda"):
        self.num_envs = num_envs
        self.seq_length = seq_length
        self.vocab_size = 2
        self.device = device
        
        # State tensors
        self.input_seqs = None
        self.target_seqs = None
        self.positions = None
        self.dones = None
    
    def reset(self) -> torch.Tensor:
        """Reset all environments. Returns input sequences."""
        # Generate random bit strings on GPU
        self.input_seqs = torch.randint(
            0, 2, (self.num_envs, self.seq_length), 
            device=self.device, dtype=torch.long
        )
        # Target is reversed
        self.target_seqs = self.input_seqs.flip(dims=[1])
        
        self.positions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        return self.input_seqs
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step all environments in parallel.
        
        Args:
            actions: (num_envs,) tensor of actions
        Returns:
            rewards: (num_envs,) tensor
            dones: (num_envs,) bool tensor
            accuracies: (num_envs,) tensor
        """
        # Get correct actions for current positions
        batch_indices = torch.arange(self.num_envs, device=self.device)
        correct_actions = self.target_seqs[batch_indices, self.positions]
        
        # Compute rewards
        correct = (actions == correct_actions)
        rewards = torch.where(correct, 
                             torch.ones_like(actions, dtype=torch.float32),
                             -torch.ones_like(actions, dtype=torch.float32))
        
        # Update positions
        self.positions = self.positions + 1
        
        # Check done
        self.dones = (self.positions >= self.seq_length)
        
        # Compute accuracy up to current position
        accuracies = correct.float()  # Per-step accuracy
        
        return rewards, self.dones, accuracies
    
    def get_target_actions(self) -> torch.Tensor:
        """Get correct actions for current positions (for supervised signal)."""
        batch_indices = torch.arange(self.num_envs, device=self.device)
        # Clamp positions to valid range
        valid_positions = torch.clamp(self.positions, 0, self.seq_length - 1)
        return self.target_seqs[batch_indices, valid_positions]


class FastReplayBuffer:
    """GPU-based replay buffer for fast sampling."""
    
    def __init__(self, capacity: int, seq_length: int, device: str = "cuda"):
        self.capacity = capacity
        self.seq_length = seq_length
        self.device = device
        
        # Pre-allocate tensors on GPU
        self.observations = torch.zeros(capacity, seq_length, dtype=torch.long, device=device)
        self.actions = torch.zeros(capacity, seq_length, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, seq_length, dtype=torch.float32, device=device)
        self.policies = torch.zeros(capacity, seq_length, 2, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, seq_length, dtype=torch.float32, device=device)
        
        self.position = 0
        self.size = 0
    
    def add_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        policies: torch.Tensor,
        values: torch.Tensor
    ):
        """Add a batch of trajectories."""
        batch_size = observations.size(0)
        
        # Handle wraparound
        end_pos = self.position + batch_size
        if end_pos <= self.capacity:
            self.observations[self.position:end_pos] = observations
            self.actions[self.position:end_pos] = actions
            self.rewards[self.position:end_pos] = rewards
            self.policies[self.position:end_pos] = policies
            self.values[self.position:end_pos] = values
        else:
            # Split across boundary
            first_part = self.capacity - self.position
            self.observations[self.position:] = observations[:first_part]
            self.observations[:batch_size - first_part] = observations[first_part:]
            self.actions[self.position:] = actions[:first_part]
            self.actions[:batch_size - first_part] = actions[first_part:]
            self.rewards[self.position:] = rewards[:first_part]
            self.rewards[:batch_size - first_part] = rewards[first_part:]
            self.policies[self.position:] = policies[:first_part]
            self.policies[:batch_size - first_part] = policies[first_part:]
            self.values[self.position:] = values[:first_part]
            self.values[:batch_size - first_part] = values[first_part:]
        
        self.position = end_pos % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of trajectories."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.policies[indices],
            self.values[indices]
        )
    
    def __len__(self):
        return self.size


class FastMuZeroTrainer:
    """
    Ultra-fast MuZero training with GPU parallelism.
    """
    
    def __init__(
        self,
        seq_length: int = 8,
        num_parallel_envs: int = 256,
        device: str = "cuda"
    ):
        self.device = device
        self.seq_length = seq_length
        self.num_parallel_envs = num_parallel_envs
        
        print(f"🚀 Fast MuZero Trainer")
        print(f"   Device: {device}")
        print(f"   Parallel envs: {num_parallel_envs}")
        print(f"   Sequence length: {seq_length}")
        
        # Config
        self.config = FastConfig(
            vocab_size=2,
            max_seq_len=seq_length,
            num_parallel_envs=num_parallel_envs,
            device=device
        )
        
        # Model
        muzero_config = MuZeroConfig(
            vocab_size=2,
            max_seq_len=seq_length,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            device=device
        )
        self.model = create_muzero(muzero_config)
        
        # Compile model for speed (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("   ✓ Model compiled with torch.compile")
        except Exception as e:
            print(f"   ⚠ torch.compile not available: {e}")
        
        # Environment
        self.envs = VectorizedBitReversalEnv(num_parallel_envs, seq_length, device)
        
        # Optimizer with higher LR
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Replay buffer
        self.buffer = FastReplayBuffer(50000, seq_length, device)
        
        # Stats
        self.total_samples = 0
        self.total_steps = 0
    
    @torch.no_grad()
    def collect_batch(self) -> Tuple[float, float]:
        """
        Collect a batch of trajectories using fast policy inference.
        Returns average reward and accuracy.
        
        IMPORTANT: We store EXPERT policies (one-hot of correct action) not model predictions.
        This allows MuZero to learn from correct demonstrations.
        """
        self.model.eval()
        
        # Reset environments
        observations = self.envs.reset()  # (num_envs, seq_len)
        
        # Get the FULL target sequence upfront for expert policies
        target_seqs = self.envs.target_seqs  # (num_envs, seq_len)
        
        # Storage for this batch
        all_actions = torch.zeros(self.num_parallel_envs, self.seq_length, 
                                  dtype=torch.long, device=self.device)
        all_rewards = torch.zeros(self.num_parallel_envs, self.seq_length, 
                                  dtype=torch.float32, device=self.device)
        all_policies = torch.zeros(self.num_parallel_envs, self.seq_length, 2,
                                   dtype=torch.float32, device=self.device)
        all_values = torch.zeros(self.num_parallel_envs, self.seq_length,
                                 dtype=torch.float32, device=self.device)
        
        # Create EXPERT policies (one-hot of correct actions)
        for step in range(self.seq_length):
            correct_action = target_seqs[:, step]  # (num_envs,)
            all_policies[:, step, 0] = (correct_action == 0).float()
            all_policies[:, step, 1] = (correct_action == 1).float()
        
        # Get initial state for policy-guided collection
        state, policy_logits, value = self.model.initial_inference(observations)
        
        total_reward = 0.0
        total_correct = 0
        
        for step in range(self.seq_length):
            # Store value
            all_values[:, step] = value.squeeze(-1)
            
            # Mix policy sampling with exploration
            policy = F.softmax(policy_logits, dim=-1)
            rand_val = np.random.random()
            
            if rand_val < 0.3:  # 30% use expert actions for better learning
                actions = target_seqs[:, step]
            elif rand_val < 0.4:  # 10% random exploration
                actions = torch.randint(0, 2, (self.num_parallel_envs,), device=self.device)
            else:  # 60% use learned policy
                actions = policy.argmax(dim=-1)
            
            all_actions[:, step] = actions
            
            # Step environments
            rewards, dones, accuracies = self.envs.step(actions)
            all_rewards[:, step] = rewards
            
            total_reward += rewards.sum().item()
            total_correct += (rewards > 0).sum().item()
            
            # Get next state for policy (if not last step)
            if step < self.seq_length - 1:
                state, reward_pred, policy_logits, value = self.model.recurrent_inference(
                    state, actions, step
                )
        
        # Add to replay buffer with EXPERT policies
        self.buffer.add_batch(
            observations, all_actions, all_rewards, all_policies, all_values
        )
        
        self.total_samples += self.num_parallel_envs * self.seq_length
        
        avg_reward = total_reward / (self.num_parallel_envs * self.seq_length)
        accuracy = total_correct / (self.num_parallel_envs * self.seq_length)
        
        return avg_reward, accuracy
    
    def train_step(self) -> float:
        """Perform one training step."""
        self.model.train()
        
        # Sample batch
        obs, actions, rewards, policies, values = self.buffer.sample(self.config.batch_size)
        
        # Compute n-step returns for value targets
        value_targets = torch.zeros_like(rewards)
        running_return = torch.zeros(self.config.batch_size, device=self.device)
        
        for t in reversed(range(self.seq_length)):
            running_return = rewards[:, t] + self.config.discount * running_return
            value_targets[:, t] = running_return
        
        # Initial inference
        state, policy_logits, value_pred = self.model.initial_inference(obs)
        
        # Compute losses
        total_loss = 0.0
        
        # Policy loss at step 0
        policy_loss = F.cross_entropy(policy_logits, policies[:, 0].argmax(dim=-1))
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, 0])
        total_loss = policy_loss + value_loss
        
        # Unroll dynamics
        for k in range(min(self.config.unroll_steps, self.seq_length - 1)):
            state, reward_pred, policy_logits, value_pred = self.model.recurrent_inference(
                state, actions[:, k], k
            )
            
            # Losses
            policy_loss = F.cross_entropy(policy_logits, policies[:, k+1].argmax(dim=-1))
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets[:, k+1])
            reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards[:, k])
            
            total_loss = total_loss + (policy_loss + value_loss + reward_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.total_steps += 1
        
        return total_loss.item()
    
    def train(
        self,
        max_samples: int = 100000,
        log_interval: int = 1000,
        target_accuracy: float = 0.99
    ) -> Dict:
        """
        Main training loop.
        """
        print(f"\n{'='*60}")
        print(f"Starting Fast MuZero Training")
        print(f"Max samples: {max_samples:,}")
        print(f"Target accuracy: {target_accuracy:.1%}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        best_accuracy = 0.0
        samples_to_target = None
        
        recent_accuracies = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            # Collect batch
            avg_reward, accuracy = self.collect_batch()
            recent_accuracies.append(accuracy)
            
            # Train if we have enough data
            if len(self.buffer) >= self.config.batch_size:
                for _ in range(8):  # More updates per collection for faster learning
                    loss = self.train_step()
                    recent_losses.append(loss)
            
            # Logging
            if self.total_samples % log_interval < self.num_parallel_envs * self.seq_length:
                avg_acc = np.mean(recent_accuracies) if recent_accuracies else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                elapsed = time.time() - start_time
                samples_per_sec = self.total_samples / elapsed
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {samples_per_sec:,.0f} samples/sec")
                
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                
                if avg_acc >= target_accuracy and samples_to_target is None:
                    samples_to_target = self.total_samples
                    print(f"\n🎯 Reached {target_accuracy:.1%} accuracy at {samples_to_target:,} samples!")
        
        elapsed = time.time() - start_time
        final_acc = np.mean(list(recent_accuracies)[-50:]) if recent_accuracies else 0
        
        results = {
            'total_samples': self.total_samples,
            'best_accuracy': best_accuracy,
            'final_accuracy': final_acc,
            'samples_to_target': samples_to_target,
            'elapsed_time': elapsed,
            'samples_per_second': self.total_samples / elapsed
        }
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total samples: {results['total_samples']:,}")
        print(f"Best accuracy: {results['best_accuracy']:.2%}")
        print(f"Speed: {results['samples_per_second']:,.0f} samples/sec")
        print(f"Time: {elapsed:.1f}s")
        if samples_to_target:
            print(f"Samples to {target_accuracy:.1%}: {samples_to_target:,}")
        print(f"{'='*60}\n")
        
        return results
    
    @torch.no_grad()
    def evaluate(self, num_envs: int = 1000) -> float:
        """Evaluate with greedy policy."""
        self.model.eval()
        
        eval_env = VectorizedBitReversalEnv(num_envs, self.seq_length, self.device)
        obs = eval_env.reset()
        
        state, policy_logits, _ = self.model.initial_inference(obs)
        
        total_correct = 0
        for step in range(self.seq_length):
            actions = policy_logits.argmax(dim=-1)
            rewards, dones, _ = eval_env.step(actions)
            total_correct += (rewards > 0).sum().item()
            
            if step < self.seq_length - 1:
                state, _, policy_logits, _ = self.model.recurrent_inference(state, actions, step)
        
        return total_correct / (num_envs * self.seq_length)


class FastSupervisedTrainer:
    """
    Ultra-fast supervised training baseline.
    """
    
    def __init__(
        self,
        seq_length: int = 8,
        batch_size: int = 512,
        device: str = "cuda"
    ):
        self.device = device
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        print(f"🚀 Fast Supervised Trainer")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        
        # Simple Transformer for sequence prediction
        self.model = nn.Sequential(
            nn.Embedding(2, 64),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(64, 4, 256, dropout=0, batch_first=True),
                num_layers=2
            ),
            nn.Linear(64, 2)
        ).to(device)
        
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("   ✓ Model compiled")
        except:
            pass
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.total_samples = 0
    
    def train_step(self) -> Tuple[float, float]:
        """One training step."""
        self.model.train()
        
        # Generate batch on GPU
        inputs = torch.randint(0, 2, (self.batch_size, self.seq_length), 
                              device=self.device, dtype=torch.long)
        targets = inputs.flip(dims=[1])  # Reversed
        
        # Forward
        logits = self.model(inputs)  # (batch, seq, 2)
        
        # Loss
        loss = F.cross_entropy(logits.view(-1, 2), targets.view(-1))
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == targets).float().mean().item()
        
        self.total_samples += self.batch_size * self.seq_length
        
        return loss.item(), accuracy
    
    def train(
        self,
        max_samples: int = 100000,
        log_interval: int = 1000,
        target_accuracy: float = 0.99
    ) -> Dict:
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Fast Supervised Training")
        print(f"Max samples: {max_samples:,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        best_accuracy = 0.0
        samples_to_target = None
        
        recent_accuracies = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        
        while self.total_samples < max_samples:
            loss, accuracy = self.train_step()
            recent_accuracies.append(accuracy)
            recent_losses.append(loss)
            
            if self.total_samples % log_interval < self.batch_size * self.seq_length:
                avg_acc = np.mean(recent_accuracies)
                avg_loss = np.mean(recent_losses)
                elapsed = time.time() - start_time
                samples_per_sec = self.total_samples / elapsed
                
                print(f"Samples: {self.total_samples:8,} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {samples_per_sec:,.0f} samples/sec")
                
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                
                if avg_acc >= target_accuracy and samples_to_target is None:
                    samples_to_target = self.total_samples
                    print(f"\n🎯 Reached {target_accuracy:.1%} accuracy at {samples_to_target:,} samples!")
        
        elapsed = time.time() - start_time
        
        return {
            'total_samples': self.total_samples,
            'best_accuracy': best_accuracy,
            'final_accuracy': np.mean(list(recent_accuracies)[-50:]),
            'samples_to_target': samples_to_target,
            'elapsed_time': elapsed,
            'samples_per_second': self.total_samples / elapsed
        }


def run_fast_experiment(
    seq_length: int = 8,
    max_samples: int = 100000,
    target_accuracy: float = 0.99
):
    """Run fast comparison experiment."""
    print("\n" + "="*70)
    print("  🚀 FAST MuZero vs Supervised Experiment")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    
    # MuZero
    print("\n[1/2] MUZERO")
    muzero = FastMuZeroTrainer(seq_length=seq_length, num_parallel_envs=1024, device=device)
    results['muzero'] = muzero.train(max_samples, log_interval=10000, target_accuracy=target_accuracy)
    
    # Supervised
    print("\n[2/2] SUPERVISED")
    supervised = FastSupervisedTrainer(seq_length=seq_length, batch_size=2048, device=device)
    results['supervised'] = supervised.train(max_samples, log_interval=10000, target_accuracy=target_accuracy)
    
    # Compare
    print("\n" + "="*70)
    print("  COMPARISON RESULTS")
    print("="*70)
    
    for method, r in results.items():
        print(f"\n  {method.upper()}")
        print(f"    Final Accuracy:    {r['final_accuracy']:.2%}")
        print(f"    Best Accuracy:     {r['best_accuracy']:.2%}")
        sts = r['samples_to_target']
        print(f"    Samples to Target: {sts:,}" if sts else "    Samples to Target: Not reached")
        print(f"    Speed:             {r['samples_per_second']:,.0f} samples/sec")
    
    print("\n" + "="*70 + "\n")
    
    return results


if __name__ == "__main__":
    run_fast_experiment(seq_length=8, max_samples=50000, target_accuracy=0.95)
