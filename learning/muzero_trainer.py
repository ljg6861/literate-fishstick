"""
MuZero Training Loop

This module implements the complete MuZero training pipeline:
- Self-play with MCTS
- Replay buffer for trajectory storage
- Training with MuZero loss function
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
import random

from .muzero_transformer import MuZeroConfig, MuZeroTransformer, create_muzero
from .mcts_muzero import MCTSMuZero, MCTSConfig
from .seq2seq_envs import create_env


@dataclass
class Trajectory:
    """A single trajectory from self-play."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    action_probs: List[np.ndarray] = field(default_factory=list)
    root_values: List[float] = field(default_factory=list)
    
    def __len__(self):
        return len(self.observations)


class ReplayBuffer:
    """
    Replay buffer for storing trajectories.
    
    Samples positions from trajectories for training.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.total_samples = 0
    
    def add(self, trajectory: Trajectory):
        """Add a trajectory to the buffer."""
        self.buffer.append(trajectory)
        self.total_samples += len(trajectory)
    
    def sample(self, batch_size: int, unroll_steps: int, td_steps: int) -> Dict:
        """
        Sample a batch of positions with unrolled targets.
        
        Args:
            batch_size: Number of positions to sample
            unroll_steps: Number of steps to unroll for training
            td_steps: Number of steps for TD target computation
        Returns:
            Dictionary with training data
        """
        if len(self.buffer) == 0:
            return None
        
        batch = {
            'observations': [],
            'actions': [],
            'target_values': [],
            'target_rewards': [],
            'target_policies': [],
        }
        
        for _ in range(batch_size):
            # Sample a trajectory
            trajectory = random.choice(self.buffer)
            
            # Sample a position in the trajectory
            max_pos = len(trajectory) - unroll_steps - 1
            if max_pos <= 0:
                pos = 0
            else:
                pos = random.randint(0, max_pos)
            
            # Get observation at position
            batch['observations'].append(trajectory.observations[0])  # Initial observation
            
            # Unroll actions, rewards, policies, values
            actions = []
            target_values = []
            target_rewards = []
            target_policies = []
            
            for k in range(unroll_steps + 1):
                step_idx = pos + k
                
                if step_idx < len(trajectory):
                    # Get action (for k > 0)
                    if k > 0 and step_idx - 1 < len(trajectory.actions):
                        actions.append(trajectory.actions[step_idx - 1])
                    elif k > 0:
                        actions.append(0)  # Padding
                    
                    # Get policy target
                    if step_idx < len(trajectory.action_probs):
                        target_policies.append(trajectory.action_probs[step_idx])
                    else:
                        target_policies.append(np.ones(2) / 2)  # Uniform
                    
                    # Get reward target
                    if step_idx < len(trajectory.rewards):
                        target_rewards.append(trajectory.rewards[step_idx])
                    else:
                        target_rewards.append(0.0)
                    
                    # Compute value target (n-step return)
                    value = 0.0
                    discount = 1.0
                    for n in range(td_steps):
                        if step_idx + n < len(trajectory.rewards):
                            value += discount * trajectory.rewards[step_idx + n]
                            discount *= 0.99  # Discount factor
                    
                    # Bootstrap with root value
                    if step_idx + td_steps < len(trajectory.root_values):
                        value += discount * trajectory.root_values[step_idx + td_steps]
                    
                    target_values.append(value)
                else:
                    # Padding for positions beyond trajectory
                    if k > 0:
                        actions.append(0)
                    target_policies.append(np.ones(2) / 2)
                    target_rewards.append(0.0)
                    target_values.append(0.0)
            
            batch['actions'].append(actions)
            batch['target_values'].append(target_values)
            batch['target_rewards'].append(target_rewards)
            batch['target_policies'].append(target_policies)
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


@dataclass
class TrainingStats:
    """Training statistics."""
    total_samples: int = 0
    total_episodes: int = 0
    total_steps: int = 0
    
    recent_losses: List[float] = field(default_factory=list)
    recent_accuracies: List[float] = field(default_factory=list)
    recent_rewards: List[float] = field(default_factory=list)
    
    def avg_loss(self) -> float:
        return np.mean(self.recent_losses[-100:]) if self.recent_losses else 0.0
    
    def avg_accuracy(self) -> float:
        return np.mean(self.recent_accuracies[-100:]) if self.recent_accuracies else 0.0
    
    def avg_reward(self) -> float:
        return np.mean(self.recent_rewards[-100:]) if self.recent_rewards else 0.0


class MuZeroTrainer:
    """
    Complete MuZero training pipeline.
    
    Alternates between:
    1. Self-play: Generate trajectories with MCTS
    2. Training: Update networks with MuZero loss
    """
    
    def __init__(
        self,
        task: str = "reversal",
        seq_length: int = 8,
        device: str = None
    ):
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Using device: {self.device}")
        
        # Environment
        self.env = create_env(task, seq_length)
        
        # Model configuration
        self.config = MuZeroConfig(
            vocab_size=self.env.vocab_size,
            max_seq_len=seq_length,
            device=device
        )
        
        # Create model
        self.model = create_muzero(self.config)
        
        # MCTS
        self.mcts_config = MCTSConfig(num_simulations=50)
        self.mcts = MCTSMuZero(self.model, self.mcts_config)
        
        # Training
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Statistics
        self.stats = TrainingStats()
    
    def self_play_episode(self) -> Trajectory:
        """
        Generate one episode using MCTS.
        
        Returns:
            Trajectory containing all states, actions, rewards, policies, values
        """
        trajectory = Trajectory()
        
        obs = self.env.reset()
        trajectory.observations.append(obs.copy())
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.long).to(self.device)
            
            # Run MCTS
            action_probs, root_value = self.mcts.search(obs_tensor, step)
            
            # Store MCTS results
            trajectory.action_probs.append(action_probs)
            trajectory.root_values.append(root_value)
            
            # Select action
            action = self.mcts.select_action(action_probs, deterministic=False)
            trajectory.actions.append(action)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            trajectory.rewards.append(reward)
            episode_reward += reward
            
            if not done:
                trajectory.observations.append(obs.copy())
            
            step += 1
        
        return trajectory, episode_reward, info.get('accuracy', 0)
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns:
            Loss value
        """
        # Sample from replay buffer
        batch = self.replay_buffer.sample(
            batch_size=self.config.batch_size,
            unroll_steps=self.config.unroll_steps,
            td_steps=self.config.td_steps
        )
        
        if batch is None:
            return 0.0
        
        # Prepare tensors
        observations = torch.tensor(
            np.array(batch['observations']), 
            dtype=torch.long
        ).to(self.device)
        
        actions = torch.tensor(
            np.array(batch['actions']), 
            dtype=torch.long
        ).to(self.device)
        
        target_values = torch.tensor(
            np.array(batch['target_values']), 
            dtype=torch.float32
        ).to(self.device)
        
        target_rewards = torch.tensor(
            np.array(batch['target_rewards']), 
            dtype=torch.float32
        ).to(self.device)
        
        target_policies = torch.tensor(
            np.array(batch['target_policies']), 
            dtype=torch.float32
        ).to(self.device)
        
        # Initial inference
        state, policy_logits, value = self.model.initial_inference(observations)
        
        # Compute initial losses
        policy_loss = F.cross_entropy(
            policy_logits, 
            target_policies[:, 0],
            reduction='mean'
        )
        value_loss = F.mse_loss(value.squeeze(-1), target_values[:, 0])
        
        total_loss = policy_loss + value_loss
        
        # Unroll dynamics and accumulate losses
        for k in range(1, self.config.unroll_steps + 1):
            # Get actions for this step
            action = actions[:, k - 1]
            
            # Recurrent inference
            state, reward, policy_logits, value = self.model.recurrent_inference(
                state, action, k
            )
            
            # Accumulate losses
            policy_loss = F.cross_entropy(
                policy_logits, 
                target_policies[:, k],
                reduction='mean'
            )
            value_loss = F.mse_loss(value.squeeze(-1), target_values[:, k])
            reward_loss = F.mse_loss(reward.squeeze(-1), target_rewards[:, k])
            
            # Scale losses for later steps (as in MuZero paper)
            scale = 1.0 / self.config.unroll_steps
            total_loss += scale * (policy_loss + value_loss + reward_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(
        self,
        max_samples: int = 10000,
        eval_interval: int = 100,
        target_accuracy: float = 0.99
    ) -> Dict:
        """
        Main training loop.
        
        Args:
            max_samples: Maximum number of samples to train on
            eval_interval: How often to evaluate (in episodes)
            target_accuracy: Stop when this accuracy is reached
        Returns:
            Training results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting MuZero Training")
        print(f"Task: {self.env.__class__.__name__}")
        print(f"Sequence length: {self.env.seq_length}")
        print(f"Max samples: {max_samples}")
        print(f"Target accuracy: {target_accuracy:.1%}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        best_accuracy = 0.0
        samples_to_target = None
        
        while self.stats.total_samples < max_samples:
            # Self-play
            trajectory, episode_reward, accuracy = self.self_play_episode()
            self.replay_buffer.add(trajectory)
            
            # Update stats
            self.stats.total_episodes += 1
            self.stats.total_samples += len(trajectory)
            self.stats.recent_rewards.append(episode_reward)
            self.stats.recent_accuracies.append(accuracy)
            
            # Training steps
            if len(self.replay_buffer) >= self.config.batch_size:
                for _ in range(4):  # Multiple training steps per episode
                    loss = self.train_step()
                    self.stats.recent_losses.append(loss)
                    self.stats.total_steps += 1
            
            # Logging
            if self.stats.total_episodes % eval_interval == 0:
                avg_acc = self.stats.avg_accuracy()
                avg_rew = self.stats.avg_reward()
                avg_loss = self.stats.avg_loss()
                
                print(f"Episode {self.stats.total_episodes:5d} | "
                      f"Samples: {self.stats.total_samples:6d} | "
                      f"Acc: {avg_acc:.2%} | "
                      f"Reward: {avg_rew:+.2f} | "
                      f"Loss: {avg_loss:.4f}")
                
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                
                # Check if target reached
                if avg_acc >= target_accuracy and samples_to_target is None:
                    samples_to_target = self.stats.total_samples
                    print(f"\n🎯 Reached {target_accuracy:.1%} accuracy at {samples_to_target} samples!")
        
        elapsed = time.time() - start_time
        
        results = {
            'total_samples': self.stats.total_samples,
            'total_episodes': self.stats.total_episodes,
            'total_steps': self.stats.total_steps,
            'best_accuracy': best_accuracy,
            'final_accuracy': self.stats.avg_accuracy(),
            'samples_to_target': samples_to_target,
            'elapsed_time': elapsed
        }
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total samples: {results['total_samples']}")
        print(f"Best accuracy: {results['best_accuracy']:.2%}")
        print(f"Time: {elapsed:.1f}s")
        if samples_to_target:
            print(f"Samples to {target_accuracy:.1%}: {samples_to_target}")
        print(f"{'='*60}\n")
        
        return results
    
    def evaluate(self, num_episodes: int = 100) -> float:
        """
        Evaluate the model without exploration.
        
        Args:
            num_episodes: Number of episodes to evaluate
        Returns:
            Average accuracy
        """
        self.model.eval()
        self.mcts_config.temperature = 0.0  # Deterministic
        
        accuracies = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            step = 0
            
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.long).to(self.device)
                action_probs, _ = self.mcts.search(obs_tensor, step)
                action = self.mcts.select_action(action_probs, deterministic=True)
                obs, _, done, info = self.env.step(action)
                step += 1
            
            accuracies.append(info.get('accuracy', 0))
        
        self.mcts_config.temperature = 1.0  # Restore
        self.model.train()
        
        return np.mean(accuracies)
    
    def save(self, path: str):
        """Save model and training state."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.stats = checkpoint['stats']


# Test code
if __name__ == "__main__":
    print("Testing MuZero Trainer...")
    
    trainer = MuZeroTrainer(
        task="reversal",
        seq_length=4,  # Start small for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Quick training test
    results = trainer.train(max_samples=500, eval_interval=10)
    
    print(f"\nFinal evaluation: {trainer.evaluate(100):.2%}")
