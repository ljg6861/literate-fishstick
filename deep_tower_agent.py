"""
Deep Tower Agent - DQN-based reinforcement learning agent.

This replaces the tabular Q-learning agent with a neural network
that can generalize across continuous state spaces.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
from typing import Optional, Tuple

from dqn_model import DuelingDQN, PrioritizedReplayBuffer


class DeepTowerAgent:
    """
    Deep Q-Network agent for tower building.
    
    Key improvements over tabular Q-learning:
    - Continuous state representation (no bucketization)
    - Neural network generalizes to unseen states
    - Experience replay for stable training
    - Target network for stable Q-value targets
    - Dueling architecture for better value estimation
    """
    
    def __init__(self, screen_width: int = 800, device: str = "cpu"):
        self.device = torch.device(device)
        self.screen_width = screen_width
        
        # State: [rel_x, angle, height, next_shape]
        # Continuous values, normalized
        self.state_dim = 4
        
        # Actions: (X position, rotation angle) pairs
        # X positions: 100 to 700 in steps of 50 (13 positions)
        # Rotations: -30°, -15°, 0°, 15°, 30° (5 angles)
        margin = 100
        self.x_positions = list(range(margin, screen_width - margin + 1, 50))
        self.rotations = [-0.52, -0.26, 0.0, 0.26, 0.52]  # radians: ~-30°, -15°, 0°, 15°, 30°
        
        # Build action list as (x, rotation) tuples
        self.actions = []
        for x in self.x_positions:
            for rot in self.rotations:
                self.actions.append((x, rot))
        
        self.action_dim = len(self.actions)  # 13 * 5 = 65 actions
        
        # Networks
        self.policy_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        self.batch_size = 64
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.02
        
        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.tau = 0.005   # Soft update rate for target network
        
        # Training stats
        self.training_steps = 0
        self.losses = []
        
    def get_state_vector(self, top_block_x: Optional[float], 
                         top_block_angle: float,
                         center_x: float, 
                         height: int = 0,
                         next_block_shape: int = 0) -> np.ndarray:
        """
        Convert game state to continuous feature vector.
        
        Unlike tabular Q-learning, we don't discretize - we normalize
        to [-1, 1] range for neural network input.
        """
        if top_block_x is None:
            # Initial state: centered, no angle, ground level
            return np.array([0.0, 0.0, 0.0, next_block_shape / 3.0], dtype=np.float32)
        
        # Relative X position normalized to [-1, 1]
        rel_x = (top_block_x - center_x) / (self.screen_width / 2)
        rel_x = np.clip(rel_x, -1.0, 1.0)
        
        # Angle normalized to [-1, 1] (assuming max +/- 45 degrees)
        angle_norm = math.degrees(top_block_angle) / 45.0
        angle_norm = np.clip(angle_norm, -1.0, 1.0)
        
        # Height normalized (log scale for better representation)
        height_norm = np.log1p(height) / 5.0  # log(150) ≈ 5
        height_norm = np.clip(height_norm, 0.0, 1.0)
        
        # Shape type normalized to [0, 1]
        shape_norm = next_block_shape / 3.0
        
        return np.array([rel_x, angle_norm, height_norm, shape_norm], dtype=np.float32)
    
    def choose_action(self, state: np.ndarray) -> tuple:
        """
        Select action using epsilon-greedy policy.
        
        Returns (x_position, rotation_angle) tuple for block placement.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            
        return self.actions[action_idx]
    
    def store_transition(self, state: np.ndarray, action: tuple, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        action_idx = self.actions.index(action)
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.
        
        Returns the loss value or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch with priorities
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # TD-error for priority updates
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Weighted MSE loss
        loss = (weights * (current_q - target_q) ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target()
        
        self.training_steps += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def _soft_update_target(self):
        """Soft update target network: θ' = τ*θ + (1-τ)*θ'"""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + 
                                   (1 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
        return {
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "training_steps": self.training_steps,
            "avg_loss": avg_loss
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
