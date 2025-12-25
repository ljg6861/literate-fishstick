"""
Deep Q-Network (DQN) components for Tower Builder AI.

This module implements:
- DQN: Neural network for Q-value approximation
- ReplayBuffer: Experience storage for stable training
- PrioritizedReplayBuffer: Priority-based sampling for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """
    Deep Q-Network: MLP that maps states to Q-values for each action.
    
    Architecture: state_dim → 128 → 64 → action_dim
    Uses ReLU activations and LayerNorm for stable training.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, action_dim)
        
        # Initialize weights with Xavier
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values for each action."""
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN: Separates value and advantage streams.
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    This helps distinguish state value from action advantages.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layer
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        
        # Value stream
        self.value_fc = nn.Linear(128, 64)
        self.value_ln = nn.LayerNorm(64)
        self.value_out = nn.Linear(64, 1)
        
        # Advantage stream
        self.adv_fc = nn.Linear(128, 64)
        self.adv_ln = nn.LayerNorm(64)
        self.adv_out = nn.Linear(64, action_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with value/advantage decomposition."""
        features = F.relu(self.ln1(self.fc1(x)))
        
        # Value stream
        value = F.relu(self.value_ln(self.value_fc(features)))
        value = self.value_out(value)
        
        # Advantage stream
        advantage = F.relu(self.adv_ln(self.adv_fc(features)))
        advantage = self.adv_out(advantage)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    
    Stores (state, action, reward, next_state, done) transitions
    and samples random minibatches for training.
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay with proportional prioritization.
    
    Samples transitions based on TD-error priority:
    - High TD-error = more surprising = sample more often
    - Uses importance sampling weights to correct bias
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add transition with max priority (will be updated after training)."""
        transition = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample based on priorities with importance sampling weights."""
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Buffer is empty")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(n, size=min(batch_size, n), 
                                   p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Build batch
        batch = [self.buffer[i] for i in indices]
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # Small epsilon for stability
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)
