import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, config):
        self.buffer = deque(maxlen=config.window_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return self._encode_batch(batch)
        
    def _encode_batch(self, batch):
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, config):
        self.capacity = config.window_size
        self.alpha = config.priority_exponent
        self.beta = config.priority_sampling
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0: return None
        
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(n, size=min(batch_size, n), p=probs, replace=False)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)
