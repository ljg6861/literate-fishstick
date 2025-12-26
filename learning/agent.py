
import torch
import torch.optim as optim
import numpy as np
import random
from .network import DuelingDQN
from .storage import PrioritizedReplayBuffer

class GeneralAgent:
    """
    Generic RL Agent (MuZero-style interface).
    
    Role:
    - Receives Observation (vector)
    - Outputs Action (index)
    - Learns from (Obs, Act, Reward, NextObs)
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.policy_net = DuelingDQN(config).to(self.device)
        self.target_net = DuelingDQN(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(config)
        
        # State
        self.epsilon = config.epsilon_init
        self.training_steps = 0
        self.losses = []
        
    def act(self, observation, training=True):
        """Select action via Epsilon-Greedy (or MCTS if integrated later)."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_space_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        
    def train_step(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return None
            
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.config.batch_size)
            
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.config.discount * next_q * (1 - dones)
            
        loss = (weights * (current_q - target_q) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Priority update
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Soft Update
        tau = 0.005
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
            
        self.training_steps += 1
        self.losses.append(loss.item())
        
        # Decay Epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        
        return loss.item()
        
    def save(self, path):
        torch.save({
            'model': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'steps': self.training_steps,
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model'])
        self.target_net.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        self.training_steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
