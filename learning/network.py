import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, config.action_space_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    def __init__(self, config):
        super(DuelingDQN, self).__init__()
        
        # Shared
        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        
        # Value
        self.value_fc = nn.Linear(config.hidden_dim, 64)
        self.value_ln = nn.LayerNorm(64)
        self.value_out = nn.Linear(64, 1)
        
        # Advantage
        self.adv_fc = nn.Linear(config.hidden_dim, 64)
        self.adv_ln = nn.LayerNorm(64)
        self.adv_out = nn.Linear(64, config.action_space_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = F.relu(self.ln1(self.fc1(x)))
        
        val = F.relu(self.value_ln(self.value_fc(features)))
        val = self.value_out(val)
        
        adv = F.relu(self.adv_ln(self.adv_fc(features)))
        adv = self.adv_out(adv)
        
        return val + adv - adv.mean(dim=-1, keepdim=True)
