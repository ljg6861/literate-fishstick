"""
MuZero with Transformer-based Dynamics Network

This module implements the core MuZero architecture using Transformers:
- RepresentationNetwork: Encodes input sequence to latent state
- DynamicsNetwork: Predicts next state and reward given (state, action)
- PredictionNetwork: Outputs policy and value from latent state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MuZeroConfig:
    """Configuration for MuZero Transformer."""
    # Model architecture
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    
    # Task configuration
    vocab_size: int = 2  # Binary for bit-reversal, or N for sorting
    max_seq_len: int = 16
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    
    # MuZero specific
    unroll_steps: int = 5  # K in the paper
    td_steps: int = 5  # n-step returns
    discount: float = 1.0  # Episodic task
    
    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + self.dropout(attn_out))
        
        # FFN with residual
        x = self.ln2(x + self.ffn(x))
        return x


class RepresentationNetwork(nn.Module):
    """
    Encodes the input sequence into an initial latent state s_0.
    Uses a Transformer encoder to process the sequence.
    """
    
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)  # +1 for padding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        
        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output projection to latent state
        self.ln_out = nn.LayerNorm(config.d_model)
        
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_seq: (batch, seq_len) - token indices
        Returns:
            latent_state: (batch, seq_len, d_model) - latent representation
        """
        # Embed tokens
        x = self.token_embed(input_seq)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return self.ln_out(x)


class DynamicsNetwork(nn.Module):
    """
    Predicts the next latent state and immediate reward given current state and action.
    Uses Transformer self-attention to model state transitions.
    
    (s_t, a_{t+1}) -> (s_{t+1}, r_{t+1})
    """
    
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        # Action embedding
        self.action_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # State-action fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model)
        )
        
        # Transformer for dynamics
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1)
        )
        
        self.ln_out = nn.LayerNorm(config.d_model)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, seq_len, d_model) - current latent state
            action: (batch,) - action index
            step: int - current step in sequence (for positional injection)
        Returns:
            next_state: (batch, seq_len, d_model) - predicted next state
            reward: (batch, 1) - predicted reward
        """
        batch_size = state.size(0)
        
        # Embed action
        action_emb = self.action_embed(action)  # (batch, d_model)
        
        # Inject action at the current step position
        # Create action-augmented state
        action_expanded = action_emb.unsqueeze(1).expand(-1, state.size(1), -1)  # (batch, seq_len, d_model)
        
        # Fuse state and action
        fused = torch.cat([state, action_expanded], dim=-1)  # (batch, seq_len, d_model*2)
        x = self.fusion(fused)  # (batch, seq_len, d_model)
        
        # Apply Transformer dynamics
        for layer in self.layers:
            x = layer(x)
        
        next_state = self.ln_out(x)
        
        # Predict reward from pooled state
        pooled = next_state.mean(dim=1)  # (batch, d_model)
        reward = self.reward_head(pooled)  # (batch, 1)
        
        return next_state, reward


class PredictionNetwork(nn.Module):
    """
    Outputs policy and value from a latent state.
    
    s -> (π, v)
    """
    
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        # Shared processing
        self.shared = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model)
        )
        
        # Policy head - outputs distribution over actions (vocab)
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.vocab_size)
        )
        
        # Value head - outputs scalar value
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1)
        )
    
    def forward(self, state: torch.Tensor, step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, seq_len, d_model) - latent state
            step: int - current step, used to select which position to predict from
        Returns:
            policy: (batch, vocab_size) - action probabilities
            value: (batch, 1) - state value
        """
        # Pool state for predictions
        # Use mean pooling for global context
        pooled = state.mean(dim=1)  # (batch, d_model)
        
        # Shared features
        features = self.shared(pooled)
        
        # Policy and value heads
        policy_logits = self.policy_head(features)  # (batch, vocab_size)
        value = self.value_head(features)  # (batch, 1)
        
        return policy_logits, value


class MuZeroTransformer(nn.Module):
    """
    Complete MuZero model combining all three networks.
    """
    
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        self.representation = RepresentationNetwork(config)
        self.dynamics = DynamicsNetwork(config)
        self.prediction = PredictionNetwork(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initial inference from observation.
        
        Args:
            observation: (batch, seq_len) - input sequence
        Returns:
            state: latent state
            policy: action probabilities
            value: state value
        """
        state = self.representation(observation)
        policy_logits, value = self.prediction(state)
        return state, policy_logits, value
    
    def recurrent_inference(self, state: torch.Tensor, action: torch.Tensor, step: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recurrent inference for planning.
        
        Args:
            state: current latent state
            action: action to take
            step: current step
        Returns:
            next_state: predicted next state
            reward: predicted reward
            policy: action probabilities for next state
            value: value of next state
        """
        next_state, reward = self.dynamics(state, action, step)
        policy_logits, value = self.prediction(next_state, step + 1)
        return next_state, reward, policy_logits, value
    
    def to_device(self, device: str = None):
        """Move model to device."""
        if device is None:
            device = self.config.device
        return self.to(device)


def create_muzero(config: MuZeroConfig = None) -> MuZeroTransformer:
    """Factory function to create MuZero model."""
    if config is None:
        config = MuZeroConfig()
    
    model = MuZeroTransformer(config)
    return model.to(config.device)


# Test code
if __name__ == "__main__":
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    config = MuZeroConfig(vocab_size=2, max_seq_len=8)
    model = create_muzero(config)
    
    # Test initial inference
    batch_size = 4
    seq_len = 8
    obs = torch.randint(0, 2, (batch_size, seq_len)).to(config.device)
    
    state, policy, value = model.initial_inference(obs)
    print(f"Initial state shape: {state.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test recurrent inference
    action = torch.randint(0, 2, (batch_size,)).to(config.device)
    next_state, reward, next_policy, next_value = model.recurrent_inference(state, action)
    print(f"Next state shape: {next_state.shape}")
    print(f"Reward shape: {reward.shape}")
    
    print("\n✓ MuZero Transformer initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
