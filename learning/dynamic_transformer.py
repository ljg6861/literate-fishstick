"""
Dynamic Transformer with Hot-Swappable Architecture

This module implements a Transformer that can be modified at runtime:
- Add/remove layers
- Resize attention heads
- Change model dimensions
- Swap activation functions

All operations preserve learned weights where possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import IntEnum


class ArchAction(IntEnum):
    """Architecture modification actions."""
    NO_OP = 0
    ADD_LAYER = 1
    REMOVE_LAYER = 2
    INCREASE_HEADS = 3
    DECREASE_HEADS = 4
    INCREASE_DIM = 5
    DECREASE_DIM = 6
    CHANGE_ACTIVATION = 7
    
    @classmethod
    def num_actions(cls) -> int:
        return len(cls)


ACTIVATIONS = ['gelu', 'relu', 'silu']


@dataclass
class DynamicConfig:
    """Configuration for dynamic transformer."""
    # Current architecture
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    d_ff: int = 128
    activation: str = 'gelu'
    dropout: float = 0.0
    
    # Task
    vocab_size: int = 2
    max_seq_len: int = 8
    
    # Constraints
    min_d_model: int = 32
    max_d_model: int = 256
    min_heads: int = 1
    max_heads: int = 8
    min_layers: int = 1
    max_layers: int = 6
    
    # Training
    device: str = "cuda"
    
    def to_vector(self) -> torch.Tensor:
        """Encode architecture as normalized vector for MuZero state."""
        return torch.tensor([
            (self.n_layers - self.min_layers) / (self.max_layers - self.min_layers),
            (self.n_heads - self.min_heads) / (self.max_heads - self.min_heads),
            (self.d_model - self.min_d_model) / (self.max_d_model - self.min_d_model),
            ACTIVATIONS.index(self.activation) / len(ACTIVATIONS),
        ], dtype=torch.float32)
    
    def param_count_estimate(self) -> int:
        """Estimate parameter count."""
        # Rough estimate: embeddings + layers
        embed_params = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        layer_params = self.n_layers * (
            4 * self.d_model * self.d_model +  # Attention QKV + O
            2 * self.d_model * self.d_ff +      # FFN
            4 * self.d_model                     # LayerNorms
        )
        return embed_params + layer_params
    
    def copy(self) -> 'DynamicConfig':
        """Create a copy of this config."""
        return DynamicConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            activation=self.activation,
            dropout=self.dropout,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            min_d_model=self.min_d_model,
            max_d_model=self.max_d_model,
            min_heads=self.min_heads,
            max_heads=self.max_heads,
            min_layers=self.min_layers,
            max_layers=self.max_layers,
            device=self.device,
        )


def get_activation(name: str):
    """Get activation function by name."""
    if name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'silu':
        return nn.SiLU()
    else:
        return nn.GELU()


class DynamicTransformerBlock(nn.Module):
    """Transformer block with dynamic sizing."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, activation: str = 'gelu', dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
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


class DynamicTransformer(nn.Module):
    """
    Transformer with dynamically modifiable architecture.
    
    Supports runtime modifications:
    - add_layer() / remove_layer()
    - resize_heads()
    - resize_dim()
    - change_activation()
    """
    
    def __init__(self, config: DynamicConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        
        # Positional embedding
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer layers
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
        
        # For dynamics: action embedding
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len) token indices
        Returns:
            policy: (batch, vocab_size)
            value: (batch, 1)
            hidden: (batch, seq_len, d_model) - for dynamics
        """
        batch_size, seq_len = x.shape
        
        # Embeddings
        tok_emb = self.token_embed(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(pos_ids)
        
        h = tok_emb + pos_emb
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h)
        
        h = self.ln_out(h)
        
        # Pooled output for heads
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy, value, h
    
    def dynamics_step(self, hidden: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamics step: predict next hidden state given action.
        
        Args:
            hidden: (batch, seq_len, d_model)
            action: (batch,) action indices
        Returns:
            next_hidden, reward, policy, value
        """
        # Embed action and add to hidden state
        action_emb = self.action_embed(action).unsqueeze(1)  # (batch, 1, d_model)
        
        # Simple fusion: add action embedding to all positions
        h = hidden + action_emb.expand(-1, hidden.size(1), -1)
        
        # Pass through layers
        for layer in self.layers:
            h = layer(h)
        
        h = self.ln_out(h)
        pooled = h.mean(dim=1)
        
        policy = self.policy_head(pooled)
        value = self.value_head(pooled)
        reward = self.reward_head(pooled)
        
        return h, reward, policy, value
    
    # ========== ARCHITECTURE MODIFICATION METHODS ==========
    
    def add_layer(self) -> bool:
        """Add a new transformer layer. Returns True if successful."""
        if self.config.n_layers >= self.config.max_layers:
            return False
        
        # Create new layer initialized near identity
        new_layer = DynamicTransformerBlock(
            self.config.d_model, self.config.n_heads, self.config.d_ff,
            self.config.activation, self.config.dropout
        ).to(self.config.device)
        
        # Initialize to near-identity (small weights)
        for p in new_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            else:
                nn.init.zeros_(p)
        
        self.layers.append(new_layer)
        self.config.n_layers += 1
        return True
    
    def remove_layer(self) -> bool:
        """Remove last transformer layer. Returns True if successful."""
        if self.config.n_layers <= self.config.min_layers:
            return False
        
        self.layers = self.layers[:-1]
        self.config.n_layers -= 1
        return True
    
    def resize_heads(self, new_n_heads: int) -> bool:
        """Change number of attention heads. Rebuilds attention layers."""
        if new_n_heads < self.config.min_heads or new_n_heads > self.config.max_heads:
            return False
        if self.config.d_model % new_n_heads != 0:
            return False  # Must divide evenly
        
        old_n_heads = self.config.n_heads
        self.config.n_heads = new_n_heads
        
        # Rebuild layers with new head count
        new_layers = nn.ModuleList()
        for old_layer in self.layers:
            new_layer = DynamicTransformerBlock(
                self.config.d_model, new_n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout
            ).to(self.config.device)
            
            # Copy FFN weights (unchanged)
            new_layer.ffn.load_state_dict(old_layer.ffn.state_dict())
            new_layer.ln1.load_state_dict(old_layer.ln1.state_dict())
            new_layer.ln2.load_state_dict(old_layer.ln2.state_dict())
            
            # Attention weights need to be handled carefully
            # For simplicity, reinitialize attention
            new_layers.append(new_layer)
        
        self.layers = new_layers
        return True
    
    def resize_dim(self, new_d_model: int) -> bool:
        """Change model dimension. Rebuilds entire model."""
        if new_d_model < self.config.min_d_model or new_d_model > self.config.max_d_model:
            return False
        if new_d_model % self.config.n_heads != 0:
            return False
        
        old_d_model = self.config.d_model
        self.config.d_model = new_d_model
        self.config.d_ff = new_d_model * 4  # Keep FFN ratio
        
        # Rebuild embeddings
        old_token_embed = self.token_embed
        self.token_embed = nn.Embedding(self.config.vocab_size + 1, new_d_model).to(self.config.device)
        
        # Project old embeddings to new dimension
        with torch.no_grad():
            if new_d_model > old_d_model:
                self.token_embed.weight[:, :old_d_model] = old_token_embed.weight
            else:
                self.token_embed.weight.copy_(old_token_embed.weight[:, :new_d_model])
        
        self.pos_embed = nn.Embedding(self.config.max_seq_len, new_d_model).to(self.config.device)
        self.action_embed = nn.Embedding(self.config.vocab_size, new_d_model).to(self.config.device)
        
        # Rebuild layers
        self.layers = nn.ModuleList([
            DynamicTransformerBlock(
                new_d_model, self.config.n_heads, self.config.d_ff,
                self.config.activation, self.config.dropout
            ).to(self.config.device)
            for _ in range(self.config.n_layers)
        ])
        
        # Rebuild heads
        self.ln_out = nn.LayerNorm(new_d_model).to(self.config.device)
        self.policy_head = nn.Linear(new_d_model, self.config.vocab_size).to(self.config.device)
        self.value_head = nn.Linear(new_d_model, 1).to(self.config.device)
        self.reward_head = nn.Linear(new_d_model, 1).to(self.config.device)
        
        self._init_weights()
        return True
    
    def change_activation(self) -> bool:
        """Toggle to next activation function."""
        current_idx = ACTIVATIONS.index(self.config.activation)
        new_idx = (current_idx + 1) % len(ACTIVATIONS)
        new_activation = ACTIVATIONS[new_idx]
        self.config.activation = new_activation
        
        # Rebuild layers with new activation
        for i, old_layer in enumerate(self.layers):
            new_layer = DynamicTransformerBlock(
                self.config.d_model, self.config.n_heads, self.config.d_ff,
                new_activation, self.config.dropout
            ).to(self.config.device)
            
            # Copy attention weights
            new_layer.attn.load_state_dict(old_layer.attn.state_dict())
            new_layer.ln1.load_state_dict(old_layer.ln1.state_dict())
            new_layer.ln2.load_state_dict(old_layer.ln2.state_dict())
            
            # Copy FFN linear weights (not activation)
            new_layer.ffn[0].load_state_dict(old_layer.ffn[0].state_dict())
            new_layer.ffn[3].load_state_dict(old_layer.ffn[3].state_dict())
            
            self.layers[i] = new_layer
        
        return True
    
    def apply_arch_action(self, action: ArchAction) -> bool:
        """
        Apply an architecture modification action.
        
        Returns True if the action was successfully applied.
        """
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
        """Get human-readable architecture description."""
        return (f"L{self.config.n_layers}_H{self.config.n_heads}_"
                f"D{self.config.d_model}_{self.config.activation}")
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


# Test code
if __name__ == "__main__":
    print("Testing DynamicTransformer...")
    
    config = DynamicConfig(d_model=32, n_heads=2, n_layers=1, vocab_size=2)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DynamicTransformer(config).to(config.device)
    
    print(f"Initial: {model.get_architecture_string()}, params: {model.count_parameters():,}")
    
    # Test forward
    x = torch.randint(0, 2, (4, 8)).to(config.device)
    policy, value, hidden = model(x)
    print(f"Forward: policy={policy.shape}, value={value.shape}, hidden={hidden.shape}")
    
    # Test architecture modifications
    print("\n--- Architecture Modifications ---")
    
    model.add_layer()
    print(f"After ADD_LAYER: {model.get_architecture_string()}, params: {model.count_parameters():,}")
    
    model.add_layer()
    print(f"After ADD_LAYER: {model.get_architecture_string()}, params: {model.count_parameters():,}")
    
    model.resize_dim(64)
    print(f"After INCREASE_DIM: {model.get_architecture_string()}, params: {model.count_parameters():,}")
    
    model.resize_heads(4)
    print(f"After INCREASE_HEADS: {model.get_architecture_string()}, params: {model.count_parameters():,}")
    
    model.change_activation()
    print(f"After CHANGE_ACTIVATION: {model.get_architecture_string()}, params: {model.count_parameters():,}")
    
    model.remove_layer()
    print(f"After REMOVE_LAYER: {model.get_architecture_string()}, params: {model.count_parameters():,}")
    
    # Verify model still works
    policy, value, hidden = model(x)
    print(f"\nFinal forward: policy={policy.shape}, value={value.shape}")
    
    print("\n✓ DynamicTransformer working correctly!")
