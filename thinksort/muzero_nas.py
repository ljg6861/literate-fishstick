#!/usr/bin/env python3
"""
MuZero Neural Architecture Search for ThinkSort

Implements:
1. NAS Search Space (dim, heads, ff, activation, recurrence)
2. Meta-Reward Function: R = Acc - λ₁(Params) - λ₂(Steps)
3. MCTS-based architecture search
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import time
from copy import deepcopy

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SwiGLU Activation (optional upgrade from GELU)
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation: x * SiLU(gate)"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


# =============================================================================
# Architecture Configuration
# =============================================================================

@dataclass
class ArchConfig:
    """Searchable architecture configuration."""
    dim: int = 128
    heads: int = 8
    ff_mult: int = 4
    recurrent: int = 4
    activation: str = 'gelu'  # 'gelu', 'swiglu', 'relu'
    # vocab: int = 10 # REMOVED
    
    # === HPO Parameters ===
    lr: float = 1e-4
    batch_scale: float = 1.0  # Multiplier for base batch size
    
    @property
    def ff_dim(self) -> int:
        if self.activation == 'swiglu':
            return self.dim * self.ff_mult * 2  # SwiGLU needs 2x
        return self.dim * self.ff_mult
    
    def count_params(self) -> int:
        """Estimate parameter count."""
        d = self.dim
        h = self.heads
        ff = self.ff_dim
        
        # Embedding: Linear(1, dim) -> 1 * dim + dim (bias) = 2 * dim
        params = 2 * d
        
        # Universal Block (QKV + out + FF + norms)
        params += 3 * d * d  # QKV
        params += d * d  # out
        params += d * ff + ff * d  # FF
        params += 4 * d  # LayerNorms
        
        # Pointer head
        params += 2 * d * d  # query/key projs
        
        # Value head
        params += d * d + d  # value MLP
        
        return params
    
    
    def to_tensor(self) -> torch.Tensor:
        """Convert config to normalized tensor for predictor."""
        # Normalize roughly to [0, 1] range
        features = [
            self.dim / 512.0,
            self.heads / 16.0,
            self.ff_mult / 8.0,
            self.recurrent / 8.0,
            1.0 if self.activation == 'swiglu' else 0.0,
            self.lr * 100.0,
            self.batch_scale / 10.0
        ]
        return torch.tensor(features, dtype=torch.float32)


# =============================================================================
# Learned Architecture Model (Step 4)
# =============================================================================

class ArchitecturePredictor(nn.Module):
    """
    Predicts (Value, Policy) for a given architecture state.
    Value = expected reward
    Policy = logits for which mutation is promising
    """
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, action_dim)
        
    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.value_head(h), self.policy_head(h)


# =============================================================================
# Meta-Reward Function (Step 3: OOD Focus)
# =============================================================================

def compute_reward(
    acc_train: float,
    acc_ood: float,
    params: int,
    train_steps: int,
    lambda_params: float = 1e-6,
    lambda_steps: float = 1e-4
) -> float:
    """
    Multi-objective reward: OOD Generalization > In-Domain Accuracy > Efficiency.
    
    R = Acc_OOD + 0.5 * Acc_ID - penalties
    """
    param_penalty = lambda_params * (params / 100_000)
    step_penalty = lambda_steps * (train_steps / 1000)
    
    # Prioritize OOD generalization hugely
    return acc_ood + 0.5 * acc_train - param_penalty - step_penalty


# =============================================================================
# MCTS Node for NAS
# =============================================================================

@dataclass
class MCTSNode:
    """MCTS tree node for architecture search."""
    config: ArchConfig
    parent: Optional['MCTSNode'] = None
    action_from_parent_idx: Optional[int] = None # Index in ACTIONS list
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0 # From learned policy
    
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def ucb_score(self, c_puct: float = 1.4) -> float:
        """Upper Confidence Bound score."""
        if self.visits == 0:
            # If never visited, use prior to encourage exploration
            # Higher prior = higher initial score
            return self.prior * c_puct
        
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.value + exploration
    
    def best_child(self, c_puct: float = 1.4) -> Tuple[int, 'MCTSNode']:
        """Select best child by UCB score."""
        best_action = None
        best_node = None
        best_score = -float('inf')
        
        for action_idx, child in self.children.items():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_node = child
        
        return best_action, best_node
    
    def expand(self, priors: torch.Tensor):
        """Expand node with all possible actions, initializing with priors."""
        # priors is (num_actions,) tensor of probabilities
        for i, action in enumerate(ACTIONS):
            if i not in self.children:
                new_config = apply_action(self.config, action)
                child = MCTSNode(
                    config=new_config,
                    parent=self,
                    action_from_parent_idx=i,
                    prior=priors[i].item()
                )
                self.children[i] = child


# =============================================================================
# MuZero NAS Controller
# =============================================================================

class MuZeroNAS:
    """
    MuZero-style Neural Architecture Search.
    
    Uses a LEARNED MODEL to guide MCTS.
    1. Search: Run MCTS using model predictions.
    2. Act: Select best candidate to actually train/eval.
    3. Learn: Update model on real (config, reward) data.
    """
    
    def __init__(
        self,
        initial_config: ArchConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        burst_steps: int = 2000,
        train_lengths: tuple = (4, 8, 16),
        lambda_params: float = 1e-6,
        lambda_steps: float = 1e-4,
        seed_model: Optional[nn.Module] = None
    ):
        self.device = device
        self.burst_steps = burst_steps
        self.train_lengths = train_lengths
        self.lambda_params = lambda_params
        self.lambda_steps = lambda_steps
        self.seed_model = seed_model
        
        # Initialize root
        self.root = MCTSNode(config=initial_config or ArchConfig())
        
        # Learned Predictor
        # Input features: 7
        self.predictor = ArchitecturePredictor(7, len(ACTIONS)).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        
        # Expand root with uniform priors initially
        self.root.expand(torch.ones(len(ACTIONS)) / len(ACTIONS))
        
        # History
        self.history: List[Dict] = []
        self.best_config: ArchConfig = self.root.config
        self.best_reward: float = -float('inf')
    
    def train_predictor(self):
        """Train predictor on history."""
        if not self.history:
            return
            
        self.predictor.train()
        losses = []
        
        # Create batch
        configs = []
        rewards = []
        actions = [] # For policy target? 
        # We can train policy head to predict "which action led to high reward"
        # Or simpler: AlphaZero style -> train policy against MCTS visit counts.
        # But we don't store visit counts from search yet.
        # Let's just train Value Head for now to fit Reward.
        
        for item in self.history:
            # We need to reconstruct config tensor
            # Ideally store it
            cfg = ArchConfig(**item['config_obj']) # Rehydrate
            configs.append(cfg.to_tensor())
            rewards.append(item['reward'])
            
        x = torch.stack(configs).to(self.device)
        y = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        
        for _ in range(50): # Small Epochs
            v_pred, _ = self.predictor(x)
            loss = F.mse_loss(v_pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        # print(f"   🧠 Model Trained. Loss: {np.mean(losses):.4f}")
        
    def evaluate(self, config: ArchConfig) -> Tuple[float, float, int]:
        """
        Evaluate architecture:
        Returns (acc_train, acc_ood, params).
        """
        from model import Config, UniversalPointerNet
        from trainer import Trainer
        from morpher import morph_model
        
        # Build model logic
        model_config = Config(
            dim=config.dim,
            heads=config.heads,
            ff=config.ff_dim,
            recurrent_steps=config.recurrent,
            vocab=config.vocab,
            train_lengths=self.train_lengths,
            samples_per_length=int(32 * config.batch_scale), # Fast burst
            lr=config.lr,
            device=self.device
        )
        
        # Morph or Init
        if self.seed_model is not None:
             try:
                 model = morph_model(self.seed_model, model_config)
             except Exception as e:
                 print(f"Morph failed: {e}. Falling back to random init.")
                 model = UniversalPointerNet(model_config).to(self.device)
        else:
             model = UniversalPointerNet(model_config).to(self.device)
        
        trainer = Trainer(model_config)
        trainer.model = model 
        trainer.optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=model_config.lr,
            weight_decay=0.01
        )
        
        # Train burst
        for _ in range(self.burst_steps):
            trainer.train_step() # This now includes search-based updates!
        
        # Eval In-Domain
        accs_id = [trainer.evaluate(n, 100) for n in self.train_lengths]
        avg_acc_id = np.mean(accs_id)
        
        # Eval OOD (2x, 4x max train length)
        max_len = max(self.train_lengths)
        ood_lens = [max_len * 2, max_len * 4]
        accs_ood = [trainer.evaluate(n, 100) for n in ood_lens]
        avg_acc_ood = np.mean(accs_ood)
        
        params = config.count_params()
        
        # Cleanup
        train_step_method = None
        del trainer.optimizer
        del trainer.model
        del trainer
        del model
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return avg_acc_id, avg_acc_ood, params

    def run_mcts_sim(self, num_sims: int = 50):
        """
        Run MCTS purely in latent/model space to find promising node using `predictor`.
        Does NOT run real evaluation.
        """
        self.predictor.eval()
        
        for _ in range(num_sims):
            node = self.root
            search_path = [node]
            
            # Select
            while node.children:
                _, node = node.best_child()
                search_path.append(node)
                
            # Expand & Evaluate (using Model, not Real Env)
            x = node.config.to_tensor().unsqueeze(0).to(self.device)
            with torch.no_grad():
                val_pred, priors_logits = self.predictor(x)
                
            val = val_pred.item()
            priors = F.softmax(priors_logits.squeeze(0), dim=0)
            
            # Expand node
            node.expand(priors.cpu())
            
            # Backpropagate (simulated value)
            for n in search_path:
                n.visits += 1
                n.value_sum += val

    def search_step(self) -> Dict:
        """
        1. Train model on past data.
        2. Run MCTS simulations.
        3. Pick best candidate from MCTS.
        4. Real Evaluate.
        5. Update Tree & History.
        """
        # 1. Train
        if len(self.history) > 2:
            self.train_predictor()
            
        # 2. Run MCTS simulations (Mental Planning)
        self.run_mcts_sim(num_sims=50)
        
        # 3. Select best leaf to actually evaluate
        # We want the node visited most by MCTS? Or highest value?
        # Standard MuZero acts based on visit counts.
        # But we are in search mode. We want to evaluate the frontier.
        # Let's pick the child of root with highest UCB that HASN'T been evaluated?
        # Or just traverse the tree greedily using visit counts to a leaf?
        
        node = self.root
        while node.children:
            # Pick most visited child
            best_child = max(node.children.values(), key=lambda c: c.visits)
            node = best_child
            
        # If this node is already evaluated (in history?), we might be stuck.
        # Actually our Tree persists.
        # But wait, we just want to find ONE good config to try next.
        # Let's traverse using UCB to a leaf.
        node = self.root
        while node.children:
             _, node = node.best_child()
             
        # Now we are at a leaf. `node`.
        config = node.config
        
        print(f"\n🔍 Evaluating Candidate: dim={config.dim}, heads={config.heads}, ff={config.ff_mult}x")
        
        # 4. Evaluate Reality
        start = time.time()
        acc_id, acc_ood, params = self.evaluate(config)
        elapsed = time.time() - start
        
        # Compute Reward (Step 3)
        reward = compute_reward(
            acc_id, acc_ood, params, self.burst_steps,
            self.lambda_params, self.lambda_steps
        )
        
        print(f"   ID: {acc_id:.1%} | OOD: {acc_ood:.1%} | Reward: {reward:.4f} | Time: {elapsed:.1f}s")
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_config = config
            print("   🌟 NEW BEST ARCHITECTURE!")
            
        # 5. Backpropagate REAL reward (correcting the tree values)
        # Note: MCTS previous backprop was "simulated". Now we have truth.
        # In true MuZero, we wouldn't overwrite, but here mixing is fine.
        temp = node
        while temp:
            temp.visits += 1
            temp.value_sum += reward
            temp = temp.parent
            
        result = {
            'config_obj': config.to_dict(), # Store dict for serialization
            'reward': reward,
            'time': elapsed
        }
        self.history.append(result)
        return result

    def search(self, num_iterations: int = 10) -> ArchConfig:
        print(f"\n🧠 MuZero NAS (Learned Model + OOD Focus)")
        for i in range(num_iterations):
            print(f"\n[Search Step {i+1}/{num_iterations}]")
            self.search_step()
            
        return self.best_config
        
if __name__ == "__main__":
    # Test run
    run_nas(num_iterations=5, burst_steps=200)

ACTIONS = [
    'expand_dim',      # dim *= 2
    'shrink_dim',      # dim //= 2
    'add_head',        # heads += 4
    'remove_head',     # heads -= 4
    'expand_ff',       # ff_mult += 2
    'shrink_ff',       # ff_mult -= 2
    'add_recurrent',   # recurrent += 2
    'remove_recurrent',# recurrent -= 2
    'swap_gelu',       # activation = 'gelu'
    'swap_swiglu',     # activation = 'swiglu'
    'inc_lr',          # lr *= 2
    'dec_lr',          # lr /= 2
    'inc_batch',       # batch_scale *= 1.5
    'dec_batch',       # batch_scale /= 1.5
    'noop',            # No change
]


def apply_action(config: ArchConfig, action: str) -> ArchConfig:
    """Apply NAS action to architecture config."""
    new_config = ArchConfig(
        dim=config.dim,
        heads=config.heads,
        ff_mult=config.ff_mult,
        recurrent=config.recurrent,
        activation=config.activation,
        lr=config.lr,
        batch_scale=config.batch_scale
    )
    
    if action == 'expand_dim' and new_config.dim < 512:
        new_config.dim *= 2
        new_config.heads = min(new_config.heads, new_config.dim // 8)
    elif action == 'shrink_dim' and new_config.dim > 64:
        new_config.dim //= 2
        new_config.heads = min(new_config.heads, new_config.dim // 8)
    elif action == 'add_head' and new_config.heads < new_config.dim // 8:
        new_config.heads += 4
    elif action == 'remove_head' and new_config.heads > 4:
        new_config.heads -= 4
    elif action == 'expand_ff' and new_config.ff_mult < 8:
        new_config.ff_mult += 2
    elif action == 'shrink_ff' and new_config.ff_mult > 2:
        new_config.ff_mult -= 2
    elif action == 'add_recurrent' and new_config.recurrent < 8:
        new_config.recurrent += 2
    elif action == 'remove_recurrent' and new_config.recurrent > 2:
        new_config.recurrent -= 2
    elif action == 'swap_gelu':
        new_config.activation = 'gelu'
    elif action == 'swap_swiglu':
        new_config.activation = 'swiglu'
    elif action == 'inc_lr' and new_config.lr < 1e-2:
        new_config.lr *= 2.0
    elif action == 'dec_lr' and new_config.lr > 1e-5:
        new_config.lr /= 2.0
    elif action == 'inc_batch' and new_config.batch_scale < 64.0:
        new_config.batch_scale *= 1.5
    elif action == 'dec_batch' and new_config.batch_scale > 0.1:
        new_config.batch_scale /= 1.5
    
    # Ensure heads divides dim
    while new_config.dim % new_config.heads != 0:
        new_config.heads -= 1
    
    return new_config


# =============================================================================
# Meta-Reward Function
# =============================================================================

def compute_reward(
    accuracy: float,
    params: int,
    train_steps: int,
    lambda_params: float = 1e-6,
    lambda_steps: float = 1e-4
) -> float:
    """
    Multi-objective reward: performance vs efficiency.
    
    R = Accuracy - λ₁(Params/100k) - λ₂(Steps/1k)
    """
    param_penalty = lambda_params * (params / 100_000)
    step_penalty = lambda_steps * (train_steps / 1000)
    return accuracy - param_penalty - step_penalty


# =============================================================================
# MCTS Node for NAS
# =============================================================================

@dataclass
class MCTSNode:
    """MCTS tree node for architecture search."""
    config: ArchConfig
    parent: Optional['MCTSNode'] = None
    action_from_parent: Optional[str] = None
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 1.0 / len(ACTIONS)
    
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def ucb_score(self, c_puct: float = 1.4) -> float:
        """Upper Confidence Bound score."""
        if self.visits == 0:
            return float('inf')
        
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.value + exploration
    
    def best_child(self, c_puct: float = 1.4) -> Tuple[str, 'MCTSNode']:
        """Select best child by UCB score."""
        best_action = None
        best_node = None
        best_score = -float('inf')
        
        for action, child in self.children.items():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child
        
        return best_action, best_node
    
    def expand(self):
        """Expand node with all possible actions."""
        for action in ACTIONS:
            if action not in self.children:
                new_config = apply_action(self.config, action)
                child = MCTSNode(
                    config=new_config,
                    parent=self,
                    action_from_parent=action
                )
                self.children[action] = child


# =============================================================================
# MuZero NAS Controller
# =============================================================================

class MuZeroNAS:
    """
    MuZero-style Neural Architecture Search.
    
    Uses MCTS to explore architecture space, with model training
    as the environment simulation.
    """
    
    def __init__(
        self,
        initial_config: ArchConfig = None,
        device: str = "cuda",
        burst_steps: int = 2000,
        train_lengths: tuple = (4, 8, 16),
        lambda_params: float = 1e-6,
        lambda_steps: float = 1e-4,
        seed_model: Optional[nn.Module] = None
    ):
        self.device = device
        self.burst_steps = burst_steps
        self.train_lengths = train_lengths
        self.lambda_params = lambda_params
        self.lambda_steps = lambda_steps
        self.seed_model = seed_model
        
        # Initialize root
        self.root = MCTSNode(
            config=initial_config or ArchConfig()
        )
        self.root.expand()
        
        # History
        self.history: List[Dict] = []
        self.best_config: ArchConfig = None
        self.best_reward: float = -float('inf')
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node via UCB."""
        while node.children:
            _, node = node.best_child()
            
        # We reached a leaf (no children).
        # If it has been visited, we must expand to explore deeper
        if node.visits > 0:
            node.expand()
            # If expansion created children, pick one to evaluate
            if node.children:
                _, node = node.best_child()
                
        return node
    
    def evaluate(self, config: ArchConfig) -> Tuple[float, int]:
        """
        Evaluate architecture by training for burst_steps.
        Uses Network Morphism if a parent model exists.
        
        Returns (accuracy, params).
        """
        from model import Config, UniversalPointerNet
        from trainer import Trainer
        from morpher import morph_model
        
        # Build model logic
        model_config = Config(
            dim=config.dim,
            heads=config.heads,
            ff=config.ff_dim,
            recurrent_steps=config.recurrent,
            vocab=config.vocab,
            train_lengths=self.train_lengths,
            samples_per_length=int(32 * config.batch_scale), # Fast burst scaled
            lr=config.lr,
            device=self.device
        )
        
        # If we have a seed model, morph it!
        if self.seed_model is not None:
             model = morph_model(self.seed_model, model_config)
        else:
             model = UniversalPointerNet(model_config).to(self.device)
        
        # Use a temporary trainer
        trainer = Trainer(model_config)
        trainer.model = model # Swap in our morphed model
        # Re-init optimizer matching new parameters
        trainer.optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=model_config.lr,
            weight_decay=0.01
        )
        
        # Train burst
        for _ in range(self.burst_steps):
            trainer.train_step()
        
        # Evaluate on all train lengths
        accs = [trainer.evaluate(n, 100) for n in self.train_lengths]
        avg_acc = np.mean(accs)
        
        params = config.count_params()
        
        # Cleanup
        # Explicitly delete objects to break ref cycles immediately
        train_step_method = None # clear any bound methods
        del trainer.optimizer
        del trainer.model
        del trainer
        del model
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return avg_acc, params
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        while node:
            node.visits += 1
            node.value_sum += reward
            node = node.parent
    
    def search_step(self) -> Dict:
        """
        One MCTS search iteration:
        1. Select leaf
        2. Evaluate (train burst)
        3. Backpropagate reward
        """
        # Select
        leaf = self.select(self.root)
        config = leaf.config
        
        print(f"\n🔍 Evaluating: dim={config.dim}, heads={config.heads}, "
              f"ff={config.ff_mult}x, rec={config.recurrent}, act={config.activation}")
        
        # Evaluate
        start = time.time()
        accuracy, params = self.evaluate(config)
        elapsed = time.time() - start
        
        # Compute reward
        reward = compute_reward(
            accuracy, params, self.burst_steps,
            self.lambda_params, self.lambda_steps
        )
        
        print(f"   Acc: {accuracy:.2%} | Params: {params:,} | Reward: {reward:.4f} | Time: {elapsed:.1f}s")
        
        # Track best
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_config = config
            print(f"   🌟 NEW BEST!")
        
        # Backpropagate
        self.backpropagate(leaf, reward)
        
        # Expand leaf for future
        leaf.expand()
        
        # Record
        result = {
            'config': config.to_dict(),
            'accuracy': accuracy,
            'params': params,
            'reward': reward,
            'time': elapsed
        }
        self.history.append(result)
        
        return result
    
    def search(self, num_iterations: int = 20) -> ArchConfig:
        """Run full NAS search."""
        print("\n" + "="*60)
        print("MuZero Neural Architecture Search")
        print(f"Iterations: {num_iterations} | Burst: {self.burst_steps} steps")
        print("="*60)
        
        for i in range(num_iterations):
            print(f"\n[Iteration {i+1}/{num_iterations}]")
            self.search_step()
        
        # Summary
        print("\n" + "="*60)
        print("NAS COMPLETE")
        print("="*60)
        print(f"\nBest Architecture:")
        print(f"  dim: {self.best_config.dim}")
        print(f"  heads: {self.best_config.heads}")
        print(f"  ff_mult: {self.best_config.ff_mult}")
        print(f"  recurrent: {self.best_config.recurrent}")
        print(f"  activation: {self.best_config.activation}")
        print(f"\nBest Reward: {self.best_reward:.4f}")
        
        return self.best_config


# =============================================================================
# Main Entry Point
# =============================================================================

def run_nas(
    num_iterations: int = 20,
    burst_steps: int = 2000,
    device: str = "cuda"
) -> ArchConfig:
    """Run NAS search and return best architecture."""
    nas = MuZeroNAS(
        device=device,
        burst_steps=burst_steps
    )
    return nas.search(num_iterations)


if __name__ == "__main__":
    best = run_nas(num_iterations=10, burst_steps=1000)
    print(f"\nRun full training with: {best.to_dict()}")
