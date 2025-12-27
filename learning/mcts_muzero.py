"""
MCTS with Learned Dynamics (MuZero-style)

This module implements Monte Carlo Tree Search using a learned dynamics model
instead of a simulator. The dynamics model predicts next states and rewards,
allowing planning in latent space.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    discount: float = 1.0
    
    # Temperature for action selection
    temperature: float = 1.0
    
    # Value bounds for normalization
    min_value: float = -1.0
    max_value: float = 1.0


class MinMaxStats:
    """Track min/max values for normalization."""
    
    def __init__(self):
        self.minimum = float('inf')
        self.maximum = float('-inf')
    
    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
    
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree.
    
    Each node represents a state in the search tree.
    """
    state: torch.Tensor  # Latent state tensor
    reward: float = 0.0  # Reward for reaching this node
    
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    parent: Optional['MCTSNode'] = None
    action_from_parent: Optional[int] = None
    
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    is_expanded: bool = False
    
    # Store predicted values
    predicted_value: float = 0.0
    
    @property
    def value(self) -> float:
        """Mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float, min_max_stats: MinMaxStats) -> float:
        """
        Upper Confidence Bound score for node selection.
        
        UCB = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        # Exploration bonus
        pb_c = math.log((parent_visit_count + 19652 + 1) / 19652) + c_puct
        pb_c *= math.sqrt(parent_visit_count) / (1 + self.visit_count)
        exploration = pb_c * self.prior
        
        # Value (normalized)
        if self.visit_count > 0:
            value_score = min_max_stats.normalize(self.value)
        else:
            value_score = 0.0
        
        return value_score + exploration


class MCTSMuZero:
    """
    Monte Carlo Tree Search using learned dynamics.
    
    Uses a MuZero model to:
    1. Get initial state from observation
    2. Predict next states and rewards via dynamics network
    3. Evaluate states via prediction network
    """
    
    def __init__(self, model, config: MCTSConfig):
        """
        Args:
            model: MuZeroTransformer instance
            config: MCTS configuration
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.action_space_size = model.config.vocab_size
    
    @torch.no_grad()
    def search(self, observation: torch.Tensor, step: int = 0) -> Tuple[np.ndarray, float]:
        """
        Run MCTS from an observation.
        
        Args:
            observation: (seq_len,) or (1, seq_len) input sequence
            step: Current step in the generation (for dynamics)
        Returns:
            action_probs: Visit count distribution over actions
            root_value: Estimated value of root state
        """
        # Ensure proper shape
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        observation = observation.to(self.device)
        
        # Get initial inference
        state, policy_logits, value = self.model.initial_inference(observation)
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        root_value = value.item()
        
        # Create root node
        root = MCTSNode(
            state=state,
            predicted_value=root_value
        )
        
        # Add exploration noise to root prior
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.action_space_size)
        priors = (1 - self.config.root_exploration_fraction) * policy_probs + \
                 self.config.root_exploration_fraction * noise
        
        # Expand root
        self._expand_node(root, priors, root_value)
        
        # Min-max stats for value normalization
        min_max_stats = MinMaxStats()
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            current_step = step
            
            # SELECT: Traverse tree to a leaf
            while node.is_expanded:
                action, child = self._select_child(node, min_max_stats)
                node = child
                search_path.append(node)
                current_step += 1
            
            # EXPAND & EVALUATE
            parent = search_path[-2] if len(search_path) > 1 else None
            parent_action = node.action_from_parent if parent else None
            
            if parent is not None and parent_action is not None:
                # Get state prediction from dynamics network
                action_tensor = torch.tensor([parent_action], device=self.device)
                next_state, reward, policy_logits, value = self.model.recurrent_inference(
                    parent.state, action_tensor, current_step - 1
                )
                
                # Update node with predictions
                node.state = next_state
                node.reward = reward.item()
                node.predicted_value = value.item()
                
                # Expand node
                policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
                self._expand_node(node, policy_probs, value.item())
                
                leaf_value = value.item()
            else:
                leaf_value = node.predicted_value
            
            # BACKUP
            self._backup(search_path, leaf_value, min_max_stats)
        
        # Compute action probabilities from visit counts
        visit_counts = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(self.action_space_size)
        ])
        
        # Apply temperature
        if self.config.temperature > 0:
            visit_counts = visit_counts ** (1 / self.config.temperature)
        
        action_probs = visit_counts / (visit_counts.sum() + 1e-8)
        
        return action_probs, root.value
    
    def _expand_node(self, node: MCTSNode, priors: np.ndarray, value: float):
        """Expand a node by adding children with prior probabilities."""
        node.is_expanded = True
        node.predicted_value = value
        
        for action in range(self.action_space_size):
            child = MCTSNode(
                state=None,  # Will be filled during traversal
                prior=priors[action],
                parent=node,
                action_from_parent=action
            )
            node.children[action] = child
    
    def _select_child(self, node: MCTSNode, min_max_stats: MinMaxStats) -> Tuple[int, MCTSNode]:
        """Select the child with highest UCB score."""
        best_score = float('-inf')
        best_action = 0
        best_child = None
        
        for action, child in node.children.items():
            score = child.ucb_score(node.visit_count, self.config.c_puct, min_max_stats)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _backup(self, search_path: List[MCTSNode], value: float, min_max_stats: MinMaxStats):
        """Backup value through the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            min_max_stats.update(node.value)
            
            # Discount value as we go up (for continuing tasks)
            value = node.reward + self.config.discount * value
    
    def select_action(self, action_probs: np.ndarray, deterministic: bool = False) -> int:
        """Select action from probability distribution."""
        if deterministic:
            return int(np.argmax(action_probs))
        else:
            return int(np.random.choice(len(action_probs), p=action_probs))


class MCTSResults:
    """Container for MCTS search results."""
    
    def __init__(self, action_probs: np.ndarray, value: float, selected_action: int):
        self.action_probs = action_probs
        self.value = value
        self.selected_action = selected_action


# Test code
if __name__ == "__main__":
    from muzero_transformer import MuZeroConfig, create_muzero
    
    print("Testing MCTS with learned dynamics...")
    
    # Create model
    config = MuZeroConfig(vocab_size=2, max_seq_len=8)
    model = create_muzero(config)
    
    # Create MCTS
    mcts_config = MCTSConfig(num_simulations=20)
    mcts = MCTSMuZero(model, mcts_config)
    
    # Test search
    obs = torch.randint(0, 2, (8,))
    action_probs, value = mcts.search(obs)
    
    print(f"Observation: {obs.numpy()}")
    print(f"Action probs: {action_probs}")
    print(f"Root value: {value:.4f}")
    print(f"Selected action: {mcts.select_action(action_probs)}")
    
    print("\n✓ MCTS working correctly!")
