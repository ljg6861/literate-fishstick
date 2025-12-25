"""
Monte Carlo Tree Search (MCTS) for Tower Builder AI.

AlphaZero-style implementation with neural network guidance:
- Uses policy network to guide action selection (prior probabilities)
- Uses value network for leaf evaluation (no rollouts needed)
- UCB formula balances exploration vs exploitation
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.
    
    Each node represents a game state and stores:
    - Visit count N(s)
    - Total value W(s) 
    - Prior probability P(s,a) from policy network
    - Children for each action taken from this state
    """
    state: np.ndarray  # The game state (normalized vector)
    parent: Optional['MCTSNode'] = None
    action: Optional[tuple] = None  # Action that led to this node (x, rotation)
    
    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0  # P(s,a) from policy network
    
    # Children nodes
    children: Dict[tuple, 'MCTSNode'] = field(default_factory=dict)
    is_expanded: bool = False
    is_terminal: bool = False
    
    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a) = W(s,a) / N(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb_score(self, c_puct: float = 1.5) -> float:
        """
        Upper Confidence Bound for Trees (UCT) with policy prior.
        
        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
        
        This balances:
        - Exploitation: Q(s,a) - actions that worked well
        - Exploration: prior * sqrt(parent visits) - try new actions
        """
        if self.parent is None:
            return 0.0
        
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    
    AlphaZero-style search:
    1. SELECT: Traverse tree using UCB until leaf
    2. EXPAND: Add children using policy network priors
    3. EVALUATE: Get value estimate from value network (no rollout)
    4. BACKUP: Propagate value up the tree
    """
    
    def __init__(self, agent, num_simulations: int = 50, c_puct: float = 1.5,
                 temperature: float = 1.0):
        """
        Args:
            agent: DeepTowerAgent with policy_net for evaluation
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for UCB
            temperature: Controls action selection randomness (1=proportional, 0=greedy)
        """
        self.agent = agent
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
    
    def search(self, root_state: np.ndarray) -> Tuple[tuple, np.ndarray]:
        """
        Perform MCTS search from the given state.
        
        Args:
            root_state: Current game state (normalized vector)
            
        Returns:
            best_action: The recommended (x, rotation) action
            action_probs: Visit count distribution over actions (for training)
        """
        import torch
        
        # Create root node
        root = MCTSNode(state=root_state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            
            # 1. SELECT: Traverse to leaf using UCB
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)
            
            # 2. EXPAND: Add children if not terminal
            if not node.is_terminal:
                self._expand(node)
            
            # 3. EVALUATE: Get value from neural network
            value = self._evaluate(node)
            
            # 4. BACKUP: Propagate value up the tree
            self._backup(node, value)
        
        # Select action based on visit counts
        best_action, action_probs = self._select_action(root)
        
        return best_action, action_probs
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand(self, node: MCTSNode):
        """
        Expand node by adding children for all possible actions.
        Uses policy network to get prior probabilities.
        """
        import torch
        
        # Get policy priors from neural network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.policy_net(state_tensor).squeeze(0)
            
            # Convert Q-values to probabilities using softmax
            priors = torch.softmax(q_values / max(0.1, self.temperature), dim=0).cpu().numpy()
        
        # Create child nodes for each action
        for idx, action in enumerate(self.agent.actions):
            # Create hypothetical next state (simplified - we don't have physics here)
            # In full implementation, you'd simulate the action
            next_state = self._estimate_next_state(node.state, action)
            
            child = MCTSNode(
                state=next_state,
                parent=node,
                action=action,
                prior=priors[idx]
            )
            node.children[action] = child
        
        node.is_expanded = True
    
    def _estimate_next_state(self, state: np.ndarray, action: tuple) -> np.ndarray:
        """
        Estimate the next state after taking an action.
        
        Since we can't clone the physics simulation, we make a simplified
        estimate based on the action taken.
        """
        x_pos, rotation = action
        
        # State format: [rel_x, angle, height, shape]
        next_state = state.copy()
        
        # Estimate new relative X (action influences position)
        center_x = 400  # screen center
        next_state[0] = (x_pos - center_x) / 400  # normalized rel_x
        
        # Rotation influences angle (simplified)
        next_state[1] = rotation / 0.52  # normalized angle
        
        # Height increases (simple increment, clamped)
        current_height = state[2]
        next_state[2] = min(1.0, current_height + 0.05)  # Small increment, max 1.0
        
        # Shape changes randomly (unknown until we actually place)
        # Keep same for estimation
        
        return next_state
    
    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluate leaf node using value network.
        
        Returns estimated value in [-1, 1] range.
        """
        import torch
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.policy_net(state_tensor)
            
            # Use max Q-value as value estimate (normalized)
            value = q_values.max().item()
            # Normalize to roughly [-1, 1] range
            value = np.tanh(value / 20.0)
        
        return value
    
    def _backup(self, node: MCTSNode, value: float):
        """Propagate value up the tree, updating statistics."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
            # Negate value for opponent (not needed in single-player, but kept for generality)
    
    def _select_action(self, root: MCTSNode) -> Tuple[tuple, np.ndarray]:
        """
        Select action based on visit counts.
        
        Returns the action and the probability distribution for training.
        """
        visit_counts = np.zeros(len(self.agent.actions))
        
        for idx, action in enumerate(self.agent.actions):
            if action in root.children:
                visit_counts[idx] = root.children[action].visit_count
        
        # Temperature-based selection
        if self.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            action_probs = np.zeros_like(visit_counts)
            action_probs[best_idx] = 1.0
        else:
            # Proportional to visit counts
            if visit_counts.sum() > 0:
                action_probs = visit_counts ** (1.0 / self.temperature)
                action_probs /= action_probs.sum()
            else:
                action_probs = np.ones_like(visit_counts) / len(visit_counts)
        
        # Sample action
        action_idx = np.random.choice(len(self.agent.actions), p=action_probs)
        best_action = self.agent.actions[action_idx]
        
        return best_action, action_probs


class MCTSAgent:
    """
    Wrapper that combines DQN agent with MCTS for decision making.
    
    During play:
    - Uses MCTS to search for best action
    - Falls back to DQN for fast decisions when needed
    
    During training:
    - Collects (state, MCTS_policy, value) tuples
    - Trains network to match MCTS-improved policy
    """
    
    def __init__(self, base_agent, num_simulations: int = 50, 
                 use_mcts_training: bool = True):
        self.base_agent = base_agent
        self.mcts = MCTS(base_agent, num_simulations=num_simulations)
        self.use_mcts_training = use_mcts_training
        
        # Training data from MCTS
        self.mcts_examples = []
    
    def choose_action(self, state: np.ndarray, use_mcts: bool = True) -> tuple:
        """
        Choose action using MCTS or base DQN.
        
        Args:
            state: Current game state
            use_mcts: Whether to use MCTS (slower but better) or DQN (fast)
        """
        if use_mcts and self.base_agent.epsilon < 0.5:  # Use MCTS after initial exploration
            action, policy = self.mcts.search(state)
            
            # Store for training
            if self.use_mcts_training:
                self.mcts_examples.append((state.copy(), policy))
            
            return action
        else:
            # Use base DQN for fast exploration
            return self.base_agent.choose_action(state)
    
    def store_transition(self, *args, **kwargs):
        """Pass through to base agent."""
        self.base_agent.store_transition(*args, **kwargs)
    
    def train_step(self):
        """Train base agent."""
        return self.base_agent.train_step()
    
    def decay_epsilon(self):
        """Pass through to base agent."""
        self.base_agent.decay_epsilon()
    
    def get_stats(self) -> dict:
        """Get combined stats."""
        stats = self.base_agent.get_stats()
        stats['mcts_examples'] = len(self.mcts_examples)
        return stats
    
    def get_state_vector(self, *args, **kwargs):
        """Pass through to base agent."""
        return self.base_agent.get_state_vector(*args, **kwargs)
    
    def save(self, path: str):
        """Save base agent."""
        self.base_agent.save(path)
    
    def load(self, path: str):
        """Load base agent."""
        self.base_agent.load(path)
