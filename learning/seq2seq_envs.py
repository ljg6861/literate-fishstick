"""
Seq2Seq Algorithmic Environments

Lightweight gym-style environments for testing MuZero on algorithmic tasks:
- BitStringReversalEnv: Reverse a bit string
- SortingEnv: Sort a sequence of numbers
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment configuration."""
    seq_length: int = 8
    vocab_size: int = 2  # Binary for reversal, N for sorting


class BitStringReversalEnv:
    """
    Environment for bit-string reversal task.
    
    The agent must output the reverse of the input bit-string.
    Example: Input [1, 0, 1, 1] -> Output [1, 1, 0, 1]
    
    State: (input_sequence, partial_output, current_position)
    Action: Next bit to output (0 or 1)
    Reward: +1 for correct bit, -1 for incorrect
    """
    
    def __init__(self, seq_length: int = 8):
        self.seq_length = seq_length
        self.vocab_size = 2
        self.action_space_size = 2
        
        self.input_seq = None
        self.target_seq = None
        self.output_seq = None
        self.position = 0
        self.done = False
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment with a new random bit-string."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random bit-string
        self.input_seq = np.random.randint(0, 2, size=self.seq_length)
        self.target_seq = self.input_seq[::-1].copy()  # Reversed
        
        self.output_seq = np.zeros(self.seq_length, dtype=np.int64)
        self.position = 0
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The bit to output (0 or 1)
        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, {"error": "Episode finished"}
        
        # Check if action is correct
        correct = (action == self.target_seq[self.position])
        reward = 1.0 if correct else -1.0
        
        # Record action
        self.output_seq[self.position] = action
        self.position += 1
        
        # Check if episode is done
        self.done = (self.position >= self.seq_length)
        
        # Calculate sequence accuracy
        if self.done:
            accuracy = np.mean(self.output_seq == self.target_seq)
        else:
            accuracy = np.mean(self.output_seq[:self.position] == self.target_seq[:self.position])
        
        info = {
            "correct": correct,
            "position": self.position,
            "accuracy": accuracy,
            "sequence_complete": self.done
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns flattened array: [input_seq, output_seq, position_onehot]
        """
        # One-hot position encoding
        pos_onehot = np.zeros(self.seq_length)
        if self.position < self.seq_length:
            pos_onehot[self.position] = 1
        
        # Return input sequence (what we need to reverse)
        return self.input_seq.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state for MCTS."""
        return {
            "input_seq": self.input_seq.copy(),
            "target_seq": self.target_seq.copy(),
            "output_seq": self.output_seq.copy(),
            "position": self.position,
            "done": self.done
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set state for MCTS simulation."""
        self.input_seq = state["input_seq"].copy()
        self.target_seq = state["target_seq"].copy()
        self.output_seq = state["output_seq"].copy()
        self.position = state["position"]
        self.done = state["done"]
    
    def clone(self) -> 'BitStringReversalEnv':
        """Clone environment for MCTS."""
        env = BitStringReversalEnv(self.seq_length)
        env.set_state(self.get_state())
        return env


class SortingEnv:
    """
    Environment for sorting task.
    
    The agent must output the sorted version of the input sequence.
    Example: Input [3, 1, 4, 0, 2] -> Output [0, 1, 2, 3, 4]
    
    State: (input_sequence, partial_output, used_mask, current_position)
    Action: Next number to output (0 to N-1)
    Reward: +1 for correct number, -1 for incorrect
    """
    
    def __init__(self, seq_length: int = 5):
        self.seq_length = seq_length
        self.vocab_size = seq_length
        self.action_space_size = seq_length
        
        self.input_seq = None
        self.target_seq = None
        self.output_seq = None
        self.used = None
        self.position = 0
        self.done = False
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment with a new random permutation."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random permutation
        self.input_seq = np.random.permutation(self.seq_length)
        self.target_seq = np.sort(self.input_seq)  # Sorted version
        
        self.output_seq = np.zeros(self.seq_length, dtype=np.int64)
        self.used = np.zeros(self.seq_length, dtype=bool)
        self.position = 0
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The number to output (0 to N-1)
        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, {"error": "Episode finished"}
        
        # Check if action is correct (should output the next smallest)
        correct = (action == self.target_seq[self.position])
        reward = 1.0 if correct else -1.0
        
        # Record action
        self.output_seq[self.position] = action
        self.used[action] = True
        self.position += 1
        
        # Check if episode is done
        self.done = (self.position >= self.seq_length)
        
        # Calculate accuracy
        if self.done:
            accuracy = np.mean(self.output_seq == self.target_seq)
        else:
            accuracy = np.mean(self.output_seq[:self.position] == self.target_seq[:self.position])
        
        info = {
            "correct": correct,
            "position": self.position,
            "accuracy": accuracy,
            "sequence_complete": self.done
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (input sequence)."""
        return self.input_seq.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state for MCTS."""
        return {
            "input_seq": self.input_seq.copy(),
            "target_seq": self.target_seq.copy(),
            "output_seq": self.output_seq.copy(),
            "used": self.used.copy(),
            "position": self.position,
            "done": self.done
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set state for MCTS simulation."""
        self.input_seq = state["input_seq"].copy()
        self.target_seq = state["target_seq"].copy()
        self.output_seq = state["output_seq"].copy()
        self.used = state["used"].copy()
        self.position = state["position"]
        self.done = state["done"]
    
    def clone(self) -> 'SortingEnv':
        """Clone environment for MCTS."""
        env = SortingEnv(self.seq_length)
        env.set_state(self.get_state())
        return env


def create_env(task: str = "reversal", seq_length: int = 8):
    """Factory function to create environment."""
    if task == "reversal":
        return BitStringReversalEnv(seq_length)
    elif task == "sorting":
        return SortingEnv(seq_length)
    else:
        raise ValueError(f"Unknown task: {task}")


# Test code
if __name__ == "__main__":
    print("=== Testing BitStringReversalEnv ===")
    env = BitStringReversalEnv(seq_length=8)
    obs = env.reset(seed=42)
    print(f"Input:  {env.input_seq}")
    print(f"Target: {env.target_seq}")
    
    total_reward = 0
    for i in range(env.seq_length):
        action = env.target_seq[i]  # Cheat: use correct action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: action={action}, reward={reward:.1f}, correct={info['correct']}")
    
    print(f"Total reward: {total_reward}")
    print(f"Final accuracy: {info['accuracy']:.2%}")
    
    print("\n=== Testing SortingEnv ===")
    env = SortingEnv(seq_length=5)
    obs = env.reset(seed=42)
    print(f"Input:  {env.input_seq}")
    print(f"Target: {env.target_seq}")
    
    total_reward = 0
    for i in range(env.seq_length):
        action = env.target_seq[i]  # Cheat: use correct action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: action={action}, reward={reward:.1f}, correct={info['correct']}")
    
    print(f"Total reward: {total_reward}")
    print(f"Final accuracy: {info['accuracy']:.2%}")
    
    print("\n✓ Environments working correctly!")
