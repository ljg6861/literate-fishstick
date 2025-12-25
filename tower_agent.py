import random
import math

class TowerAgent:
    def __init__(self):
        self.q_table = {} # (state) -> [action_values]
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # High exploration to start
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Discretize actions: offsets from center [-50, 50]
        self.actions = list(range(-50, 51, 10)) 
        
    def get_state(self, top_block_x, center_x):
        """Convert continuous world state to discrete key."""
        # Relative position of the previous block
        if top_block_x is None:
            return "start"
            
        rel_x = int(top_block_x - center_x)
        # Bucket into 10-pixel chunks
        rel_x = round(rel_x / 10) * 10
        return str(rel_x)
        
    def choose_action(self, state):
        """Choose an action (x offset) based on state."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
            
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
            
        # Argmax
        values = self.q_table[state]
        max_val = max(values)
        best_indices = [i for i, v in enumerate(values) if v == max_val]
        action_idx = random.choice(best_indices)
        
        return self.actions[action_idx]
        
    def learn(self, state, action, reward, next_state):
        """Update Q-values."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * len(self.actions)
            
        action_idx = self.actions.index(action)
        
        old_value = self.q_table[state][action_idx]
        next_max = max(self.q_table[next_state])
        
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action_idx] = new_value
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_confidence(self, state):
        """Get the max Q-value for the current state (proxy for confidence)."""
        if state not in self.q_table:
            return 0.0
        return max(self.q_table[state])
