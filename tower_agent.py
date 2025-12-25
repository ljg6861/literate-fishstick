import random
import math

class TowerAgent:
    def __init__(self, screen_width=800):
        self.q_table = {} 
        self.learning_rate = 0.3  # Higher for faster adaptation
        self.discount_factor = 0.9  # Lower to focus more on immediate rewards
        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # Faster decay - less random exploration
        self.min_epsilon = 0.02
        
        # Actions: absolute X positions across the screen
        # Agent decides WHERE to place each block
        margin = 100
        self.actions = list(range(margin, screen_width - margin + 1, 50))  # [100, 150, 200, ..., 700]
        
    def get_state(self, top_block_x, top_block_angle, center_x, height=0, next_block_shape=0):
        """State includes position, angle, height tier, and next block shape."""
        if top_block_x is None:
            return "start"
            
        # 1. Relative X position (Bucket: 20px - coarser for better generalization)
        rel_x = int(top_block_x - center_x)
        rel_x = round(rel_x / 20) * 20
        # Clamp to prevent state explosion
        rel_x = max(-60, min(60, rel_x))
        
        # 2. Angle of the top block (Bucket: 10 degrees - coarser)
        degrees = math.degrees(top_block_angle)
        rel_angle = round(degrees / 10) * 10
        # Clamp angles
        rel_angle = max(-30, min(30, rel_angle))
        
        # 3. Height tier: low (0-20), mid (21-50), high (51+)
        if height <= 20:
            height_tier = "L"
        elif height <= 50:
            height_tier = "M"
        else:
            height_tier = "H"
        
        # 4. Next block shape type (0=square, 1=wide, 2=tall, 3=triangle)
        shape_code = ["S", "W", "T", "X"][next_block_shape]  # S=square, W=wide, T=tall, X=triangle
        
        # Combine into a string key
        return f"{rel_x}_{rel_angle}_{height_tier}_{shape_code}"
        
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
            
        if state not in self.q_table:
            return random.choice(self.actions)
            
        values = self.q_table[state]
        max_val = max(values)
        best_indices = [i for i, v in enumerate(values) if v == max_val]
        action_idx = random.choice(best_indices)
        
        return self.actions[action_idx]
        
    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * len(self.actions)
            
        action_idx = self.actions.index(action)
        
        old_value = self.q_table[state][action_idx]
        next_max = max(self.q_table[next_state])
        
        # standard Q-learning update
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action_idx] = new_value
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_confidence(self, state):
        if state not in self.q_table:
            return 0.0
        return max(self.q_table[state])