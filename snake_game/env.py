
import numpy as np
from .sim import SnakeSim

class SnakeEnv:
    def __init__(self, config):
        self.config = config
        self.sim = SnakeSim()
        self.max_score_record = 0
        
    def reset(self):
        self.sim.reset()
        return self._get_state_vector()
        
    def step(self, action):
        reward, done = self.sim.step(action)
        
        if self.sim.score > self.max_score_record:
            self.max_score_record = self.sim.score
            
        return self._get_state_vector(), reward, done
    
    def render(self, screen):
        self.sim.render(screen)
        
    def _get_state_vector(self):
        # [HeadX, HeadY, FoodX, FoodY, DangerL, DangerR, DangerU, DangerD, DirL, DirR, DirU, DirD]
        head = self.sim.head
        food = self.sim.food
        w, h = self.sim.grid_w, self.sim.grid_h
        
        # 1. Head Pos (Normalized)
        head_x = head[0] / w
        head_y = head[1] / h
        
        # 2. Food Pos (Relative to head is usually better, but absolute works)
        # Let's do Relative vector to food
        rel_food_x = (food[0] - head[0]) / w
        rel_food_y = (food[1] - head[1]) / h
        
        # 3. Dangers (Immediate surroundings)
        # Check U, D, L, R
        def is_danger(x, y):
            if x < 0 or x >= w or y < 0 or y >= h: return 1.0
            if [x, y] in self.sim.body[:-1]: return 1.0 # Self collision
            return 0.0
            
        danger_u = is_danger(head[0], head[1]-1)
        danger_d = is_danger(head[0], head[1]+1)
        danger_l = is_danger(head[0]-1, head[1])
        danger_r = is_danger(head[0]+1, head[1])
        
        # 4. Direction (One-hot)
        # 0: Up, 1: Down, 2: Left, 3: Right
        d = self.sim.direction
        dir_u = 1.0 if d == 0 else 0.0
        dir_d = 1.0 if d == 1 else 0.0
        dir_l = 1.0 if d == 2 else 0.0
        dir_r = 1.0 if d == 3 else 0.0
        
        return np.array([
            head_x, head_y, 
            rel_food_x, rel_food_y,
            danger_l, danger_r, danger_u, danger_d,
            dir_l, dir_r, dir_u, dir_d
        ], dtype=np.float32)
