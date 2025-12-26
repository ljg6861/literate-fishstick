import numpy as np
import math
import pygame
from .sim import TowerSimulation

class TowerEnv:
    """
    Wrapper for TowerSimulation to expose a standard RL interface.
    """
    def __init__(self, config):
        self.config = config
        self.width = 800
        self.height = 600
        self.sim = TowerSimulation(self.width, self.height)
        
        # State tracking
        self.prev_top_x = self.width // 2
        self.prev_top_angle = 0.0
        self.current_tower_height = 0
        self.max_height_record = 0
        
    def reset(self):
        self.sim.reset()
        self.prev_top_x = self.width // 2
        self.prev_top_angle = 0.0
        self.current_tower_height = 0
        
        return self._get_state_vector()
        
    def step(self, action_idx):
        # Decode action
        # Actions are flat index [0..64] -> (x, rot)
        n_rots = len(self.config.rotations)
        x_idx = action_idx // n_rots
        rot_idx = action_idx % n_rots
        
        place_x = self.config.x_positions[x_idx]
        rotation = self.config.rotations[rot_idx]
        
        # Execute in Sim
        current_block = self.sim.next_block
        self.sim.spawn_block(place_x, current_block) # Difficulty tier handled inside sim if needed
        # Apply rotation (hacky: set after spawn)
        if self.sim.blocks:
            self.sim.blocks[-1].angle = rotation
            
        # Fast Forward physics
        # We need to simulate until settled or fell
        tower_fell = False
        steps = 0
        max_steps = 200 # Safety limit
        
        while steps < max_steps:
             self.sim.step()
             if self.sim.game_over:
                 tower_fell = True
                 break
             
             # Check settled
             if self.sim.blocks:
                 last = self.sim.blocks[-1]
                 if last.velocity.length_squared < 0.1 and abs(last.angular_velocity) < 0.1:
                     break
             steps += 1
             
        # Calculate Reward
        reward = self._calculate_reward(tower_fell, place_x)
        done = tower_fell
        
        # Update trackers
        if not tower_fell:
            self.current_tower_height += 1
            top = self.sim.blocks[-1]
            self.prev_top_x = top.position.x
            self.prev_top_angle = top.angle
            if self.current_tower_height > self.max_height_record:
                self.max_height_record = self.current_tower_height
        
        next_state = self._get_state_vector()
        
        return next_state, reward, done
    
    def render(self, screen):
        # Calculate camera
        camera_y = 0
        if self.sim.highest_point < 200:
            camera_y = -(self.sim.highest_point - 200)
            if camera_y < 0: camera_y = 0
            
        self.sim.render(screen, scroll_y=camera_y)
        return camera_y

    def _get_state_vector(self):
        # Same logic as old DeepTowerAgent.get_state_vector
        # [rel_x, angle, height, shape, lean, stability]
        
        center_x = self.width // 2
        
        # 1. Rel X
        rel_x = (self.prev_top_x - center_x) / (self.width / 2)
        rel_x = np.clip(rel_x, -1.0, 1.0)
        
        # 2. Angle
        angle_norm = math.degrees(self.prev_top_angle) / 45.0
        angle_norm = np.clip(angle_norm, -1.0, 1.0)
        
        # 3. Height
        height_norm = np.log1p(self.current_tower_height) / 5.0
        height_norm = np.clip(height_norm, 0.0, 1.0)
        
        # 4. Shape
        next_shape = self.sim.next_block['shape'] if self.sim.next_block else 0
        shape_norm = next_shape / 3.0
        
        # 5. Lean (CoM)
        total_mass = 0
        weighted_pos = 0
        for body in self.sim.blocks:
            weighted_pos += body.position.x * body.mass
            total_mass += body.mass
        tower_com_x = weighted_pos / total_mass if total_mass > 0 else center_x
        lean_norm = (tower_com_x - center_x) / (self.width / 4)
        lean_norm = np.clip(lean_norm, -1.0, 1.0)
        
        # 6. Stability (Kinetic)
        metrics = self.sim.get_stability_metrics()
        ke = metrics['total_kinetic_energy']
        stability_norm = np.log1p(ke) / 5.0
        stability_norm = np.clip(stability_norm, 0.0, 1.0)
        
        return np.array([rel_x, angle_norm, height_norm, shape_norm, lean_norm, stability_norm], dtype=np.float32)

    def _calculate_reward(self, fell, place_x):
        if fell: return -100.0
        
        # Stats
        center_x = self.width // 2
        total_mass = 0
        weighted_pos = 0
        for body in self.sim.blocks:
            weighted_pos += body.position.x * body.mass
            total_mass += body.mass
        tower_com_x = weighted_pos / total_mass if total_mass > 0 else center_x
        
        # Components
        r_survival = 5.0
        
        dist_from_com = abs(place_x - tower_com_x)
        r_alignment = (50 - dist_from_com) / 10.0
        
        dist_lean = abs(tower_com_x - center_x)
        r_balance = -dist_lean * 0.5
        
        top_body = self.sim.blocks[-1]
        r_stability = (1.0 - abs(math.sin(top_body.angle))) * 5
        
        metrics = self.sim.get_stability_metrics()
        ke = metrics['total_kinetic_energy']
        r_kinetic = -math.log1p(max(0, ke)) * 0.5
        
        return r_survival + r_alignment + r_balance + r_stability + r_kinetic
