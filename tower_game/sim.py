import pygame
import pymunk
import pymunk.pygame_util
import random
import math

class TowerSimulation:
    # Block shape types
    SHAPE_SQUARE = 0
    SHAPE_WIDE = 1
    SHAPE_TALL = 2
    SHAPE_TRIANGLE = 3
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Physics setup
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)
        self.space.sleep_time_threshold = 0.5
        
        # Drawing setup
        self.draw_options = pymunk.pygame_util.DrawOptions(None) # Set surface later
        
        # Game state
        self.ground = None
        self.blocks = []
        self.game_over = False
        self.score = 0
        self.highest_point = height
        
        # Next block preview
        self.next_block = None
        
        self.reset()
        
    def reset(self):
        """Reset the simulation."""
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)
        
        self.blocks = []
        self.game_over = False
        self.score = 0
        self.highest_point = self.height - 50 # Ground level roughly
        
        # Create ground
        self.ground = pymunk.Segment(self.space.static_body, (50, self.height-50), (self.width-50, self.height-50), 5.0)
        self.ground.friction = 1.0
        self.space.add(self.ground)
        
        # Generate first next block
        self.next_block = self.generate_next_block()
    
    def generate_next_block(self, difficulty_tier=1):
        """
        Generate random block properties based on difficulty.
        Tier 1: Squares only (Easy)
        Tier 2: Squares + Wide/Tall (Medium)
        Tier 3: All shapes incl Triangles (Hard)
        """
        if difficulty_tier == 1:
            choices = [self.SHAPE_SQUARE]
        elif difficulty_tier == 2:
            choices = [self.SHAPE_SQUARE, self.SHAPE_WIDE, self.SHAPE_TALL]
        else:
            choices = [self.SHAPE_SQUARE, self.SHAPE_WIDE, self.SHAPE_TALL, self.SHAPE_TRIANGLE]
            
        shape_type = random.choice(choices)
        
        # Angle noise increases with difficulty
        angle_limit = 0.1 * difficulty_tier
        angle = random.uniform(-angle_limit, angle_limit)
        
        # Size based on shape
        if shape_type == self.SHAPE_SQUARE:
            w, h = 50, 50
        elif shape_type == self.SHAPE_WIDE:
            w, h = 80, 30
        elif shape_type == self.SHAPE_TALL:
            w, h = 30, 70
        else:  # TRIANGLE
            w, h = 60, 50
        
        return {'shape': shape_type, 'width': w, 'height': h, 'angle': angle}
        
    def spawn_block(self, x_pos, block_type=None, difficulty_tier=1):
        """Spawn a block at the given x position using block_type properties."""
        if self.game_over:
            return
        
        if block_type is None:
            block_type = self.next_block or self.generate_next_block(difficulty_tier)
        
        mass = 1
        w = block_type['width']
        h = block_type['height']
        angle = block_type['angle']
        shape_type = block_type['shape']
        
        if shape_type == self.SHAPE_TRIANGLE:
            # Triangle vertices
            vertices = [(-w/2, h/2), (w/2, h/2), (0, -h/2)]
            moment = pymunk.moment_for_poly(mass, vertices)
            body = pymunk.Body(mass, moment)
            body.position = (x_pos, 100)
            body.angle = angle
            shape = pymunk.Poly(body, vertices)
        else:
            moment = pymunk.moment_for_box(mass, (w, h))
            body = pymunk.Body(mass, moment)
            body.position = (x_pos, 100)
            body.angle = angle
            shape = pymunk.Poly.create_box(body, (w, h))
        
        shape.friction = 0.5
        shape.elasticity = 0.1
        
        self.space.add(body, shape)
        self.blocks.append(body)
        self.score += 1
        
        # Generate next block for preview
        self.next_block = self.generate_next_block(difficulty_tier)
        
    def step(self):
        """Advance physics one step."""
        if self.game_over:
            return

        dt = 1.0 / 60.0
        self.space.step(dt)
        
        # Check for game over
        # 1. Block falls off screen
        # 2. Block touches ground (except the first few?) - wait, Jenga logic is different
        # Let's say: Game Over if any block touches the ground *besides* the first one? 
        # Actually safer: Game Over if center of mass of any block goes below a threshold AND it's not the base.
        # Simple start: usage "lives" or "height". 
        # Let's stick to "If any block falls off the platform (left/right) or below ground."
        
        for body in self.blocks:
            # Check bounds
            if body.position.x < 0 or body.position.x > self.width:
                self.game_over = True
            if body.position.y > self.height:
                self.game_over = True
                
        # Update highest point for reward
        min_y = self.height
        for body in self.blocks:
            min_y = min(min_y, body.position.y)
        self.highest_point = min_y

    def get_stability_metrics(self):
        """
        Calculate kinetic energy and movement stats to detect wobbling.
        Returns dict with:
        - total_kinetic_energy: Sum of 0.5 * m * v^2 + 0.5 * I * w^2
        - max_velocity: Highest velocity magnitude among blocks
        """
        total_ke = 0.0
        max_v = 0.0
        
        for body in self.blocks:
            # Linear Kinetic Energy: 0.5 * m * v^2
            v_sq = body.velocity.length_squared
            ke_linear = 0.5 * body.mass * v_sq
            
            # Rotational Kinetic Energy: 0.5 * I * w^2
            ke_angular = 0.5 * body.moment * (body.angular_velocity ** 2)
            
            total_ke += ke_linear + ke_angular
            max_v = max(max_v, body.velocity.length)
            
        return {
            'total_kinetic_energy': total_ke,
            'max_velocity': max_v
        }

    def render(self, screen, scroll_y=0):
        """Draw the simulation to the screen."""
        screen.fill((30, 30, 30))
        
        # Apply camera transform
        # We want to shift everything DOWN by scroll_y (since y decreases as we go up)
        transform = pymunk.Transform.translation(0, scroll_y)
        self.draw_options.transform = transform
        self.draw_options.surface = screen
        
        self.space.debug_draw(self.draw_options)
        
        # Draw HUD (Static, not affected by camera)
        # return text surface so caller can blit it? Or just blit here.
        # Caller handles specific HUD text now.

