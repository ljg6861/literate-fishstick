import pygame
import pymunk
import pymunk.pygame_util
import random
import math

class TowerSimulation:
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
        
    def spawn_block(self, x_pos, angle=0.0):
        """Spawn a block at the given x position and angle."""
        if self.game_over:
            return
            
        mass = 1
        width = 50
        height = 50
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = (x_pos, 100) # Spawn near top
        body.angle = angle
        
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.friction = 0.5
        shape.elasticity = 0.1
        
        self.space.add(body, shape)
        self.blocks.append(body)
        self.score += 1
        
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

