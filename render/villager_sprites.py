"""
The Village - Villager Sprites
Visual representation of villagers.
Phase 2: Show illness and working state.
"""

import pygame
import math
from config import RENDER_CONFIG
from simulation.villager import Role, IllnessState


class VillagerRenderer:
    """Renders villagers as color-coded dots."""
    
    def __init__(self):
        self.config = RENDER_CONFIG
        self.base_size = 6
    
    def draw_all(self, screen: pygame.Surface, villagers: list, 
                 camera_offset: tuple = (0, 0)):
        """Draw all villagers."""
        for villager in villagers:
            self._draw_villager(screen, villager, camera_offset)
    
    def _draw_villager(self, screen: pygame.Surface, villager, camera_offset: tuple):
        """Draw a single villager."""
        # Update visual position for smooth movement
        villager.update_visual_position()
        
        # Get screen position
        x = int(villager.visual_position[0] - camera_offset[0])
        y = int(villager.visual_position[1] - camera_offset[1])
        
        # Skip if off screen
        margin = 50
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        if x < -margin or x > screen_width + margin or y < -margin or y > screen_height + margin:
            return
        
        # Calculate color based on state
        intensity = villager.state_color_intensity
        
        # Base color is grayscale based on intensity
        gray_value = int(50 + intensity * 205)
        color = (gray_value, gray_value, gray_value)
        
        # Phase 2: Ill villagers have a reddish tint
        if villager.illness == IllnessState.SEVERE:
            color = (min(255, gray_value + 50), max(0, gray_value - 50), max(0, gray_value - 50))
        elif villager.illness == IllnessState.MILD:
            color = (min(255, gray_value + 20), gray_value, max(0, gray_value - 20))
        
        # Size based on role and state
        size = self.base_size
        if villager.is_child:
            size = 4
        elif villager.is_elder:
            size = 5
        
        # Draw the villager dot
        pygame.draw.circle(screen, color, (x, y), size)
        
        # Draw role indicator
        self._draw_role_indicator(screen, villager, x, y, size)
        
        # Phase 2: Draw task direction line if working
        if villager.is_working and villager.assigned_task is not None:
            target = villager.assigned_task.target_position
            target_x = int(target[0] - camera_offset[0])
            target_y = int(target[1] - camera_offset[1])
            
            # Draw faint line toward target
            line_length = min(30, math.sqrt((target_x - x)**2 + (target_y - y)**2))
            if line_length > 5:
                dx = target_x - x
                dy = target_y - y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    end_x = x + int(dx / dist * line_length)
                    end_y = y + int(dy / dist * line_length)
                    pygame.draw.line(screen, self.config.color_dim, (x, y), (end_x, end_y), 1)
    
    def _draw_role_indicator(self, screen: pygame.Surface, villager, x: int, y: int, size: int):
        """Draw a subtle role indicator."""
        if villager.is_child:
            return  # No indicator for children
        
        indicator_offset = size + 2
        indicator_size = 2
        
        if villager.role == Role.BUILDER:
            # Small square above
            pygame.draw.rect(screen, self.config.color_accent,
                           pygame.Rect(x - 1, y - indicator_offset - 2, 3, 3), 1)
        elif villager.role == Role.FARMER:
            # Small line below
            pygame.draw.line(screen, self.config.color_accent,
                           (x - 3, y + indicator_offset), (x + 3, y + indicator_offset), 1)
        elif villager.role == Role.THINKER:
            # Small dot
            pygame.draw.circle(screen, self.config.color_accent, 
                             (x, y - indicator_offset), 1)
        elif villager.role == Role.HEALER:
            # Small cross
            pygame.draw.line(screen, self.config.color_accent,
                           (x - 2, y - indicator_offset), (x + 2, y - indicator_offset), 1)
            pygame.draw.line(screen, self.config.color_accent,
                           (x, y - indicator_offset - 2), (x, y - indicator_offset + 2), 1)
