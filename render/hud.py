"""
The Village - HUD (Heads-Up Display)
Clean, minimal information display.
"""

import pygame
from config import RENDER_CONFIG
from learning.rewards import calculate_village_stability, get_stability_description


class HUD:
    """Minimal heads-up display for village stats."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = RENDER_CONFIG
        
        # Initialize font
        pygame.font.init()
        self.font = pygame.font.SysFont('Consolas', 20)
        self.large_font = pygame.font.SysFont('Consolas', 28)
        self.small_font = pygame.font.SysFont('Consolas', 14)
        
        # HUD position (top-left, minimal)
        self.padding = 15
    
    def draw(self, screen: pygame.Surface, world):
        """Draw the minimal HUD on screen."""
        # Top-left: Evolution and Population
        evolution_text = f"Evolution {world.evolution_number}"
        pop_text = f"Population: {len(world.villagers)}"
        
        # Single line status
        stability = calculate_village_stability(world)
        status = get_stability_description(stability)
        
        # Draw evolution number (larger)
        evo_surface = self.large_font.render(evolution_text, True, self.config.color_foreground)
        screen.blit(evo_surface, (self.padding, self.padding))
        
        # Draw population below
        pop_surface = self.font.render(pop_text, True, self.config.color_accent)
        screen.blit(pop_surface, (self.padding, self.padding + 35))
        
        # Draw status
        status_surface = self.font.render(status, True, self.config.color_accent)
        screen.blit(status_surface, (self.padding, self.padding + 60))
        
        # Bottom-left: Resource summary (very compact)
        food_icon = "●" if world.resources.food > 50 else "○"
        housing_icon = "■" if world.resources.housing_capacity >= len(world.villagers) else "□"
        
        resources_text = f"{food_icon} Food: {int(world.resources.food)}   {housing_icon} Housing: {world.resources.housing_capacity}"
        res_surface = self.small_font.render(resources_text, True, self.config.color_dim)
        screen.blit(res_surface, (self.padding, self.screen_height - 30))
        
        # Bottom-right: Tick (very dim)
        tick_text = f"Tick {world.tick}"
        tick_surface = self.small_font.render(tick_text, True, self.config.color_dim)
        screen.blit(tick_surface, (self.screen_width - tick_surface.get_width() - self.padding, 
                                   self.screen_height - 30))
