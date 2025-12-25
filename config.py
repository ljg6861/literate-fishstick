"""
The Village - Configuration
All simulation parameters in one place.
"""

from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class SimulationConfig:
    """Core simulation parameters."""
    
    # Population
    initial_population: int = 40
    max_population: int = 200
    
    # Time
    ticks_per_second_normal: int = 10
    ticks_per_second_accelerated: int = 100
    render_fps: int = 60
    
    # Villager defaults
    initial_health: float = 100.0
    initial_happiness: float = 70.0
    initial_hunger: float = 20.0
    initial_skill: float = 10.0
    memory_size: int = 20  # Last N events
    
    # Aging
    child_age_threshold: float = 18.0
    elder_age_threshold: float = 65.0
    max_age: float = 85.0
    aging_per_tick: float = 0.003  # 1 tick = 1 day (Year ~333 ticks)
    
    # Needs rates (per tick)
    hunger_increase_rate: float = 0.5  # Higher daily consumption
    health_decay_when_starving: float = 0.5  # Fast death without food
    happiness_decay_homeless: float = 0.05
    happiness_decay_hungry: float = 0.05
    
    # Resources
    initial_food: float = 200.0
    initial_housing_capacity: int = 20
    food_per_farm_tick: float = 0.1  # Low yield (Scarcity)
    housing_per_building: int = 4
    
    # Buildings
    building_labor_cost: float = 800.0  # Massive project (Hardcore)
    max_builders_per_building: int = 5
    
    # Learning
    learning_rate: float = 0.1
    exploration_temperature: float = 1.0
    reward_survival: float = 0.1
    reward_fed: float = 0.2
    reward_housed: float = 0.1
    penalty_starving: float = -0.5
    penalty_homeless: float = -0.2
    
    # Persistence
    auto_save_interval_seconds: float = 300.0  # 5 minutes
    save_directory: str = "saves"
    
    # Random seed (None = random)
    seed: Optional[int] = None
    
    def get_seed(self) -> int:
        """Get or generate random seed."""
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        return self.seed


@dataclass  
class RenderConfig:
    """Rendering parameters."""
    
    # Display
    fullscreen: bool = True
    window_title: str = "The Village"
    
    # Colors (black and white palette)
    color_background: tuple = (10, 10, 10)  # Near black
    color_foreground: tuple = (240, 240, 240)  # Near white
    color_accent: tuple = (180, 180, 180)  # Gray
    color_dim: tuple = (80, 80, 80)  # Dark gray
    color_warning: tuple = (120, 120, 120)  # Medium gray for emphasis
    
    # Villager visualization
    villager_radius: int = 4
    villager_healthy_color: tuple = (240, 240, 240)  # White
    villager_hungry_color: tuple = (160, 160, 160)  # Light gray
    villager_unhappy_color: tuple = (100, 100, 100)  # Medium gray
    villager_critical_color: tuple = (60, 60, 60)  # Dark gray
    
    # Building visualization
    building_housing_color: tuple = (200, 200, 200)
    building_farm_color: tuple = (180, 180, 180)
    building_workshop_color: tuple = (160, 160, 160)
    building_school_color: tuple = (140, 140, 140)
    building_size: int = 40
    
    # HUD
    hud_font_size: int = 18
    hud_padding: int = 20
    hud_opacity: int = 200
    
    # Overlays
    overlay_font_size: int = 24
    overlay_fade_duration: float = 3.0
    overlay_display_duration: float = 5.0
    
    # Camera
    camera_smoothing: float = 0.05  # Lower = slower


# Global config instances
CONFIG = SimulationConfig()
RENDER_CONFIG = RenderConfig()
