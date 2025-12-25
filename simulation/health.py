"""
The Village - Health System
Disease spreading, illness states, and recovery mechanics.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import random
import math

if TYPE_CHECKING:
    from simulation.villager import Villager
    from simulation.world import World


class IllnessState(Enum):
    """Illness severity levels."""
    NONE = auto()
    MILD = auto()      # Slight productivity loss
    SEVERE = auto()    # Major productivity loss, can spread
    CHRONIC = auto()   # Permanent, low-level illness


@dataclass
class IllnessConfig:
    """Configuration for disease mechanics."""
    # Spreading
    spread_radius: float = 50.0  # Distance for proximity spread
    spread_chance_base: float = 0.002  # Per tick, per nearby ill villager
    spread_chance_severe: float = 0.008  # Higher for severe illness
    
    # Progression
    mild_to_severe_chance: float = 0.001  # Per tick
    severe_to_chronic_chance: float = 0.0005  # Per tick
    
    # Recovery
    natural_recovery_mild: float = 0.005  # Per tick without hospital
    natural_recovery_severe: float = 0.001  # Much harder to recover
    hospital_recovery_bonus: float = 5.0  # Multiplier with hospital treatment
    
    # Effects
    productivity_mild: float = 0.8  # 80% productivity
    productivity_severe: float = 0.4  # 40% productivity
    productivity_chronic: float = 0.7  # 70% productivity (permanent)
    
    # Crowding
    crowding_spread_multiplier: float = 1.5  # Extra spread in expanded housing


# Global config
ILLNESS_CONFIG = IllnessConfig()


def get_productivity_modifier(illness: IllnessState) -> float:
    """Get productivity modifier based on illness state."""
    modifiers = {
        IllnessState.NONE: 1.0,
        IllnessState.MILD: ILLNESS_CONFIG.productivity_mild,
        IllnessState.SEVERE: ILLNESS_CONFIG.productivity_severe,
        IllnessState.CHRONIC: ILLNESS_CONFIG.productivity_chronic,
    }
    return modifiers.get(illness, 1.0)


def spread_disease(villagers: List['Villager'], world: 'World'):
    """
    Spread disease based on proximity to ill villagers.
    Called each tick.
    """
    if not villagers:
        return
    
    # Find ill villagers
    ill_villagers = [v for v in villagers if v.illness != IllnessState.NONE]
    if not ill_villagers:
        return
    
    # Check housing crowding
    crowding_factor = 1.0
    housing_capacity = world.resources.housing_capacity
    if housing_capacity > 0:
        occupancy_ratio = len(villagers) / housing_capacity
        if occupancy_ratio > 1.0:
            crowding_factor = ILLNESS_CONFIG.crowding_spread_multiplier
    
    # For each healthy villager, check if they get infected
    for villager in villagers:
        if villager.illness != IllnessState.NONE:
            continue
        
        # Count nearby ill villagers
        for ill in ill_villagers:
            distance = math.sqrt(
                (villager.position[0] - ill.position[0]) ** 2 +
                (villager.position[1] - ill.position[1]) ** 2
            )
            
            if distance < ILLNESS_CONFIG.spread_radius:
                # Calculate spread chance
                if ill.illness == IllnessState.SEVERE:
                    chance = ILLNESS_CONFIG.spread_chance_severe
                else:
                    chance = ILLNESS_CONFIG.spread_chance_base
                
                chance *= crowding_factor
                
                if random.random() < chance:
                    villager.illness = IllnessState.MILD
                    villager.illness_duration = 0
                    break  # Only get sick once per tick


def progress_illness(villager: 'Villager'):
    """
    Progress or recover from illness.
    Called each tick for ill villagers.
    """
    if villager.illness == IllnessState.NONE:
        return
    
    villager.illness_duration += 1
    
    if villager.illness == IllnessState.MILD:
        # Can progress to severe
        if random.random() < ILLNESS_CONFIG.mild_to_severe_chance:
            villager.illness = IllnessState.SEVERE
            villager.illness_duration = 0
            
    elif villager.illness == IllnessState.SEVERE:
        # Can become chronic
        if random.random() < ILLNESS_CONFIG.severe_to_chronic_chance:
            villager.illness = IllnessState.CHRONIC
            villager.illness_duration = 0


def attempt_recovery(villager: 'Villager', has_hospital: bool, hospital_quality: float = 1.0):
    """
    Attempt natural or hospital-assisted recovery.
    Returns True if recovered.
    """
    if villager.illness == IllnessState.NONE:
        return True
    
    if villager.illness == IllnessState.CHRONIC:
        # Chronic illness doesn't fully recover
        return False
    
    # Base recovery chance
    if villager.illness == IllnessState.MILD:
        chance = ILLNESS_CONFIG.natural_recovery_mild
    else:
        chance = ILLNESS_CONFIG.natural_recovery_severe
    
    # Hospital bonus
    if has_hospital:
        chance *= ILLNESS_CONFIG.hospital_recovery_bonus * hospital_quality
    
    # Health affects recovery
    health_factor = villager.health / 100.0
    chance *= health_factor
    
    if random.random() < chance:
        villager.illness = IllnessState.NONE
        villager.illness_duration = 0
        return True
    
    return False


def apply_illness_effects(villager: 'Villager'):
    """Apply ongoing effects of illness."""
    if villager.illness == IllnessState.NONE:
        return
    
    # Update productivity modifier
    villager.productivity_modifier = get_productivity_modifier(villager.illness)
    
    # Health drain for severe illness
    if villager.illness == IllnessState.SEVERE:
        villager.health -= 0.05  # Slow drain
        
    # Happiness affected by illness
    if villager.illness == IllnessState.SEVERE:
        villager.happiness -= 0.1
    elif villager.illness == IllnessState.MILD:
        villager.happiness -= 0.02


def get_village_health_status(villagers: List['Villager']) -> dict:
    """Get aggregate health statistics for the village."""
    if not villagers:
        return {
            'healthy': 0,
            'mild': 0,
            'severe': 0,
            'chronic': 0,
            'infection_rate': 0.0
        }
    
    counts = {
        IllnessState.NONE: 0,
        IllnessState.MILD: 0,
        IllnessState.SEVERE: 0,
        IllnessState.CHRONIC: 0,
    }
    
    for v in villagers:
        counts[v.illness] += 1
    
    total = len(villagers)
    infected = total - counts[IllnessState.NONE]
    
    return {
        'healthy': counts[IllnessState.NONE],
        'mild': counts[IllnessState.MILD],
        'severe': counts[IllnessState.SEVERE],
        'chronic': counts[IllnessState.CHRONIC],
        'infection_rate': infected / total if total > 0 else 0.0
    }


def trigger_outbreak(villagers: List['Villager'], severity: float = 0.3):
    """
    Trigger a disease outbreak (player intervention or disaster).
    Infects a percentage of the population.
    """
    if not villagers:
        return 0
    
    target_count = max(1, int(len(villagers) * severity))
    healthy = [v for v in villagers if v.illness == IllnessState.NONE]
    
    if not healthy:
        return 0
    
    infected = 0
    for v in random.sample(healthy, min(target_count, len(healthy))):
        v.illness = IllnessState.MILD
        v.illness_duration = 0
        infected += 1
    
    return infected
