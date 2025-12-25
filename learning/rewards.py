"""
The Village - Reward System
Calculates rewards for villager actions and states.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.villager import Villager
    from simulation.world import World
    from simulation.actions import ActionResult


def calculate_state_reward(villager: 'Villager', config) -> float:
    """
    Calculate reward based on villager's current state.
    Called every tick to provide continuous learning signal.
    """
    reward = 0.0
    
    # Survival reward
    if villager.is_alive:
        reward += config.reward_survival
    
    # Fed reward/penalty
    if villager.hunger < 30:
        reward += config.reward_fed
    elif villager.hunger > 80:
        reward += config.penalty_starving
    
    # Housed reward/penalty
    if villager.has_home:
        reward += config.reward_housed
    else:
        reward += config.penalty_homeless
    
    # Health bonus
    if villager.health > 80:
        reward += 0.05
    elif villager.health < 30:
        reward -= 0.1
    
    # Happiness bonus
    if villager.happiness > 70:
        reward += 0.03
    elif villager.happiness < 30:
        reward -= 0.05
    
    return reward


def calculate_action_reward(
    villager: 'Villager',
    action_result: 'ActionResult',
    world: 'World'
) -> float:
    """
    Calculate total reward for an action, combining base reward with context.
    """
    reward = action_result.reward
    
    # Bonus for addressing urgent needs
    if action_result.hunger_change < 0 and villager.hunger > 70:
        reward += 0.2  # Extra reward for eating when very hungry
    
    if action_result.health_change > 0 and villager.health < 50:
        reward += 0.2  # Extra reward for healing when hurt
    
    # Penalty for ignoring urgent needs
    if villager.hunger > 90 and action_result.hunger_change >= 0:
        reward -= 0.1  # Penalty for not eating when starving
    
    # Village-level bonuses
    if len(world.villagers) > 0:
        avg_happiness = sum(v.happiness for v in world.villagers) / len(world.villagers)
        if avg_happiness > 60:
            reward += 0.02  # Small bonus when village is happy
    
    return reward


def calculate_village_stability(world: 'World') -> float:
    """
    Calculate overall village stability (0-1 scale).
    Used for HUD display and global reward modifiers.
    """
    if len(world.villagers) == 0:
        return 0.0
    
    # Factors
    population_stability = min(1.0, len(world.villagers) / 20)  # Stable above 20
    
    avg_health = sum(v.health for v in world.villagers) / len(world.villagers)
    health_factor = avg_health / 100.0
    
    avg_happiness = sum(v.happiness for v in world.villagers) / len(world.villagers)
    happiness_factor = avg_happiness / 100.0
    
    food_factor = min(1.0, world.resources.food / 200)  # Stable above 200 food
    
    housing_ratio = min(1.0, world.resources.housing_capacity / max(1, len(world.villagers)))
    
    # Weighted combination
    stability = (
        population_stability * 0.2 +
        health_factor * 0.25 +
        happiness_factor * 0.15 +
        food_factor * 0.25 +
        housing_ratio * 0.15
    )
    
    return stability


def get_stability_description(stability: float) -> str:
    """Get human-readable stability description."""
    if stability > 0.8:
        return "Thriving"
    elif stability > 0.6:
        return "Stable"
    elif stability > 0.4:
        return "Struggling"
    elif stability > 0.2:
        return "Critical"
    else:
        return "Collapse"
