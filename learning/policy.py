"""
The Village - Learning Policy
Multi-armed bandit style action selection and weight updates.
"""

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.villager import Villager
    from simulation.actions import Action


def softmax(weights: dict, temperature: float = 1.0) -> dict:
    """
    Compute softmax probabilities for action weights.
    Higher temperature = more exploration.
    """
    # Prevent overflow by subtracting max
    max_weight = max(weights.values()) if weights else 0
    
    exp_weights = {}
    for action, weight in weights.items():
        exp_weights[action] = math.exp((weight - max_weight) / max(0.01, temperature))
    
    total = sum(exp_weights.values())
    
    probabilities = {}
    for action, exp_weight in exp_weights.items():
        probabilities[action] = exp_weight / total if total > 0 else 1.0 / len(weights)
    
    return probabilities


def select_action(
    villager: 'Villager',
    available_actions: list['Action'],
    temperature: float = 1.0
) -> 'Action':
    """
    Select an action using softmax over weights, limited to available actions.
    """
    if not available_actions:
        from simulation.actions import Action
        return Action.WANDER
    
    # Filter weights to available actions
    available_weights = {
        action: villager.action_weights.get(action, 1.0)
        for action in available_actions
    }
    
    # Apply needs-based urgency modifiers
    from simulation.actions import Action
    
    # Boost eating when hungry
    if villager.hunger > 60 and Action.EAT in available_weights:
        available_weights[Action.EAT] *= (1 + villager.hunger / 50)
    
    # Boost resting when unhealthy
    if villager.health < 50 and Action.REST in available_weights:
        available_weights[Action.REST] *= (1 + (100 - villager.health) / 50)
    
    # Compute probabilities
    probs = softmax(available_weights, temperature)
    
    # Sample action
    r = random.random()
    cumulative = 0.0
    
    for action, prob in probs.items():
        cumulative += prob
        if r <= cumulative:
            return action
    
    # Fallback
    return list(available_weights.keys())[-1]


def update_weights(
    villager: 'Villager',
    action: 'Action',
    reward: float,
    learning_rate: float = 0.1
):
    """
    Update action weights based on reward signal.
    Simple reinforcement: weight += lr * reward
    """
    current_weight = villager.action_weights.get(action, 1.0)
    
    # Update with reward
    new_weight = current_weight + learning_rate * reward
    
    # Clamp to reasonable range
    new_weight = max(0.1, min(10.0, new_weight))
    
    villager.action_weights[action] = new_weight


def calculate_exploration_temperature(tick: int, base_temp: float = 1.0) -> float:
    """
    Calculate exploration temperature that decreases over time.
    Early: high exploration. Late: more exploitation.
    """
    # Temperature decays but never below 0.3
    decay = 1.0 / (1.0 + tick / 10000)
    return max(0.3, base_temp * decay)


def get_action_preferences(villager: 'Villager') -> dict:
    """
    Get human-readable action preference summary.
    Useful for debugging and observability.
    """
    probs = softmax(villager.action_weights, temperature=1.0)
    
    # Sort by probability
    sorted_prefs = sorted(probs.items(), key=lambda x: -x[1])
    
    return {action.name: f"{prob*100:.1f}%" for action, prob in sorted_prefs}
