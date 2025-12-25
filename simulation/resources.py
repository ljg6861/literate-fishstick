"""
The Village - Resources
Resource management and scarcity effects.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.villager import Villager


@dataclass
class Resources:
    """Village resource pools."""
    
    food: float = 500.0
    housing_capacity: int = 20
    work_capacity: float = 0.0
    knowledge: float = 0.0
    
    # Tracking
    food_produced_today: float = 0.0
    food_consumed_today: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'food': self.food,
            'housing_capacity': self.housing_capacity,
            'work_capacity': self.work_capacity,
            'knowledge': self.knowledge,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Resources':
        """Deserialize from dictionary."""
        return cls(
            food=data.get('food', 500.0),
            housing_capacity=data.get('housing_capacity', 20),
            work_capacity=data.get('work_capacity', 0.0),
            knowledge=data.get('knowledge', 0.0),
        )
    
    def reset_daily_tracking(self):
        """Reset daily production/consumption tracking."""
        self.food_produced_today = 0.0
        self.food_consumed_today = 0.0
    
    @property
    def food_shortage(self) -> bool:
        """Check if there's a food shortage."""
        return self.food < 50
    
    @property
    def food_critical(self) -> bool:
        """Check if food situation is critical."""
        return self.food < 10
    
    @property
    def knowledge_level(self) -> str:
        """Get qualitative knowledge level."""
        if self.knowledge < 10:
            return "Primitive"
        elif self.knowledge < 50:
            return "Basic"
        elif self.knowledge < 100:
            return "Developing"
        elif self.knowledge < 200:
            return "Advanced"
        else:
            return "Flourishing"


def apply_scarcity_effects(villager: 'Villager', resources: Resources, config) -> list[str]:
    """
    Apply scarcity effects to a villager based on resource state.
    Returns list of effect messages.
    """
    effects = []
    
    # Hunger effects
    if villager.hunger > 80:
        villager.health -= config.health_decay_when_starving
        villager.happiness -= config.happiness_decay_hungry
        effects.append("starving")
    elif villager.hunger > 60:
        villager.happiness -= config.happiness_decay_hungry * 0.5
        effects.append("hungry")
    
    # Homeless effects
    if not villager.has_home:
        villager.happiness -= config.happiness_decay_homeless
        if villager.health > 50:  # Slower health decay for homeless
            villager.health -= 0.02
        effects.append("homeless")
    
    # Low knowledge effects (village-wide, affects learning speed)
    if resources.knowledge < 10:
        # Learning is slower in primitive villages
        pass  # Handled in learning actions
    
    # Clamp values
    villager.health = max(0, min(100, villager.health))
    villager.happiness = max(0, min(100, villager.happiness))
    
    return effects


def calculate_housing_assignments(villagers: list['Villager'], housing_capacity: int) -> int:
    """
    Assign housing to villagers. Returns number of housed villagers.
    Priority: families with children, then elderly, then others.
    """
    from simulation.villager import Role
    
    # Sort by priority
    def priority(v):
        if v.role == Role.CHILD:
            return 0  # Children housed first (with families)
        elif v.age > 60:
            return 1  # Elderly next
        else:
            return 2  # Others
    
    sorted_villagers = sorted(villagers, key=priority)
    
    housed_count = 0
    for i, villager in enumerate(sorted_villagers):
        if i < housing_capacity:
            villager.has_home = True
            housed_count += 1
        else:
            villager.has_home = False
    
    return housed_count
