"""
The Village - Actions
All possible actions a villager can take.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.villager import Villager
    from simulation.world import World


class Action(Enum):
    """All possible villager actions."""
    EAT = auto()
    REST = auto()
    WORK = auto()
    BUILD = auto()
    LEARN = auto()
    WANDER = auto()
    SOCIALIZE = auto()


@dataclass
class ActionResult:
    """Result of performing an action."""
    success: bool
    reward: float
    message: str
    hunger_change: float = 0.0
    health_change: float = 0.0
    happiness_change: float = 0.0
    skill_change: float = 0.0


def can_eat(villager: 'Villager', world: 'World') -> bool:
    """Check if villager can eat."""
    return world.resources.food >= 1.0


def do_eat(villager: 'Villager', world: 'World') -> ActionResult:
    """Consume food to reduce hunger."""
    if world.resources.food < 1.0:
        return ActionResult(
            success=False,
            reward=-0.1,
            message="No food available"
        )
    
    world.resources.food -= 1.0
    hunger_reduction = min(villager.hunger, 30.0)
    
    return ActionResult(
        success=True,
        reward=0.3 if villager.hunger > 50 else 0.1,
        message="Ate food",
        hunger_change=-hunger_reduction,
        happiness_change=2.0 if villager.hunger > 70 else 0.5
    )


def can_rest(villager: 'Villager', world: 'World') -> bool:
    """Check if villager can rest."""
    return villager.has_home


def do_rest(villager: 'Villager', world: 'World') -> ActionResult:
    """Rest at home to regain health."""
    if not villager.has_home:
        return ActionResult(
            success=False,
            reward=-0.05,
            message="No home to rest in"
        )
    
    return ActionResult(
        success=True,
        reward=0.1,
        message="Rested at home",
        health_change=5.0,
        happiness_change=1.0,
        hunger_change=2.0  # Resting makes you hungry
    )


def can_work(villager: 'Villager', world: 'World') -> bool:
    """Check if villager can work."""
    from simulation.villager import Role
    return villager.role != Role.CHILD and villager.health > 20


def do_work(villager: 'Villager', world: 'World') -> ActionResult:
    """Work based on role to produce resources."""
    from simulation.villager import Role
    
    if villager.role == Role.CHILD:
        return ActionResult(
            success=False,
            reward=0.0,
            message="Children cannot work"
        )
    
    efficiency = (villager.skill_level / 100.0) * (villager.health / 100.0)
    
    if villager.role == Role.FARMER:
        food_produced = 2.0 * efficiency
        world.resources.food += food_produced
        return ActionResult(
            success=True,
            reward=0.2,
            message=f"Farmed {food_produced:.1f} food",
            hunger_change=3.0,
            skill_change=0.1
        )
    
    elif villager.role == Role.WORKER:
        world.resources.work_capacity += efficiency
        return ActionResult(
            success=True,
            reward=0.15,
            message="Produced work",
            hunger_change=4.0,
            skill_change=0.1
        )
    
    elif villager.role == Role.THINKER:
        knowledge_gain = 0.5 * efficiency
        world.resources.knowledge += knowledge_gain
        return ActionResult(
            success=True,
            reward=0.1,
            message="Advanced knowledge",
            hunger_change=2.0,
            skill_change=0.2
        )
    
    else:  # BUILDER default work
        return ActionResult(
            success=True,
            reward=0.1,
            message="General work",
            hunger_change=3.0,
            skill_change=0.05
        )


def can_build(villager: 'Villager', world: 'World') -> bool:
    """Check if villager can contribute to building."""
    from simulation.villager import Role
    return (villager.role == Role.BUILDER or villager.role == Role.WORKER) and \
           villager.health > 30 and \
           world.has_active_construction()


def do_build(villager: 'Villager', world: 'World') -> ActionResult:
    """Contribute labor to construction."""
    building = world.get_active_construction()
    if building is None:
        return ActionResult(
            success=False,
            reward=0.0,
            message="Nothing to build"
        )
    
    efficiency = (villager.skill_level / 100.0) * (villager.health / 100.0)
    labor_contributed = 2.0 * efficiency
    
    completed = world.contribute_to_building(building, labor_contributed)
    
    reward = 0.5 if completed else 0.2
    message = f"Building completed: {building.building_type.name}" if completed else "Contributed to construction"
    
    return ActionResult(
        success=True,
        reward=reward,
        message=message,
        hunger_change=5.0,
        skill_change=0.15
    )


def can_learn(villager: 'Villager', world: 'World') -> bool:
    """Check if villager can learn."""
    return any(b.building_type.name == "SCHOOL" and b.is_complete 
               for b in world.buildings)


def do_learn(villager: 'Villager', world: 'World') -> ActionResult:
    """Learn at the school to increase skills."""
    has_school = any(b.building_type.name == "SCHOOL" and b.is_complete 
                     for b in world.buildings)
    
    if not has_school:
        # Can still learn, just slower
        return ActionResult(
            success=True,
            reward=0.05,
            message="Self-study",
            skill_change=0.05,
            hunger_change=1.0
        )
    
    return ActionResult(
        success=True,
        reward=0.15,
        message="Studied at school",
        skill_change=0.3,
        hunger_change=1.5
    )


def can_wander(villager: 'Villager', world: 'World') -> bool:
    """Wandering is always possible."""
    return True


def do_wander(villager: 'Villager', world: 'World') -> ActionResult:
    """Wander around the village."""
    import random
    
    # Move to a random nearby position
    dx = random.uniform(-20, 20)
    dy = random.uniform(-20, 20)
    
    new_x = max(50, min(world.width - 50, villager.position[0] + dx))
    new_y = max(50, min(world.height - 50, villager.position[1] + dy))
    villager.position = (new_x, new_y)
    
    return ActionResult(
        success=True,
        reward=0.0,
        message="Wandered",
        hunger_change=0.5
    )


def can_socialize(villager: 'Villager', world: 'World') -> bool:
    """Check if there are other villagers nearby."""
    return len(world.villagers) > 1


def do_socialize(villager: 'Villager', world: 'World') -> ActionResult:
    """Interact with other villagers."""
    return ActionResult(
        success=True,
        reward=0.05,
        message="Socialized",
        happiness_change=3.0,
        hunger_change=0.5
    )


# Action registry
ACTION_HANDLERS = {
    Action.EAT: (can_eat, do_eat),
    Action.REST: (can_rest, do_rest),
    Action.WORK: (can_work, do_work),
    Action.BUILD: (can_build, do_build),
    Action.LEARN: (can_learn, do_learn),
    Action.WANDER: (can_wander, do_wander),
    Action.SOCIALIZE: (can_socialize, do_socialize),
}


def get_available_actions(villager: 'Villager', world: 'World') -> list[Action]:
    """Get all actions currently available to a villager."""
    available = []
    for action, (can_do, _) in ACTION_HANDLERS.items():
        if can_do(villager, world):
            available.append(action)
    return available


def execute_action(action: Action, villager: 'Villager', world: 'World') -> ActionResult:
    """Execute an action and return the result."""
    _, do_action = ACTION_HANDLERS[action]
    return do_action(villager, world)
