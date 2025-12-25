"""
The Village - Villager Agent
Core villager class with state, behavior loop, and learning.
Phase 2: Added illness, productivity, and task assignment.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Optional, TYPE_CHECKING, Any
import random
import math

if TYPE_CHECKING:
    from simulation.world import World
    from simulation.tasks import TaskAssignment


class Role(Enum):
    """Villager roles."""
    CHILD = auto()
    WORKER = auto()
    BUILDER = auto()
    FARMER = auto()
    THINKER = auto()
    HEALER = auto()  # New: works at hospitals


# Import IllnessState from health module to avoid duplicate enum
from simulation.health import IllnessState


@dataclass
class MemoryEvent:
    """A remembered event."""
    tick: int
    action: str
    outcome: str
    reward: float


@dataclass
class Villager:
    """A villager agent in the simulation."""
    
    id: int
    age: float = 20.0
    health: float = 100.0
    happiness: float = 70.0
    hunger: float = 20.0
    role: Role = Role.WORKER
    has_home: bool = True
    skill_level: float = 10.0
    position: tuple[float, float] = (0.0, 0.0)
    
    # Target position for smooth movement
    target_position: tuple[float, float] = field(default=None)
    
    # Phase 2: Illness system
    illness: IllnessState = IllnessState.NONE
    illness_duration: int = 0
    productivity_modifier: float = 1.0
    
    # Phase 2: Task assignment
    assigned_task: Optional[Any] = None  # TaskAssignment
    
    # Learning weights for action selection
    action_weights: dict = field(default_factory=dict)
    
    # Memory of recent events
    memory: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Visual interpolation
    _visual_position: tuple[float, float] = field(default=None)
    
    def __post_init__(self):
        if self.target_position is None:
            self.target_position = self.position
        if self._visual_position is None:
            self._visual_position = self.position
        
        # Initialize action weights if empty
        if not self.action_weights:
            from simulation.actions import Action
            self.action_weights = {action: 1.0 for action in Action}
    
    @property
    def is_alive(self) -> bool:
        return self.health > 0
    
    @property
    def is_child(self) -> bool:
        return self.age < 18
    
    @property
    def is_elder(self) -> bool:
        return self.age > 65
    
    @property
    def is_ill(self) -> bool:
        return self.illness != IllnessState.NONE
    
    @property
    def needs_food(self) -> bool:
        return self.hunger > 50
    
    @property
    def needs_rest(self) -> bool:
        return self.health < 70
    
    @property
    def is_working(self) -> bool:
        """Check if villager has an active work task."""
        if self.assigned_task is None:
            return False
        from simulation.tasks import TaskType
        return self.assigned_task.task_type not in [TaskType.IDLE, TaskType.REST]
    
    @property
    def state_color_intensity(self) -> float:
        """
        Calculate intensity for visual representation.
        1.0 = fully healthy, 0.0 = critical
        """
        # Base intensity from health/hunger/happiness
        health_factor = self.health / 100.0
        hunger_factor = 1.0 - (self.hunger / 100.0)
        happiness_factor = self.happiness / 100.0
        
        base = (health_factor * 0.5 + hunger_factor * 0.3 + happiness_factor * 0.2)
        
        # Illness reduces intensity
        if self.illness == IllnessState.SEVERE:
            base *= 0.5
        elif self.illness == IllnessState.MILD:
            base *= 0.7
        elif self.illness == IllnessState.CHRONIC:
            base *= 0.8
        
        return base
    
    def get_status(self) -> str:
        """Get a human-readable status string."""
        if self.health < 20:
            return "critical"
        elif self.illness == IllnessState.SEVERE:
            return "severely ill"
        elif self.illness == IllnessState.MILD:
            return "ill"
        elif self.hunger > 80:
            return "starving"
        elif self.hunger > 60:
            return "hungry"
        elif not self.has_home:
            return "homeless"
        elif self.happiness < 30:
            return "unhappy"
        elif self.health < 50:
            return "unwell"
        elif self.illness == IllnessState.CHRONIC:
            return "chronic"
        else:
            return "healthy"
    
    def update_visual_position(self, smoothing: float = 0.1):
        """Interpolate visual position toward target for smooth movement."""
        if self._visual_position is None:
            self._visual_position = self.position
        
        dx = self.position[0] - self._visual_position[0]
        dy = self.position[1] - self._visual_position[1]
        
        self._visual_position = (
            self._visual_position[0] + dx * smoothing,
            self._visual_position[1] + dy * smoothing
        )
    
    @property
    def visual_position(self) -> tuple[float, float]:
        """Get interpolated visual position."""
        return self._visual_position if self._visual_position else self.position
    
    def add_memory(self, tick: int, action: str, outcome: str, reward: float):
        """Add an event to memory."""
        self.memory.append(MemoryEvent(tick, action, outcome, reward))
    
    def age_tick(self, aging_rate: float):
        """Process aging for one tick."""
        self.age += aging_rate
        
        # Update role based on age
        if self.age < 18:
            self.role = Role.CHILD
        elif self.role == Role.CHILD:
            # Assign role when becoming adult
            self.role = random.choice([Role.WORKER, Role.BUILDER, Role.FARMER, Role.THINKER, Role.HEALER])
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        from simulation.actions import Action
        
        task_data = None
        if self.assigned_task is not None:
            task_data = self.assigned_task.to_dict()
        
        return {
            'id': self.id,
            'age': self.age,
            'health': self.health,
            'happiness': self.happiness,
            'hunger': self.hunger,
            'role': self.role.name,
            'has_home': self.has_home,
            'skill_level': self.skill_level,
            'position': list(self.position),
            'target_position': list(self.target_position) if self.target_position else None,
            'illness': self.illness.name,
            'illness_duration': self.illness_duration,
            'productivity_modifier': self.productivity_modifier,
            'assigned_task': task_data,
            'action_weights': {a.name: w for a, w in self.action_weights.items()},
            'memory': [
                {'tick': m.tick, 'action': m.action, 'outcome': m.outcome, 'reward': m.reward}
                for m in self.memory
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Villager':
        """Deserialize from dictionary."""
        from simulation.actions import Action
        from simulation.tasks import TaskAssignment
        
        villager = cls(
            id=data['id'],
            age=data.get('age', 20.0),
            health=data.get('health', 100.0),
            happiness=data.get('happiness', 70.0),
            hunger=data.get('hunger', 20.0),
            role=Role[data.get('role', 'WORKER')],
            has_home=data.get('has_home', True),
            skill_level=data.get('skill_level', 10.0),
            position=tuple(data.get('position', [0.0, 0.0])),
            illness=IllnessState[data.get('illness', 'NONE')],
            illness_duration=data.get('illness_duration', 0),
            productivity_modifier=data.get('productivity_modifier', 1.0),
        )
        
        if data.get('target_position'):
            villager.target_position = tuple(data['target_position'])
        
        # Restore task
        if data.get('assigned_task'):
            villager.assigned_task = TaskAssignment.from_dict(data['assigned_task'])
        
        # Restore action weights
        if 'action_weights' in data:
            villager.action_weights = {
                Action[name]: weight 
                for name, weight in data['action_weights'].items()
            }
        
        # Restore memory
        if 'memory' in data:
            for m in data['memory']:
                villager.memory.append(MemoryEvent(
                    m['tick'], m['action'], m['outcome'], m['reward']
                ))
        
        return villager


def create_villager(
    villager_id: int,
    world_width: float,
    world_height: float,
    config
) -> Villager:
    """Create a new villager with randomized initial state."""
    
    # Random position within world bounds
    margin = 100
    x = random.uniform(margin, world_width - margin)
    y = random.uniform(margin, world_height - margin)
    
    # Randomize age (mostly adults, some children)
    if random.random() < 0.2:
        age = random.uniform(1, 17)
    else:
        age = random.uniform(18, 50)
    
    # Determine role based on age
    if age < 18:
        role = Role.CHILD
    else:
        role = random.choice([Role.WORKER, Role.BUILDER, Role.FARMER, Role.THINKER, Role.HEALER])
    
    return Villager(
        id=villager_id,
        age=age,
        health=config.initial_health + random.uniform(-10, 10),
        happiness=config.initial_happiness + random.uniform(-20, 20),
        hunger=config.initial_hunger + random.uniform(-5, 15),
        role=role,
        has_home=True,
        skill_level=config.initial_skill + random.uniform(-5, 10),
        position=(x, y),
        target_position=(x, y),
    )


def create_child(parent1: Villager, parent2: Villager, villager_id: int) -> Villager:
    """Create a child villager from two parents."""
    # Child spawns near parents
    x = (parent1.position[0] + parent2.position[0]) / 2 + random.uniform(-20, 20)
    y = (parent1.position[1] + parent2.position[1]) / 2 + random.uniform(-20, 20)
    
    # Inherit some skill from parents
    inherited_skill = (parent1.skill_level + parent2.skill_level) / 4
    
    return Villager(
        id=villager_id,
        age=0.0,
        health=100.0,
        happiness=80.0,
        hunger=30.0,
        role=Role.CHILD,
        has_home=parent1.has_home or parent2.has_home,
        skill_level=inherited_skill,
        position=(x, y),
        target_position=(x, y),
    )
