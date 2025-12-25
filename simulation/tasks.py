"""
The Village - Task Assignment System
Villagers are assigned to tasks and move with purpose.
Fixed: Villagers now get new destinations when completing tasks.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Any, TYPE_CHECKING
import math
import random

if TYPE_CHECKING:
    from simulation.villager import Villager
    from simulation.world import World


class TaskType(Enum):
    """Types of tasks villagers can be assigned to."""
    IDLE = auto()
    GATHER_FOOD = auto()
    GATHER_MATERIALS = auto()
    CONSTRUCT = auto()
    HEAL = auto()
    RESEARCH = auto()
    REST = auto()
    FARM = auto()


@dataclass
class TaskAssignment:
    """A task assigned to a villager."""
    task_type: TaskType
    target_position: tuple[float, float]
    target_id: Optional[int] = None
    priority: float = 1.0
    duration: int = 0
    completed: bool = False  # Mark when task is done
    
    def to_dict(self) -> dict:
        return {
            'task_type': self.task_type.name,
            'target_position': list(self.target_position),
            'target_id': self.target_id,
            'priority': self.priority,
            'duration': self.duration
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TaskAssignment':
        return cls(
            task_type=TaskType[data['task_type']],
            target_position=tuple(data['target_position']),
            target_id=data.get('target_id'),
            priority=data.get('priority', 1.0),
            duration=data.get('duration', 0)
        )


class TaskManager:
    """Manages task assignments for the village."""
    
    def __init__(self):
        pass
    
    def assign_tasks(self, villagers: List['Villager'], world: 'World'):
        """Assign tasks to villagers based on village needs."""
        from simulation.villager import Role
        from simulation.buildings import BuildingType
        
        needs = self._calculate_needs(world)
        
        # Get available workers
        available = [v for v in villagers 
                     if v.is_alive and not v.is_child and v.health > 20]
        
        assigned_ids = set()
        
        # Priority 1: Construction
        if needs['construction'] > 0:
            builders = [v for v in available 
                       if v.role == Role.BUILDER and v.id not in assigned_ids]
            construction_site = world.get_active_construction()
            
            if construction_site:
                for builder in builders[:needs['construction']]:
                    self._assign_to_construction(builder, construction_site, world)
                    assigned_ids.add(builder.id)
        
        # Priority 2: Hospitals
        if needs['healing'] > 0:
            hospitals = [b for b in world.buildings 
                        if b.building_type == BuildingType.HOSPITAL and b.is_complete]
            if hospitals:
                healers = [v for v in available 
                          if v.id not in assigned_ids and not v.is_ill][:needs['healing']]
                for healer in healers:
                    self._assign_to_healing(healer, hospitals[0], world)
                    assigned_ids.add(healer.id)
        
        # Priority 3: Farming
        farms = [b for b in world.buildings 
                if b.building_type == BuildingType.FARM and b.is_complete]
        if farms and needs['farming'] > 0:
            farmers = [v for v in available 
                      if v.role == Role.FARMER and v.id not in assigned_ids]
            for i, farmer in enumerate(farmers[:needs['farming']]):
                farm = farms[i % len(farms)]
                self._assign_to_farming(farmer, farm, world)
                assigned_ids.add(farmer.id)
        
        # Priority 4: Resource nodes for remaining
        unassigned = [v for v in available if v.id not in assigned_ids]
        for villager in unassigned:
            self._assign_to_resource_node(villager, world)
    
    def _calculate_needs(self, world: 'World') -> dict:
        """Calculate task needs based on village state."""
        from simulation.health import get_village_health_status
        
        health_status = get_village_health_status(world.villagers)
        
        needs = {
            'construction': 0,
            'healing': 0,
            'research': 0,
            'farming': 0,
            'food_gathering': 0
        }
        
        if world.has_active_construction():
            needs['construction'] = 5
        
        ill_count = health_status['mild'] + health_status['severe']
        if ill_count > 0:
            needs['healing'] = max(1, ill_count // 5)
        
        from simulation.buildings import BuildingType
        farm_count = world.building_manager.count_by_type(BuildingType.FARM)
        needs['farming'] = farm_count * 2
        
        needs['research'] = 2
        
        if world.resources.food < 300:
            needs['food_gathering'] = 15
        elif world.resources.food < 600:
            needs['food_gathering'] = 8
        
        return needs
    
    def _get_random_position_near(self, center: tuple, world: 'World', radius: float = 80) -> tuple:
        """Get a random position near a center point."""
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(20, radius)
        x = center[0] + math.cos(angle) * dist
        y = center[1] + math.sin(angle) * dist
        # Keep in bounds
        x = max(50, min(world.width - 50, x))
        y = max(50, min(world.height - 50, y))
        return (x, y)
    
    def _assign_to_construction(self, villager: 'Villager', building, world: 'World'):
        """Assign villager to construction."""
        target = self._get_random_position_near(building.position, world, 60)
        villager.assigned_task = TaskAssignment(
            task_type=TaskType.CONSTRUCT,
            target_position=target,
            target_id=building.id
        )
    
    def _assign_to_healing(self, villager: 'Villager', hospital, world: 'World'):
        """Assign villager to work at hospital."""
        target = self._get_random_position_near(hospital.position, world, 50)
        villager.assigned_task = TaskAssignment(
            task_type=TaskType.HEAL,
            target_position=target,
            target_id=hospital.id
        )
    
    def _assign_to_research(self, villager: 'Villager', research_center, world: 'World'):
        """Assign villager to research."""
        target = self._get_random_position_near(research_center.position, world, 50)
        villager.assigned_task = TaskAssignment(
            task_type=TaskType.RESEARCH,
            target_position=target,
            target_id=research_center.id
        )
    
    def _assign_to_farming(self, villager: 'Villager', farm, world: 'World'):
        """Assign villager to farm work."""
        target = self._get_random_position_near(farm.position, world, 100)
        villager.assigned_task = TaskAssignment(
            task_type=TaskType.FARM,
            target_position=target,
            target_id=farm.id
        )
    
    def _assign_to_resource_node(self, villager: 'Villager', world: 'World'):
        """Assign villager to a world resource node."""
        if not world.resource_nodes:
            # Fallback to random if no nodes (shouldn't happen)
            target = (random.uniform(100, world.width-100), random.uniform(100, world.height-100))
            villager.assigned_task = TaskAssignment(TaskType.GATHER_FOOD, target)
            return

        # Simple: Pick a node and go to it
        # Prefer food if hungry, otherwise any node
        node_type = 'FOOD' if world.resources.food < 300 else random.choice(['FOOD', 'MATERIAL'])
        nodes = [n for n in world.resource_nodes if n.node_type == node_type]
        if not nodes: nodes = world.resource_nodes
        
        node = random.choice(nodes)
        target = self._get_random_position_near(node.position, world, 40)
        
        villager.assigned_task = TaskAssignment(
            task_type=TaskType.GATHER_FOOD if node.node_type == 'FOOD' else TaskType.GATHER_MATERIALS,
            target_position=target,
            target_id=node.id
        )

    def _assign_to_food_gathering(self, villager: 'Villager', world: 'World'):
        """Old method, redirected to resource nodes."""
        self._assign_to_resource_node(villager, world)


def move_toward_task(villager: 'Villager', world: 'World', speed: float = 3.0) -> bool:
    """
    Move villager toward their task target.
    Returns True if at destination.
    """
    if villager.assigned_task is None:
        return True
    
    target = villager.assigned_task.target_position
    current = villager.position
    
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    distance = math.sqrt(dx * dx + dy * dy)
    
    if distance < 10:  # Close enough
        return True
    
    # Move toward target
    move_dist = min(speed, distance)
    new_x = current[0] + (dx / distance) * move_dist
    new_y = current[1] + (dy / distance) * move_dist
    
    villager.position = (new_x, new_y)
    villager.target_position = target
    
    return False


def execute_task(villager: 'Villager', world: 'World') -> float:
    """
    Execute the villager's current task if at destination.
    Returns productivity output.
    """
    if villager.assigned_task is None:
        return 0.0
    
    task = villager.assigned_task
    task.duration += 1
    
    productivity = villager.productivity_modifier * (villager.skill_level / 100 + 0.5)
    
    if task.task_type == TaskType.GATHER_FOOD:
        food_gathered = 0.3 * productivity
        world.resources.food += food_gathered
        
        # After gathering for a while, get a new spot
        if task.duration > 50:
            task.completed = True
        return food_gathered
        
    elif task.task_type == TaskType.FARM:
        food_produced = 0.5 * productivity
        world.resources.food += food_produced
        
        # Farmers move around the farm
        if task.duration > 30:
            task.completed = True
        return food_produced
        
    elif task.task_type == TaskType.CONSTRUCT:
        building = world.get_active_construction()
        if building:
            labor = 1.0 * productivity
            world.contribute_to_building(building, labor)
            if task.duration > 40:
                task.completed = True
            return labor
        else:
            task.completed = True
            
    elif task.task_type == TaskType.HEAL:
        if task.duration > 60:
            task.completed = True
        return productivity
        
    elif task.task_type == TaskType.RESEARCH:
        # Generate insight points
        world.resources.knowledge += 0.01 * productivity
        if task.duration > 80:
            task.completed = True
        return productivity
    
    return 0.0


def wander(villager: 'Villager', world: 'World'):
    """Random wandering for idle villagers."""
    # More active wandering
    dx = random.uniform(-5, 5)
    dy = random.uniform(-5, 5)
    
    new_x = max(50, min(world.width - 50, villager.position[0] + dx))
    new_y = max(50, min(world.height - 50, villager.position[1] + dy))
    
    villager.position = (new_x, new_y)


def process_villager_movement(villager: 'Villager', world: 'World'):
    """Process a villager's movement and task execution."""
    if villager.assigned_task is None:
        wander(villager, world)
        return
    
    task = villager.assigned_task
    
    # Check if task is completed and needs new assignment
    if task.completed:
        # Get a new task of the same type but different position
        reassign_task(villager, task.task_type, world)
        # DON'T return - continue to move toward new task!
        task = villager.assigned_task
    
    if task is None or task.task_type == TaskType.IDLE:
        wander(villager, world)
    else:
        at_destination = move_toward_task(villager, world, speed=3.0)
        if at_destination:
            execute_task(villager, world)
            # Add small random movement while working to appear active
            dx = random.uniform(-2, 2)
            dy = random.uniform(-2, 2)
            villager.position = (
                max(50, min(world.width - 50, villager.position[0] + dx)),
                max(50, min(world.height - 50, villager.position[1] + dy))
            )


def reassign_task(villager: 'Villager', task_type: TaskType, world: 'World'):
    """Reassign a villager to a new instance of the same task type."""
    from simulation.buildings import BuildingType
    
    margin = 100
    
    if task_type in [TaskType.GATHER_FOOD, TaskType.GATHER_MATERIALS]:
        # Move to a DIFFERENT node of same type
        node_type = 'FOOD' if task_type == TaskType.GATHER_FOOD else 'MATERIAL'
        nodes = [n for n in world.resource_nodes if n.node_type == node_type]
        if not nodes: nodes = world.resource_nodes
        
        node = random.choice(nodes)
        target = (node.position[0] + random.uniform(-30, 30), 
                 node.position[1] + random.uniform(-30, 30))
        
        villager.assigned_task = TaskAssignment(
            task_type=task_type,
            target_position=target,
            target_id=node.id
        )
    
    elif task_type == TaskType.FARM:
        farms = [b for b in world.buildings 
                if b.building_type == BuildingType.FARM and b.is_complete]
        if farms:
            farm = random.choice(farms)
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(20, 100)
            target = (
                max(50, min(world.width - 50, farm.position[0] + math.cos(angle) * dist)),
                max(50, min(world.height - 50, farm.position[1] + math.sin(angle) * dist))
            )
            villager.assigned_task = TaskAssignment(
                task_type=TaskType.FARM,
                target_position=target,
                target_id=farm.id
            )
        else:
            reassign_task(villager, TaskType.GATHER_FOOD, world)
    
    elif task_type == TaskType.CONSTRUCT:
        building = world.get_active_construction()
        if building:
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(20, 60)
            target = (building.position[0] + math.cos(angle) * dist,
                     building.position[1] + math.sin(angle) * dist)
            villager.assigned_task = TaskAssignment(
                task_type=TaskType.CONSTRUCT,
                target_position=target,
                target_id=building.id
            )
        else:
            reassign_task(villager, TaskType.GATHER_FOOD, world)
    
    elif task_type in [TaskType.HEAL, TaskType.RESEARCH]:
        # Keep at building but move around it
        reassign_task(villager, TaskType.GATHER_FOOD, world)
    
    else:
        # Default to gathering
        reassign_task(villager, TaskType.GATHER_FOOD, world)
