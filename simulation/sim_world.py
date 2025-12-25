import os
print(f"DEBUG: world.py loaded from {os.path.abspath(__file__)}")

"""
The Village - World State
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import random
import math

from config import CONFIG
from simulation.villager import Villager, create_villager, create_child, Role, IllnessState
from simulation.resources import Resources, apply_scarcity_effects, calculate_housing_assignments
from simulation.buildings import Building, BuildingManager, BuildingType
from simulation.actions import Action, get_available_actions, execute_action
from simulation.health import (
    spread_disease, progress_illness, attempt_recovery, 
    apply_illness_effects, get_village_health_status, trigger_outbreak
)
from simulation.tasks import TaskManager, TaskType, process_villager_movement
from learning.policy import select_action, update_weights, calculate_exploration_temperature
from learning.rewards import calculate_state_reward, calculate_action_reward, calculate_village_stability
from learning.research import ResearchState, generate_insight
from learning.planner import PlannerState, run_planner_decision, detect_crisis


@dataclass
class ResourceNode:
    """A static point of interest for resource gathering."""
    id: int
    node_type: str  # 'FOOD', 'MATERIAL'
    position: tuple[float, float]
    amount: float = 100.0
    capacity: int = 5  # Max villagers at once
    current_users: int = 0

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'node_type': self.node_type,
            'position': self.position,
            'amount': self.amount,
            'capacity': self.capacity,
            'current_users': self.current_users
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ResourceNode':
        return cls(**data)


@dataclass
class WorldStats:
    """Statistics tracking for the world."""
    total_ticks: int = 0
    total_births: int = 0
    total_deaths: int = 0
    peak_population: int = 0
    total_food_produced: float = 0.0
    total_food_consumed: float = 0.0
    buildings_completed: int = 0
    # Phase 2 stats
    illness_deaths: int = 0
    villagers_healed: int = 0
    
    def to_dict(self) -> dict:
        return {
            'total_ticks': self.total_ticks,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'peak_population': self.peak_population,
            'total_food_produced': self.total_food_produced,
            'total_food_consumed': self.total_food_consumed,
            'buildings_completed': self.buildings_completed,
            'illness_deaths': self.illness_deaths,
            'villagers_healed': self.villagers_healed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WorldStats':
        stats = cls()
        for key, value in data.items():
            if hasattr(stats, key):
                setattr(stats, key, value)
        return stats


@dataclass
class VillageHistory:
    """Track major events for cultural memory."""
    first_hospital_tick: Optional[int] = None
    first_research_tick: Optional[int] = None
    crises_survived: List[str] = field(default_factory=list)
    population_peaks: List[int] = field(default_factory=list)
    major_outbreaks: int = 0
    
    def to_dict(self) -> dict:
        return {
            'first_hospital_tick': self.first_hospital_tick,
            'first_research_tick': self.first_research_tick,
            'crises_survived': self.crises_survived,
            'population_peaks': self.population_peaks,
            'major_outbreaks': self.major_outbreaks,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VillageHistory':
        return cls(
            first_hospital_tick=data.get('first_hospital_tick'),
            first_research_tick=data.get('first_research_tick'),
            crises_survived=data.get('crises_survived', []),
            population_peaks=data.get('population_peaks', []),
            major_outbreaks=data.get('major_outbreaks', 0),
        )


@dataclass
class EvolutionMemory:
    """Learned knowledge that persists across evolutions."""
    evolution_number: int = 1
    learned_action_weights: Dict[str, float] = field(default_factory=dict)
    extinction_causes: List[str] = field(default_factory=list)
    best_peak_population: int = 0
    best_survival_ticks: int = 0
    
    # Phase 2: cultural biases from history
    hospital_priority_bias: float = 1.0  # Increases after plague deaths
    
    def record_extinction(self, cause: str, stats: WorldStats):
        """Record data from an extinction event."""
        self.extinction_causes.append(cause)
        self.best_peak_population = max(self.best_peak_population, stats.peak_population)
        self.best_survival_ticks = max(self.best_survival_ticks, stats.total_ticks)
        
        # Learn from illness deaths
        if stats.illness_deaths > stats.total_deaths * 0.3:
            self.hospital_priority_bias += 0.5
    
    def learn_from_villagers(self, villagers: List[Villager]):
        if not villagers:
            return
        
        from simulation.actions import Action
        
        weight_sums = {a: 0.0 for a in Action}
        total_weight = 0.0
        
        for villager in villagers:
            experience = villager.skill_level * (villager.age / 50)
            for action, weight in villager.action_weights.items():
                weight_sums[action] += weight * experience
            total_weight += experience
        
        if total_weight > 0:
            self.learned_action_weights = {
                a.name: weight_sums[a] / total_weight 
                for a in Action
            }
    
    def get_starting_weights(self) -> Dict[str, float]:
        if not self.learned_action_weights:
            return {}
        from simulation.actions import Action
        return {
            Action[name]: weight 
            for name, weight in self.learned_action_weights.items()
            if name in Action.__members__
        }
    
    def to_dict(self) -> dict:
        return {
            'evolution_number': self.evolution_number,
            'learned_action_weights': self.learned_action_weights,
            'extinction_causes': self.extinction_causes,
            'best_peak_population': self.best_peak_population,
            'best_survival_ticks': self.best_survival_ticks,
            'hospital_priority_bias': self.hospital_priority_bias,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EvolutionMemory':
        return cls(
            evolution_number=data.get('evolution_number', 1),
            learned_action_weights=data.get('learned_action_weights', {}),
            extinction_causes=data.get('extinction_causes', []),
            best_peak_population=data.get('best_peak_population', 0),
            best_survival_ticks=data.get('best_survival_ticks', 0),
            hospital_priority_bias=data.get('hospital_priority_bias', 1.0),
        )


class World:
    """The simulation world containing all state."""
    
    def __init__(self, width: float = 1920, height: float = 1080, config=None):
        self.width = width
        self.height = height
        self.config = config or CONFIG
        
        # Core state
        self.villagers: List[Villager] = []
        self.resources = Resources(
            food=self.config.initial_food,
            housing_capacity=self.config.initial_housing_capacity
        )
        self.building_manager = BuildingManager()
        
        # Phase 2: Task manager
        self.task_manager = TaskManager()
        
        # Phase 2B: Research state
        self.research = ResearchState()
        
        # Phase 2C: The Planner
        self.planner = PlannerState()
        
        # Resource nodes for purposeful movement
        self.resource_nodes: List[ResourceNode] = []
        self._next_node_id: int = 1
        
        # Evolution memory
        self.evolution = EvolutionMemory()
        
        # History
        self.history = VillageHistory()
        
        # Statistics
        self.stats = WorldStats()
        self.tick = 0
        
        # Health status cache
        self._health_status: dict = {}
        
        # Event log
        self.event_log: List[str] = []
        self._log_max_size = 1000
        
        # Random state
        self._rng = random.Random()
        self._next_villager_id = 1
        
        # Overlay messages queue
        self.pending_messages: List[str] = []
        
        # Tracking
        self._ticks_without_food = 0
        self._last_task_assignment = 0
    
    @property
    def evolution_number(self) -> int:
        return self.evolution.evolution_number
    
    @property
    def is_extinct(self) -> bool:
        return len(self.villagers) == 0
    
    def initialize(self, seed: Optional[int] = None):
        """Initialize the world with starting population."""
        if seed is not None:
            self._rng.seed(seed)
            random.seed(seed)
        
        starting_weights = self.evolution.get_starting_weights()
        
        for _ in range(self.config.initial_population):
            villager = create_villager(
                self._next_villager_id,
                self.width,
                self.height,
                self.config
            )
            
            if starting_weights:
                villager.action_weights = starting_weights.copy()
            
            self._next_villager_id += 1
            self.villagers.append(villager)
        
        # Initial buildings
        self.building_manager.add_building(BuildingType.HOUSING, (self.width//2, self.height//2))
        
        # Initial resource nodes
        self._spawn_initial_resource_nodes()
        
        self.is_extinct = False
        self.pending_messages = ["The first settlers have arrived."]
        
        self.building_manager.buildings[0].construction_progress = 100
        
        # Later evolutions get more starting buildings
        if self.evolution_number > 1:
            self.building_manager.add_building(
                BuildingType.FARM,
                self.building_manager.find_building_position(self.width, self.height)
            )
            self.building_manager.buildings[1].construction_progress = 100
            self.resources.food = self.config.initial_food * 1.5
        
        # If learned from plague, start with hospital
        if self.evolution.hospital_priority_bias > 1.5:
            self.building_manager.add_building(
                BuildingType.HOSPITAL,
                self.building_manager.find_building_position(self.width, self.height)
            )
            self.building_manager.buildings[-1].construction_progress = 100
        
        self._log_event(f"Evolution {self.evolution_number} started with {len(self.villagers)} villagers")

    def _spawn_initial_resource_nodes(self):
        """Create initial resource nodes across the map."""
        # Clear existing nodes if any
        self.resource_nodes = []
        self._next_node_id = 1
        
        # 4 food nodes
        corners = [
            (300, 300), (self.width-300, 300),
            (300, self.height-300), (self.width-300, self.height-300)
        ]
        for pos in corners:
            self.add_resource_node('FOOD', pos)
        
        # 2 material nodes near center
        self.add_resource_node('MATERIAL', (self.width//2 - 250, self.height//2 + 150))
        self.add_resource_node('MATERIAL', (self.width//2 + 250, self.height//2 - 150))

    def add_resource_node(self, node_type: str, position: tuple[float, float]):
        """Add a new resource node."""
        node = ResourceNode(
            id=self._next_node_id,
            node_type=node_type,
            position=position
        )
        self._next_node_id += 1
        self.resource_nodes.append(node)
        return node
    
    def restart_evolution(self):
        """Restart after extinction."""
        cause = "unknown"
        if self.resources.food <= 0:
            cause = "starvation"
        elif self.stats.illness_deaths > self.stats.total_deaths * 0.3:
            cause = "plague"
        elif self.resources.housing_capacity < 5:
            cause = "no shelter"
        
        self.evolution.record_extinction(cause, self.stats)
        self.evolution.evolution_number += 1
        
        self._log_event(f"Evolution {self.evolution_number - 1} ended: {cause}")
        
        # Reset
        self.villagers = []
        self.resources = Resources(
            food=self.config.initial_food,
            housing_capacity=self.config.initial_housing_capacity
        )
        self.building_manager = BuildingManager()
        self.task_manager = TaskManager()
        self.history = VillageHistory()
        self.stats = WorldStats()
        self.tick = 0
        self._next_villager_id = 1
        self._ticks_without_food = 0
        self._last_task_assignment = 0
        
        self.initialize()
        self.pending_messages.append(f"Evolution {self.evolution_number} begins...")
    
    @property
    def buildings(self) -> List[Building]:
        return self.building_manager.buildings
    
    def has_active_construction(self) -> bool:
        return len(self.building_manager.get_incomplete_buildings()) > 0
    
    def get_active_construction(self) -> Optional[Building]:
        incomplete = self.building_manager.get_incomplete_buildings()
        return incomplete[0] if incomplete else None
    
    def contribute_to_building(self, building: Building, labor: float) -> bool:
        completed = building.add_labor(labor)
        if completed:
            self.stats.buildings_completed += 1
            self._log_event(f"Building completed: {building.building_type.name}")
            
            blueprint = building.get_blueprint()
            self.resources.housing_capacity += blueprint.housing_bonus
            
            # Track history
            if building.building_type == BuildingType.HOSPITAL and self.history.first_hospital_tick is None:
                self.history.first_hospital_tick = self.tick
            elif building.building_type == BuildingType.RESEARCH_CENTER and self.history.first_research_tick is None:
                self.history.first_research_tick = self.tick
            
            self.pending_messages.append(f"A {building.building_type.name.lower().replace('_', ' ')} has been completed.")
        
        return completed
    
    def tick_simulation(self) -> bool:
        """Advance the simulation by one tick."""
        self.tick += 1
        self.stats.total_ticks += 1
        
        if self.is_extinct:
            self.restart_evolution()
            return False
        
        if self.resources.food <= 0:
            self._ticks_without_food += 1
        else:
            self._ticks_without_food = 0
        
        # Update health status
        self._health_status = get_village_health_status(self.villagers)
        
        # Phase 2: Disease spreading
        spread_disease(self.villagers, self)
        
        # Assign tasks more frequently
        if self.tick - self._last_task_assignment > 20:
            self.task_manager.assign_tasks(self.villagers, self)
            self._last_task_assignment = self.tick
        
        # Update housing
        calculate_housing_assignments(self.villagers, self.resources.housing_capacity)
        
        temperature = calculate_exploration_temperature(self.tick, self.config.exploration_temperature)
        
        # Process villagers
        deaths = []
        has_hospital = self.building_manager.has_hospital()
        
        for villager in self.villagers:
            if not villager.is_alive:
                deaths.append(villager)
                continue
            
            # Age
            villager.age_tick(self.config.aging_per_tick)
            
            # Hunger
            villager.hunger = min(100, villager.hunger + self.config.hunger_increase_rate)
            
            # Phase 2: Illness progression
            if villager.is_ill:
                progress_illness(villager)
                apply_illness_effects(villager)
                attempt_recovery(villager, has_hospital)
            
            # Scarcity effects
            apply_scarcity_effects(villager, self.resources, self.config)
            
            # Check death
            if villager.health <= 0:
                deaths.append(villager)
                if villager.illness in [IllnessState.SEVERE, IllnessState.MILD]:
                    self.stats.illness_deaths += 1
                continue
            
            if villager.age > self.config.max_age:
                deaths.append(villager)
                continue
            
            # Phase 2: Task-based movement (using new function)
            process_villager_movement(villager, self)
            
            # Learning (simplified for task-based system)
            available = get_available_actions(villager, self)
            action = select_action(villager, available, temperature)
            result = execute_action(action, villager, self)
            
            villager.hunger = max(0, min(100, villager.hunger + result.hunger_change))
            villager.health = max(0, min(100, villager.health + result.health_change))
            villager.happiness = max(0, min(100, villager.happiness + result.happiness_change))
            villager.skill_level = max(0, min(100, villager.skill_level + result.skill_change))
            
            state_reward = calculate_state_reward(villager, self.config)
            action_reward = calculate_action_reward(villager, result, self)
            total_reward = state_reward + action_reward
            
            update_weights(villager, action, total_reward, self.config.learning_rate)
            villager.add_memory(self.tick, action.name, result.message, total_reward)
        
        # Learn from experienced before removing dead
        experienced = [v for v in self.villagers if v.skill_level > 20 or v.age > 30]
        if experienced:
            self.evolution.learn_from_villagers(experienced)
        
        # Remove dead
        for dead in deaths:
            if dead in self.villagers:
                self.villagers.remove(dead)
            self.stats.total_deaths += 1
        
        # Reproduction
        self._process_reproduction()
        
        # Building decisions
        self._process_building_decisions()
        
        # Building effects
        self._apply_building_effects()
        
        # Phase 2B: Generate research insight
        generate_insight(self, 1.0)
        
        # Phase 2C: Run planner decisions periodically
        if self.tick % 100 == 0:
            run_planner_decision(self, self.tick)
        
        # Stats
        self.stats.peak_population = max(self.stats.peak_population, len(self.villagers))
        
        # Events
        self._check_for_events()
        
        return True
    
    def _process_reproduction(self):
        if len(self.villagers) < 2:
            return
        if len(self.villagers) >= self.config.max_population:
            return
        
        stability = calculate_village_stability(self)
        min_stability = 0.3 if len(self.villagers) < 20 else 0.4
        if stability < min_stability:
            return
        
        min_food = 50 if len(self.villagers) < 20 else 100
        if self.resources.food < min_food:
            return
        
        adults = [v for v in self.villagers if 18 <= v.age <= 45 and v.health > 50 and not v.is_ill]
        if len(adults) < 2:
            return
        
        birth_chance = 0.003 * stability
        if stability > 0.7 and self.resources.food > 300:
            birth_chance *= 2
        
        if self._rng.random() < birth_chance:
            parent1, parent2 = self._rng.sample(adults, 2)
            child = create_child(parent1, parent2, self._next_villager_id)
            
            starting_weights = self.evolution.get_starting_weights()
            if starting_weights:
                child.action_weights = starting_weights.copy()
            
            child.position = (
                max(50, min(self.width - 50, child.position[0])),
                max(50, min(self.height - 50, child.position[1]))
            )
            
            self._next_villager_id += 1
            self.villagers.append(child)
            self.stats.total_births += 1
            self.pending_messages.append("A child has been born.")
    
    def _process_building_decisions(self):
        if len(self.building_manager.get_incomplete_buildings()) >= 1:
            return
        
        # Use simple comfort check for basic buildings
        stability = calculate_village_stability(self)
        pop = len(self.villagers)
        housing = self.resources.housing_capacity
        
        # High level decision from Planner
        building_type_name = self.planner.decide_building_priority(self)
        if not building_type_name:
            return
            
        try:
            building_type = BuildingType[building_type_name]
        except (KeyError, ValueError):
            return
 
        # Double check if we actually need it based on resources
        if building_type == BuildingType.HOUSING and pop < housing * 0.7:
            return
        if building_type == BuildingType.FARM and self.resources.food > 400:
            return
 
        # Find position based on district rules
        position = self._find_district_position(building_type)
        if position:
            self.building_manager.add_building(building_type, position)
            self.pending_messages.append(f"Construction started: {building_type.name.lower().replace('_', ' ')}")
 
    def _find_district_position(self, building_type: BuildingType) -> tuple[float, float]:
        """Find a position for a building based on its 'district'."""
        margin = 150
        center_x = self.width // 2
        center_y = self.height // 2
        
        if building_type in [BuildingType.FARM, BuildingType.ADVANCED_FARM]:
            # Farms to the North-East
            base_x = self.width - margin - random.uniform(0, 300)
            base_y = margin + random.uniform(0, 300)
        elif building_type in [BuildingType.HOUSING, BuildingType.EXPANDED_HOUSING]:
            # Housing to the South-West
            base_x = margin + random.uniform(0, 300)
            base_y = self.height - margin - random.uniform(0, 300)
        else:
            # Service buildings near center
            base_x = center_x + random.uniform(-200, 200)
            base_y = center_y + random.uniform(-200, 200)
            
        # Ensure spacing
        for b in self.building_manager.buildings:
            dist = math.sqrt((b.position[0]-base_x)**2 + (b.position[1]-base_y)**2)
            if dist < 60:
                # Try a slightly different spot
                base_x += random.uniform(-40, 40)
                base_y += random.uniform(-40, 40)
        
        return (max(50, min(self.width-50, base_x)), 
                max(50, min(self.height-50, base_y)))
    
    def _apply_building_effects(self):
        bonuses = self.building_manager.calculate_bonuses()
        
        if bonuses['food_production'] > 0:
            food_produced = bonuses['food_production'] * 0.15
            self.resources.food += food_produced
            self.stats.total_food_produced += food_produced
        
        if bonuses['knowledge'] > 0:
            self.resources.knowledge += bonuses['knowledge'] * 0.01
    
    def _check_for_events(self):
        # Food warnings
        if self.resources.food < 20 and self.tick % 200 == 0:
            self.pending_messages.append("Famine threatens the village.")
        elif self.resources.food < 50 and self.tick % 500 == 0:
            self.pending_messages.append("Food supplies are running low.")
        
        # Health warnings
        if self._health_status.get('infection_rate', 0) > 0.3 and self.tick % 300 == 0:
            self.pending_messages.append("Illness spreads through the village.")
            self.history.major_outbreaks += 1
        
        # Population
        if len(self.villagers) < 10 and self.tick % 200 == 0:
            self.pending_messages.append("The village struggles to survive.")
        
        if self.resources.food > 500 and self.tick % 300 == 0:
            self.pending_messages.append("The village prospers.")
    
    def _log_event(self, message: str):
        self.event_log.append(f"[{self.tick}] {message}")
        if len(self.event_log) > self._log_max_size:
            self.event_log.pop(0)
    
    def get_next_message(self) -> Optional[str]:
        if self.pending_messages:
            return self.pending_messages.pop(0)
        return None
    
    # Player intervention methods
    def trigger_disease_outbreak(self):
        """Player triggers an outbreak."""
        infected = trigger_outbreak(self.villagers, severity=0.3)
        self.pending_messages.append(f"A mysterious illness has struck {infected} villagers.")
        self.history.major_outbreaks += 1
    
    def trigger_resource_shortage(self):
        """Player triggers food shortage."""
        self.resources.food = max(0, self.resources.food * 0.3)
        self.pending_messages.append("A blight has destroyed much of the food supply.")
    
    def trigger_knowledge_breakthrough(self):
        """Player triggers knowledge bonus."""
        self.resources.knowledge += 20
        self.pending_messages.append("A breakthrough in knowledge occurs!")
    
    def to_dict(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'tick': self.tick,
            'villagers': [v.to_dict() for v in self.villagers],
            'resources': self.resources.to_dict(),
            'building_manager': self.building_manager.to_dict(),
            'stats': self.stats.to_dict(),
            'evolution': self.evolution.to_dict(),
            'history': self.history.to_dict(),
            'research': self.research.to_dict(),
            'planner': self.planner.to_dict(),
            'resource_nodes': [n.to_dict() for n in self.resource_nodes],
            'next_node_id': self._next_node_id,
            'next_villager_id': self._next_villager_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict, config=None) -> 'World':
        world = cls(
            width=data.get('width', 1920),
            height=data.get('height', 1080),
            config=config
        )
        
        world.tick = data.get('tick', 0)
        world.villagers = [Villager.from_dict(v) for v in data.get('villagers', [])]
        world.resources = Resources.from_dict(data.get('resources', {}))
        world.building_manager = BuildingManager.from_dict(data.get('building_manager', {}))
        world.stats = WorldStats.from_dict(data.get('stats', {}))
        world._next_villager_id = data.get('next_villager_id', 1)
        
        if 'evolution' in data:
            world.evolution = EvolutionMemory.from_dict(data['evolution'])
        if 'history' in data:
            world.history = VillageHistory.from_dict(data['history'])
        if 'research' in data:
            world.research = ResearchState.from_dict(data['research'])
        if 'planner' in data:
            world.planner = PlannerState.from_dict(data['planner'])
        if 'resource_nodes' in data:
            world.resource_nodes = [ResourceNode.from_dict(n) for n in data['resource_nodes']]
        if 'next_node_id' in data:
            world._next_node_id = data['next_node_id']
        
        return world
