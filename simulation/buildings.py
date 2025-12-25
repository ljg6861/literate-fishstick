"""
The Village - Buildings
Building types, construction, and effects.
Phase 2: Added Hospital, Research Center, Advanced Farm, Expanded Housing.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List
import random


class BuildingType(Enum):
    """Types of buildings that can be constructed."""
    HOUSING = auto()
    FARM = auto()
    WORKSHOP = auto()
    SCHOOL = auto()
    # Phase 2 buildings
    HOSPITAL = auto()
    RESEARCH_CENTER = auto()
    ADVANCED_FARM = auto()
    EXPANDED_HOUSING = auto()


@dataclass
class BuildingBlueprint:
    """Blueprint defining building properties."""
    building_type: BuildingType
    labor_cost: float
    housing_bonus: int = 0
    food_production_bonus: float = 0.0
    work_capacity_bonus: float = 0.0
    knowledge_bonus: float = 0.0
    healing_capacity: int = 0  # Phase 2: hospital capacity
    research_output: float = 0.0  # Phase 2: insight generation
    disease_risk_modifier: float = 1.0  # Phase 2: crowding risk
    requires_research: bool = False  # Phase 2: needs unlock
    description: str = ""


from config import CONFIG


# Building blueprints
BLUEPRINTS = {
    BuildingType.HOUSING: BuildingBlueprint(
        building_type=BuildingType.HOUSING,
        labor_cost=CONFIG.building_labor_cost * 0.8,
        housing_bonus=4,
        description="Provides shelter for villagers"
    ),
    BuildingType.FARM: BuildingBlueprint(
        building_type=BuildingType.FARM,
        labor_cost=CONFIG.building_labor_cost,
        food_production_bonus=1.0,
        description="Increases food production"
    ),
    BuildingType.WORKSHOP: BuildingBlueprint(
        building_type=BuildingType.WORKSHOP,
        labor_cost=CONFIG.building_labor_cost * 1.2,
        work_capacity_bonus=2.0,
        description="Increases work efficiency"
    ),
    BuildingType.SCHOOL: BuildingBlueprint(
        building_type=BuildingType.SCHOOL,
        labor_cost=CONFIG.building_labor_cost * 1.5,
        knowledge_bonus=0.5,
        description="Enables learning and skill gain"
    ),
    BuildingType.HOSPITAL: BuildingBlueprint(
        building_type=BuildingType.HOSPITAL,
        labor_cost=CONFIG.building_labor_cost * 2.0,
        healing_capacity=5,
        description="Treats illness and speeds recovery"
    ),
    BuildingType.RESEARCH_CENTER: BuildingBlueprint(
        building_type=BuildingType.RESEARCH_CENTER,
        labor_cost=CONFIG.building_labor_cost * 2.5,
        research_output=1.0,
        description="Generates insights for new tech"
    ),
    BuildingType.ADVANCED_FARM: BuildingBlueprint(
        building_type=BuildingType.ADVANCED_FARM,
        labor_cost=CONFIG.building_labor_cost * 1.2,
        food_production_bonus=2.0,  # 2x regular farm
        requires_research=True,
        description="High-yield farming (requires research)"
    ),
    BuildingType.EXPANDED_HOUSING: BuildingBlueprint(
        building_type=BuildingType.EXPANDED_HOUSING,
        labor_cost=CONFIG.building_labor_cost,
        housing_bonus=8,  # 2x regular housing
        disease_risk_modifier=1.5,  # Higher disease spread
        description="Dense housing (higher disease risk)"
    ),
}


@dataclass
class Building:
    """A building in the village."""
    
    id: int
    building_type: BuildingType
    position: tuple[float, float]
    construction_progress: float = 0.0
    labor_cost: float = 100.0
    
    # Phase 2: operational state
    workers_assigned: int = 0
    load: float = 0.0  # 0-1, how busy the building is
    
    # Visual properties
    size: float = 40.0
    
    @property
    def is_complete(self) -> bool:
        return self.construction_progress >= self.labor_cost
    
    @property
    def progress_percent(self) -> float:
        return min(100.0, (self.construction_progress / self.labor_cost) * 100)
    
    def add_labor(self, amount: float) -> bool:
        """Add construction labor. Returns True if building just completed."""
        was_complete = self.is_complete
        self.construction_progress += amount
        return not was_complete and self.is_complete
    
    def get_blueprint(self) -> BuildingBlueprint:
        """Get the blueprint for this building type."""
        return BLUEPRINTS[self.building_type]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'building_type': self.building_type.name,
            'position': self.position,
            'construction_progress': self.construction_progress,
            'labor_cost': self.labor_cost,
            'workers_assigned': self.workers_assigned,
            'load': self.load,
            'size': self.size,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Building':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            building_type=BuildingType[data['building_type']],
            position=tuple(data['position']),
            construction_progress=data.get('construction_progress', 0.0),
            labor_cost=data.get('labor_cost', 100.0),
            workers_assigned=data.get('workers_assigned', 0),
            load=data.get('load', 0.0),
            size=data.get('size', 40.0),
        )


class BuildingManager:
    """Manages village buildings."""
    
    def __init__(self):
        self.buildings: List[Building] = []
        self._next_id: int = 1
        self._pending_construction: Optional[BuildingType] = None
    
    def add_building(self, building_type: BuildingType, position: tuple[float, float]) -> Building:
        """Create and add a new building."""
        blueprint = BLUEPRINTS[building_type]
        building = Building(
            id=self._next_id,
            building_type=building_type,
            position=position,
            labor_cost=blueprint.labor_cost
        )
        self._next_id += 1
        self.buildings.append(building)
        return building
    
    def get_incomplete_buildings(self) -> List[Building]:
        """Get all buildings under construction."""
        return [b for b in self.buildings if not b.is_complete]
    
    def get_complete_buildings(self) -> List[Building]:
        """Get all completed buildings."""
        return [b for b in self.buildings if b.is_complete]
    
    def get_buildings_by_type(self, building_type: BuildingType) -> List[Building]:
        """Get all completed buildings of a type."""
        return [b for b in self.buildings 
                if b.building_type == building_type and b.is_complete]
    
    def count_by_type(self, building_type: BuildingType) -> int:
        """Count completed buildings of a type."""
        return sum(1 for b in self.buildings 
                   if b.building_type == building_type and b.is_complete)
    
    def has_hospital(self) -> bool:
        """Check if village has a functioning hospital."""
        return self.count_by_type(BuildingType.HOSPITAL) > 0
    
    def has_research_center(self) -> bool:
        """Check if village has a research center."""
        return self.count_by_type(BuildingType.RESEARCH_CENTER) > 0
    
    def get_total_healing_capacity(self) -> int:
        """Get total hospital healing capacity."""
        total = 0
        for b in self.get_buildings_by_type(BuildingType.HOSPITAL):
            total += b.get_blueprint().healing_capacity
        return total
    
    def get_total_research_output(self) -> float:
        """Get total research output."""
        total = 0.0
        for b in self.get_buildings_by_type(BuildingType.RESEARCH_CENTER):
            total += b.get_blueprint().research_output
        return total
    
    def calculate_bonuses(self) -> dict:
        """Calculate total bonuses from all completed buildings."""
        bonuses = {
            'housing': 0,
            'food_production': 0.0,
            'work_capacity': 0.0,
            'knowledge': 0.0,
            'healing_capacity': 0,
            'research_output': 0.0,
            'disease_risk_modifier': 1.0,
        }
        
        housing_risk_sum = 0.0
        housing_count = 0
        
        for building in self.get_complete_buildings():
            blueprint = building.get_blueprint()
            bonuses['housing'] += blueprint.housing_bonus
            bonuses['food_production'] += blueprint.food_production_bonus
            bonuses['work_capacity'] += blueprint.work_capacity_bonus
            bonuses['knowledge'] += blueprint.knowledge_bonus
            bonuses['healing_capacity'] += blueprint.healing_capacity
            bonuses['research_output'] += blueprint.research_output
            
            # Track disease risk from housing
            if blueprint.housing_bonus > 0:
                housing_risk_sum += blueprint.disease_risk_modifier * blueprint.housing_bonus
                housing_count += blueprint.housing_bonus
        
        # Average disease risk modifier
        if housing_count > 0:
            bonuses['disease_risk_modifier'] = housing_risk_sum / housing_count
        
        return bonuses
    
    def should_start_construction(self, resources, population: int, 
                                   health_status: dict = None,
                                   has_research_unlock: bool = False) -> Optional[BuildingType]:
        """
        Determine if the village should start building something.
        Phase 2: considers health status and research.
        """
        # Don't start if already building too many
        if len(self.get_incomplete_buildings()) >= 2:
            return None
        
        bonuses = self.calculate_bonuses()
        current_capacity = 20 + bonuses['housing']
        
        hospital_count = self.count_by_type(BuildingType.HOSPITAL)
        farm_count = self.count_by_type(BuildingType.FARM)
        research_count = self.count_by_type(BuildingType.RESEARCH_CENTER)
        
        # Phase 2: Consider illness
        if health_status:
            infection_rate = health_status.get('infection_rate', 0)
            severe_count = health_status.get('severe', 0)
            
            # Critical: need hospital if high infection and no hospital
            if infection_rate > 0.2 and hospital_count == 0:
                return BuildingType.HOSPITAL
            
            # Build more hospitals if many severely ill
            if severe_count > 5 and hospital_count < 2:
                return BuildingType.HOSPITAL
        
        # Priority 1: Housing if overcrowded
        if population > current_capacity * 0.8:
            # Choose expanded housing if available via research
            if has_research_unlock and random.random() < 0.3:
                return BuildingType.EXPANDED_HOUSING
            return BuildingType.HOUSING
        
        # Priority 2: Farms if food is low
        if resources.food < 200:
            if has_research_unlock and farm_count >= 2:
                return BuildingType.ADVANCED_FARM
            elif farm_count < 4:
                return BuildingType.FARM
        
        # Priority 3: Research center (one is enough early on)
        if research_count == 0 and population > 25:
            return BuildingType.RESEARCH_CENTER
        
        # Priority 4: Hospital if none (preventive)
        if hospital_count == 0 and population > 20:
            return BuildingType.HOSPITAL
        
        # Expand based on prosperity
        if resources.food > 500:
            options = [BuildingType.HOUSING, BuildingType.FARM]
            if hospital_count < 2:
                options.append(BuildingType.HOSPITAL)
            if research_count < 2:
                options.append(BuildingType.RESEARCH_CENTER)
            return random.choice(options)
        
        return None
    
    def find_building_position(self, world_width: float, world_height: float) -> tuple[float, float]:
        """Find a good position for a new building."""
        margin = 100
        spacing = 80
        
        index = len(self.buildings)
        cols = int((world_width - 2 * margin) / spacing)
        
        row = index // max(1, cols)
        col = index % max(1, cols)
        
        x = margin + col * spacing + random.uniform(-10, 10)
        y = margin + row * spacing + random.uniform(-10, 10)
        
        x = max(margin, min(world_width - margin, x))
        y = max(margin, min(world_height - margin, y))
        
        return (x, y)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'buildings': [b.to_dict() for b in self.buildings],
            'next_id': self._next_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BuildingManager':
        """Deserialize from dictionary."""
        manager = cls()
        manager.buildings = [Building.from_dict(b) for b in data.get('buildings', [])]
        manager._next_id = data.get('next_id', 1)
        return manager
