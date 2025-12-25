"""
The Village - Research System
Phase 2B: Research specialization and insight points.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import random


class ResearchCategory(Enum):
    """Categories of research specialization."""
    AGRICULTURE = auto()    # Farm efficiency
    MEDICINE = auto()       # Hospital effectiveness
    ENGINEERING = auto()    # Construction speed
    EDUCATION = auto()      # Skill growth rate


@dataclass
class ResearchState:
    """Tracks research progress and specializations."""
    insight_points: float = 0.0
    specializations: Dict[ResearchCategory, int] = field(default_factory=dict)
    pending_insights: float = 0.0  # Accumulated before decision
    
    # Track what the AI has chosen
    research_decisions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Initialize all categories at 0
        for cat in ResearchCategory:
            if cat not in self.specializations:
                self.specializations[cat] = 0
    
    def add_insight(self, amount: float):
        """Add insight points from research centers."""
        self.pending_insights += amount
    
    def can_specialize(self) -> bool:
        """Check if we have enough insight to specialize."""
        return self.pending_insights >= 10.0
    
    def apply_insight(self, category: ResearchCategory) -> bool:
        """
        Apply pending insight to a category.
        This is a permanent choice - represents institutional decision.
        """
        if not self.can_specialize():
            return False
        
        self.specializations[category] += 1
        self.insight_points += self.pending_insights
        cost = 10.0
        self.pending_insights -= cost
        
        self.research_decisions.append(category.name)
        return True
    
    def get_bonus(self, category: ResearchCategory) -> float:
        """
        Get the bonus multiplier for a category.
        Each level gives +10% bonus.
        """
        level = self.specializations.get(category, 0)
        return 1.0 + (level * 0.1)
    
    def get_farm_bonus(self) -> float:
        """Get farm production multiplier."""
        return self.get_bonus(ResearchCategory.AGRICULTURE)
    
    def get_hospital_bonus(self) -> float:
        """Get hospital effectiveness multiplier."""
        return self.get_bonus(ResearchCategory.MEDICINE)
    
    def get_construction_bonus(self) -> float:
        """Get construction speed multiplier."""
        return self.get_bonus(ResearchCategory.ENGINEERING)
    
    def get_education_bonus(self) -> float:
        """Get skill growth multiplier."""
        return self.get_bonus(ResearchCategory.EDUCATION)
    
    def to_dict(self) -> dict:
        return {
            'insight_points': self.insight_points,
            'specializations': {k.name: v for k, v in self.specializations.items()},
            'pending_insights': self.pending_insights,
            'research_decisions': self.research_decisions,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ResearchState':
        state = cls()
        state.insight_points = data.get('insight_points', 0.0)
        state.pending_insights = data.get('pending_insights', 0.0)
        state.research_decisions = data.get('research_decisions', [])
        
        if 'specializations' in data:
            for name, level in data['specializations'].items():
                try:
                    cat = ResearchCategory[name]
                    state.specializations[cat] = level
                except KeyError:
                    pass
        
        return state


def generate_insight(world, output_rate: float) -> float:
    """
    Generate insight from research centers.
    Called each tick when researchers are working.
    """
    # Get research output from buildings
    research_output = world.building_manager.get_total_research_output()
    
    if research_output > 0:
        insight = output_rate * research_output * 0.01
        world.research.add_insight(insight)
        return insight
    
    return 0.0
