"""
The Village - The Planner
Phase 2C: Institutional decision-making agent with cultural memory.
The Planner makes village-level decisions and develops biases over time.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
import random


@dataclass
class CrisisMemory:
    """Record of a past crisis."""
    tick: int
    crisis_type: str  # 'plague', 'famine', 'collapse'
    severity: float  # 0-1
    resolved: bool
    resolution_method: Optional[str] = None


@dataclass 
class PlannerState:
    """
    The Planner - a constrained policy module for village-level decisions.
    Not an omniscient AI, but a biased decision-maker that learns from history.
    """
    
    # Short-term memory (recent events)
    short_term_memory: Deque[CrisisMemory] = field(default_factory=lambda: deque(maxlen=10))
    
    # Long-term bias vector (cultural preferences)
    # Higher values = more priority
    building_bias: Dict[str, float] = field(default_factory=dict)
    research_bias: Dict[str, float] = field(default_factory=dict)
    
    # Tracked metrics for learning
    population_history: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    stability_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    # Decision tracking
    decisions_made: int = 0
    last_decision_tick: int = 0
    
    def __post_init__(self):
        # Initialize biases
        if not self.building_bias:
            self.building_bias = {
                'HOUSING': 1.0,
                'FARM': 1.0,
                'HOSPITAL': 1.0,
                'RESEARCH_CENTER': 1.0,
                'WORKSHOP': 1.0,
                'SCHOOL': 1.0,
            }
        
        if not self.research_bias:
            from learning.research import ResearchCategory
            self.research_bias = {
                cat.name: 1.0 for cat in ResearchCategory
            }
    
    def record_crisis(self, tick: int, crisis_type: str, severity: float):
        """Record a crisis in memory."""
        crisis = CrisisMemory(
            tick=tick,
            crisis_type=crisis_type,
            severity=severity,
            resolved=False
        )
        self.short_term_memory.append(crisis)
        
        # Adjust biases based on crisis type
        self._learn_from_crisis(crisis_type, severity)
    
    def _learn_from_crisis(self, crisis_type: str, severity: float):
        """
        Cultural inertia: the village learns to overweight solutions
        to past crises, even when conditions change.
        """
        if crisis_type == 'plague':
            # Overweight hospitals after plague
            self.building_bias['HOSPITAL'] += severity * 0.5
            self.research_bias['MEDICINE'] = self.research_bias.get('MEDICINE', 1.0) + severity * 0.3
            
        elif crisis_type == 'famine':
            # Overweight farms after famine
            self.building_bias['FARM'] += severity * 0.5
            self.research_bias['AGRICULTURE'] = self.research_bias.get('AGRICULTURE', 1.0) + severity * 0.3
            
        elif crisis_type == 'housing_crisis':
            self.building_bias['HOUSING'] += severity * 0.5
            
        elif crisis_type == 'stagnation':
            # Underinvestment in progress
            self.building_bias['RESEARCH_CENTER'] += severity * 0.3
            self.building_bias['SCHOOL'] += severity * 0.3
    
    def update_metrics(self, population: int, stability: float):
        """Update tracked metrics."""
        self.population_history.append(population)
        self.stability_history.append(stability)
    
    def get_population_variance(self) -> float:
        """Calculate population variance (lower is better for stability)."""
        if len(self.population_history) < 2:
            return 0.0
        
        values = list(self.population_history)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def get_average_stability(self) -> float:
        """Get average stability over recent history."""
        if not self.stability_history:
            return 0.5
        return sum(self.stability_history) / len(self.stability_history)
    
    def decide_building_priority(self, world) -> Optional[str]:
        """
        Decide what building to prioritize.
        Uses biased weights modified by current conditions.
        """
        from simulation.buildings import BuildingType
        from simulation.health import get_village_health_status
        
        # Get current conditions
        health_status = get_village_health_status(world.villagers)
        pop = len(world.villagers)
        food = world.resources.food
        housing = world.resources.housing_capacity
        
        # Start with base biases
        scores = dict(self.building_bias)
        
        # Modify based on current needs (urgency)
        if food < 100:
            scores['FARM'] += 2.0
        elif food < 200:
            scores['FARM'] += 1.0
        
        if pop > housing * 0.8:
            scores['HOUSING'] += 2.0
        
        if health_status['infection_rate'] > 0.2:
            scores['HOSPITAL'] += 2.0
        elif health_status['infection_rate'] > 0.1:
            scores['HOSPITAL'] += 1.0
        
        # Research priority when stable
        avg_stability = self.get_average_stability()
        if avg_stability > 0.7:
            scores['RESEARCH_CENTER'] += 1.0
        
        # Normalize and select probabilistically
        total = sum(scores.values())
        if total == 0:
            return None
        
        r = random.random() * total
        cumulative = 0.0
        for building, score in scores.items():
            cumulative += score
            if r <= cumulative:
                self.decisions_made += 1
                return building
        
        return list(scores.keys())[0]
    
    def decide_research_allocation(self, world) -> Optional[str]:
        """
        Decide which research category to invest in.
        Reflects institutional values and past experiences.
        """
        from learning.research import ResearchCategory
        
        if not hasattr(world, 'research') or not world.research.can_specialize():
            return None
        
        # Get current conditions
        from simulation.health import get_village_health_status
        health_status = get_village_health_status(world.villagers)
        
        scores = dict(self.research_bias)
        
        # Modify based on conditions
        if health_status['infection_rate'] > 0.1:
            scores['MEDICINE'] = scores.get('MEDICINE', 1.0) + 1.0
        
        if world.resources.food < 200:
            scores['AGRICULTURE'] = scores.get('AGRICULTURE', 1.0) + 1.0
        
        if world.has_active_construction():
            scores['ENGINEERING'] = scores.get('ENGINEERING', 1.0) + 0.5
        
        # Select probabilistically
        total = sum(scores.values())
        if total == 0:
            return None
        
        r = random.random() * total
        cumulative = 0.0
        for cat, score in scores.items():
            cumulative += score
            if r <= cumulative:
                return cat
        
        return list(scores.keys())[0]
    
    def get_cultural_summary(self) -> str:
        """Generate a summary of the village's cultural leanings."""
        # Find highest biases
        top_building = max(self.building_bias.items(), key=lambda x: x[1])
        top_research = max(self.research_bias.items(), key=lambda x: x[1])
        
        building_desc = {
            'HOSPITAL': 'healthcare-focused',
            'FARM': 'agricultural',
            'HOUSING': 'expansionist', 
            'RESEARCH_CENTER': 'intellectual',
            'WORKSHOP': 'industrious',
            'SCHOOL': 'educational',
        }
        
        research_desc = {
            'MEDICINE': 'medical advancement',
            'AGRICULTURE': 'farming techniques',
            'ENGINEERING': 'construction methods',
            'EDUCATION': 'knowledge preservation',
        }
        
        culture = building_desc.get(top_building[0], 'balanced')
        focus = research_desc.get(top_research[0], 'general knowledge')
        
        return f"A {culture} society focused on {focus}"
    
    def to_dict(self) -> dict:
        return {
            'short_term_memory': [
                {
                    'tick': c.tick,
                    'crisis_type': c.crisis_type,
                    'severity': c.severity,
                    'resolved': c.resolved,
                }
                for c in self.short_term_memory
            ],
            'building_bias': self.building_bias,
            'research_bias': self.research_bias,
            'decisions_made': self.decisions_made,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PlannerState':
        state = cls()
        
        if 'short_term_memory' in data:
            for m in data['short_term_memory']:
                crisis = CrisisMemory(
                    tick=m['tick'],
                    crisis_type=m['crisis_type'],
                    severity=m['severity'],
                    resolved=m.get('resolved', False)
                )
                state.short_term_memory.append(crisis)
        
        state.building_bias = data.get('building_bias', state.building_bias)
        state.research_bias = data.get('research_bias', state.research_bias)
        state.decisions_made = data.get('decisions_made', 0)
        
        return state


def detect_crisis(world) -> Optional[tuple]:
    """
    Detect if the village is in crisis.
    Returns (crisis_type, severity) or None.
    """
    from simulation.health import get_village_health_status
    from learning.rewards import calculate_village_stability
    
    health_status = get_village_health_status(world.villagers)
    stability = calculate_village_stability(world)
    
    # Plague detection
    if health_status['infection_rate'] > 0.4:
        return ('plague', health_status['infection_rate'])
    
    # Famine detection
    if world.resources.food < 20:
        return ('famine', 1.0 - (world.resources.food / 20))
    elif world.resources.food < 50 and len(world.villagers) > 20:
        return ('famine', 0.5)
    
    # Housing crisis
    housing_ratio = len(world.villagers) / max(1, world.resources.housing_capacity)
    if housing_ratio > 1.2:
        return ('housing_crisis', min(1.0, housing_ratio - 1.0))
    
    # Stagnation (no growth, no progress)
    if stability > 0.8 and len(world.villagers) < 30:
        return ('stagnation', 0.3)
    
    return None


def run_planner_decision(world, tick: int):
    """
    Run the planner's decision-making process.
    Called periodically to make institutional decisions.
    """
    from learning.rewards import calculate_village_stability
    
    planner = world.planner
    
    # Update metrics
    stability = calculate_village_stability(world)
    planner.update_metrics(len(world.villagers), stability)
    
    # Detect and record crises
    crisis = detect_crisis(world)
    if crisis:
        crisis_type, severity = crisis
        # Only record if not already in recent memory of same type
        recent_types = [c.crisis_type for c in planner.short_term_memory]
        if crisis_type not in recent_types[-3:]:  # Last 3 crises
            planner.record_crisis(tick, crisis_type, severity)
    
    # Research decisions (when enough insight accumulated)
    if hasattr(world, 'research') and world.research.can_specialize():
        category_name = planner.decide_research_allocation(world)
        if category_name:
            from learning.research import ResearchCategory
            try:
                category = ResearchCategory[category_name]
                if world.research.apply_insight(category):
                    world.pending_messages.append(
                        f"The village invests in {category_name.lower().replace('_', ' ')} research."
                    )
            except KeyError:
                pass
    
    planner.last_decision_tick = tick
