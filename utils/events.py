"""
The Village - Events
Event definitions and tracking.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class EventType(Enum):
    """Types of events that can occur."""
    BIRTH = auto()
    DEATH = auto()
    BUILDING_STARTED = auto()
    BUILDING_COMPLETED = auto()
    FOOD_SHORTAGE = auto()
    FOOD_SURPLUS = auto()
    POPULATION_MILESTONE = auto()
    KNOWLEDGE_MILESTONE = auto()
    VILLAGE_STABLE = auto()
    VILLAGE_STRUGGLING = auto()


@dataclass
class Event:
    """A simulation event."""
    tick: int
    event_type: EventType
    message: str
    data: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            'tick': self.tick,
            'event_type': self.event_type.name,
            'message': self.message,
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        return cls(
            tick=data['tick'],
            event_type=EventType[data['event_type']],
            message=data['message'],
            data=data.get('data')
        )


class EventTracker:
    """Tracks events for analysis."""
    
    def __init__(self, max_events: int = 1000):
        self.events: list[Event] = []
        self.max_events = max_events
    
    def record(self, tick: int, event_type: EventType, message: str, 
               data: Optional[dict] = None):
        """Record an event."""
        event = Event(tick, event_type, message, data)
        self.events.append(event)
        
        # Trim old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_recent(self, count: int = 10) -> list[Event]:
        """Get most recent events."""
        return self.events[-count:]
    
    def count_by_type(self, event_type: EventType, since_tick: int = 0) -> int:
        """Count events of a type since a tick."""
        return sum(1 for e in self.events 
                   if e.event_type == event_type and e.tick >= since_tick)
    
    def get_statistics(self, since_tick: int = 0) -> dict:
        """Get event statistics."""
        stats = {}
        for event_type in EventType:
            stats[event_type.name] = self.count_by_type(event_type, since_tick)
        return stats
    
    def to_dict(self) -> dict:
        return {
            'events': [e.to_dict() for e in self.events],
            'max_events': self.max_events
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EventTracker':
        tracker = cls(max_events=data.get('max_events', 1000))
        tracker.events = [Event.from_dict(e) for e in data.get('events', [])]
        return tracker
