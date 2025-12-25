"""
The Village - Event Logger
Structured logging for debugging and observability.
"""

import logging
import os
from datetime import datetime


class EventLogger:
    """Logger for simulation events."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Create logs directory
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("village")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"village_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Also log to console in debug mode (optional)
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(formatter)
        # self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_tick(self, tick: int, population: int, food: float, stability: float):
        """Log periodic tick summary."""
        if tick % 100 == 0:  # Log every 100 ticks
            self.debug(
                f"Tick {tick}: pop={population}, food={food:.1f}, stability={stability:.2f}"
            )
    
    def log_event(self, tick: int, event_type: str, details: str):
        """Log a simulation event."""
        self.info(f"[Tick {tick}] {event_type}: {details}")
    
    def log_learning(self, tick: int, villager_id: int, action: str, 
                     reward: float, new_weight: float):
        """Log learning update."""
        self.debug(
            f"[Tick {tick}] Learning: V{villager_id} action={action} "
            f"reward={reward:.3f} weight={new_weight:.3f}"
        )


# Global logger instance
def get_logger() -> EventLogger:
    """Get the global event logger."""
    return EventLogger()
