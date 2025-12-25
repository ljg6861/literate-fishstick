"""
The Village - Text Overlays
Ambient text messages that fade in and out.
"""

import pygame
from typing import Optional
from config import RENDER_CONFIG


class TextOverlay:
    """A single text overlay message."""
    
    def __init__(self, message: str, duration: float = 5.0, fade_duration: float = 1.0):
        self.message = message
        self.duration = duration
        self.fade_duration = fade_duration
        self.elapsed = 0.0
        self.alpha = 0
    
    @property
    def is_finished(self) -> bool:
        return self.elapsed >= self.duration
    
    def update(self, dt: float):
        """Update overlay state."""
        self.elapsed += dt
        
        # Calculate alpha based on phase
        if self.elapsed < self.fade_duration:
            # Fade in
            self.alpha = int(255 * (self.elapsed / self.fade_duration))
        elif self.elapsed > self.duration - self.fade_duration:
            # Fade out
            remaining = self.duration - self.elapsed
            self.alpha = int(255 * (remaining / self.fade_duration))
        else:
            # Full visibility
            self.alpha = 255
        
        self.alpha = max(0, min(255, self.alpha))


class OverlayManager:
    """Manages text overlays on screen."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = RENDER_CONFIG
        
        # Initialize font
        pygame.font.init()
        self.font = pygame.font.SysFont('Georgia', self.config.overlay_font_size)
        
        # Current overlay
        self.current_overlay: Optional[TextOverlay] = None
        
        # Message queue
        self.message_queue: list[str] = []
        
        # Cooldown between messages
        self.cooldown = 0.0
        self.cooldown_duration = 2.0  # Seconds between messages
    
    def queue_message(self, message: str):
        """Add a message to the queue."""
        if message not in self.message_queue:
            self.message_queue.append(message)
    
    def update(self, dt: float):
        """Update overlay state."""
        # Update cooldown
        if self.cooldown > 0:
            self.cooldown -= dt
        
        # Update current overlay
        if self.current_overlay:
            self.current_overlay.update(dt)
            if self.current_overlay.is_finished:
                self.current_overlay = None
                self.cooldown = self.cooldown_duration
        
        # Start next message if available
        if self.current_overlay is None and self.cooldown <= 0 and self.message_queue:
            message = self.message_queue.pop(0)
            self.current_overlay = TextOverlay(
                message,
                duration=self.config.overlay_display_duration,
                fade_duration=self.config.overlay_fade_duration
            )
    
    def draw(self, screen: pygame.Surface):
        """Draw current overlay if any."""
        if self.current_overlay is None or self.current_overlay.alpha <= 0:
            return
        
        # Render text
        text_surface = self.font.render(
            self.current_overlay.message,
            True,
            self.config.color_foreground
        )
        
        # Apply alpha
        text_surface.set_alpha(self.current_overlay.alpha)
        
        # Center horizontally, position in lower third
        x = (self.screen_width - text_surface.get_width()) // 2
        y = int(self.screen_height * 0.7)
        
        screen.blit(text_surface, (x, y))
