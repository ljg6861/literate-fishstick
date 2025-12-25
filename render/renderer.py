"""
The Village - Main Renderer
Pygame-based rendering system.
"""

import pygame
from typing import Optional
from config import RENDER_CONFIG
from render.hud import HUD
from render.overlays import OverlayManager
from render.villager_sprites import VillagerRenderer
from render.building_sprites import BuildingRenderer


class Renderer:
    """Main rendering system using Pygame."""
    
    def __init__(self):
        self.config = RENDER_CONFIG
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        
        # Screen dimensions (set on init)
        self.width = 0
        self.height = 0
        
        # Camera
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.target_camera_x = 0.0
        self.target_camera_y = 0.0
        
        # Sub-renderers
        self.hud: Optional[HUD] = None
        self.overlay_manager: Optional[OverlayManager] = None
        self.villager_renderer: Optional[VillagerRenderer] = None
        self.building_renderer: Optional[BuildingRenderer] = None
        
        # State
        self.initialized = False
    
    def initialize(self) -> tuple[int, int]:
        """Initialize Pygame and create window. Returns (width, height)."""
        pygame.init()
        pygame.font.init()
        
        # Get display info for fullscreen
        display_info = pygame.display.Info()
        self.width = display_info.current_w
        self.height = display_info.current_h
        
        # Create fullscreen window
        if self.config.fullscreen:
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            )
        else:
            self.width = 1280
            self.height = 720
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption(self.config.window_title)
        
        # Hide mouse cursor for clean display
        pygame.mouse.set_visible(False)
        
        # Create clock for FPS management
        self.clock = pygame.time.Clock()
        
        # Initialize sub-renderers
        self.hud = HUD(self.width, self.height)
        self.overlay_manager = OverlayManager(self.width, self.height)
        self.villager_renderer = VillagerRenderer()
        self.building_renderer = BuildingRenderer()
        
        self.initialized = True
        
        return (self.width, self.height)
    
    def shutdown(self):
        """Clean up Pygame resources."""
        if self.initialized:
            pygame.quit()
            self.initialized = False
    
    def process_events(self) -> dict:
        """
        Process Pygame events.
        Returns dict with 'quit' and other event flags.
        """
        result = {
            'quit': False,
            'pause_toggle': False,
            'speed_up': False,
            'speed_down': False,
            'save': False,
            # Phase 2: Player interventions
            'trigger_outbreak': False,
            'trigger_shortage': False,
            'trigger_breakthrough': False,
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result['quit'] = True
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    result['quit'] = True
                elif event.key == pygame.K_SPACE:
                    result['pause_toggle'] = True
                elif event.key == pygame.K_UP or event.key == pygame.K_EQUALS:
                    result['speed_up'] = True
                elif event.key == pygame.K_DOWN or event.key == pygame.K_MINUS:
                    result['speed_down'] = True
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    result['save'] = True
                # Phase 2: Player interventions
                elif event.key == pygame.K_1:
                    result['trigger_outbreak'] = True
                elif event.key == pygame.K_2:
                    result['trigger_shortage'] = True
                elif event.key == pygame.K_3:
                    result['trigger_breakthrough'] = True
        
        return result
    
    def update_camera(self, world, dt: float):
        """Update camera to follow village center."""
        if len(world.villagers) == 0:
            return
        
        # Calculate center of all villagers
        avg_x = sum(v.position[0] for v in world.villagers) / len(world.villagers)
        avg_y = sum(v.position[1] for v in world.villagers) / len(world.villagers)
        
        # Target camera position (centered on village)
        self.target_camera_x = avg_x - self.width / 2
        self.target_camera_y = avg_y - self.height / 2
        
        # Smooth camera movement
        smoothing = self.config.camera_smoothing
        self.camera_x += (self.target_camera_x - self.camera_x) * smoothing
        self.camera_y += (self.target_camera_y - self.camera_y) * smoothing
    
    def render(self, world, dt: float):
        """Render the full frame."""
        if not self.initialized:
            return
        
        # Clear screen
        self.screen.fill(self.config.color_background)
        
        # Camera offset
        camera_offset = (self.camera_x, self.camera_y)
        
        # Draw resource nodes
        self.building_renderer.draw_resource_nodes(
            self.screen,
            world.resource_nodes,
            camera_offset
        )

        # Draw buildings
        self.building_renderer.draw_all(
            self.screen, 
            world.buildings, 
            camera_offset
        )
        
        # Draw villagers
        self.villager_renderer.draw_all(
            self.screen,
            world.villagers,
            camera_offset
        )
        
        # Draw HUD
        self.hud.draw(self.screen, world)
        
        # Process and draw overlays
        message = world.get_next_message()
        if message:
            self.overlay_manager.queue_message(message)
        
        self.overlay_manager.update(dt)
        self.overlay_manager.draw(self.screen)
        
        # Update display
        pygame.display.flip()
    
    def tick(self) -> float:
        """
        Tick the clock and return delta time in seconds.
        Also enforces FPS limit.
        """
        return self.clock.tick(60) / 1000.0
    
    def draw_pause_indicator(self):
        """Draw pause indicator on screen."""
        if not self.initialized:
            return
        
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))
        
        # Pause text
        font = pygame.font.SysFont('Consolas', 48)
        text = font.render("PAUSED", True, self.config.color_foreground)
        x = (self.width - text.get_width()) // 2
        y = (self.height - text.get_height()) // 2
        self.screen.blit(text, (x, y))
        
        # Speed controls hint
        hint_font = pygame.font.SysFont('Consolas', 18)
        hint = hint_font.render("SPACE to resume | UP/DOWN for speed | ESC to quit", 
                                True, self.config.color_accent)
        hx = (self.width - hint.get_width()) // 2
        hy = y + 60
        self.screen.blit(hint, (hx, hy))
        
        pygame.display.flip()
    
    def draw_speed_indicator(self, speed_multiplier: float):
        """Draw current speed indicator briefly."""
        if not self.initialized:
            return
        
        font = pygame.font.SysFont('Consolas', 24)
        text = font.render(f"Speed: {speed_multiplier:.1f}x", True, self.config.color_foreground)
        
        # Top right corner
        x = self.width - text.get_width() - 20
        y = 20
        
        self.screen.blit(text, (x, y))
