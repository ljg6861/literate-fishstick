"""
The Village - Main Entry Point
Runs the village simulation.
"""

import sys
import time
import argparse
from config import CONFIG, RENDER_CONFIG
from simulation.world import World
from render.renderer import Renderer
from persistence.save_manager import SaveManager
from utils.logger import get_logger
from learning.rewards import calculate_village_stability


class VillageSimulation:
    """Main simulation controller."""
    
    def __init__(self, seed: int = None, load_save: bool = True):
        self.config = CONFIG
        self.render_config = RENDER_CONFIG
        
        # Set seed if provided
        if seed is not None:
            self.config.seed = seed
        
        # Core components
        self.world: World = None
        self.renderer = Renderer()
        self.save_manager = SaveManager()
        self.logger = get_logger()
        
        # State
        self.running = False
        self.paused = False
        self.speed_multiplier = 1.0
        self.ticks_per_frame = 1
        
        # Timing
        self.accumulated_time = 0.0
        self.tick_interval = 1.0 / self.config.ticks_per_second_normal
        
        # Try to load existing save
        self.load_save = load_save
    
    def initialize(self):
        """Initialize the simulation."""
        self.logger.info("Initializing The Village...")
        
        # Initialize renderer first to get screen dimensions
        screen_width, screen_height = self.renderer.initialize()
        self.logger.info(f"Screen: {screen_width}x{screen_height}")
        
        # Try to load existing save
        if self.load_save:
            try:
                self.world = self.save_manager.load_latest(config=self.config)
                if self.world:
                    self.world.width = screen_width
                    self.world.height = screen_height
                    self.logger.info(f"Loaded save: tick {self.world.tick}")
            except Exception as e:
                self.logger.warning(f"Could not load save: {e}")
                self.world = None
        
        # Create new world if no save loaded
        if self.world is None:
            self.world = World(
                width=screen_width,
                height=screen_height,
                config=self.config
            )
            seed = self.config.get_seed()
            self.world.initialize(seed=seed)
            self.logger.info(f"Created new world with seed {seed}")
        
        self.running = True
        self.logger.info("Initialization complete")
    
    def run(self):
        """Main simulation loop."""
        self.logger.info("Starting simulation loop")
        
        try:
            while self.running:
                # Get frame delta time
                dt = self.renderer.tick()
                
                # Process input events
                events = self.renderer.process_events()
                self._handle_events(events)
                
                if not self.running:
                    break
                
                # Update simulation if not paused
                if not self.paused:
                    self._update_simulation(dt)
                
                # Update camera
                self.renderer.update_camera(self.world, dt)
                
                # Render
                self.renderer.render(self.world, dt)
                
                # Draw speed indicator if not 1x
                if self.speed_multiplier != 1.0:
                    self.renderer.draw_speed_indicator(self.speed_multiplier)
                
                # Draw pause indicator if paused
                if self.paused:
                    self.renderer.draw_pause_indicator()
                
                # Auto-save check
                if self.save_manager.should_auto_save(dt):
                    self._auto_save()
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        finally:
            self._shutdown()
    
    def _handle_events(self, events: dict):
        """Handle input events."""
        if events['quit']:
            self.running = False
        
        if events['pause_toggle']:
            self.paused = not self.paused
            self.logger.info(f"Paused: {self.paused}")
        
        if events['speed_up']:
            self._change_speed(1.5)
        
        if events['speed_down']:
            self._change_speed(0.667)
        
        if events['save']:
            self._manual_save()
        
        # Phase 2: Player interventions
        if events.get('trigger_outbreak'):
            self.world.trigger_disease_outbreak()
            self.logger.info("Player triggered disease outbreak")
        
        if events.get('trigger_shortage'):
            self.world.trigger_resource_shortage()
            self.logger.info("Player triggered resource shortage")
        
        if events.get('trigger_breakthrough'):
            self.world.trigger_knowledge_breakthrough()
            self.logger.info("Player triggered knowledge breakthrough")
    
    def _change_speed(self, factor: float):
        """Change simulation speed."""
        self.speed_multiplier *= factor
        self.speed_multiplier = max(0.1, min(10.0, self.speed_multiplier))
        
        # Update ticks per frame based on speed
        if self.speed_multiplier > 1:
            self.ticks_per_frame = int(self.speed_multiplier)
        else:
            self.ticks_per_frame = 1
        
        self.logger.info(f"Speed: {self.speed_multiplier:.1f}x")
    
    def _update_simulation(self, dt: float):
        """Update simulation by appropriate number of ticks."""
        # Accumulate time
        self.accumulated_time += dt * self.speed_multiplier
        
        # Calculate how many ticks to run
        ticks_to_run = int(self.accumulated_time / self.tick_interval)
        self.accumulated_time -= ticks_to_run * self.tick_interval
        
        # Cap ticks per frame to prevent spiral of death
        ticks_to_run = min(ticks_to_run, 20)
        
        # Run ticks
        for _ in range(ticks_to_run):
            self.world.tick_simulation()
            
            # Log periodically
            if self.world.tick % 100 == 0:
                stability = calculate_village_stability(self.world)
                self.logger.log_tick(
                    self.world.tick,
                    len(self.world.villagers),
                    self.world.resources.food,
                    stability
                )
            
            # Check for extinction
            if len(self.world.villagers) == 0:
                self.logger.warning("Village extinct!")
                self.world.pending_messages.append("The village has fallen silent...")
    
    def _auto_save(self):
        """Perform auto-save."""
        try:
            filepath = self.save_manager.save(self.world, "autosave.json")
            self.logger.info(f"Auto-saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
    
    def _manual_save(self):
        """Perform manual save."""
        try:
            filepath = self.save_manager.save(self.world)
            self.logger.info(f"Saved to {filepath}")
            self.world.pending_messages.append("Game saved.")
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
    
    def _shutdown(self):
        """Clean shutdown."""
        self.logger.info("Shutting down...")
        
        # Final save
        try:
            self.save_manager.save(self.world, "autosave.json")
            self.logger.info("Final save complete")
        except Exception as e:
            self.logger.error(f"Final save failed: {e}")
        
        # Cleanup
        self.renderer.shutdown()
        self.logger.info("Shutdown complete")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="The Village - A simulation")
    parser.add_argument('--seed', type=int, help='Random seed for new world')
    parser.add_argument('--new', action='store_true', help='Start new world (ignore saves)')
    parser.add_argument('--windowed', action='store_true', help='Run in windowed mode')
    
    args = parser.parse_args()
    
    # Apply windowed mode if requested
    if args.windowed:
        RENDER_CONFIG.fullscreen = False
    
    # Create and run simulation
    sim = VillageSimulation(
        seed=args.seed,
        load_save=not args.new
    )
    
    sim.initialize()
    sim.run()


if __name__ == "__main__":
    main()
