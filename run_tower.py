import pygame
import time
from tower_sim import TowerSimulation
from tower_agent import TowerAgent

def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tower Builder AI")
    clock = pygame.time.Clock()
    
    # Visualization settings
    camera_y = 0
    target_camera_y = 0
    font = pygame.font.SysFont("Arial", 20)
    
    sim = TowerSimulation(width, height)
    agent = TowerAgent()
    
    episodes = 0
    max_height_record = 0
    
    # Speed control
    steps_per_frame = 1  # 1 = Normal speed
    
    running = True
    
    # Training Loop
    while running:
        # 1. Reset
        sim.reset()
        current_tower_height = 0
        
        # Base state
        prev_top_x = width // 2
        state = agent.get_state(prev_top_x, width // 2)
        
        # Episode Loop
        while not sim.game_over:
            # Input handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    sim.game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        steps_per_frame = min(100, steps_per_frame + 1)
                    elif event.key == pygame.K_DOWN:
                        steps_per_frame = max(1, steps_per_frame - 1)
                    elif event.key == pygame.K_RIGHT: 
                        steps_per_frame = 50 # Fast forward preset
                    elif event.key == pygame.K_LEFT:
                        steps_per_frame = 1 # Normal preset
            
            if not running:
                break
            
            # FAST TRAINING LOOP
            # We run multiple logic steps per render frame
            for _ in range(steps_per_frame):
                if sim.game_over:
                    break
                    
                # 2. Agent chooses action
                action_offset = agent.choose_action(state)
                place_x = prev_top_x + action_offset
                
                # Clamp
                place_x = max(50, min(width-50, place_x))
                
                # 3. Spawn
                sim.spawn_block(place_x)
                
                # 4. Simulate physics delay
                # Instead of rendering every step, we just step physics high-speed
                settle_steps = 60
                tower_fell = False
                
                for _ in range(settle_steps):
                    sim.step()
                    if sim.game_over:
                        tower_fell = True
                        break
                
                # 5. Measure Reward
                if tower_fell:
                    reward = -100
                    next_state = "game_over"
                    agent.learn(state, action_offset, reward, next_state)
                else:
                    reward = 10 + (current_tower_height * 0.5)
                    if sim.blocks:
                        top_body = sim.blocks[-1]
                        new_top_x = top_body.position.x
                        next_state = agent.get_state(new_top_x, width // 2)
                        agent.learn(state, action_offset, reward, next_state)
                        state = next_state
                        prev_top_x = new_top_x
                        current_tower_height += 1
            
            # --- RENDERING (Once per frame) ---
            
            # Update camera target to follow top of tower
            # Keep top of tower ~200px from top of screen
            # highest_point is y-coord (lower is higher)
            desired_y = -(sim.highest_point - 400)
            if desired_y < 0: desired_y = 0
            
            # Smooth camera
            target_camera_y = desired_y
            camera_y += (target_camera_y - camera_y) * 0.1
            
            sim.render(screen, scroll_y=camera_y)
            
            # Draw HUD
            # Background panel for text
            start_y = 10
            line_height = 25
            
            stats = [
                f"Episode: {episodes}",
                f"Tower Height: {(height - sim.highest_point)/50:.1f} blocks", 
                f"Max Record: {max_height_record}",
                f"Confidence: {agent.get_confidence(state):.2f}",
                f"Epsilon: {agent.epsilon:.2f}",
                f"Speed: {steps_per_frame}x (Arrow Keys)",
            ]
            
            for i, line in enumerate(stats):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (10, start_y + i * line_height))
                
            pygame.display.flip()
            clock.tick(60)
        
        # End of Episode
        episodes += 1
        agent.decay_epsilon()
        
        if current_tower_height > max_height_record:
            max_height_record = current_tower_height


if __name__ == "__main__":
    main()
