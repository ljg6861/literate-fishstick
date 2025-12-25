"""
Tower Builder AI - Deep Learning Training Loop

This script trains a DQN agent to stack blocks in a physics simulation.
Uses PyTorch for neural network training with experience replay.
Optionally uses MCTS for better action selection (AlphaZero-style).
"""

import pygame
import math
import random
from tower_sim import TowerSimulation
from deep_tower_agent import DeepTowerAgent
from mcts import MCTSAgent


def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tower Builder AI - Deep RL")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    sim = TowerSimulation(width, height)
    base_agent = DeepTowerAgent(screen_width=width)
    
    # Wrap with MCTS for improved decision making
    agent = MCTSAgent(base_agent, num_simulations=30, use_mcts_training=True)
    use_mcts = False  # Toggle with 'M' key
    
    episodes = 0
    max_height_record = 0
    total_rewards = []
    
    # Training batch frequency (higher = faster sim, less frequent training)
    train_every = 8  # Train every N blocks placed
    blocks_placed = 0
    
    # Speed control (default to turbo mode for fast training)
    steps_per_frame = 50
    
    running = True
    
    while running:
        sim.reset()
        current_tower_height = 0
        episode_reward = 0
        
        # Base state variables
        prev_top_x = width // 2
        prev_top_angle = 0.0
        last_place_x = None
        same_x_counter = 0
        camera_y = 0
        
        # Get next block shape for state
        next_shape = sim.next_block['shape'] if sim.next_block else 0
        
        # Initial State (continuous vector)
        state = agent.get_state_vector(prev_top_x, prev_top_angle, width // 2, 
                                        current_tower_height, next_shape)
        
        while not sim.game_over:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    sim.game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT: 
                        steps_per_frame = 50  # Turbo mode
                        use_mcts = False      # Auto-disable MCTS for speed
                        print("Turbo mode (MCTS OFF)")
                    elif event.key == pygame.K_LEFT:
                        steps_per_frame = 1   # Watch mode
                        use_mcts = True       # Auto-enable MCTS for smarter play
                        print("Watch mode (MCTS ON)")
                    elif event.key == pygame.K_s:
                        # Save checkpoint
                        agent.save("tower_agent_checkpoint.pt")
                        print(f"Saved checkpoint at episode {episodes}")
            
            if not running: 
                break
            
            # --- Logic Loop ---
            for _ in range(steps_per_frame):
                if sim.game_over: 
                    break
                    
                # 1. Choose Action (now returns (x_position, rotation) tuple)
                current_block = sim.next_block
                action = agent.choose_action(state, use_mcts=use_mcts)
                place_x, place_rotation = action
                
                # 2. Spawn Block with agent-chosen rotation
                spawn_y = sim.highest_point - 100
                sim.spawn_block(place_x, current_block)
                if sim.blocks:
                    sim.blocks[-1].position = (place_x, spawn_y)
                    sim.blocks[-1].angle = place_rotation  # Apply agent's rotation
                
                # 3. Fast Physics Settle
                max_settle_frames = 100
                tower_fell = False
                
                for settle_i in range(max_settle_frames):
                    sim.step()
                    if sim.game_over:
                        tower_fell = True
                        break
                    
                    # Watch mode rendering (less frequent, no delay)
                    if steps_per_frame == 1 and settle_i % 10 == 0:
                        render_frame(sim, screen, font, agent, episodes, 
                                   max_height_record, current_tower_height, 
                                   state, camera_y, use_mcts)
                        pygame.display.flip()
                    
                    # Early exit if settled
                    if sim.blocks:
                        last_block = sim.blocks[-1]
                        if (last_block.velocity.length < 0.5 and 
                            abs(last_block.angular_velocity) < 0.1):
                            break
                
                # 4. Calculate Reward
                if tower_fell:
                    reward = -100.0
                    next_shape = 0
                    next_state = agent.get_state_vector(None, 0, width // 2, 0, next_shape)
                    agent.store_transition(state, action, reward, next_state, True)
                    episode_reward += reward
                else:
                    # Reward shaping
                    r_survival = 5.0
                    
                    # Alignment reward
                    dist_from_center = abs(place_x - prev_top_x)
                    r_alignment = (50 - dist_from_center) / 10.0
                    
                    # Stability reward
                    top_body = sim.blocks[-1]
                    current_angle = top_body.angle
                    r_stability = (1.0 - abs(math.sin(current_angle))) * 5
                    
                    # Height bonus (encourage going higher)
                    r_height = min(current_tower_height / 10.0, 5.0)
                    
                    # Stacking penalty
                    if place_x == last_place_x:
                        same_x_counter += 1
                    else:
                        same_x_counter = 1
                        last_place_x = place_x
                    
                    r_stacking = -10 if same_x_counter > 3 else 0
                    
                    reward = r_survival + r_alignment + r_stability + r_height + r_stacking
                    
                    # Update state
                    new_top_x = top_body.position.x
                    new_top_angle = top_body.angle
                    next_shape = sim.next_block['shape'] if sim.next_block else 0
                    
                    next_state = agent.get_state_vector(new_top_x, new_top_angle, 
                                                        width // 2, 
                                                        current_tower_height + 1, 
                                                        next_shape)
                    
                    agent.store_transition(state, action, reward, next_state, False)
                    
                    state = next_state
                    prev_top_x = new_top_x
                    prev_top_angle = new_top_angle
                    current_tower_height += 1
                    episode_reward += reward
                
                # 5. Training step
                blocks_placed += 1
                if blocks_placed % train_every == 0:
                    loss = agent.train_step()
            
            # Update camera
            top_in_screen = sim.highest_point + camera_y
            if top_in_screen < 100:
                camera_y = -(sim.highest_point - 200)
            if camera_y < 0: 
                camera_y = 0
            
            # Render
            render_frame(sim, screen, font, agent, episodes, max_height_record,
                        current_tower_height, state, camera_y, use_mcts)
            pygame.display.flip()
            clock.tick(60)
        
        # Episode end
        episodes += 1
        agent.decay_epsilon()
        total_rewards.append(episode_reward)
        
        if current_tower_height > max_height_record:
            max_height_record = current_tower_height
            # Auto-save on new record
            agent.save("tower_agent_best.pt")
        
        # Print progress every 10 episodes
        if episodes % 10 == 0:
            stats = agent.get_stats()
            avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
            print(f"Ep {episodes} | Height: {current_tower_height} | "
                  f"Record: {max_height_record} | ε: {stats['epsilon']:.3f} | "
                  f"Loss: {stats['avg_loss']:.4f} | Avg Reward: {avg_reward:.1f}")


def render_frame(sim, screen, font, agent, episodes, max_height_record,
                 current_tower_height, state, camera_y, use_mcts=False):
    """Render a single frame with HUD."""
    sim.render(screen, scroll_y=camera_y)
    
    stats = agent.get_stats()
    mcts_status = "MCTS: ON" if use_mcts else "MCTS: OFF"
    
    # HUD
    hud_lines = [
        f"Episode: {episodes} | Record: {max_height_record}",
        f"Height: {current_tower_height} | {mcts_status}",
        f"ε: {stats['epsilon']:.3f} | Buffer: {stats['buffer_size']}",
        f"Loss: {stats['avg_loss']:.4f}",
        f"[←] Watch | [→] Turbo | [M] MCTS | [S] Save"
    ]
    
    for i, txt in enumerate(hud_lines):
        text_surface = font.render(txt, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10 + i * 18))


if __name__ == "__main__":
    main()