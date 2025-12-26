
import pygame
from .config import MuZeroConfig
from learning.agent import GeneralAgent
from .env import TowerEnv

def run_tower(screen=None):
    standalone = False
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        standalone = True

    config = MuZeroConfig()
    env = TowerEnv(config)
    agent = GeneralAgent(config)
    
    # Try loading existing model
    try:
        agent.load("generic_agent.pt")
        print("Loaded existing agent.")
    except:
        print("Starting fresh agent.")
        
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    running = True
    episodes = 0
    turbo_mode = True
    
    while running:
        # Reset Env
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT: 
                        turbo_mode = True
                    elif event.key == pygame.K_LEFT:
                        turbo_mode = False
                    elif event.key == pygame.K_s:
                        agent.save("generic_agent.pt")
                        print("Saved.")
                    elif event.key == pygame.K_ESCAPE and not standalone:
                        return

            if not running: break
            
            # 1. Agent Act
            action = agent.act(obs)
            
            # 2. Env Step
            next_obs, reward, done = env.step(action)
            episode_reward += reward
            
            # 3. Agent Store & Train
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            
            obs = next_obs
            
            # Render
            if not turbo_mode:
                screen.fill((30, 30, 30))
                env.render(screen)
                
                # HUD
                lines = [
                    f"Ep: {episodes} | H: {env.current_tower_height} | Max: {env.max_height_record}",
                    f"Epsilon: {agent.epsilon:.3f}",
                    f"Loss: {loss if loss else 0.0:.4f}",
                    "[ESC] Return" if not standalone else ""
                ]
                for i, txt in enumerate(lines):
                    s = font.render(txt, True, (255, 255, 255))
                    screen.blit(s, (10, 10 + i*20))
                    
                pygame.display.flip()
                clock.tick(60) # Watch mode speed
            else:
                # Fast render (every few frames? or just blank?)
                # To keep window responsive we pump events, but maybe dont draw
                # Let's draw minimal for now to see it working
                if episodes % 5 == 0:
                     pygame.event.pump()
                     screen.fill((30, 30, 30))
                     env.render(screen)
                     lines = [f"TURBO MODE | Ep: {episodes} | Max: {env.max_height_record}"]
                     s = font.render(lines[0], True, (255, 255, 0))
                     screen.blit(s, (10, 10))
                     pygame.display.flip()

        episodes += 1
        if episodes % 10 == 0:
            print(f"Ep {episodes} | Reward: {episode_reward:.1f} | Max Height: {env.max_height_record}")

    if standalone:
        pygame.quit()

if __name__ == "__main__":
    run_tower()
