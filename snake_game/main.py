
import pygame
from .config import SnakeConfig
from learning.agent import GeneralAgent
from .env import SnakeEnv

def run_snake(screen=None):
    # Standalone setup if no screen provided
    standalone = False
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        standalone = True
        
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    config = SnakeConfig()
    env = SnakeEnv(config)
    agent = GeneralAgent(config)
    
    try:
        agent.load("snake_agent.pt")
        print("Loaded Snake agent.")
    except:
        print("Starting fresh Snake agent.")
        
    running = True
    episodes = 0
    turbo_mode = True
    
    while running:
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT: turbo_mode = True
                    elif event.key == pygame.K_LEFT: turbo_mode = False
                    elif event.key == pygame.K_s:
                        agent.save("snake_agent.pt")
                        print("Saved Snake Agent.")
                    elif event.key == pygame.K_ESCAPE and not standalone:
                        # Return to launcher
                        return
                        
            if not running: break
            
            # AI
            action = agent.act(obs)
            next_obs, reward, done = env.step(action)
            
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            
            obs = next_obs
            total_reward += reward
            
            # Render
            if not turbo_mode:
                screen.fill((20, 20, 20))
                env.render(screen)
                
                # HUD
                hud = [
                   f"Ep: {episodes} | Score: {env.sim.score} | Max: {env.max_score_record}",
                   f"Eps: {agent.epsilon:.3f} | Loss: {loss if loss else 0:.4f}",
                   "[ESC] Return" if not standalone else ""
                ]
                for i, txt in enumerate(hud):
                    s = font.render(txt, True, (255, 255, 255))
                    screen.blit(s, (10, 10 + i*20))
                    
                pygame.display.flip()
                clock.tick(15) # Slower for snake
            else:
                 if episodes % 10 == 0:
                     pygame.event.pump()
                     screen.fill((20, 20, 20))
                     env.render(screen)
                     s = font.render(f"TURBO | Ep: {episodes}", True, (255, 200, 0))
                     screen.blit(s, (10, 10))
                     pygame.display.flip()
                     
        episodes += 1
        if episodes % 20 == 0:
            print(f"Snake Ep {episodes} | Rew: {total_reward:.1f} | Score: {env.sim.score}")
            
    if standalone:
        pygame.quit()

if __name__ == "__main__":
    run_snake()
