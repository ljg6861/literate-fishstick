
import random
import pygame

class SnakeSim:
    def __init__(self, grid_w=20, grid_h=15, cell_size=40):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        
        self.reset()
        
    def reset(self):
        # Snake starts in middle
        self.head = [self.grid_w // 2, self.grid_h // 2]
        self.body = [list(self.head), [self.head[0], self.head[1]+1], [self.head[0], self.head[1]+2]]
        self.direction = 0 # 0: Up, 1: Down, 2: Left, 3: Right
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self._spawn_food()
        
    def _spawn_food(self):
        while True:
            self.food = [random.randint(0, self.grid_w-1), random.randint(0, self.grid_h-1)]
            if self.food not in self.body:
                break
                
    def step(self, action):
        """
        Action: 0: Up, 1: Down, 2: Left, 3: Right
        Returns: (reward, done)
        """
        if self.game_over: return 0, True
        
        # Prevent 180 turn
        if (action == 0 and self.direction == 1) or \
           (action == 1 and self.direction == 0) or \
           (action == 2 and self.direction == 3) or \
           (action == 3 and self.direction == 2):
            action = self.direction # Ignore invalid turn
            
        self.direction = action
        
        # Move Head
        dx, dy = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1
        
        new_head = [self.head[0] + dx, self.head[1] + dy]
        
        # Check Collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_w or 
            new_head[1] < 0 or new_head[1] >= self.grid_h or 
            new_head in self.body[:-1]): # Ignore tail for self-collision (it moves)
            
            self.game_over = True
            return -100, True # Death penalty
            
        self.head = new_head
        self.body.insert(0, new_head)
        
        # Check Food
        reward = 0
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._spawn_food()
        else:
            self.body.pop() # Remove tail
            
            # Survival/Distance reward shaping could go here
            # But simple is better for now:
            reward = -0.1 # Step penalty to encourage speed
            
        self.steps += 1
        if self.steps > 100 * (len(self.body) - 2): # Starvation limit
            self.game_over = True
            return -50, True
            
        return reward, False

    def render(self, screen):
        # Draw Checkerboard bg
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                rect = (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                color = (40, 40, 40) if (x+y)%2==0 else (50, 50, 50)
                pygame.draw.rect(screen, color, rect)
                
        # Draw Apple
        fx, fy = self.food
        f_rect = (fx*self.cell_size+5, fy*self.cell_size+5, self.cell_size-10, self.cell_size-10)
        pygame.draw.rect(screen, (200, 50, 50), f_rect)
        
        # Draw Snake
        for part in self.body:
            px, py = part
            p_rect = (px*self.cell_size, py*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, (50, 200, 100), p_rect)
            pygame.draw.rect(screen, (30, 150, 80), p_rect, 2) # Border
