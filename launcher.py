
import pygame
import sys

# Game Imports
import tower_game.main
import snake_game.main

def draw_button(screen, rect, text, font, color=(50, 50, 50), hover_color=(70, 70, 70)):
    mouse_pos = pygame.mouse.get_pos()
    is_hover = rect.collidepoint(mouse_pos)
    
    pygame.draw.rect(screen, hover_color if is_hover else color, rect)
    pygame.draw.rect(screen, (200, 200, 200), rect, 2)
    
    txt_surf = font.render(text, True, (255, 255, 255))
    txt_rect = txt_surf.get_rect(center=rect.center)
    screen.blit(txt_surf, txt_rect)
    
    return is_hover

def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("AI Game Launcher")
    
    font_title = pygame.font.SysFont("Arial", 48, bold=True)
    font_btn = pygame.font.SysFont("Arial", 24)
    
    running = True
    while running:
        screen.fill((20, 20, 30))
        
        # Title
        title = font_title.render("AI Game Launcher", True, (0, 255, 100))
        screen.blit(title, title.get_rect(center=(width//2, 100)))
        
        # Buttons
        btn_width, btn_height = 300, 60
        tower_rect = pygame.Rect((width-btn_width)//2, 250, btn_width, btn_height)
        snake_rect = pygame.Rect((width-btn_width)//2, 350, btn_width, btn_height)
        quit_rect = pygame.Rect((width-btn_width)//2, 500, btn_width, btn_height)
        
        hover_tower = draw_button(screen, tower_rect, "Tower Builder", font_btn, color=(40, 60, 100), hover_color=(60, 80, 140))
        hover_snake = draw_button(screen, snake_rect, "Snake AI", font_btn, color=(40, 100, 40), hover_color=(60, 140, 60))
        hover_quit = draw_button(screen, quit_rect, "Quit", font_btn, color=(100, 40, 40), hover_color=(140, 60, 60))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if hover_tower:
                        print("Allocating Tower Game...")
                        tower_game.main.run_tower(screen)
                        pygame.display.set_caption("AI Game Launcher") # Restore caption
                    elif hover_snake:
                        print("Allocating Snake Game...")
                        snake_game.main.run_snake(screen)
                        pygame.display.set_caption("AI Game Launcher")
                    elif hover_quit:
                        running = False
                        
        pygame.display.flip()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
