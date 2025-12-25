"""
The Village - Building Sprites
Visual representation of buildings with stacking.
Phase 2: Hospital, Research Center icons.
"""

import pygame
import math
from config import RENDER_CONFIG
from simulation.buildings import BuildingType


class BuildingRenderer:
    """Renders buildings as stacked icons with counts."""
    
    def __init__(self):
        self.config = RENDER_CONFIG
        pygame.font.init()
        self.count_font = pygame.font.SysFont('Consolas', 14, bold=True)
    
    def get_building_shape(self, building_type: BuildingType) -> str:
        """Get shape type for building."""
        shapes = {
            BuildingType.HOUSING: "house",
            BuildingType.FARM: "rectangle",
            BuildingType.WORKSHOP: "hexagon",
            BuildingType.SCHOOL: "triangle",
            BuildingType.HOSPITAL: "cross",
            BuildingType.RESEARCH_CENTER: "diamond",
            BuildingType.ADVANCED_FARM: "rectangle",
            BuildingType.EXPANDED_HOUSING: "house",
        }
        return shapes.get(building_type, "square")
    
    def draw_stacked_buildings(self, screen: pygame.Surface, buildings: list, 
                                camera_offset: tuple = (0, 0)):
        """Draw buildings grouped by type with counts."""
        from collections import defaultdict
        
        complete_by_type = defaultdict(list)
        incomplete = []
        
        for building in buildings:
            if building.is_complete:
                complete_by_type[building.building_type].append(building)
            else:
                incomplete.append(building)
        
        # Draw consolidated complete buildings in a row at top
        x_offset = 250
        y_pos = 30
        spacing = 80
        
        for building_type, building_list in complete_by_type.items():
            if building_list:
                count = len(building_list)
                self._draw_building_icon(screen, x_offset, y_pos, building_type, count)
                x_offset += spacing
        
        # Draw incomplete buildings at their actual positions
        for building in incomplete:
            self._draw_construction(screen, building, camera_offset)
    
    def _draw_building_icon(self, screen: pygame.Surface, x: float, y: float,
                            building_type: BuildingType, count: int):
        """Draw a building icon with count."""
        size = 35
        color = self.config.color_foreground
        
        shape = self.get_building_shape(building_type)
        
        if shape == "house":
            # House shape
            base_rect = pygame.Rect(int(x) - size//2, int(y), size, size - 10)
            pygame.draw.rect(screen, color, base_rect, 2)
            roof_points = [
                (int(x) - size//2 - 5, int(y)),
                (int(x), int(y) - 15),
                (int(x) + size//2 + 5, int(y))
            ]
            pygame.draw.polygon(screen, color, roof_points, 2)
            
        elif shape == "rectangle":
            # Farm
            rect = pygame.Rect(int(x) - size//2 - 5, int(y) - 5, size + 10, size - 15)
            pygame.draw.rect(screen, color, rect, 2)
            for i in range(3):
                line_x = int(x) - size//2 + 5 + i * 12
                pygame.draw.line(screen, color, (line_x, int(y) - 2), (line_x, int(y) + size - 18), 1)
                
        elif shape == "hexagon":
            # Workshop
            points = []
            for i in range(6):
                angle = math.pi / 3 * i - math.pi / 6
                px = x + (size//2) * math.cos(angle)
                py = y + 5 + (size//2) * math.sin(angle)
                points.append((int(px), int(py)))
            pygame.draw.polygon(screen, color, points, 2)
            
        elif shape == "triangle":
            # School
            points = [
                (int(x), int(y) - 10),
                (int(x) - size//2, int(y) + size - 15),
                (int(x) + size//2, int(y) + size - 15)
            ]
            pygame.draw.polygon(screen, color, points, 2)
            
        elif shape == "cross":
            # Hospital - cross shape
            arm_width = size // 3
            # Vertical arm
            pygame.draw.rect(screen, color, 
                           pygame.Rect(int(x) - arm_width//2, int(y) - 5, arm_width, size - 5), 2)
            # Horizontal arm
            pygame.draw.rect(screen, color,
                           pygame.Rect(int(x) - size//2, int(y) + 5, size, arm_width), 2)
                           
        elif shape == "diamond":
            # Research Center
            points = [
                (int(x), int(y) - size//2 + 5),
                (int(x) + size//2 - 5, int(y) + 5),
                (int(x), int(y) + size//2 + 5),
                (int(x) - size//2 + 5, int(y) + 5)
            ]
            pygame.draw.polygon(screen, color, points, 2)
        
        # Draw count
        if count > 1:
            count_text = f"x{count}"
            count_surface = self.count_font.render(count_text, True, self.config.color_accent)
            count_x = int(x) - count_surface.get_width() // 2
            count_y = int(y) + size - 5
            screen.blit(count_surface, (count_x, count_y))
        
        # Draw building type label
        label = building_type.name.replace('_', ' ').title()
        # Shorten long names
        if len(label) > 10:
            label = label[:8] + ".."
        label_surface = self.count_font.render(label, True, self.config.color_dim)
        label_x = int(x) - label_surface.get_width() // 2
        label_y = int(y) + size + 8
        screen.blit(label_surface, (label_x, label_y))
    
    def _draw_construction(self, screen: pygame.Surface, building, camera_offset: tuple):
        """Draw a building under construction at its world position."""
        x = building.position[0] - camera_offset[0]
        y = building.position[1] - camera_offset[1]
        
        size = 50  # Larger for better visibility
        progress = building.progress_percent / 100.0
        
        # Draw scaffolding/hollow box
        rect = pygame.Rect(int(x) - size//2, int(y) - size//2, size, size)
        pygame.draw.rect(screen, (100, 100, 100), rect, 2)
        
        # Draw filling progress from bottom
        fill_height = int(size * progress)
        if fill_height > 0:
            fill_rect = pygame.Rect(int(x) - size//2, int(y) + size//2 - fill_height, 
                                    size, fill_height)
            pygame.draw.rect(screen, self.config.color_accent, fill_rect)
        
        # Draw glow if progress is happening
        if progress > 0 and progress < 1.0:
            glow_size = size + 4 + int(2 * math.sin(pygame.time.get_ticks() * 0.01))
            glow_rect = pygame.Rect(int(x) - glow_size//2, int(y) - glow_size//2, glow_size, glow_size)
            pygame.draw.rect(screen, self.config.color_accent, glow_rect, 1)
        
        progress_text = f"{int(building.progress_percent)}%"
        text_surface = self.count_font.render(progress_text, True, self.config.color_foreground)
        screen.blit(text_surface, (int(x) - text_surface.get_width()//2, int(y) + size//2 + 5))
        
        # Label what it is
        label = building.building_type.name.replace('_', ' ').title()
        label_surf = self.count_font.render(label, True, self.config.color_dim)
        screen.blit(label_surf, (int(x) - label_surf.get_width()//2, int(y) - size//2 - 20))

    def draw_resource_nodes(self, screen: pygame.Surface, nodes: list, camera_offset: tuple = (0, 0)):
        """Draw resource nodes (food, materials) on the map."""
        for node in nodes:
            x = node.position[0] - camera_offset[0]
            y = node.position[1] - camera_offset[1]
            
            color = (0, 255, 100) if node.node_type == 'FOOD' else (200, 200, 200)
            size = 15
            
            # Simple circle icon
            pygame.draw.circle(screen, color, (int(x), int(y)), size, 1)
            pygame.draw.circle(screen, color, (int(x), int(y)), 4)
            
            # Label
            label = "Food" if node.node_type == 'FOOD' else "Materials"
            label_surf = self.count_font.render(label, True, self.config.color_dim)
            screen.blit(label_surf, (int(x) - label_surf.get_width()//2, int(y) + size + 2))
    
    def draw_all(self, screen: pygame.Surface, buildings: list, camera_offset: tuple = (0, 0)):
        """Draw all buildings using stacked display."""
        self.draw_stacked_buildings(screen, buildings, camera_offset)
