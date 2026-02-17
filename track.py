import pygame
from utils import SCREEN_WIDTH, SCREEN_HEIGHT, ASPHALT, BARRIER_COLOR, WHITE, GRASS_GREEN

class Track:
    """
    Track class representing the race track walls and checkpoints.
    """
    def __init__(self):
        """
        Initialize the track.
        """
        self.walls = []
        self.checkpoints = [] # For valid lap detection (optional for v0)
        self.start_line = None
        self._create_track()

    def _create_track(self):
        """
        Define the track geometry (walls, start line).
        """
        # Create a simple loop track
        # Outer boundaries
        outer_margin = 50
        self.walls.append(pygame.Rect(0, 0, SCREEN_WIDTH, outer_margin)) # Top
        self.walls.append(pygame.Rect(0, SCREEN_HEIGHT - outer_margin, SCREEN_WIDTH, outer_margin)) # Bottom
        self.walls.append(pygame.Rect(0, 0, outer_margin, SCREEN_HEIGHT)) # Left
        self.walls.append(pygame.Rect(SCREEN_WIDTH - outer_margin, 0, outer_margin, SCREEN_HEIGHT)) # Right

        # Inner block (island)
        inner_margin = 250
        island_rect = pygame.Rect(
            inner_margin, 
            inner_margin, 
            SCREEN_WIDTH - 2 * inner_margin, 
            SCREEN_HEIGHT - 2 * inner_margin
        )
        self.walls.append(island_rect)

        # Start Line (Bottom straight, right side)
        # x, y, w, h
        self.start_line = pygame.Rect(
            SCREEN_WIDTH / 2, 
            SCREEN_HEIGHT - inner_margin, 
            10, 
            inner_margin - outer_margin
        )

    def draw(self, surface):
        """
        Draw the track walls and surface.
        
        Args:
            surface (pygame.Surface): The surface to draw on.
        """
        surface.fill(ASPHALT)
        
        # Draw Walls (Grass)
        for wall in self.walls:
            pygame.draw.rect(surface, GRASS_GREEN, wall)
            pygame.draw.rect(surface, (0, 50, 0), wall, 2) # Darker green outline

        # Draw Start Line
        pygame.draw.rect(surface, WHITE, self.start_line)
    
    def check_collision(self, car_rect):
        """
        Check if the car has collided with any wall.
        
        Args:
            car_rect (pygame.Rect): The car's bounding box.
            
        Returns:
            bool: True if collision detected, False otherwise.
        """
        index = car_rect.collidelist(self.walls)
        return index != -1
