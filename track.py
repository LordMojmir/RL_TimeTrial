import pygame
from utils import SCREEN_WIDTH, SCREEN_HEIGHT, ASPHALT, BARRIER_COLOR, WHITE, GRASS_GREEN

class Track:
    def __init__(self):
        self.walls = []
        self.checkpoints = [] # For valid lap detection (optional for v0)
        self.start_line = None
        self._create_track()

    def _create_track(self):
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
        surface.fill(ASPHALT)
        
        # Draw Walls (Grass)
        for wall in self.walls:
            pygame.draw.rect(surface, GRASS_GREEN, wall)
            pygame.draw.rect(surface, (0, 50, 0), wall, 2) # Darker green outline

        # Draw Start Line
        pygame.draw.rect(surface, WHITE, self.start_line)
    
    def check_collision(self, car_rect):
        # collided_vals = car_rect.collidelistall(self.walls)
        # return len(collided_vals) > 0
        
        # More precise: check if any wall collides with the car rect
        # Note: car_rect here should be the passed AABB from the car
        index = car_rect.collidelist(self.walls)
        return index != -1
