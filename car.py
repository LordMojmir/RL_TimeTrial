import pygame
import math
from utils import *

class Car:
    def __init__(self, x, y):
        # Physics state
        self.x = x
        self.y = y
        self.angle = 0  # Degrees
        self.speed = 0
        self.velocity_x = 0
        self.velocity_y = 0
        
        # Appearance
        self.width = 40
        self.height = 20
        self.color = RED
        
        # Collision
        self.rect = pygame.Rect(x, y, self.width, self.height)
        
        # Lap timing
        self.lap_start_time = 0
        self.current_lap_time = 0
        self.best_lap_time = float('inf')
        self.on_track = True

    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        # Steering
        if keys[pygame.K_LEFT]:
            self.angle += CAR_TURN_SPEED * (self.speed / CAR_MAX_SPEED) # Turn slower if slow
        if keys[pygame.K_RIGHT]:
            self.angle -= CAR_TURN_SPEED * (self.speed / CAR_MAX_SPEED)

        # Acceleration / Braking
        if keys[pygame.K_UP]:
            self.speed += CAR_ACCELERATION
        elif keys[pygame.K_DOWN]:
            self.speed -= CAR_BRAKING
        else:
            # Friction
            if self.speed > 0:
                self.speed -= CAR_FRICTION
            elif self.speed < 0:
                self.speed += CAR_FRICTION
            if abs(self.speed) < CAR_FRICTION:
                self.speed = 0

        # Cap speed
        self.speed = max(min(self.speed, CAR_MAX_SPEED), -CAR_MAX_SPEED/2)

    def update(self):
        # Calculate velocity vector based on angle and speed
        rad_angle = math.radians(self.angle)
        self.velocity_x = math.cos(rad_angle) * self.speed
        self.velocity_y = -math.sin(rad_angle) * self.speed # Y is flipped in pygame
        
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Update rect for rendering and collision (AABB)
        # Note: To be more precise we should rotate the rect, but for V0 AABB is acceptable
        # or we update the center
        self.rect.center = (self.x, self.y)

    def draw(self, surface):
        # Rotate the car image (or rect)
        # Since we are drawing primitives, we create a surface, draw rect, rotate surface
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, self.color, (0, 0, self.width, self.height))
        pygame.draw.polygon(car_surface, BLACK, [(self.width-5, 0), (self.width, self.height/2), (self.width-5, self.height)]) # Front indicator
        
        rotated_surface = pygame.transform.rotate(car_surface, self.angle)
        new_rect = rotated_surface.get_rect(center=(self.x, self.y))
        
        surface.blit(rotated_surface, new_rect.topleft)
        
        # Update self.rect to match the visible rotated bounding box for collision interactions
        self.rect = new_rect

    def get_data(self):
        # For RL later
        return [self.x, self.y, self.speed, self.angle]
    
    def reset(self, x, y):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.rect.center = (x, y)
