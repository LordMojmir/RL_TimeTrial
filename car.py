import pygame
import math
from utils import *

class Car:
    """
    Car class representing the player/agent vehicle.
    Handles physics, input, and rendering.
    """
    def __init__(self, x, y):
        """
        Initialize the car.
        
        Args:
            x (float): Initial x position.
            y (float): Initial y position.
        """
        # Physics state
        self.x = x
        self.y = y
        self.angle = 0  # Degrees
        self.speed = 0
        self.velocity_x = 0
        self.velocity_y = 0
        
        # Appearance
        self.width = 60
        self.height = 30
        self.color = RED
        try:
            self.image = pygame.image.load("car.png")
            self.image = pygame.transform.scale(self.image, (self.width, self.height))
        except FileNotFoundError:
            # Fallback if image not found
            print("Warning: car.png not found. Using fallback rendering.")
            self.image = None
        
        # Collision
        self.rect = pygame.Rect(x, y, self.width, self.height)
        
        # Lap timing
        self.lap_start_time = 0
        self.current_lap_time = 0
        self.best_lap_time = float('inf')
        self.on_track = True

    def handle_input(self):
        """
        Handle keyboard input for manual control.
        """
        keys = pygame.key.get_pressed()
        
        # Steering
        if keys[pygame.K_LEFT]:
            self.angle += CAR_TURN_SPEED * (self.speed / CAR_MAX_SPEED) # Turn slower if slow
        if keys[pygame.K_RIGHT]:
            self.angle -= CAR_TURN_SPEED * (self.speed / CAR_MAX_SPEED)

        # Constant Acceleration (match RL env)
        self.speed += CAR_ACCELERATION
        
        # Friction
        if self.speed > 0:
            self.speed -= CAR_FRICTION

        # Cap speed
        self.speed = max(min(self.speed, CAR_MAX_SPEED), 0)

    def update(self):
        """
        Update car physics (velocity, position, bounding box).
        """
        # Calculate velocity vector based on angle and speed
        rad_angle = math.radians(self.angle)
        self.velocity_x = math.cos(rad_angle) * self.speed
        self.velocity_y = -math.sin(rad_angle) * self.speed # Y is flipped in pygame
        
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Update rect for rendering and collision (AABB)
        # Note: AABB is an approximation for rotated rectangles, but sufficient for this specific game
        self.rect.center = (self.x, self.y)

    def draw(self, surface):
        """
        Draw the car on the given surface.
        
        Args:
            surface (pygame.Surface): The surface to draw on.
        """
        if self.image:
            # Rotate the car image
            rotated_surface = pygame.transform.rotate(self.image, self.angle)
            new_rect = rotated_surface.get_rect(center=(self.x, self.y))
            surface.blit(rotated_surface, new_rect.topleft)
        else:
            # Fallback drawing (Red Block)
            # This handles the case if image loading failed
            car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.rect(car_surface, self.color, (0, 0, self.width, self.height))
            # Front indicator
            pygame.draw.polygon(car_surface, BLACK, [(self.width-5, 0), (self.width, self.height/2), (self.width-5, self.height)]) 
            
            rotated_surface = pygame.transform.rotate(car_surface, self.angle)
            new_rect = rotated_surface.get_rect(center=(self.x, self.y))
            surface.blit(rotated_surface, new_rect.topleft)
        
    def cast_rays(self, walls):
        """
        Cast rays to detect walls for RL observation.
        
        Args:
            walls (list of pygame.Rect): List of wall rectangles.
            
        Returns:
            list: Normalized distances (0.0 to 1.0) for each ray.
        """
        # Cast rays in a fan
        # Return distances to walls
        start = pygame.Vector2(self.x, self.y)
        observations = []
        # 5 rays: -60, -30, 0, 30, 60 degrees relative to car angle
        angles = [-60, -30, 0, 30, 60]
        
        max_view_dist = 300
        
        for angle_offset in angles:
            ray_angle = math.radians(self.angle + angle_offset)
            direction = pygame.Vector2(math.cos(ray_angle), -math.sin(ray_angle))
            
            closest_dist = max_view_dist
            end = start + direction * max_view_dist
            
            # Check intersection with each wall rect
            for wall in walls:
                clipped = wall.clipline(start, end)
                if clipped:
                    p1 = pygame.Vector2(clipped[0])
                    p2 = pygame.Vector2(clipped[1])
                    
                    d1 = start.distance_to(p1)
                    d2 = start.distance_to(p2)
                    
                    dist = min(d1, d2)
                    if dist < closest_dist:
                        closest_dist = dist
            
            observations.append(closest_dist / max_view_dist) # Normalize
            
        return observations

    def get_data(self):
        """
        Get normalized state data for RL.
        
        Returns:
            list: [normalized_speed, normalized_angle]
        """
        # For RL later - basic state
        return [self.speed / CAR_MAX_SPEED, self.angle / 360.0]

    def reset(self, x, y):
        """
        Reset car to a specific position.
        
        Args:
            x (float): New x position.
            y (float): New y position.
        """
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.rect.center = (x, y)
