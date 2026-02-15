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
        self.width = 60
        self.height = 30
        self.color = RED
        self.image = pygame.image.load("car.png")
        self.image = pygame.transform.scale(self.image, (self.width, self.height))
        
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

        # Constant Acceleration (match RL env)
        self.speed += CAR_ACCELERATION
        
        # Friction
        if self.speed > 0:
            self.speed -= CAR_FRICTION

        # Cap speed
        self.speed = max(min(self.speed, CAR_MAX_SPEED), 0)

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
        rotated_surface = pygame.transform.rotate(self.image, self.angle)
        new_rect = rotated_surface.get_rect(center=(self.x, self.y))
        
        surface.blit(rotated_surface, new_rect.topleft)
        
        # Update self.rect to match the visible rotated bounding box for collision interactions
        self.rect = new_rect


    def cast_rays(self, walls):
        # Cast rays in a fan
        # Return distances to walls
        start = pygame.Vector2(self.x, self.y)
        observations = []
        # 5 rays: -60, -30, 0, 30, 60 degrees relative to car angle
        angles = [-60, -30, 0, 30, 60]
        
        max_view_dist = 300
        
        for angle_offset in angles:
            ray_angle = math.radians(self.angle + angle_offset)
            # Ray direction
            # Note: y is inverted in pygame (down is positive), so sin is negative for "up"
            direction = pygame.Vector2(math.cos(ray_angle), -math.sin(ray_angle))
            
            # Simple raymarching or line intersection
            # For simplicity in V0, let's do a coarse step check or line intersection with walls
            # Line intersection is better
            
            closest_dist = max_view_dist
            
            end = start + direction * max_view_dist
            
            # Check intersection with each wall rect
            # We treat walls as 4 lines
            for wall in walls:
                # Expand wall to lines? Or use clipline
                # rect.clipline returns the segment of the line inside the rect
                # If there is a segment, we take the distance to the start of the segment
                
                # However, walls are just rects. 
                # If we are inside the track (island), the walls are the boundaries. 
                # If we are outside, we are crashing.
                
                # Check if the line start->end intersects the wall rect
                clipped = wall.clipline(start, end)
                if clipped:
                    # clipped is ((x1, y1), (x2, y2))
                    # We want the point closest to start
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
        # For RL later - basic state
        return [self.speed / CAR_MAX_SPEED, self.angle / 360.0]

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.rect.center = (x, y)
