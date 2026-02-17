import pygame

"""
Configuration constants and utility values for the game.
"""

# Screen
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
CAPTION = "Monaco Time Trial"

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 200)
ASPHALT = (40, 40, 40)
BARRIER_COLOR = (200, 200, 0) # Yellowish
GRASS_GREEN = (30, 160, 30)

# Car Physics
CAR_ACCELERATION = 0.2
CAR_FRICTION = 0.05
CAR_BRAKING = 0.3
CAR_TURN_SPEED = 3
CAR_MAX_SPEED = 10
