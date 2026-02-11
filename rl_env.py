import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game import Game
from utils import *

class CarRacingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        self.game = Game(headless=(render_mode != "human"))
        self.game.state = "PLAYING"
        # We need to decouple the game loop for Gym
        # ideally Game should act as the world state holder
        
        self.render_mode = render_mode
        
        # Action Space: [Steering]
        # Steering: -1 (Left) to 1 (Right)
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        
        # Observation Space: 
        # 5 Raycasts (normalized 0-1)
        # Speed (normalized)
        # Angle (normalized? maybe not needed if raycasts are relative, but good to have)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        
        # Initial observation
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Action is [steering]
        steering = float(action[0])
        accel = 1.0 # Always max acceleration
        
        car = self.game.car
        
        # Apply physics manually instead of keyboard input
        # Steering
        if abs(steering) > 0.05:
             car.angle -= steering * CAR_TURN_SPEED * (car.speed / CAR_MAX_SPEED)

        # Constant Acceleration
        car.speed += CAR_ACCELERATION * accel
        
        # Friction (Air resistance)
        if car.speed > 0:
            car.speed -= CAR_FRICTION
        elif car.speed < 0:
            car.speed += CAR_FRICTION
        if abs(car.speed) < CAR_FRICTION:
            car.speed = 0
                
        # Cap speed (No negative speed allowed in RL)
        car.speed = max(min(car.speed, CAR_MAX_SPEED), 0)
        
        # Update Physics
        car.update()
        
        # Check collisions / Termination
        terminated = False
        truncated = False
        reward = -0.1 # Base Time penalty
        info = {}
        
        if self.game.track.check_collision(car.rect):
            terminated = True
            reward = -500 # Strong penalty for crashing
        
        # Check Lap Progress / Reward
        if not terminated:
            # 1. Reward speed (Distance driven proxy)
            # Reduced to 0.1 to prioritize survival
            reward += (car.speed / CAR_MAX_SPEED) * 0.1
            
            # 2. Wall Proximity Penalty
            # we need to cast rays to know how close we are
            rays = car.cast_rays(self.game.track.walls)
            min_dist = min(rays)
            
            # Rays are 0-1. 0 is hit, 1 is max_view
            if min_dist < 0.2:
                reward -= 0.5
            if min_dist < 0.1:
                reward -= 1.0 
                
            # 3. Lap Line
            if car.rect.colliderect(self.game.track.start_line):
                if not self.game.crossed_start_line:
                    self.game.crossed_start_line = True
                    current_time = pygame.time.get_ticks() 
                    
                    # Only reward if we have started a lap previously
                    # Basic check: if lap time > 3s (very fast lap, but avoids instant trigger)
                    if self.game.lap_start_time != 0:
                        lap_time = (current_time - self.game.lap_start_time) / 1000.0
                        if lap_time > 3.0: 
                            reward += 1000
                            # Reset for next lap
                            self.game.lap_start_time = current_time
                            info['lap_time'] = lap_time
                    else:
                         # First crossing
                         self.game.lap_start_time = current_time
                         reward += 10
            else:
                self.game.crossed_start_line = False
        
        # Get obs for next step
        observation = self._get_obs()
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        rays = self.game.car.cast_rays(self.game.track.walls)
        car_data = self.game.car.get_data() # [speed_norm, angle_norm]
        
        obs = np.array(rays + car_data, dtype=np.float32)
        return obs

    def render(self):
        self.game.draw()
        
    def close(self):
        pygame.quit()
