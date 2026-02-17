import pygame
import time
from utils import *
from car import Car
from track import Track

class Game:
    """
    Main Game Class.
    Handles the game loop, events, updates, and rendering.
    """
    def __init__(self, headless=False):
        pygame.init()
        self.headless = headless
        if not headless:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption(CAPTION)
            self.font = pygame.font.SysFont("Verdana", 24)
        else:
            self.screen = None
            
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.track = Track()
        
        # Start position: Middle of bottom straight
        start_x = SCREEN_WIDTH / 2 - 100
        start_y = SCREEN_HEIGHT - 150
        self.car = Car(start_x, start_y)
        
        self.lap_start_time = 0
        self.best_lap = float('inf')
        self.last_lap_time = 0
        
        # Game State
        self.state = "MENU" # MENU or PLAYING
        
        # Lap logic state
        self.crossed_start_line = False 

    def handle_input(self):
        """
        Handle user input events (Keyboard).
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if self.state == "MENU":
                    if event.key == pygame.K_RETURN:
                        self.state = "PLAYING"
                        self.reset()
                    if event.key == pygame.K_t:
                        # Start training
                        from train import train
                        train()
                        # Restore screen after training returns
                        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                    if event.key == pygame.K_w:
                        # Watch agent
                        from train import watch
                        watch()
                        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                elif self.state == "PLAYING":
                    if event.key == pygame.K_r:
                        self.reset()
                    if event.key == pygame.K_ESCAPE:
                        self.state = "MENU"

        if self.state == "PLAYING":
            self.car.handle_input()

    def update(self):
        """
        Update game state.
        """
        if self.state != "PLAYING":
            return
            
        self.car.update()
        
        # Collision with walls
        if self.track.check_collision(self.car.rect):
             self.reset()
        
        # Lap Timing Logic
        # Check if car crosses start line
        if self.car.rect.colliderect(self.track.start_line):
           if not self.crossed_start_line:
                self.crossed_start_line = True
                current_time = time.time()
                
                if self.lap_start_time != 0:
                     lap_time = current_time - self.lap_start_time
                     if lap_time > 1.0: # Avoid instant double trigger
                        self.last_lap_time = lap_time
                        if lap_time < self.best_lap:
                             self.best_lap = lap_time
                
                self.lap_start_time = current_time
        else:
            self.crossed_start_line = False

    def draw(self):
        """
        Render the game.
        """
        if not self.screen:
            return

        self.track.draw(self.screen)
        self.car.draw(self.screen)
        
        if self.state == "MENU":
            self.draw_menu()
        else:
            self.draw_ui()
        
        pygame.display.flip()

    def draw_ui(self):
        """
        Draw HUD (Heads Up Display).
        """
        current = 0
        if self.lap_start_time != 0:
            current = time.time() - self.lap_start_time
        
        text_color = WHITE
        
        # Draw background panel for UI
        pygame.draw.rect(self.screen, BLACK, (10, 10, 300, 100))
        pygame.draw.rect(self.screen, WHITE, (10, 10, 300, 100), 2)
        
        cur_text = self.font.render(f"Current: {current:.2f}s", True, text_color)
        self.screen.blit(cur_text, (20, 20))
        
        best_text = "Best: --"
        if self.best_lap != float('inf'):
            best_text = f"Best: {self.best_lap:.2f}s"
        
        best_surf = self.font.render(best_text, True, text_color)
        self.screen.blit(best_surf, (20, 50))
        
        # Controls Hint
        hint = self.font.render("Arrows to drive. R to reset. ESC for Menu.", True, GRAY)
        self.screen.blit(hint, (20, 80))

    def draw_menu(self):
        """
        Draw Main Menu.
        """
        # Semi-transparent overlay
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        s.set_alpha(128)
        s.fill(BLACK)
        self.screen.blit(s, (0,0))
        
        title = self.font.render("MONACO TIME TRIAL", True, WHITE)
        start_msg = self.font.render("Press ENTER to Start", True, WHITE)
        train_msg = self.font.render("Press T to Train Agent", True, WHITE)
        watch_msg = self.font.render("Press W to Watch Agent", True, WHITE)
        quit_msg = self.font.render("Press ESC to Quit", True, WHITE)
        
        self.screen.blit(title, (SCREEN_WIDTH/2 - title.get_width()/2, SCREEN_HEIGHT/3))
        self.screen.blit(start_msg, (SCREEN_WIDTH/2 - start_msg.get_width()/2, SCREEN_HEIGHT/2))
        self.screen.blit(train_msg, (SCREEN_WIDTH/2 - train_msg.get_width()/2, SCREEN_HEIGHT/2 + 40))
        self.screen.blit(watch_msg, (SCREEN_WIDTH/2 - watch_msg.get_width()/2, SCREEN_HEIGHT/2 + 80))
        self.screen.blit(quit_msg, (SCREEN_WIDTH/2 - quit_msg.get_width()/2, SCREEN_HEIGHT/2 + 120))

    def reset(self):
        """
        Reset the game state (car position, lap timer).
        """
        start_x = SCREEN_WIDTH / 2 - 100
        start_y = SCREEN_HEIGHT - 150
        self.car.reset(start_x, start_y)
        self.lap_start_time = 0
        self.crossed_start_line = False

    def run(self):
        """
        Main Loop.
        """
        while self.running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
