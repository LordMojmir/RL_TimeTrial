import pygame
import time
from utils import *
from car import Car
from track import Track

import numpy as np

class Game:
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
        
        # Training Config
        self.parallel_cars = 1000
        
        # Lap logic state
        self.crossed_start_line = False 

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if self.state == "MENU":
                    if event.key == pygame.K_RETURN:
                        self.state = "PLAYING"
                        self.reset()
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
        if self.state != "PLAYING":
            return
            
        self.car.update()
        
        # Collision with walls
        # Collision with walls
        if self.track.check_collision(self.car.rect):
             self.reset()
        
        # Lap Timing Logic
        # Check if car crosses start line
        # Simple check: collision with start line rect
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
        self.track.draw(self.screen)
        self.car.draw(self.screen)
        
        if self.state == "MENU":
            self.draw_menu()
        else:
            self.draw_ui()
        
        pygame.display.flip()

    def draw_ui(self):
        # Current Lap Time
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
        # Controls Hint
        hint = self.font.render("Arrows to drive. R to reset. ESC for Menu.", True, GRAY)
        self.screen.blit(hint, (20, 80))

    def draw_menu(self):
        # Semi-transparent overlay
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        s.set_alpha(128)
        s.fill(BLACK)
        self.screen.blit(s, (0,0))
        
        title = self.font.render("MONACO TIME TRIAL", True, WHITE)
        start_msg = self.font.render("Press ENTER to Start", True, WHITE)
        quit_msg = self.font.render("Press ESC to Quit", True, WHITE)
        
        self.screen.blit(title, (SCREEN_WIDTH/2 - title.get_width()/2, SCREEN_HEIGHT/3))
        self.screen.blit(start_msg, (SCREEN_WIDTH/2 - start_msg.get_width()/2, SCREEN_HEIGHT/2))
        self.screen.blit(quit_msg, (SCREEN_WIDTH/2 - quit_msg.get_width()/2, SCREEN_HEIGHT/2 + 50))

    def reset(self):
        start_x = SCREEN_WIDTH / 2 - 100
        start_y = SCREEN_HEIGHT - 150
        self.car.reset(start_x, start_y)
        self.lap_start_time = 0
        self.crossed_start_line = False



    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if self.state == "MENU":
                    if event.key == pygame.K_RETURN:
                        self.state = "PLAYING"
                        self.reset()
                    if event.key == pygame.K_t:
                        self.start_training()
                    if event.key == pygame.K_w:
                        self.watch_agent()
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_UP:
                        self.parallel_cars = min(self.parallel_cars + 1, 16)
                    if event.key == pygame.K_DOWN:
                        self.parallel_cars = max(self.parallel_cars - 1, 1)
                        
                elif self.state == "PLAYING":
                    if event.key == pygame.K_r:
                        self.reset()
                    if event.key == pygame.K_ESCAPE:
                        self.state = "MENU"
        
        if self.state == "PLAYING":
            self.car.handle_input()

    def load_model(self):
        # We need an agent instance to load into?
        # Or just flag that we want to load? 
        # For simplicity, let's create a global agent or just confirm loading.
        # But we don't have an agent instance in Game usually (only in start_training).
        # We should probably persist the agent in Game if we want to Load then Train/Watch.
        
        # NOTE: For this architecture, let's just create a dummy agent to check file exists or just set a flag
        # self.agent_path = "ppo_model.pth"
        pass 

    def watch_agent(self):
        import gymnasium as gym
        from rl_env import CarRacingEnv
        from agent import PPOAgent
        
        env = CarRacingEnv(render_mode="human")
        agent = PPOAgent(num_inputs=7, num_outputs=2)
        try:
            agent.load("ppo_model.pth")
            print("Model loaded.")
        except:
            print("No model found, using random weights.")
            
        obs, _ = env.reset()
        done = False
        
        running = True
        while running:
             for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        self.state = "MENU"
             
             action, _, _ = agent.select_action(obs)
             obs, reward, terminated, truncated, _ = env.step(action)
             done = terminated or truncated
             
             if done:
                 obs, _ = env.reset()
                 
        env.close()
        if self.running and not self.headless:
             self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    def start_training(self):
        # Import here to avoid circular dependencies if any, or just for clarity
        import gymnasium as gym
        from rl_env import CarRacingEnv
        from agent import PPOAgent
        
        num_envs = self.parallel_cars
        
        # Manual List of Envs instead of SyncVectorEnv to allow custom rendering of ALL cars
        # We pass headless=True to all because WE (the main Game) will handle rendering
        envs = [CarRacingEnv(render_mode="rgb_array") for _ in range(num_envs)]
        
        # Agent input 7, output 1 (Steering only)
        agent = PPOAgent(num_inputs=7, num_outputs=1)
        
        # Try load if exists
        try:
            agent.load("ppo_model.pth")
            print("Resuming training from ppo_model.pth")
        except:
            print("Starting new training.")

        # Training Config
        num_steps = 2048
        global_step = 0
        epoch_count = 0
        self.best_reward_so_far = -float('inf')
        self.best_times = []
        car_histories = [[] for _ in range(num_envs)] # List of lists of tuples (x,y,angle)
        
        # Reset all
        next_obs = np.array([env.reset()[0] for env in envs])
        next_done = np.zeros(num_envs)
        
        print("Starting Training...")
        
        training_running = True
        while training_running:
            # Data collection
            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            values = []
            
            for step in range(num_steps):
                global_step += 1
                
                # Check for exit/render events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        training_running = False
                        self.running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            training_running = False
                            self.state = "MENU"
                            break
                        if event.key == pygame.K_s:
                            agent.save("ppo_model.pth")
                            print("Model saved.")
                if not training_running: break
                
                # Action
                action, log_prob, value = agent.select_action(next_obs)
                
                # Step all envs manually
                step_rewards = []
                step_dones = []
                step_next_obs = []
                infos = []
                
                for i, env in enumerate(envs):
                    # action[i] is [steering] (shape 1)
                    
                    act = action[i] # This is [steering]
                    
                    obs, reward, terminated, truncated, info = env.step(act)
                    done = terminated or truncated
                    
                    if done:
                        obs, _ = env.reset()
                        
                    step_rewards.append(reward)
                    step_dones.append(done)
                    step_next_obs.append(obs)
                    infos.append(info)
                
                # Turn back to numpy arrays
                obs_batch = np.array(step_next_obs)
                reward_batch = np.array(step_rewards)
                done_batch = np.array(step_dones)
                
                states.append(next_obs)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward_batch)
                dones.append(next_done)
                values.append(value)
                
                next_obs = obs_batch
                next_done = done_batch
                
                # RENDER ALL CARS
                self.track.draw(self.screen) # Draw background
                
                # Check for input to toggle Replay Mode
                keys = pygame.key.get_pressed()
                if keys[pygame.K_r] and len(self.best_times) > 0:
                     self.replay_best_runs()

                for i, env in enumerate(envs):
                    env.game.car.draw(self.screen) # Draw each car
                    
                    # RECORD HISTORY
                    # We need to access the car's current state
                    c = env.game.car
                    # If this car is active (not done yet this episode, or better, just record everything)
                    # We'll reset history on env.reset() but we don't have easy hook here unless we check done
                    if len(car_histories) <= i:
                        car_histories.append([])
                    
                    car_histories[i].append((c.x, c.y, c.angle))
                
                # Check for lap times
                for i, info in enumerate(infos):
                    if 'lap_time' in info:
                        l_time = info['lap_time']
                        # Record the history for this successful lap
                        # The history in car_histories[i] is for the current episode. 
                        # Since we reset on done, and lap completion doesn't necessarily mean done (unless we force it),
                        # ASsuming "one lap" episode for simplicity or we just take current buffer.
                        # Wait, env.reset() clears history? No, we need to clear it manually.
                        
                        # Copy current history
                        lap_history = list(car_histories[i])
                        
                        # Add to leaderboard
                        self.best_times.append({'time': l_time, 'epoch': epoch_count, 'car': i, 'history': lap_history})
                        # Sort by time (asc)
                        self.best_times.sort(key=lambda x: x['time'])
                        self.best_times = self.best_times[:3]
                        print(f"Car {i} finished lap in {l_time:.2f}s")
                
                # Draw UI (Leaderboard & Stats)
                # Epoch
                epoch_text = self.font.render(f"Epoch: {epoch_count} | Cars: {num_envs}", True, WHITE)
                self.screen.blit(epoch_text, (10, 10))
                
                # Leaderboard
                lb_title = self.font.render("Best 3 Times (Press R to Replay):", True, WHITE)
                self.screen.blit(lb_title, (SCREEN_WIDTH - 300, 10))
                
                for idx, entry in enumerate(self.best_times):
                    # entry is dict
                    time_str = f"{idx+1}. {entry['time']:.2f}s (Ep {entry['epoch']})"
                    lb_text = self.font.render(time_str, True, WHITE)
                    self.screen.blit(lb_text, (SCREEN_WIDTH - 300, 40 + idx * 30))
                
                pygame.display.flip()
                
                # Clear history for cars that are done
                for i, done in enumerate(step_dones):
                    if done:
                        car_histories[i] = []

            if not training_running: break

            # Compute GAE and Update
            _, _, next_value = agent.select_action(next_obs)
            
            # Helper for GAE (needs list of rewards [steps, envs])
            # My GAE function expects 1D lists or needs update.
            # agent.compute_gae was written for 1D.
            # We need to compute GAE for each env separately or vectorize GAE.
            # Simple approach: Loop over envs to compute returns, then flatten.
            
            # Reshape data to [envs, steps] for easier GAE
            # rewards: [steps, envs] -> [envs, steps]
            r_T = np.array(rewards).T
            d_T = np.array(dones).T
            v_T = np.array(values).T
            nv_T = next_value # (envs,)
            
            # We need to update compute_gae in agent to handle this or loop here
            # Let's loop here for simplicity and robustness
            all_returns = []
            
            for i in range(num_envs):
                # rewards for env i
                env_rewards = r_T[i]
                env_dones = d_T[i] # actually masks? "dones" is usually "is done". GAE needs "mask" (1-done)
                env_values = v_T[i]
                env_next_val = nv_T[i]
                
                env_masks = 1 - env_dones
                
                # We need list inputs for my simple agent.compute_gae
                env_returns = agent.compute_gae(env_next_val, env_rewards.tolist(), env_masks.tolist(), env_values.tolist())
                all_returns.append(env_returns)
            
            # Now flatten everything for PPO update
            # states: [steps, envs, 7] -> [steps * envs, 7]
            b_states = np.array(states).reshape(-1, 7)
            b_actions = np.array(actions).reshape(-1, 1) # Action dim 1
            b_log_probs = np.array(log_probs).reshape(-1)
            b_returns = np.array(all_returns).T.reshape(-1) # Tanspose back to [steps, envs] then flatten
            b_values = np.array(values).reshape(-1)
            b_advantages = b_returns - b_values
            
            agent.update(b_states, b_actions, b_log_probs, b_returns, b_advantages)
            
            epoch_count += 1
            avg_reward = np.mean(rewards)
            print(f"Epoch {epoch_count} Complete. Avg Reward: {avg_reward}")
            
            # Save if best
            if avg_reward > self.best_reward_so_far:
                self.best_reward_so_far = avg_reward
                agent.save("ppo_model.pth")
                print(f"New Best Reward! Model Saved. ({avg_reward:.2f})")
            
            # periodic save backup
            if epoch_count % 10 == 0:
                 agent.save("ppo_checkpoint.pth")

        for env in envs: 
            env.close()
            
        # Reset window for normal play if we exit training
        if self.running and not self.headless:
             self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    def replay_best_runs(self):
        # Playback the best runs recorded in self.best_times
        print("Replaying Best Runs...")
        import time
        from car import Car
        
        # Create ghost cars
        ghosts = []
        for entry in self.best_times:
             c = Car(0, 0) # Dummy car for rendering, x/y overriden by history
             ghosts.append({'car': c, 'history': entry['history'], 'color': (0, 255, 255)}) # Cyan for ghosts
             
        # Find max length
        max_steps = max([len(g['history']) for g in ghosts])
        
        replay_running = True
        step = 0
        
        while replay_running:
             for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     replay_running = False
                     self.running = False
                 if event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_ESCAPE or event.key == pygame.K_r:
                         replay_running = False
             
             self.track.draw(self.screen)
             
             # Draw Ghosts
             for g in ghosts:
                 hist = g['history']
                 if step < len(hist):
                     x, y, angle = hist[step]
                     g['car'].x = x
                     g['car'].y = y
                     g['car'].angle = angle
                     
                     # Draw uses self.x/y/angle so we are good.
                     g['car'].draw(self.screen)
             
             message_y = 50
             msg = self.font.render("REPLAYING BEST RUNS (Press R to return)", True, (255, 255, 0))
             self.screen.blit(msg, (SCREEN_WIDTH/2 - msg.get_width()/2, message_y))
             
             # Replay Timer
             replay_time = step / 60.0 # Assuming 60 FPS
             timer_msg = self.font.render(f"Replay Time: {replay_time:.2f}s", True, WHITE)
             self.screen.blit(timer_msg, (SCREEN_WIDTH/2 - timer_msg.get_width()/2, message_y + 30))
             
             pygame.display.flip()
             self.clock.tick(60)
             
             step += 1
             if step >= max_steps:
                 step = 0 # Loop
                 time.sleep(1) # Pause before restart

    def run(self):
        while self.running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
