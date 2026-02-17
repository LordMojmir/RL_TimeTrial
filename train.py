import pygame
import numpy as np
import time
from rl_env import CarRacingEnv
from agent import PPOAgent
from utils import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE

def train():
    parallel_cars = 10
    num_envs = parallel_cars
    
    # Manual List of Envs to allow custom rendering of ALL cars
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
    best_reward_so_far = -float('inf')
    best_times = []
    car_histories = [[] for _ in range(num_envs)] # List of lists of tuples (x,y,angle)
    
    # Reset all
    next_obs = np.array([env.reset()[0] for env in envs])
    next_done = np.zeros(num_envs)
    
    print("Starting Training...")
    
    # Initialize screen for rendering
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Training Visualization")
    font = pygame.font.SysFont("Verdana", 24)
    clock = pygame.time.Clock()
    
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
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        training_running = False
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
            # We use the first env's track to draw the background since they are all the same
            envs[0].game.track.draw(screen) 
            
            # Check for input to toggle Replay Mode
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r] and len(best_times) > 0:
                 replay_best_runs(screen, font, best_times, envs[0].game.track)

            for i, env in enumerate(envs):
                env.game.car.draw(screen) # Draw each car
                
                # RECORD HISTORY
                c = env.game.car
                if len(car_histories) <= i:
                    car_histories.append([])
                
                car_histories[i].append((c.x, c.y, c.angle))
            
            # Check for lap times
            for i, info in enumerate(infos):
                if 'lap_time' in info:
                    l_time = info['lap_time']
                    lap_history = list(car_histories[i])
                    
                    # Add to leaderboard
                    best_times.append({'time': l_time, 'epoch': epoch_count, 'car': i, 'history': lap_history})
                    # Sort by time (asc)
                    best_times.sort(key=lambda x: x['time'])
                    best_times = best_times[:3]
                    print(f"Car {i} finished lap in {l_time:.2f}s")
            
            # Draw UI (Leaderboard & Stats)
            epoch_text = font.render(f"Epoch: {epoch_count} | Cars: {num_envs}", True, WHITE)
            screen.blit(epoch_text, (10, 10))
            
            lb_title = font.render("Best 3 Times (Press R to Replay):", True, WHITE)
            screen.blit(lb_title, (SCREEN_WIDTH - 300, 10))
            
            for idx, entry in enumerate(best_times):
                time_str = f"{idx+1}. {entry['time']:.2f}s (Ep {entry['epoch']})"
                lb_text = font.render(time_str, True, WHITE)
                screen.blit(lb_text, (SCREEN_WIDTH - 300, 40 + idx * 30))
            
            pygame.display.flip()
            # Limit FPS for visualization if needed, but usually we want to train fast
            # clock.tick(60) 
            
            # Clear history for cars that are done
            for i, done in enumerate(step_dones):
                if done:
                    car_histories[i] = []

        if not training_running: break

        # Compute GAE and Update
        _, _, next_value = agent.select_action(next_obs)
        
        # Reshape data to [envs, steps] for easier GAE
        r_T = np.array(rewards).T
        d_T = np.array(dones).T
        v_T = np.array(values).T
        nv_T = next_value # (envs,)
        
        all_returns = []
        
        for i in range(num_envs):
            env_rewards = r_T[i]
            env_dones = d_T[i]
            env_values = v_T[i]
            env_next_val = nv_T[i]
            
            env_masks = 1 - env_dones
            
            env_returns = agent.compute_gae(env_next_val, env_rewards.tolist(), env_masks.tolist(), env_values.tolist())
            all_returns.append(env_returns)
        
        # Flatten for PPO update
        b_states = np.array(states).reshape(-1, 7)
        b_actions = np.array(actions).reshape(-1, 1)
        b_log_probs = np.array(log_probs).reshape(-1)
        b_returns = np.array(all_returns).T.reshape(-1)
        b_values = np.array(values).reshape(-1)
        b_advantages = b_returns - b_values
        
        agent.update(b_states, b_actions, b_log_probs, b_returns, b_advantages)
        
        epoch_count += 1
        avg_reward = np.mean(rewards)
        print(f"Epoch {epoch_count} Complete. Avg Reward: {avg_reward}")
        
        if avg_reward > best_reward_so_far:
            best_reward_so_far = avg_reward
            agent.save("ppo_model.pth")
            print(f"New Best Reward! Model Saved. ({avg_reward:.2f})")
        
        if epoch_count % 10 == 0:
             agent.save("ppo_checkpoint.pth")

    for env in envs: 
        env.close()
    # pygame.quit() # Don't quit, return to menu

def replay_best_runs(screen, font, best_times, track):
    print("Replaying Best Runs...")
    from car import Car
    
    # Create ghost cars
    ghosts = []
    for entry in best_times:
         c = Car(0, 0) # Dummy car for rendering, x/y overriden by history
         ghosts.append({'car': c, 'history': entry['history']})
         
    # Find max length
    max_steps = max([len(g['history']) for g in ghosts])
    
    replay_running = True
    step = 0
    clock = pygame.time.Clock()
    
    while replay_running:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 replay_running = False
                 # If we quit here during replay, we probably want to stop everything?
                 # But for now let's just exit replay
             if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_ESCAPE or event.key == pygame.K_r:
                     replay_running = False
         
         track.draw(screen)
         
         # Draw Ghosts
         for g in ghosts:
             hist = g['history']
             if step < len(hist):
                 x, y, angle = hist[step]
                 g['car'].x = x
                 g['car'].y = y
                 g['car'].angle = angle
                 g['car'].draw(screen)
         
         message_y = 50
         msg = font.render("REPLAYING BEST RUNS (Press R to return)", True, (255, 255, 0))
         screen.blit(msg, (SCREEN_WIDTH/2 - msg.get_width()/2, message_y))
         
         replay_time = step / 60.0
         timer_msg = font.render(f"Replay Time: {replay_time:.2f}s", True, WHITE)
         screen.blit(timer_msg, (SCREEN_WIDTH/2 - timer_msg.get_width()/2, message_y + 30))
         
         pygame.display.flip()
         clock.tick(60)
         
         step += 1
         if step >= max_steps:
             step = 0 # Loop
             time.sleep(1)

def watch():
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
    clock = pygame.time.Clock()
    
    while running:
         # Need to handle events to prevent freezing
         for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
         
         # The env has a game instance, but we need to step the agent
         action, _, _ = agent.select_action(obs)
         obs, reward, terminated, truncated, _ = env.step(action)
         done = terminated or truncated
         
         if done:
             obs, _ = env.reset()
             
         clock.tick(60)
             
    env.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    else:
        train()
    pygame.quit()
