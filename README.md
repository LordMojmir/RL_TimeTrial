# RL TimeTrial: Monaco

Welcome to the **RL TimeTrial**, a 2D top-down racing game developed using Pygame where you can train a Reinforcement Learning (RL) agent using Proximal Policy Optimization (PPO) to master navigating the track!

The project lets you not only play the game and train an AI, but also features a dedicated **Agent vs. Player** mode where you can race your trained agent side-by-side using custom car sprites.

---

## Features

- **Top-Down Racing Physics:** A custom `Car` object featuring constant acceleration, frictionless steering logic, and Raycasting logic to provide spatial awareness to AI models.
- **Custom Environment:** A Gym-compatible environment `CarRacingEnv` designed specifically to train our car agent on lap progression and wall-avoidance proxies.
- **PPO RL Agent:** A hand-written `PPOAgent` with an integrated Actor-Critic network implementation, computing GAE (Generalized Advantage Estimation) natively using PyTorch.
- **Agent vs. Player "Duel Mode":** Test your skills against the trained neural network by racing alongside it using your keyboard arrows.

---

## Setup and Installation

### Prerequisites

Ensure you have **Python 3.10+** and a working package manager (e.g., `pip` or `uv`). 

This project requires the following libraries:
- `pygame`
- `numpy`
- `torch`
- `gymnasium`

If you are using `uv`, you can run the files seamlessly if requirements are met in a virtual environment. Otherwise, standard `pip install` works:
```bash
pip install pygame numpy torch gymnasium
```

---

##  How to Run

To launch the game's Main Menu, execute the following from the root directory:

```bash
python main.py
```
*(Or if using your `uv` environment: `uv run python main.py`)*

### Control Layout

- **Main Menu Navigation:**
  - `Enter`: Start Standard Player Trial Mode
  - `T`: Start Training the Agent (`train()`)
  - `W`: Watch/Duel Mode (`watch()`)
  - `Esc`: Quit Application
- **Standard Driving/Duel Mode Controls:**
  - `Left Arrow`: Turn Left
  - `Right Arrow`: Turn Right
  - *(Acceleration is automatic)*
- **Training Mode Shortcuts:**
  - `S`: Save model checkpoint manually.
  - `R`: Replay best ghost times dynamically.
  - `Esc`: Stop training and return.

---

##  Codebase Architecture

Here is a breakdown of what each module in this repository is responsible for:

### 1. Game & Engine
- **`main.py`**: The entry point of the script. Simply instantiates and executes the core `Game` loop.
- **`game.py`**: Controls the foundational structure of the application. Handles logic for user keyboard input, tracks time, and rendering main menus.
- **`utils.py`**: Defines all global configurations, physical variables (e.g. `CAR_ACCELERATION`), RGB constants, and framerate definitions.
- **`car.py`**: Encapsulates all physics interactions involving the vehicle. Calculates rotation velocity logic, manages sprite transformations, and calculates normalized Raycasts.
- **`track.py`**: Defines the map geometry, wall collision checks, line segments for RL calculation, and checkpoints/start-line bounding boxes.

### 2. Reinforcement Learning
- **`rl_env.py`**: Wraps Pygame within a formal `gymnasium.Env` (`CarRacingEnv`). This provides the translation layer where `observations` (car speed, angle, and 5 raycast distances) and `actions` (a normalized steering array) occur so the agent understands its boundaries and rewards.
- **`train.py`**: The heart of the training loop and observer routines. Modifies environments natively to collect action distributions, rewards, and applies GAE to pass batch tensors into the model. Now houses the fully functioning *Agent vs. Player* `watch()` system!
- **`agent.py`**: Defines the neural network `ActorCritic` architecture, relying strictly on native feedforward networks, PyTorch `Normal` Distributions, and the Proximal Policy Optimization equations.

### 3. Assets
- **`car.png`**: The player sprite skin (usually Blue/Default).
- **`car2.png`**: The trained Agent sprite skin (Red).
- **`ppo_model.pth`** *(Generated)*: The compiled weights matrix dictating your agent's driving capability. By default, `train.py` saves to this output after achieving a sequence of new personal-best times.

---

##  How PPO Learns

During training mode, 10 `envs` are instantiated in parallel to mass-collect states. 
The RL `CarRacingEnv` handles reward modeling via:
- High penalty (`-500`) for colliding with walls.
- Minor positional penalty deductions for proximity clipping using raycasts (`min_dist`).
- Lap success bonuses (`+1000`) designed uniquely to encourage crossing the `track.start_line`. 

After 2048 steps, the batched variables feed into `.update()` scaling the Agent's probability distribution to repeat rewarded steering angles!
