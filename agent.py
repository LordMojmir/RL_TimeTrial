import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    """
    A foundational Actor-Critic neural network used for Continuous PPO control.
    Contains two separate Multilayer Perceptrons (MLPs) sharing no weights.
    """
    def __init__(self, num_inputs, num_outputs, hidden_size=64, std=0.0):
        """
        Initializes the actor and critic networks.
        
        Args:
            num_inputs (int): Dimension of the observation space.
            num_outputs (int): Dimension of the action space.
            hidden_size (int): Node count for hidden layers.
            std (float): Initial standard deviation constraint for continuous actions.
        """
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
    def forward(self, x):
        """
        Passes state x through the networks to return a continuous action distribution
        and the critic's value baseline.
        
        Args:
            x (torch.Tensor): The observation batch tensor.
            
        Returns:
            Normal: PyTorch Normal distribution defining continuous continuous probabilities.
            torch.Tensor: Evaluated state value.
        """
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

class PPOAgent:
    """
    Reinforcement learning agent applying the Proximal Policy Optimization (PPO) 
    strategy with Generalized Advantage Estimation (GAE).
    """
    def __init__(self, num_inputs, num_outputs, lr=3e-4):
        """
        Initializes the PPO agent, hyper-parameters, and Adam optimizer.
        
        Args:
            num_inputs (int): The shape width of the environments observation space.
            num_outputs (int): The shape width of the environments continuous action space.
            lr (float): The learning rate for the networks.
        """
        self.model = ActorCritic(num_inputs, num_outputs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.num_epochs = 10
        self.batch_size = 64

    def select_action(self, state):
        """
        Samples an action probabilistically from the Actor given an observation.
        
        Args:
            state (np.array/list): Observation representing the environment state.
            
        Returns:
            np.array: Action numerical choice.
            np.array: The log probability of sampling that action from the actor distribution.
            np.array: The Critic network's estimated value for the given state.
        """
        # State can be single (7,) or batch (N, 7)
        state = torch.FloatTensor(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        dist, value = self.model(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().numpy().flatten()

    def update(self, states, actions, log_probs, returns, advantages):
        """
        Executes the primary PPO gradient descent sequence.
        
        Calculates the ratio between new probabilities and old probabilities, clips to
        ensure small/safe trust region updates, and calculates value/entropy losses.
        
        Args:
            states (np.array): Batched recorded states.
            actions (np.array): Batched recorded actions taken.
            log_probs (np.array): Batched log probabilities of those actions.
            returns (np.array): Actual discount returns evaluated by GAE.
            advantages (np.array): Difference between GAE returns and Critic baseline.
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        
        # PPO Epochs (Repeatedly optimizing on collected trajectory data)
        for _ in range(self.num_epochs):
            dist, value = self.model(states)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
            
            # --- PPO Advantage Math ---
            # Ratio checks how different the updated policy is from the old policy.
            # E.g if probability increased, ratio > 1
            ratio = (new_log_probs - old_log_probs).exp()
            
            # surr1: Standard reinforcement formula: shift probabilities proportional to Advantage
            surr1 = ratio * advantages
            
            # surr2: PPO specific: Clamp the ratio so policy cannot drift too far per update (Trust Region)
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            # We minimize the negative objective (equivalent to maximizing the expected reward)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # The critic learns by trying to make its value prediction approach the actual GAE returns
            critic_loss = (returns - value).pow(2).mean()
            
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def compute_gae(self, next_value, rewards, masks, values):
        """
        Computes Generalized Advantage Estimation (GAE).
        
        Args:
            next_value (float): Critic's extrapolated prediction of next state value.
            rewards (list): Tracked historical rewards.
            masks (list): 0 if episode finished, otherwise 1.
            values (list): Tracked historical Critic predictions.
            
        Returns:
            list: The actual bootstrapped target returns for the given trajectory.
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            # --- GAE Math ---
            # delta (Temporal Difference Error): Looks at Immediate Reward + (Discounted Next State Value) - Expected Value
            # Basically asks: Was this single step better or worse than the Critic predicted?
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            
            # Accumulate discounting backwards through time utilizing lambda to smooth variance tradeoffs
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
        
    def save(self, filename):
        """Saves PyTorch parameters to disk."""
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        """Loads PyTorch parameters onto the ActorCritic model."""
        self.model.load_state_dict(torch.load(filename))
