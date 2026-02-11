import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=64, std=0.0):
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
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

class PPOAgent:
    def __init__(self, num_inputs, num_outputs, lr=3e-4):
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
        # State can be single (7,) or batch (N, 7)
        state = torch.FloatTensor(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        dist, value = self.model(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().numpy().flatten()

    def update(self, states, actions, log_probs, returns, advantages):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        
        # PPO Epochs
        for _ in range(self.num_epochs):
            dist, value = self.model(states)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value).pow(2).mean()
            
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def compute_gae(self, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
        
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
