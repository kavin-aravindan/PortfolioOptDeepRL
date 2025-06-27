# TD3.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            states,
            actions,
            rewards.reshape(-1, 1),
            next_states,
            dones.reshape(-1, 1),
        )

class GaussianNoise:
    def __init__(self, action_dim: int, mu: float = 0.0, sigma: float = 0.1):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(self.mu, self.sigma, size=self.action_dim)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.noise = GaussianNoise(action_dim)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)
        return action.cpu().numpy().flatten()

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Action with exploration noise."""
        action = self.select_action(state)
        raw = action + self.noise.sample()
        action = np.clip(raw, -self.max_action, self.max_action)
        return action, raw

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        self.total_it += 1

        # Sample a batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute target actions with clipped noise
        noise = (
            torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip).to(self.device)
        next_actions = (
            self.actor_target(next_states) + noise
        ).clamp(-self.max_action, self.max_action)

        # Compute target Q-values
        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + (1 - dones) * self.gamma * target_q.detach()

        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_loss = -self.critic.q1(torch.cat([states, self.actor(states)], dim=1)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Update targets
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)