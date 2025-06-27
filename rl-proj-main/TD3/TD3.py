import torch
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np
import backtrader as bt
from utils import state_to_tensor
from neural import Actor, Critic


class AnnealedGaussianProcess:
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0
        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size
        self.storage = self.buffer

    def add(self, s, a, r, s_next, terminal):
        self.buffer.append((s, a, r, s_next, terminal))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = map(list, zip(*samples))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(terminals)
        )


class TD3:
    def __init__(
        self, env,
        n_assets, in_feat,
        actor_model=None, critic_model=None,
        buffer_size=1000000, batch_size=64,
        gamma=0.99, tau=1e-3,
        lr_actor=1e-4, lr_critic=1e-3,
        ou_theta=0.15, ou_sigma=0.2,
        weight_decay=1e-2,
        policy_noise=0.2, noise_clip=0.5, policy_delay=2,
        hidden_dim=128
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.in_feat = in_feat
        self.n_assets = n_assets
        self.batch_size  = batch_size
        self.gamma       = gamma
        self.tau         = tau
        self.loss_func   = nn.MSELoss()
        # actor + target
        self.actor_model        = actor_model  if actor_model  else Actor(n_assets, in_feat, h=hidden_dim)
        self.actor_target_model = copy.deepcopy(self.actor_model)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr_actor)
        # critics + targets
        self.critic_model1        = critic_model  if critic_model  else Critic(n_assets, in_feat, h=hidden_dim)
        self.critic_model2        = copy.deepcopy(self.critic_model1)
        self.critic_target1_model = copy.deepcopy(self.critic_model1)
        self.critic_target2_model = copy.deepcopy(self.critic_model1)
        self.critic_optim = torch.optim.Adam(
            list(self.critic_model1.parameters()) + list(self.critic_model2.parameters()),
            lr_critic, weight_decay=weight_decay
        )
        # to device
        for net in [self.actor_model, self.actor_target_model,
                    self.critic_model1, self.critic_model2,
                    self.critic_target1_model, self.critic_target2_model]:
            net.to(self.device)
        # replay buffer + exploration
        self.buffer = ReplayBuffer(buffer_size)
        self.OU     = OrnsteinUhlenbeckProcess(theta=ou_theta, size=n_assets, sigma=ou_sigma)
        # TD3 hyperparams
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_delay = policy_delay
        self.total_it     = 0

        # metrics
        self.all_step_rewards = []
        self.episode_sums     = []
        self.episode_means    = []
        self.episode_vars     = []
        self.actor_losses           = []
        self.actor_target_losses    = []
        self.critic_losses          = []
        self.critic_target_losses   = []

    def act(self, state):
        s = state_to_tensor(state).to(self.device)
        w, _ = self.actor_model.get_action(s)
        # noise = torch.from_numpy(self.OU.sample()).float().to(self.device)
        # w = w + noise
        return w.squeeze().cpu().detach().numpy()

    def episode(self):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.env.simulate(self)
        sum_r=np.sum(self.rewards); mean_r=np.mean(self.rewards); var_r=np.var(self.rewards)
        self.episode_sums.append(sum_r); self.episode_means.append(mean_r); self.episode_vars.append(var_r)
        
        if len(self.buffer.buffer) > self.batch_size:
            a_l,c_l=self.update_policy()
            self.actor_losses.append(a_l); self.critic_losses.append(c_l)

        return self.rewards

    def step(self, state, reward):
        # self.all_step_rewards.append(reward)
        s = state_to_tensor(state).to(self.device)
        w, _ = self.actor_model.get_action(s)
        self.states.append(s.detach().cpu().numpy().squeeze(0))
        self.actions.append(w.detach().cpu().numpy().squeeze(0))
        self.rewards.append(reward)
        if len(self.states) > 1:
            self.buffer.add(
                self.states[-2], self.actions[-2], self.rewards[-2],
                self.states[-1], False
            )

    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1 - self.tau) + s.data * self.tau)

    def update_policy(self):
        self.total_it += 1
        # sample batch
        states, actions, rewards, next_states, terminals = self.buffer.sample(self.batch_size)
        s  = torch.from_numpy(states).float().to(self.device)
        a  = torch.from_numpy(actions).float().to(self.device)
        r  = torch.from_numpy(rewards).float().to(self.device)
        s2 = torch.from_numpy(next_states).float().to(self.device)
        d  = torch.from_numpy(terminals).float().to(self.device)
        # critic update
        with torch.no_grad():
            a2, _ = self.actor_target_model(s2)
            noise = (torch.randn_like(a2) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = a2 + noise
            q1_t = self.critic_target1_model(s2, a2)
            q2_t = self.critic_target2_model(s2, a2)
            q_t  = torch.min(q1_t, q2_t)
            target_q = r + self.gamma * q_t * (1 - d)
        q1 = self.critic_model1(s, a)
        q2 = self.critic_model2(s, a)
        critic_loss = self.loss_func(q1, target_q) + self.loss_func(q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # delayed actor update
        if self.total_it % self.policy_delay == 0:
            a_pred, _ = self.actor_model(s)
            actor_loss = -self.critic_model1(s, a_pred).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            # soft update targets
            self.soft_update(self.actor_target_model,      self.actor_model)
            self.soft_update(self.critic_target1_model, self.critic_model1)
            self.soft_update(self.critic_target2_model, self.critic_model2)
        
        return actor_loss.item() if self.total_it % self.policy_delay == 0 else 0, critic_loss.item()