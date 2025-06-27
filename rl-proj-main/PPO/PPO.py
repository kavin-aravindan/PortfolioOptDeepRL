import torch
import torch.nn as nn
import numpy as np
import backtrader as bt
from utils import state_to_tensor



class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1=400, hidden2=300):
        super().__init__()
        self.state_size=state_size

        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self.init_weights()
        self.layers = nn.Sequential(self.fc1, nn.ReLU(),
                                    self.fc2, nn.ReLU(),
                                    self.fc3, nn.Tanh())
        self.log_std = nn.Parameter(torch.randn(action_size))
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.fc3.weight.uniform_(-3e-3, 3e-3)
            self.fc3.bias.uniform_(-3e-3, 3e-3)
            self.fc1.weight.uniform_(-1/self.state_size, 1/self.state_size)
            self.fc2.weight.uniform_(-1/400, 1/400)
        
    def forward(self, s):
        mean = self.layers(s)
        std = torch.clamp(torch.exp(self.log_std), min=1e-3, max=1.0)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_size, hidden1=400, hidden2=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, s):
        return self.net(s)

class PPO():
    def __init__(self, env, state_size, action_size, gamma=0.99, clip=0.2, lr_actor=0.0003, lr_critic=0.001, batch_size=64, epochs=10):
        self.env = env
        self.gamma = gamma
        self.clip = clip
        self.gae_lambda = 0.95
        self.entropy_coeff = 0.01
        self.batch_size = batch_size
        self.epochs = epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._clear_trajectory_buffers()

    def _clear_trajectory_buffers(self):
        self.trajectory_obs = []
        self.trajectory_actions = []
        self.trajectory_log_probs_old = []
        self.trajectory_rewards = []
        self.trajectory_values_old = []
        self.trajectory_dones = []


    def act_and_get_details(self, state):
        input = state_to_tensor(state).to(self.device)
        input = input.view(input.size(0), -1)

        with torch.no_grad():
            mean, std = self.actor(input)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            # normalize action
            action = action / (torch.sum(torch.abs(action), dim=-1, keepdim=True) + 1e-8)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            value = self.critic(input)

        action_numpy = action.squeeze().cpu().numpy()
        return {
            'action_numpy': action_numpy,
            'obs_tensor': state_to_tensor(state).cpu(),
            'action_tensor': action.cpu(),
            'log_prob_tensor': log_prob.cpu(),
            'value_tensor': value.cpu(),
        }


    def store_transition(self, obs_tensor, action_tensor, log_prob_old_tensor, reward, value_old_tensor, done):
        self.trajectory_obs.append(obs_tensor)
        self.trajectory_actions.append(action_tensor)
        self.trajectory_log_probs_old.append(log_prob_old_tensor)
        self.trajectory_rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.trajectory_values_old.append(value_old_tensor)
        self.trajectory_dones.append(torch.tensor([1 if done else 0], dtype=torch.float32))


    def _calculate_gae(self, rewards_tensor, values_old_list_cpu, dones_tensor, last_value_estimate_cpu):
        num_steps = len(rewards_tensor)
        advantages = torch.zeros(num_steps, device=self.device) # Calculate on device

        values_for_gae = torch.cat(
            [v.to(self.device).view(-1) for v in values_old_list_cpu] + [last_value_estimate_cpu.view(-1).to(self.device)]
        ).squeeze() # Shape (num_steps + 1)

        rewards_tensor_dev = rewards_tensor.to(self.device)
        dones_tensor_dev = dones_tensor.to(self.device)

        last_gae_lam = 0
        for t in reversed(range(num_steps)):
            delta = rewards_tensor_dev[t] + self.gamma * values_for_gae[t+1] * (1.0 - dones_tensor_dev[t]) - values_for_gae[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - dones_tensor_dev[t]) * last_gae_lam
        
        rtgs_critic_targets = advantages + torch.cat([v.to(self.device) for v in values_old_list_cpu]).squeeze()
        return advantages, rtgs_critic_targets


    def evaluate(self, obs, acts):
        input = obs.view(obs.size(0), -1).to(self.device)
        values = self.critic(input)
        mean, std = self.actor(input)
        #try:
        dist = torch.distributions.Normal(mean, std)
        #print("mean",mean)
        #print("std", std)
       # except:
        #    print(obs)
         #   print(mean)
          #  print(std)
           # dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(acts).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return values, log_probs, entropy

    def episode(self):
        self._clear_trajectory_buffers()

        self.env.simulate(self) # (rollout) this calls self.step() for each step in the episode and populates the trajectory
        if self.trajectory_rewards == []:
          print("skipping")
          return []
        # print("reward is ")
        # print(self.trajectory_rewards)

        # Update policy and critic
        aloss, closs, actions = self.update()

        return [r.item() for r in self.trajectory_rewards], aloss, closs, actions # Return the rewards of the episode


    def update(self):
        obs_tensor = torch.cat(self.trajectory_obs, dim=0).to(self.device)
        #print(obs_tensor)
        acts_tensor = torch.cat(self.trajectory_actions, dim=0).to(self.device)
        #print(acts_tensor)
        log_probs_old_tensor = torch.cat(self.trajectory_log_probs_old, dim=0).squeeze(-1).to(self.device)
        #print(log_probs_old_tensor)
        rewards_tensor = torch.cat(self.trajectory_rewards, dim=0).squeeze(-1)
        batch_dones = torch.cat(self.trajectory_dones, dim=0).squeeze(-1)

        with torch.no_grad():
            last_value_estimate = torch.tensor([0.0], dtype=torch.float32)
            # if not batch_dones[-1] == 1:
            #     last_obs = obs_tensor[-1].view(1, -1).to(self.device)
            #     last_value_estimate = self.critic(last_obs).cpu()

            advantages, rtgs = self._calculate_gae(
                rewards_tensor, self.trajectory_values_old, batch_dones, last_value_estimate
            )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses, critic_losses = [],[]
        for _ in range(self.epochs):
            V, log_probs, entropy = self.evaluate(obs_tensor, acts_tensor)
            #print(log_probs)
            ratio = (log_probs - log_probs_old_tensor).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            #print(ratio)

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = self.entropy_coeff * entropy.mean()
            actor_loss = actor_loss - entropy_loss
       #     print(actor_loss)
            self.actor_optimizer.zero_grad()
           # with torch.autograd.detect_anomaly():
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())

            critic_loss = nn.MSELoss()(V.squeeze(), rtgs)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())
    
        
        return np.mean(np.array(actor_losses)), np.mean(np.array(critic_losses)), self.trajectory_actions
