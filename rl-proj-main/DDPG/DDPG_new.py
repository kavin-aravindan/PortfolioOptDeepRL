import torch
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np
import backtrader as bt
from utils import state_to_tensor

# stolen code. Check
class AnnealedGaussianProcess():
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

class ReplayBuffer():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size
        
    def add(self, s, a, r, s_next, terminal):
        self.buffer.append((s,a,r,s_next,terminal))
    
    #check this
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

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1=400, hidden2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self.init_weights()
        self.layers = nn.Sequential(self.fc1, nn.ReLU(),
                                    self.fc2, nn.ReLU(),
                                    self.fc3, nn.Tanh())
        
        # remove tanh. can't deal with noise addition in a continuous + discrete space
        
    def init_weights(self):
        with torch.no_grad():
            self.fc3.weight.uniform_(-3e-3, 3e-3)
            self.fc3.bias.uniform_(-3e-3, 3e-3)
            self.fc1.weight.uniform_(-1/np.sqrt(self.fc1.in_features), 1/np.sqrt(self.fc1.in_features))
            self.fc2.weight.uniform_(-1/np.sqrt(self.fc2.in_features), 1/np.sqrt(self.fc2.in_features))
        
    
    def forward(self, s):
        # with torch.autograd.set_detect_anomaly(True):
        return self.layers(s)
    
    def get_action(self, state):
        return self.forward(state)
        unnormalised_action =  self.forward(state)
        action = unnormalised_action / (torch.sum(torch.abs(unnormalised_action), dim=-1, keepdim=True) + 1e-8)
        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden1=400, hidden2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1+action_size, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights()
        self.layers_1 = nn.Sequential(self.fc1, nn.ReLU())
        self.layers_2 = nn.Sequential(self.fc2, nn.ReLU(),
                                      self.fc3)
        
    def init_weights(self):
        with torch.no_grad():
            self.fc3.weight.uniform_(-3e-3, 3e-3)
            self.fc3.bias.uniform_(-3e-3, 3e-3)
            self.fc1.weight.uniform_(-1/np.sqrt(self.fc1.in_features), 1/np.sqrt(self.fc1.in_features))
            self.fc2.weight.uniform_(-1/np.sqrt(self.fc2.in_features), 1/np.sqrt(self.fc2.in_features))    
        
    def forward(self, s, a):
        # with torch.autograd.set_detect_anomaly(True):
        # print(f"{s.shape=}")
        # print(f"{a.shape=}")
        y = self.layers_1(s)
        return self.layers_2(torch.cat((y,a), dim=-1))
        
class DDPG():
    """
    Deep Deterministic Policy Gradient (DDPG) algorithm for reinforcement learning.
    """
    def __init__(self, env, state_size, action_size, actor_model=None, critic_model=None, buffer_size=1000000, batch_size=64, gamma=0.99,
                 tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, ou_theta=0.15, ou_sigma=0.2, weight_decay=1e-2):
        
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.loss_func = nn.MSELoss()        
        
        if actor_model:
            self.actor_model = actor_model
        else:
            self.actor_model = Actor(self.state_size, self.action_size)
        self.actor_target_model = copy.deepcopy(self.actor_model)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr_actor)
        
        if critic_model:
            self.critic_model = critic_model
        else:
            self.critic_model = Critic(self.state_size, self.action_size)
        self.critic_target_model = copy.deepcopy(self.critic_model)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr_critic, weight_decay=weight_decay)  
        
        self.actor_model.to(self.device)
        self.actor_target_model.to(self.device)
        self.critic_model.to(self.device)
        self.critic_target_model.to(self.device)
        
        self.buffer = ReplayBuffer(buffer_size)
        self.OU = OrnsteinUhlenbeckProcess(size=action_size, theta=ou_theta, sigma=ou_sigma)
        
    def act(self, state):
        input = state_to_tensor(state).to(self.device)
        # print("input shpae", input.shape)
        input = input.view(input.size(0), -1)
        w = self.actor_model.get_action(input)
        # add noise
        noise = torch.from_numpy(self.OU.sample()).float().to(self.device)
        w += noise
        # normalise w 
        w = w / (torch.sum(torch.abs(w), dim=-1, keepdim=True) + 1e-8)
        return w.squeeze().cpu().detach().numpy()
    
    def episode(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.losses_actor = []
        self.losses_critic = []

        self.env.simulate(self)
        if self.buffer.size > self.batch_size:
            # update policy
            critic_loss, actor_loss = self.update_policy()
            # self.losses_actor.append(actor_loss)
            # self.losses_critic.append(critic_loss)
            # soft update the target networks
            self.soft_update(self.critic_target_model, self.critic_model)
        
        return self.rewards, critic_loss, actor_loss, self.actions
    
    def step(self, state, reward):
        input = state_to_tensor(state).to(self.device)
        input = input.view(input.size(0), -1)
        w = self.actor_model.get_action(input)
        value = self.critic_model(input, w)
        
        self.states.append(input)
        self.actions.append(w)
        self.values.append(value)
        self.rewards.append(reward)
        
        if len(self.states) > 1:
            # add data to replay buffer
            self.buffer.add(self.states[-2].squeeze(0).detach().cpu().numpy(),
                            self.actions[-2].squeeze(0).detach().cpu().numpy(),
                            self.rewards[-2],
                            self.states[-1].squeeze(0).detach().cpu().numpy(), False)
    
    # def get_action(self, state):
    #     state = torch.from_numpy(state).float().to(self.device)
    #     noise = torch.from_numpy(self.OU.sample()).float().to(self.device)
    #     # print(f"{state=}")
    #     # print(f"{noise=}")
        
    #     self.actor_model.eval()
    #     with torch.no_grad():
    #         result = self.actor_model(state) 
    #     self.actor_model.train()
    #     result += noise
        
    #     n = self.action_size // 4
        
    #     # output is of size 4 * n stocks
    #     # take the first n stocks and apply softmax
    #     weights = torch.softmax(result[:n], dim=0)
    #     # reshape the 3 * n stocks to (n, 3)
    #     positions = result[n:].reshape(-1, 3)
    #     # apply softmax to the positions
    #     positions = torch.softmax(positions, dim=1)
    #     # perform argmax to get the position
    #     final_pos = torch.argmax(positions, dim=1)
    #     final_pos -= 1
        
    #     raw_action = result.detach().cpu().numpy()
    #     action = np.concatenate((weights.detach().cpu().numpy(), final_pos.detach().cpu().numpy()), axis=0)
    #     return action, raw_action

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)  
        
    
    def update_policy(self):
        states, raw_actions, rewards, next_states, terminals = self.buffer.sample(self.batch_size)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        raw_actions_tensor = torch.from_numpy(raw_actions).float().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        terminals_tensor = torch.from_numpy(terminals.astype(np.float32)).float().to(self.device).reshape(-1, 1)
        
        # update critic
        self.actor_target_model.eval()
        self.critic_target_model.eval()
        with torch.no_grad():
            raw_actions_next = self.actor_target_model(next_states_tensor)
            next_q_vals_calc = self.critic_target_model(next_states_tensor, raw_actions_next)

            rewards_for_sum = rewards_tensor.view(self.batch_size, 1)
            next_q_for_sum = next_q_vals_calc.view(self.batch_size, 1)
            terminals_for_sum = terminals_tensor.view(self.batch_size, 1)

            # compute target reward
            target_q_vals = rewards_for_sum + self.gamma * next_q_for_sum * (1.0 - terminals_for_sum)
        
        # calc loss            
        self.critic_model.train()
        q_vals = self.critic_model(states_tensor, raw_actions_tensor)
        # Inside DDPG.update_policy()

        q_vals = q_vals.view(self.batch_size, 1)

        critic_loss = self.loss_func(target_q_vals, q_vals)
        
        # optimiser step
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        #update actor
        self.actor_model.train()
        raw_pred_actions = self.actor_model(states_tensor)
        #calc loss
        policy_loss = - self.critic_model(states_tensor, raw_pred_actions).mean()
        
        # optimiser step
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        
        # return actor and critic losses
        return critic_loss.item(), policy_loss.item()
        
    
# def train_loop_DDPG(model, env, num_episodes, num_timesteps):
#     for episode in range(num_episodes):
#         # get inital state??
#         s_init = None
#         s = s_init
#         for t in range(num_timesteps):
#             a = model.get_action(s)
#             # something to get details from env
#             r, s_next = env.step(a)
            
#             # add data to replay buffer
#             if t < num_timesteps - 1:
#                 model.buffer.add(s, a, r, s_next, False)
#             else:
#                 model.buffer.add(s, a, r, s_next, True)

#             # update policy
#             model.update_policy()
            
#             # soft update the target networks
#             model.soft_update(model.actor_target_model, model.actor_model)
#             model.soft_update(model.critic_target_model, model.critic_model)
            
            

