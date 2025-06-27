import torch
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np

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
                                    self.fc3)
        
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
        y = self.layers_1(s)
        return self.layers_2(torch.cat((y,a), dim=1))
        
class DDPG():
    """
    Deep Deterministic Policy Gradient (DDPG) algorithm for reinforcement learning.
    """
    def __init__(self, state_size, action_size, actor_model=None, critic_model=None, buffer_size=1000000, batch_size=64, gamma=0.99,
                 tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, ou_theta=0.15, ou_sigma=0.2, weight_decay=1e-2):
        
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        noise = torch.from_numpy(self.OU.sample()).float().to(self.device)
        print(f"{state=}")
        print(f"{noise=}")
        
        self.actor_model.eval()
        with torch.no_grad():
            result = self.actor_model(state) 
        self.actor_model.train()
        result += noise
        
        n = self.action_size // 4
        
        # output is of size 4 * n stocks
        # take the first n stocks and apply softmax
        weights = torch.softmax(result[:n], dim=0)
        # reshape the 3 * n stocks to (n, 3)
        positions = result[n:].reshape(-1, 3)
        # apply softmax to the positions
        positions = torch.softmax(positions, dim=1)
        # perform argmax to get the position
        final_pos = torch.argmax(positions, dim=1)
        
        raw_action = result.detach().cpu().numpy()
        action = np.concatenate((weights.detach().cpu().numpy(), final_pos.detach().cpu().numpy()), axis=0)
        return action, raw_action

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
            next_q_vals = self.critic_target_model(next_states_tensor, raw_actions_next)
            # compute target reward
            target_q_vals = rewards_tensor + self.gamma * next_q_vals * (1.0 - terminals_tensor)
            
        # calc loss            
        self.critic_model.train()
        q_vals = self.critic_model(states_tensor, raw_actions_tensor)
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
        
    
def train_loop_DDPG(model, env, num_episodes, num_timesteps):
    for episode in range(num_episodes):
        # get inital state??
        s_init = None
        s = s_init
        for t in range(num_timesteps):
            a = model.get_action(s)
            # something to get details from env
            r, s_next = env.step(a)
            
            # add data to replay buffer
            if t < num_timesteps - 1:
                model.buffer.add(s, a, r, s_next, False)
            else:
                model.buffer.add(s, a, r, s_next, True)

            # update policy
            model.update_policy()
            
            # soft update the target networks
            model.soft_update(model.actor_target_model, model.actor_model)
            model.soft_update(model.critic_target_model, model.critic_model)
            
            

