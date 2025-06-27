import backtrader as bt
import torch
from utils import state_to_tensor


class Reinforce:
    def __init__(self, env, actor, critic, discount_factor=0.99):
        self.env = env
        self.discount_factor = discount_factor
        self.actor = actor
        self.critic = critic

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
    
    def act(self, state):
        input = state_to_tensor(state)
        w, log_prob = self.actor.get_action(input)
        return w.squeeze().detach().numpy()

    def step(self, state, reward):
        input = state_to_tensor(state)
        w, log_prob = self.actor.get_action(input)
        value = self.critic(input)
        
        self.actions.append(w)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
    
    def episode(self):
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

        self.env.simulate(self)
        self.update_policy()
        self.update_critic()

        return self.rewards
    
    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []

        for r in reversed(self.rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for log_prob, value, R in zip(self.log_probs, self.values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)

        self.actor_optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        self.actor_optimizer.step()
    
    def update_critic(self):
        returns = torch.tensor(self.rewards, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values = torch.stack(self.values).squeeze()
        critic_loss = (returns - values).pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
