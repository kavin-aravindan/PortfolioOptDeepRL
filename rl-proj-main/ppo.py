import torch
import torch.nn as nn
import numpy as np
from gym_env import portfolioEnv
from data_utils import preprocess

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1=400, hidden2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self.layers = nn.Sequential(self.fc1, nn.ReLU(),
                                    self.fc2, nn.ReLU(),
                                    self.fc3)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, s):
        mean = self.layers(s)
        std = torch.clamp(torch.exp(self.log_std), min=1e-3, max=1.0)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden1=400, hidden2=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, s):
        return self.net(s)


class PPO():
    def __init__(self, env, action_dim, obs_dim, max_steps_per_ep, time_steps_per_batch, lr=0.005, gamma=0.99, clip=0.2):
        self.env = env
        self.max_steps_per_ep = max_steps_per_ep
        self.time_steps_per_batch = time_steps_per_batch
        self.updates_per_iter = 5
        self.gamma = gamma
        self.lr = lr
        self.clip = clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.actor.to(self.device)
        self.critic.to(self.device)


    def get_action(self, obs):
        state = torch.from_numpy(obs).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            mean, std = self.actor(state)
        self.actor.train()

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob = log_prob.detach().to('cpu').numpy()

        # process action
        n = action.shape[0] // 4
        weights = torch.softmax(action[:n], dim=0)
        positions_logits = action[n:].reshape(n, 3)
        positions = torch.softmax(positions_logits, dim=1)
        positions = torch.argmax(positions, dim=1)
        real_action = np.concatenate((weights.detach().to('cpu').numpy(), positions.detach().to('cpu').numpy()), axis=0)

        action = action.detach().to('cpu').numpy()
        return action, log_prob, real_action


    def rollout(self):
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch


        t = 0
        while t < self.time_steps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_steps_per_ep):
                t += 1
                batch_obs.append(obs)
                action, log_prob, real_action = self.get_action(obs)
                obs, reward, done, _, _ = self.env.step(real_action)

                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rews.append(reward)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for reward in reversed(ep_rews):
                discounted_reward = reward + (self.gamma * discounted_reward)
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts, nograd=True):
        batch_obs = batch_obs.to(self.device)
        batch_acts = batch_acts.to(self.device)

        if nograd:
            self.actor.eval()
            self.critic.eval()
            with torch.no_grad():
                V = self.critic(batch_obs).squeeze()
                mean, std = self.actor(batch_obs)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(batch_acts).sum(dim=-1)
            
            self.actor.train()
            self.critic.train()
        else:
            V = self.critic(batch_obs).squeeze()
            mean, std = self.actor(batch_obs)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(batch_acts).sum(dim=-1)

        return V, log_probs

    def learn(self, total_timesteps):

        # logging
        all_batch_rewards = []
        all_actor_losses = []
        all_critic_losses = []
        
        t_sofar = 0
        while t_sofar < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_sofar += sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)
            V = V.detach()
            batch_rtgs = batch_rtgs.to(self.device)
            A_k = batch_rtgs - V
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            batch_log_probs = batch_log_probs.to(self.device)

            actor_losses = []
            critic_losses = []

            # update actor & critic
            for i in range(self.updates_per_iter):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts, nograd=False)
                # curr_log_probs = curr_log_probs.detach().to('cpu')
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                # update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
            
            all_actor_losses.append(np.mean(actor_losses))
            all_critic_losses.append(np.mean(critic_losses))
            all_batch_rewards.append(np.mean(batch_rtgs.detach().to('cpu').numpy()))

            print(f"Total Timesteps: {t_sofar} | Actor Loss: {np.mean(actor_losses):.4f} | Critic Loss: {np.mean(critic_losses):.4f} | Batch Reward: {np.mean(batch_rtgs.detach().to('cpu').numpy()):.4f}")


if __name__ == "__main__":
    file_path = "sp500_data/sp500_stockwise.csv"
    data_df_train = preprocess(file_path)
    NUM_STEPS = 252 * 5
    data_df_train = data_df_train.iloc[:NUM_STEPS, :]

    lookback_window_train = 60
    initial_cash_train = 100000
    env_train = portfolioEnv(data_df=data_df_train,
                             init_cash=initial_cash_train,
                             lookback_period=lookback_window_train) 

    state_size_train = env_train.observation_space.shape[0]
    action_size_train = env_train.action_space.shape[0] * 2

    ppo_agent_train = PPO(env=env_train,
                            action_dim=action_size_train,
                            obs_dim=state_size_train,
                            max_steps_per_ep=1000,
                            time_steps_per_batch=4000,
                            lr=0.005,
                            gamma=0.95,
                            clip=0.2)
    
    total_timesteps = 100000
    ppo_agent_train.learn(total_timesteps=total_timesteps)
