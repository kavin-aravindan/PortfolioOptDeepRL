import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# Assuming state_utils.py and reward_utils.py are accessible
from state_utils import make_state
from reward_utils import reward as reward_func


class portfolioEnv(gym.Env):
    
    def __init__(self, data_df, init_cash=100000.0, transaction_cost_bp = 1e-3, min_period=122, lookback_period=60, bankrupt_reward=-10.0):
        super().__init__()
        self.asset_names = list(data_df.keys())
        self.n_assets = len(self.asset_names)
        self.data_df = data_df
        self.initial_balance = init_cash
        self.transaction_cost_bp = transaction_cost_bp
        
        self.total_period = len(self.data_df)
        self.min_period = min_period
        self.lookback_period = lookback_period
        self.bankrupt_reward = bankrupt_reward
        
        if self.total_period < self.min_period:
            raise ValueError(f'Minimum period is {self.min_period}')
        
        # state space
        self.state_size = self.n_assets * self.lookback_period * 4
        # not sure what i should define the limits as here
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_size,), dtype=np.float32)
        
        # action space
        self.action_size = 2 * self.n_assets
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.action_size,), dtype=np.float32)
        
        # init other vars
        self.current_step = 0
        self.current_cash = self.initial_balance
        self.current_positions = np.zeros(self.n_assets, dtype=np.float32)
        self.current_portfolio_value = self.initial_balance
        self.current_weights = np.zeros(self.n_assets, dtype=np.float32) 
        
        print(f"PortfolioEnv Initialized: {self.n_assets} assets, {self.total_period} steps.")
        print(f"Initial Balance: {self.initial_balance:.2f}")
        print(f"Trading starts after step {self.min_period - 1}.")
        print(f"Observation space shape: {self.observation_space.shape}")
        print(f"Action space shape: {self.action_space.shape}")
        
    def _get_obs(self):
        end_idx = self.current_step + 1
        if end_idx < self.min_period:
            raise RuntimeError(f"Cannot get observation at step {self.current_step}, requires {self.min_period}.")

        current_data_dict = {
            asset: self.data_df[asset].iloc[:end_idx].tolist()
            for asset in self.asset_names
        }
        # print(f"Current data dict: {current_data_dict}")
        market_state_features = make_state(current_data_dict)
        return market_state_features.flatten()
        
        
    def _process_action(self, action):
        # convert the action returned by ddpg to appropriate scale
        # first n correspond to weights, next n to pos
        
        # need the weights to sum to 1. apply softmax
        weight_action = np.asarray(action[:self.n_assets], dtype=np.float32)
        pos_action = np.asarray(action[self.n_assets:], dtype=np.float32)
        return weight_action, pos_action
    
    def _get_info(self):
        return {
        "current_step": self.current_step,
        "current_cash": self.current_cash,
        "current_shares": self.current_positions.tolist(),
        "current_weights": self.current_weights.tolist(), 
        "portfolio_value": self.current_portfolio_value
    }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.min_period - 1
        self.current_cash = self.initial_balance
        self.current_positions = np.zeros(self.n_assets, dtype=np.float32)
        self.current_portfolio_value = self.initial_balance
        self.current_weights = np.zeros(self.n_assets, dtype=np.float32) 
        self.previous_metric_value = self.initial_balance

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
        
    def step(self, action):
        # DO WE WANT TO TERMINATE IF WE'RE BROKE????? YESS BROKE PEOPLE HAVE NO RIGHTS 
        
        # Store state from start of step
        portfolio_value_start = self.current_portfolio_value
        positions_start = np.copy(self.current_positions)
        
        terminated = False

        # Get prices for current step
        self.current_step += 1
        
        # if at end of data return last obs
        if self.current_step >= self.total_period:
            truncated = True
            terminated = False
            self.current_step -= 1
            last_observation = self._get_obs()
            self.current_step += 1
            reward = 0.0
            info = self._get_info()
            info["error"] = "Reached end of data"
            return last_observation, reward, terminated, truncated, info

        current_prices = self.data_df.iloc[self.current_step].values.astype(np.float32)
        # if np.any(current_prices <= 1e-9):
        #     current_prices = np.maximum(current_prices, 1e-9)

        # reward_calc
        position_before = np.sign(positions_start)
        # need to re-calc weights according to new portfolio value
        weights_before = self.current_weights.copy()
        # take action 
        weights_after, position_after = self._process_action(action)        
        position_after = np.sign(position_after)

        portfolio_value_after, transaction_costs = reward_func(cash_t=self.current_cash, prices_t=current_prices, w_t_1=weights_before, pos_t_1=position_before,           
            w_t_2=weights_after, pos_t_2=position_after, bp=self.transaction_cost_bp)
        
        reward = portfolio_value_after - transaction_costs

        # update current cash 

        if self.current_portfolio_value <= 1e-9:
            reward = self.bankrupt_reward
            terminated = True

        truncated = (self.current_step >= self.total_period - 1)
        
        if terminated:
            truncated = False

        observation = self._get_obs()

        info = self._get_info()
        info["reward"] = reward
        info["transaction_costs"] = transaction_costs
        info["target_weights"] = weights_after.tolist()
        info["target_positions_scaling"] = position_after.tolist()

        # Return terminated as False
        return observation, float(reward), terminated, truncated, info
        

