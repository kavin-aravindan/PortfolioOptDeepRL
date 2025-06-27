import numpy as np
import torch 
from collections import deque
import time
import pandas as pd 

from DDPG import DDPG
from gym_env import portfolioEnv
from data_utils import preprocess
from tqdm import tqdm

def train_loop_DDPG(model: DDPG,
                    env: portfolioEnv, 
                    num_episodes: int,
                    batch_size: int = 64,
                    max_steps_per_episode: int = 1000, 
                    log_interval: int = 10):

    print(f"Starting DDPG Training with portfolioEnv...")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Steps/Episode: {max_steps_per_episode if max_steps_per_episode else 'Env Limit'}")
    print(f"  Log Interval: {log_interval}")
    print(f"  Observation Space (Agent): {env.observation_space.shape}") # Verify obs space
    print("-----------------------------------")

    total_steps_taken = 0
    episode_rewards_history = []
    episode_steps_history = []
    episode_final_value_history = []
    episode_avg_costs_history = [] # Track average transaction costs per episode

    start_time = time.time()

    for episode in tqdm(range(1, num_episodes + 1)):
        # Reset env - observation contains only market features now
        observation, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_costs = [] 

        truncated = False
        terminated = False

        # Loop continues until truncated (end of data or max_steps_per_episode)
        with tqdm(total=max_steps_per_episode, position=2, leave=False) as pbar:
            while not truncated and not terminated:
                # Get action from DDPG agent using the market-feature-only observation
                action, raw_action = model.get_action(observation)
                print(f"Action: {action} | Raw Action: {raw_action}")

                #Environment Step
                next_observation, reward, terminated, truncated, info = env.step(action) 

                # Store the transition. 
                model.buffer.add(observation, raw_action, reward, next_observation, terminated)
                    

                # Update State and Counters
                print(f"Episode {episode} | Step {episode_steps} | Action: {action} | Reward: {reward:.2f} | Portfolio Value: {info.get('portfolio_value', np.nan):.2f}")
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                total_steps_taken += 1
                if 'transaction_costs' in info: # Store costs if available
                    episode_costs.append(info['transaction_costs'])

                # Training Step
                # Start training only if buffer has enough samples
                if len(model.buffer.buffer) >= batch_size:
                    model.update_policy()
                    # Soft update target networks
                    model.soft_update(model.actor_target_model, model.actor_model)
                    model.soft_update(model.critic_target_model, model.critic_model)

                # Check Max Steps 
                if max_steps_per_episode is not None and episode_steps >= max_steps_per_episode:
                    truncated = True 
                    
                pbar.update(1)


        episode_rewards_history.append(episode_reward)
        episode_steps_history.append(episode_steps)
        final_portfolio_value = info.get('portfolio_value', np.nan)
        episode_final_value_history.append(final_portfolio_value)
        avg_costs = np.mean(episode_costs) if episode_costs else 0.0
        episode_avg_costs_history.append(avg_costs)


        if episode % log_interval == 0 or episode == num_episodes:
            avg_reward = np.mean(episode_rewards_history[-log_interval:])
            avg_steps = np.mean(episode_steps_history[-log_interval:])
            avg_final_value = np.mean(episode_final_value_history[-log_interval:])
            avg_avg_costs = np.mean(episode_avg_costs_history[-log_interval:]) # Avg of episode averages

            elapsed_time = time.time() - start_time
            print(f"Ep {episode}/{num_episodes} | "
                  f"Steps (Total): {total_steps_taken} | "
                  f"Avg Reward (Last {log_interval}): {avg_reward:.3f} | "
                  f"Avg Final Value (Last {log_interval}): {avg_final_value:.2f} | "
                  f"Avg Costs (Last {log_interval}): {avg_avg_costs:.4f} | "
                  f"Time: {elapsed_time:.2f}s")

    print('-------------------------------')
    print("Training Finished.")
    env.close() 

    return {
        "rewards": episode_rewards_history,
        "steps": episode_steps_history,
        "final_values": episode_final_value_history,
        "avg_costs": episode_avg_costs_history
    }

if __name__ == '__main__':

    file_path = "sp500_data/sp500_stockwise.csv"
    data_df_train = preprocess(file_path)
    
    #truncate data to 252*5 = 1260 rows
    NUM_STEPS = 252 * 5
    data_df_train = data_df_train.iloc[:NUM_STEPS, :]

    lookback_window_train = 60
    initial_cash_train = 100000
    env_train = portfolioEnv(data_df=data_df_train,
                             init_cash=initial_cash_train,
                             lookback_period=lookback_window_train) 

    # init DDPG agent
    state_size_train = env_train.observation_space.shape[0]
    action_size_train = env_train.action_space.shape[0] * 2

    ddpg_agent_train = DDPG(state_size=state_size_train,
                            action_size=action_size_train,
                            buffer_size=100000,
                            batch_size=64,
                            gamma=0.99,
                            tau=0.005,
                            lr_actor=1e-4,
                            lr_critic=1e-3,
                            ou_theta=0.15,
                            ou_sigma=0.2,
                            weight_decay=0.0)

    num_episodes_train_run = 50
    batch_size_train_run = 64

    training_history = train_loop_DDPG(
        model=ddpg_agent_train,
        env=env_train,
        num_episodes=num_episodes_train_run,
        batch_size=batch_size_train_run,
        max_steps_per_episode=NUM_STEPS,
        log_interval=5
    )

    print("\n----------------------")
    print("Last 10 Episode Rewards:", training_history["rewards"][-10:])
    print("Last 10 Final Values:", training_history["final_values"][-10:])
    print("Last 10 Avg Costs:", training_history["avg_costs"][-10:])