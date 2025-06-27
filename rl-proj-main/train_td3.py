# train_TD3.py

import numpy as np
import torch
import time
from tqdm import tqdm

from TD3 import TD3Agent
from gym_env import portfolioEnv
from data_utils import preprocess  # returns a DataFrame keyed by asset


def train_loop_TD3(
    model: TD3Agent,
    env: portfolioEnv,
    num_episodes: int,
    max_steps_per_episode: int = 1000,
    log_interval: int = 10,
):
    print(f"Starting TD3 Training with portfolioEnv...")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch Size: {model.batch_size}")
    print(f"  Max Steps/Episode: {max_steps_per_episode if max_steps_per_episode else 'Env Limit'}")
    print(f"  Log Interval: {log_interval}")
    print(f"  Observation Space (Agent): {env.observation_space.shape}")
    print("-----------------------------------")

    total_steps = 0
    rewards_hist = []
    steps_hist = []
    final_val_hist = []
    avg_costs_hist = []

    start_time = time.time()

    for ep in tqdm(range(1, num_episodes + 1), desc="Episodes"):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_costs = []
        terminated = False
        truncated = False

        with tqdm(total=max_steps_per_episode, desc=f"Ep {ep}", leave=False) as pbar:
            while not (terminated or truncated):
                # choose action & get raw output
                action, raw_action = model.get_action(obs)
                # print(f"\nAction: {action}\nRaw:    {raw_action}")

                # step the env (Gymnasium style)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # store raw_action in replay buffer
                model.store(obs, raw_action, reward, next_obs, float(terminated))

                # train when ready
                if len(model.replay_buffer.buffer) >= model.batch_size:
                    model.train()
                    if total_steps % 100 == 0:
                        buf_size = len(model.replay_buffer.buffer)
                        print(f"[Train] step {total_steps} | buffer size {buf_size}")

                # log this step
                port_val = info.get("portfolio_value", np.nan)
                print(f"Ep {ep} | Step {ep_steps} | Reward {reward:.3f} | PortVal {port_val:.2f}")

                # update state & counters
                obs = next_obs
                ep_reward += reward
                total_steps += 1
                ep_steps += 1
                if "transaction_costs" in info:
                    ep_costs.append(info["transaction_costs"])

                # enforce max steps
                if ep_steps >= max_steps_per_episode:
                    truncated = True

                pbar.update(1)

        # end of episode
        avg_cost = np.mean(ep_costs) if ep_costs else 0.0
        rewards_hist.append(ep_reward)
        steps_hist.append(ep_steps)
        final_val_hist.append(info.get("portfolio_value", np.nan))
        avg_costs_hist.append(avg_cost)

        print(f"→ Episode {ep} done | Reward {ep_reward:.3f} | Final PortVal {final_val_hist[-1]:.2f}")

        # summary every log_interval
        if ep % log_interval == 0 or ep == num_episodes:
            recent_rewards = rewards_hist[-log_interval:]
            recent_vals    = final_val_hist[-log_interval:]
            print(
                f"[Summary] Episodes {ep-log_interval+1}–{ep} | "
                f"AvgReward {np.mean(recent_rewards):.3f} | "
                f"AvgPortVal {np.mean(recent_vals):.2f} | "
                f"AvgCost {np.mean(avg_costs_hist[-log_interval:]):.4f} | "
                f"Elapsed {time.time()-start_time:.1f}s"
            )

    # save models
    torch.save(model.actor.state_dict(),   "td3_actor.pth")
    torch.save(model.critic.state_dict(),  "td3_critic.pth")

    print("Training complete.")
    return {
        "rewards": rewards_hist,
        "steps": steps_hist,
        "final_values": final_val_hist,
        "avg_costs": avg_costs_hist,
    }


if __name__ == "__main__":
    # load & (optionally) truncate to 5 years
    df = preprocess("sp500_data/sp500_stockwise.csv")
    NUM_STEPS = 252 * 5
    df = df.iloc[:NUM_STEPS]

    env = portfolioEnv(
        data_df=df,
        init_cash=100_000,
        lookback_period=60,
    )

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        buffer_size=int(1e6),
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=1e-3,
        critic_lr=1e-3,
    )

    train_loop_TD3(
        model=agent,
        env=env,
        num_episodes=50,
        max_steps_per_episode=NUM_STEPS,
        log_interval=5,
    )