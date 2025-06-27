import argparse
from env import Env
from TD3 import TD3
from data import load_data
import numpy as np
from tqdm import tqdm
import json, os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str,       required=True, help="Experiment name")
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic",type=float, default=1e-3)
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--tau",      type=float, default=1e-3)
    p.add_argument("--noise",    type=float, default=0.2)
    p.add_argument("--delay",    type=int,   default=2)
    p.add_argument("--hidden",   type=int,   default=128)
    return p.parse_args()

def main():
    args = parse_args()
    data = load_data('data/sp500.csv')
    data = data[:5]
    env = Env(data)
    # state_size = n_assets * features * lookback (unchanged from DDPG)
    # state_size  = len(data) * 4 * 60
    # action_size = len(data)
    agent= TD3(
        env,
        n_assets=len(data), in_feat=4,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.noise,
        policy_delay=args.delay,
        hidden_dim=args.hidden
    )

    episodes = 10000
    for ep in tqdm(range(episodes)):
        rewards = agent.episode()
        print(f"Episode {ep} completed")
        print(f"{np.mean(rewards)} / {np.std(rewards)} | {10_000_000.0 + np.sum(rewards)}")

        if (ep +1) % 50 == 0:
            metrics = {
                "all_step_rewards":          agent.all_step_rewards,
                "episode_sums":              agent.episode_sums,
                "episode_means":             agent.episode_means,
                "episode_vars":              agent.episode_vars,
                "actor_losses":              agent.actor_losses,
                "critic_losses":             agent.critic_losses,
            }

            # 1) write to <exp>/metrics.json
            metrics_dir = os.path.join(f"logs/{args.exp}")
            metrics_path = os.path.join(metrics_dir, "metrics.json")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)


    
   
    metrics = {
        "all_step_rewards":          agent.all_step_rewards,
        "episode_sums":              agent.episode_sums,
        "episode_means":             agent.episode_means,
        "episode_vars":              agent.episode_vars,
        "actor_losses":              agent.actor_losses,
        "critic_losses":             agent.critic_losses,
    }

    # 3) write to <exp>/metrics.json
    metrics_dir = os.path.join(f"logs/{args.exp}")
    metrics_path = os.path.join(metrics_dir, "metrics.json")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()