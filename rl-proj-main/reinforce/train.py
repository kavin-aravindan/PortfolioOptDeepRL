from env import Env
from reinforce import Reinforce
from data import load_data
from neural import Actor, Critic
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    data = load_data()
    data = data[:5]

    actor = Actor(n=len(data), in_feat=4, h=128)
    critic = Critic(n=len(data), in_feat=4, h=128)

    env = Env(data)
    agent = Reinforce(env, actor, critic)

    mean_rewards = []
    std_rewards = []
    sum_rewards = []
    policy_losses = []
    critic_losses = []

    episodes = 200
    for episode in tqdm(range(episodes)):
        rewards = agent.episode()
        # if _ % 100 == 0:
        # print(f"Episode {_} completed")

        print(f"{np.mean(rewards)} / {np.std(rewards)} | {10_000_000.0 + np.sum(rewards)}")

        print(f"Episode {episode} completed")

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        sum_reward = 10_000_000.0 + np.sum(rewards)
        
        print(f"{mean_reward} / {std_reward} | {sum_reward}")
        print("=" * 20)


    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards)
    plt.title('Mean Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    plt.savefig('mean_rewards.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(std_rewards)
    plt.title('Standard Deviation of Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Std Reward')
    plt.grid(True)
    plt.savefig('std_rewards.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(sum_rewards)
    plt.title('Sum of Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Sum Reward')
    plt.grid(True)
    plt.savefig('sum_rewards.png')
    plt.close()


if __name__ == "__main__":
    main()
