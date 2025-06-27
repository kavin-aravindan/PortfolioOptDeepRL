from env import Env
from PPO import PPO
from data import load_data
import numpy as np
from tqdm import tqdm


def main():
    data = load_data('data/sp5.csv')
    data = data[:5]


    env = Env(data)
    agent = PPO(env, len(data) * 4 * 60, len(data))

    episodes = 200
    for _ in tqdm(range(episodes)):
        rewards, a_loss, c_loss, actions = agent.episode()
        # if _ % 100 == 0:
        print(f"Episode {_} completed")

        print(f"{np.mean(rewards)} / {np.std(rewards)} | {10_000_000.0 + np.sum(rewards)}")
        print(f"{a_loss} | {c_loss}")

        if _ == 109:
            print(actions)

if __name__ == "__main__":
    main()
