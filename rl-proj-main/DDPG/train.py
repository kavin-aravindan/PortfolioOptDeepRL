from env import Env
from DDPG_new import DDPG
from data import load_data
import numpy as np
from tqdm import tqdm


def main():
    data = load_data('data/sp5.csv')
    data = data[:5]

    env = Env(data)
    agent = DDPG(env, len(data) * 4 * 60, len(data))

    episodes = 250
    for _ in tqdm(range(episodes)):
        rewards, critic_loss, actor_loss, actions = agent.episode()
        # if _ % 100 == 0:
        print(f"Episode {_} completed")

        print(f"{np.mean(rewards)} / {np.std(rewards)} | {10_000_000.0 + np.sum(rewards)}")
        print(f"{critic_loss} | {actor_loss}")
        
    print(f"Actions: {actions}")


if __name__ == "__main__":
    main()
