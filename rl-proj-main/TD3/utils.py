import torch


def state_to_tensor(state):
    n = len(state)
    t = len(state[next(iter(state))]["prices"])
    in_feat = 4

    tensor_state = torch.zeros((n, t, in_feat))

    for i, (name, features) in enumerate(state.items()):
        prices = features["prices"]
        returns = features["returns"]
        macd = features["MACD"]
        rsi = features["RSI"]

        tensor_state[i, :, 0] = torch.tensor(prices, dtype=torch.float32)
        tensor_state[i, :, 1] = torch.tensor(returns, dtype=torch.float32)
        tensor_state[i, :, 2] = torch.tensor(macd, dtype=torch.float32)
        tensor_state[i, :, 3] = torch.tensor(rsi, dtype=torch.float32)
    
    return tensor_state.unsqueeze(0)


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def soft_update(net, target_net, tau):
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

def load_config(path):
    import yaml
    with open(path) as f:
        return type("Cfg", (), yaml.safe_load(f))


if __name__ == "__main__":
    state = {
        "AAPL": {
            "prices": [1, 2, 3, 4, 5],
            "returns": [0.1, 0.2, 0.3, 0.4, 0.5],
            "MACD": [0.01, 0.02, 0.03, 0.04, 0.05],
            "RSI": [30, 40, 50, 60, 70],
        },
        "GOOGL": {
            "prices": [1, 2, 3, 4, 5],
            "returns": [0.1, 0.2, 0.3, 0.4, 0.5],
            "MACD": [0.01, 0.02, 0.03, 0.04, 0.05],
            "RSI": [30, 40, 50, 60, 70],
        },
    }

    tensor_state = state_to_tensor(state)
    print(tensor_state.shape)
    print(tensor_state)
