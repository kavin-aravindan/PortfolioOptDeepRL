import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_tickers, in_features, hidden_dim=128):
        super(Actor, self).__init__()
        # takes input of size (n_tickers, n_days, in_features)
        # flatten to (n_tickers * n_days, in_features)
        # and then pass through an LSTM layer to get a hidden state
        # output is two tensors
        # one for weights and one for positions
        # the positions tensor is of size (n_tickers, 3)
        # the weights tensor is of size (n_tickers, 1)

        self.fc1 = nn.Linear(n_tickers * in_features, hidden_dim)
        
        self.w_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.pos_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.w_out = nn.Linear(hidden_dim, n_tickers)
        self.pos_out = nn.Linear(hidden_dim, n_tickers * 3)


    def forward(self, x):
        batch_size, n_tickers, n_days, in_features = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, n_days, n_tickers * in_features)


        x = F.relu(self.fc1(x))

        _x, _ = self.w_lstm(x)
        w = self.w_out(_x[:, -1, :])

        # # normalize w to (0, 1) and sum to 1 (not softmax)
        w = F.sigmoid(w)
        w_sum = torch.sum(w, dim=-1, keepdim=True)
        w = w / (w_sum + 1e-8)

        _x, _ = self.pos_lstm(x)
        pos = self.pos_out(_x[:, -1, :])
        pos = pos.view(batch_size, n_tickers, 3)
        
        # pos = F.softmax(pos, dim=-1)

        return w, pos


class Critic(nn.Module):
    def __init__(self, n_tickers, in_features, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_tickers * in_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        batch_size, n_tickers, n_days, in_features = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, n_days, n_tickers * in_features)

        x = F.relu(self.fc1(x))

        _x, _ = self.lstm(x)
        value = self.out(_x[:, -1, :])

        return value


if __name__ == "__main__":
    n_tickers = 3
    in_features = 4
    hidden_dim = 128

    actor = Actor(n_tickers, in_features, hidden_dim)
    critic = Critic(n_tickers, in_features, hidden_dim)

    x = torch.randn(2, n_tickers, 60, in_features)  # batch size of 32
    w, pos = actor(x)
    value = critic(x)

    print("Weights:", w.size())
    print("Positions:", pos.size())
    print("Value:", value.size())

    print(w)
