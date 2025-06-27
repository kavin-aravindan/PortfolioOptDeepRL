import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n, in_feat, h=128):
        super(Actor, self).__init__()

        self.norm = nn.LayerNorm(n * in_feat)
        self.fc1 = nn.Linear(n * in_feat, h)
        
        self.lstm = nn.LSTM(h, h, batch_first=True)
        
        self.fc_z = nn.Linear(h, n)
        self.log_std = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        b, n, t, in_feat = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, t, n * in_feat)

        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x, _ = self.lstm(x)

        z = self.fc_z(x[:, -1, :])
        std = torch.exp(self.log_std.expand_as(z))

        return z, std
    
    def get_action(self, state):
        z, std = self.forward(state)
        dist = torch.distributions.Normal(z, std)
        z_sample = dist.sample()

        w = z_sample / (torch.sum(torch.abs(z_sample), dim=-1, keepdim=True) + 1e-8)
        log_prob = dist.log_prob(w).sum(dim=-1)
        return w, log_prob


class Critic(nn.Module):
    def __init__(self, n, in_feat, h=128):
        super(Critic, self).__init__()

        self.norm = nn.LayerNorm(n * in_feat)
        self.fc = nn.Linear(n * in_feat, h)
        self.lstm = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        b, n, t, in_feat = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, t, n * in_feat)

        x = self.norm(x)
        x = F.relu(self.fc(x))

        _x, _ = self.lstm(x)
        value = self.out(_x[:, -1, :])

        return value


if __name__ == "__main__":
    n = 3
    in_feat = 4
    h = 128

    actor = Actor(n, in_feat, h)
    critic = Critic(n, in_feat, h)

    x = torch.randn(1, n, 60, in_feat)
    z, log_std = actor(x)
    value = critic(x)

    w = actor.get_action(x)
    print(w)

    print("Weights:", z.size())
    print("Log Std:", log_std.size())
    print("Value:", value.size())

    print(z)
    print(log_std)
