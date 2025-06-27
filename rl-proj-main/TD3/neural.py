# neural.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n, in_feat, h=128):
        super(Actor, self).__init__()

        # self.norm = nn.LayerNorm(n * in_feat)
        # self.fc1 = nn.Linear(n * in_feat, h)
        
        # self.lstm = nn.LSTM(h, h, batch_first=True)

        self.encoder = nn.Linear(n * in_feat * 60, h)
        
        self.fc_z = nn.Linear(h, n)
        self.log_std = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        b, n, t, in_feat = x.size()
        # x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, t * n * in_feat)

        # x = self.norm(x)
        # x = F.relu(self.fc1(x))
        # x, _ = self.lstm(x)

        # z = self.fc_z(x[:, -1, :])

        x = F.relu(self.encoder(x))
        z = self.fc_z(x)
        std = torch.exp(self.log_std.expand_as(z))

        return z, std
    
    def get_action(self, state):
        """
        Same as before: sample from Normal(z, std),
        then normalize to weights w on the simplex.
        """
        z, std = self.forward(state)
        dist = torch.distributions.Normal(z, std)
        z_sample = dist.sample()

        w = z_sample / (torch.sum(torch.abs(z_sample), dim=-1, keepdim=True) + 1e-8)
        log_prob = dist.log_prob(w).sum(dim=-1)
        return w, log_prob


class Critic(nn.Module):
    """
    Q‐network for TD3: takes both state x and action w, 
    shares the same LSTM backbone as your V‐critic, then
    concatenates an action embedding before the final head.
    """
    def __init__(self, n, in_feat, h=128):
        super(Critic, self).__init__()

        # shared state‐backbone
        # self.norm = nn.LayerNorm(n * in_feat)
        # self.fc   = nn.Linear(n * in_feat, h)
        # self.lstm = nn.LSTM(h, h, batch_first=True)
        self.encoder = nn.Linear(n * in_feat * 60, h)

        # action embedding
        self.action_fc = nn.Linear(n, h)

        # final Q‐value head
        self.out = nn.Linear(h * 2, 1)

    def forward(self, x, action):
        """
        x:      (B, n, t, in_feat)
        action: (B, n)
        returns Q(x, action): (B, 1)
        """
        b, n, t, in_feat = x.size()
        # reshape state into sequence
        # seq = x.permute(0, 2, 1, 3).reshape(b, t, n * in_feat)
        x = x.reshape(b, n * t * in_feat)

        # seq = self.norm(seq)
        # seq = F.relu(self.fc(seq))
        # seq, _ = self.lstm(seq)
        # seq = seq[:, -1, :]                # (B, h)

        seq = F.relu(self.encoder(x))       # (B, h)

        # embed the action
        a_emb = F.relu(self.action_fc(action))  # (B, h)

        # concat and output
        cat = torch.cat([seq, a_emb], dim=-1)   # (B, 2h)
        return self.out(cat)                    # (B, 1)