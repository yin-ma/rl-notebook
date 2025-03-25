import torch
from torch import nn


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.logit_net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU)
        self.mu = nn.Linear(list(hidden_sizes)[-1], act_dim)
        self.std = nn.Linear(list(hidden_sizes)[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        logits = self.logit_net(obs)
        mu = self.mu(logits)
        std = torch.exp(torch.clamp(self.std(logits), -20.0, 2.0))
        dist = torch.distributions.Normal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q_net = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.ReLU)

    def forward(self, obs, act):
        q = self.q_net(torch.cat([obs, act], dim=-1))
        return q