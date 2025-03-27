import torch
from torch import nn
import numpy as np


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes):
        super().__init__()
        self.logit_net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU)
        self.mu_net = nn.Linear(list(hidden_sizes)[-1], act_dim)
        self.log_std_net = nn.Linear(list(hidden_sizes)[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        logits = self.logit_net(obs)
        mu = self.mu_net(logits)
        std = torch.exp(torch.clamp(self.log_std_net(logits), -20.0, 2.0))
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(dim=-1)
        logp -= (2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action))).sum(dim=1)
        action = self.act_limit * torch.tanh(action)
        return action, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q_net = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.ReLU)

    def forward(self, obs, act):
        q = self.q_net(torch.cat([obs, act], dim=-1))
        return q.squeeze(-1)