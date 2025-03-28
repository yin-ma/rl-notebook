import torch
from torch import nn


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.logit_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation=nn.ReLU)

    def forward(self, obs):
        logits = self.logit_net(obs)
        return logits

    def sample(self, obs):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp

    def log_prob(self, obs, act):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act)
        return logp


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation=nn.ReLU)

    def forward(self, obs):
        v = self.v_net(obs)
        return v.squeeze(dim=-1)
