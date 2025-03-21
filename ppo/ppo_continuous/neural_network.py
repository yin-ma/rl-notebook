import torch
from torch import nn


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.logit_net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU)

        self.mu = nn.Linear(list(hidden_sizes)[-1], act_dim)
        self.std = nn.Linear(list(hidden_sizes)[-1], act_dim)

    def forward(self, obs):
        logits = self.logit_net(obs)
        mu = 2.0 * torch.tanh(self.mu(logits))
        std = torch.clamp(self.std(logits), -20, -0.5)
        cov = torch.diag_embed(torch.exp(std))
        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)

        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1])

    def forward(self, obs):
        v = self.v_net(obs)
        return v
