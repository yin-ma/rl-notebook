import scipy
import torch


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def logprobabilities(logits, a):
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(a)
    return logp


def sample_action(actor, obs):
    logits = actor(obs)
    action = torch.distributions.Categorical(logits=logits).sample()
    return logits, action
