import torch
from torch import nn
import numpy as np
import gymnasium as gym
from copy import deepcopy
import itertools

from buffer import Buffer
from network import Actor, Critic


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
buffer_size = int(1e6)
hidden_sizes = [64, 64, 64]
steps_per_epoch = 1000
epochs = 1000
gamma = 0.99
alpha = 0.20
polyak = 0.995
policy_lr = 3e-4
q_lr = 3e-4
batch_size = 100
max_ep_len = 1000

start_step = 3000
update_after = 1000
update_every = 50
update_iteration = 10


def update(env, buffer, actor, critic1, critic2, critic1_target, critic2_target, actor_optim, critic_optim):
    obs, act, rew, obs2, done = buffer.sample_batch(batch_size=batch_size, device=device)
    act_limit = env.action_space.shape[0]
    critic_params = list(critic1.parameters()) + list(critic2.parameters())
    critic_params_target = list(critic1_target.parameters()) + list(critic2_target.parameters())

    # update critic
    with torch.no_grad():
        dist = actor(obs2)
        act2 = dist.rsample()
        logp_pi = dist.log_prob(act2).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - act2 - torch.nn.functional.softplus(-2 * act2))).sum(axis=1)
        act2 = act_limit * torch.tanh(act2)

        q1_target = critic1_target(obs2, act2)
        q2_target = critic2_target(obs2, act2)
        q_target = torch.min(q1_target, q2_target)
        y = rew + gamma * (1 - done) * (q_target - alpha * logp_pi)

    q1 = critic1(obs, act)
    q2 = critic2(obs, act)

    loss_q1 = ((q1 - y) ** 2).mean()
    loss_q2 = ((q2 - y) ** 2).mean()
    loss_q = loss_q1 + loss_q2

    critic_optim.zero_grad()
    loss_q.backward()
    critic_optim.step()

    # update actor
    for p in critic_params:
        p.requires_grad = False

    dist = actor(obs)
    act = dist.rsample()
    logp_pi = dist.log_prob(act).sum(axis=-1)
    logp_pi -= (2 * (np.log(2) - act - torch.nn.functional.softplus(-2 * act))).sum(axis=1)
    act = act_limit * torch.tanh(act)

    q1 = critic1(obs, act)
    q2 = critic2(obs, act)
    q = torch.min(q1, q2)

    loss_pi = (alpha * logp_pi - q).mean()

    actor_optim.zero_grad()
    loss_pi.backward()
    actor_optim.step()

    for p in critic_params:
        p.requires_grad = True

    # update target network
    with torch.no_grad():
        for p, p_targ in zip(critic_params, critic_params_target):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)


def learn(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.shape[0]

    # init buffer, actor, critic
    buffer = Buffer(obs_dim, act_dim, buffer_size)
    actor = Actor(obs_dim, act_dim, hidden_sizes, act_limit).to(device)
    critic1 = Critic(obs_dim, act_dim, hidden_sizes).to(device)
    critic2 = Critic(obs_dim, act_dim, hidden_sizes).to(device)

    critic1_target = deepcopy(critic1).to(device)
    critic2_target = deepcopy(critic2).to(device)

    for p1, p2 in zip(critic1_target.parameters(), critic2_target.parameters()):
        p1.requires_grad = False
        p2.requires_grad = False

    critic_params = list(critic1.parameters()) + list(critic2.parameters())
    critic_params_target = list(critic1_target.parameters()) + list(critic2_target.parameters())

    actor_optim = torch.optim.Adam(actor.parameters(), lr=policy_lr)
    critic_optim = torch.optim.Adam(critic_params, lr=q_lr)

    total_steps = steps_per_epoch * epochs
    obs, _ = env.reset()
    ep_ret, ep_len = 0, 0
    avg_ret = 0

    for t in range(total_steps):
        if t > start_step:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).to(device)
                dist = actor(obs_tensor)
                action = dist.rsample()
                action = act_limit * torch.tanh(action)
                a = action.cpu().numpy()
        else:
            a = env.action_space.sample()

        obs2, rew, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_ret += rew
        ep_len += 1
        done = False if ep_len == max_ep_len else done

        buffer.store(obs, a, rew, obs2, done)
        obs = obs2

        if done or (ep_len == max_ep_len):
            obs, _ = env.reset()
            avg_ret += ep_ret
            ep_ret, ep_len = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_iteration):
                update(env, buffer, actor, critic1, critic2, critic1_target, critic2_target, actor_optim, critic_optim)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            print(f"----- ep{epoch} ----")
            print(f"avg_ret : {avg_ret:.4f}\n")
            avg_ret = 0


if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous=True)
    learn(env)

