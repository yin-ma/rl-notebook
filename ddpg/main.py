import torch
from torch import nn
import numpy as np
import gymnasium as gym
from copy import deepcopy

from buffer import Buffer
from neural_network import Actor, Critic


# hyperparameters
buffer_size = int(1e6)
hidden_sizes = [64, 64, 64]
steps_per_epoch = 2000
epochs = 500
gamma = 0.99
polyak = 0.995
policy_lr = 3e-4
q_lr = 3e-4
batch_size = 100
max_ep_len = 2000

start_step = 10000
update_after = 1000
update_every = 50
update_iteration = 10


def update(buffer, actor, critic, actor_target, critic_target, actor_optim, critic_optim):
    # data
    obs, act, rew, obs2, done = buffer.sample_batch(batch_size=batch_size)

    # update critic
    with torch.no_grad():
        y = rew + gamma * (1 - done) * critic_target(obs2, actor_target(obs2).rsample())
    loss_q = ((critic(obs, act) - y) ** 2).mean()

    critic_optim.zero_grad()
    loss_q.backward()
    critic_optim.step()

    # update actor
    for p in critic.parameters():
        p.requires_grad = False

    loss_pi = -critic(obs, actor(obs).rsample()).mean()
    actor_optim.zero_grad()
    loss_pi.backward()
    actor_optim.step()

    for p in critic.parameters():
        p.requires_grad = True

    # update target network
    with torch.no_grad():
        for p, p_targ in zip(actor.parameters(), actor_target.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

        for p, p_targ in zip(critic.parameters(), critic_target.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)


def learn(env, buffer, actor, critic, actor_target, critic_target, actor_optim, critic_optim):
    total_steps = steps_per_epoch * epochs
    obs, _ = env.reset()
    ep_len, ep_ret = 0, 0
    avg_ret = 0

    # game loop
    for t in range(total_steps):
        if t > start_step:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs)
                dist = actor(obs_tensor)
                action = dist.sample()
                a = action.numpy()
        else:
            a = env.action_space.sample()

        obs2, rew, done, _, _ = env.step(a)
        ep_ret += rew
        ep_len += 1
        done = False if ep_len == max_ep_len else done

        buffer.store(obs, a, rew, obs2, done)
        obs = obs2

        # if end of game, update env, para...
        if (done) or (ep_len == max_ep_len):
            obs, _ = env.reset()
            avg_ret += ep_ret
            ep_len = 0
            ep_ret = 0

        # if time to update ac, update ac
        if t >= update_after and t % update_every == 0:
            for _ in range(update_iteration):
                update(buffer, actor, critic, actor_target, critic_target, actor_optim, critic_optim)

        # if epoch updated, print ret
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            print(f"----- ep{epoch} ----")
            print(f"avg_ret : {avg_ret}\n")
            avg_ret = 0


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    buffer = Buffer(obs_dim, act_dim, buffer_size)

    actor = Actor(obs_dim, act_dim, hidden_sizes, act_limit)
    critic = Critic(obs_dim, act_dim, hidden_sizes)

    actor_target = deepcopy(actor)
    critic_target = deepcopy(critic)

    for p in actor_target.parameters():
        p.requires_grad = False
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_optim = torch.optim.Adam(actor.parameters(), lr=policy_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=q_lr)

    learn(env, buffer, actor, critic, actor_target, critic_target, actor_optim, critic_optim)




