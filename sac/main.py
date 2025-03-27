import torch
from torch import nn
import numpy as np
import gymnasium as gym
from copy import deepcopy

from buffer import Buffer
from network import Actor, Critic


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
buffer_size = int(1e6)
hidden_sizes = [64, 64]
steps_per_epoch = 200
epochs = 40
gamma = 0.99
polyak = 0.995
pi_lr = 1e-3
q_lr = 1e-3
batch_size = 100
max_ep_len = 200
alpha_init = 0.2
alpha_lr = 5e-3

start_step = 1000
update_after = 1000
update_every = 50
update_iteration = 50

# init
env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

buffer = Buffer(obs_dim, act_dim, buffer_size)

actor = Actor(obs_dim, act_dim, act_limit, hidden_sizes).to(device)
critic1 = Critic(obs_dim, act_dim, hidden_sizes).to(device)
critic2 = Critic(obs_dim, act_dim, hidden_sizes).to(device)
critic1_target = deepcopy(critic1)
critic2_target = deepcopy(critic2)

critic_para = list(critic1.parameters()) + list(critic2.parameters())
critic_target_para = list(critic1_target.parameters()) + list(critic2_target.parameters())

for p in critic_target_para:
    p.requires_grad = False

actor_optim = torch.optim.Adam(actor.parameters(), lr=pi_lr)
critic_optim = torch.optim.Adam(critic_para, lr=q_lr)

log_alpha = nn.Parameter(torch.as_tensor(np.log(alpha_init)))
alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
target_entropy = env.action_space.shape[0]


def update():
    obs, act, rew, obs2, done = buffer.sample_batch(batch_size=batch_size, device=device)

    alpha = torch.exp(log_alpha).detach()

    # update critic #
    with torch.no_grad():
        act2, logp2 = actor.sample(obs2)

        q1_target = critic1_target(obs2, act2)
        q2_target = critic2_target(obs2, act2)
        q_target = torch.min(q1_target, q2_target)
        y = rew + gamma * (1 - done) * (q_target - alpha * logp2)

    q1 = critic1(obs, act)
    q2 = critic2(obs, act)

    loss_q1 = ((q1 - y) ** 2).mean()
    loss_q2 = ((q2 - y) ** 2).mean()
    loss_q = loss_q1 + loss_q2

    critic_optim.zero_grad()
    loss_q.backward()
    critic_optim.step()
    # update critic #

    # update actor #
    for p in critic_para:
        p.requires_grad = False

    a, logp = actor(obs)
    q1 = critic1(obs, a)
    q2 = critic2(obs, a)
    q = torch.min(q1, q2)
    loss_pi = (alpha * logp - q).mean()

    actor_optim.zero_grad()
    loss_pi.backward()
    actor_optim.step()

    for p in critic_para:
        p.requires_grad = True
    # update actor #

    # update alpha #
    alpha_loss = (torch.exp(log_alpha) * (target_entropy - logp.detach())).mean()
    alpha_optim.zero_grad()
    alpha_loss.backward()
    alpha_optim.step()
    # update alpha #

    # update target network #
    with torch.no_grad():
        for p, p_targ in zip(critic_para, critic_target_para):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)
    # update target network #


def learn():
    total_steps = steps_per_epoch * epochs
    obs, _ = env.reset()
    ep_ret, ep_num = 0, 0
    tra_ret, tra_len = 0, 0

    for t in range(total_steps):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
        if t > start_step:
            with torch.no_grad():
                act, logp = actor.sample(obs_tensor)
                a = act.squeeze(-1).cpu().numpy()

        else:
            a = env.action_space.sample()

        obs2, rew, done, _, _ = env.step(a)
        tra_ret += rew
        tra_len += 1

        done = False if (tra_len == max_ep_len) else done
        buffer.store(obs, a, rew, obs2, done)
        obs = obs2

        if done or (tra_len == max_ep_len):
            obs, _ = env.reset()
            ep_ret += tra_ret
            ep_num += 1
            tra_len, tra_ret = 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_iteration):
                update()

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            avg_ret = ep_ret / ep_num

            print(f"----- ep{epoch} -----")
            print(f"avg_ret : {avg_ret:.4f}\n")
            ep_ret, ep_num = 0, 0


learn()
