import gymnasium as gym
import torch

from buffer import Buffer
from neural_network import Actor, Critic


# hyperparameters
steps_per_epoch = 400
epochs = 1000
gamma = 0.99
clip_ratio = 0.2
pi_lr = 3e-4
v_lr = 1e-3
train_pi_iteration = 10
train_v_iteration = 10
lam = 0.97
hidden_sizes = [64, 64, 64, 64]
buffer_size = 400


# init
env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

buffer = Buffer(obs_dim, act_dim, buffer_size, gamma=gamma, lam=lam)

actor = Actor(obs_dim, act_dim, hidden_sizes)
critic = Critic(obs_dim, hidden_sizes)

actor_optim = torch.optim.Adam(actor.parameters(), lr=pi_lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=v_lr)


def compute_loss_pi(data):
    obs, act, _, adv, logp_old = data

    logp = actor.log_prob(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    approx_kl = (logp_old - logp).mean().item()
    return loss_pi, approx_kl


def compute_loss_v(data):
    obs, _, ret, _, _ = data
    v = critic(obs)
    loss_v = ((v - ret) ** 2).mean()
    return loss_v


def update():
    data = buffer.get()

    for _ in range(train_pi_iteration):
        actor_optim.zero_grad()
        loss_pi, kl = compute_loss_pi(data)
        if kl > 1.5 * 0.01:
            break
        loss_pi.backward()
        actor_optim.step()

    for _ in range(train_v_iteration):
        critic_optim.zero_grad()
        loss_v = compute_loss_v(data)
        loss_v.backward()
        critic_optim.step()


def learn():
    obs, _ = env.reset()
    for ep in range(epochs):
        ep_ret, ep_len = 0, 0
        avg_ret, avg_len = 0, 0
        for t in range(steps_per_epoch):
            obs_tensor = torch.as_tensor(obs).unsqueeze(0)
            with torch.no_grad():
                act, logp_old = actor.sample(obs_tensor)
                val = critic(obs_tensor)
                a = act.cpu().numpy()[0]
            obs2, rew, done, _, _ = env.step(a)
            ep_ret += rew
            ep_len += 1

            buffer.store(obs, a, rew, val, logp_old)
            obs = obs2

            if done or (t == steps_per_epoch - 1):
                avg_ret += ep_ret
                avg_len += 1
                ep_ret, ep_len = 0, 0
                if done:
                    last_value = 0
                else:
                    with torch.no_grad():
                        last_value = critic(torch.as_tensor(obs))
                buffer.finish_path(last_val=last_value)
                obs, _ = env.reset()

        update()

        print(f"----- ep {ep} -----")
        print(f"avg_ret : {avg_ret / avg_len:.4f}")


if __name__ == "__main__":
    learn()