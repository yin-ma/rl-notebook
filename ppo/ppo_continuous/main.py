import gymnasium as gym
import torch
from torch import nn
import scipy
import numpy as np

from neural_network import Actor, Critic


# hyperparameters
steps_per_epoch = 2000
epochs = 800
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 3e-4
train_policy_iteration = 10
train_value_iteration = 10
lam = 0.97
hidden_sizes = [64, 64, 64]


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def rollout(env, actor, critic):
    obs_buff = np.zeros((steps_per_epoch, env.observation_space.shape[0]), dtype=np.float32)
    act_buff = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_buff = np.zeros(steps_per_epoch, dtype=np.float32)
    logp_buff = np.zeros(steps_per_epoch, dtype=np.float32)
    val_buff = np.zeros(steps_per_epoch, dtype=np.float32)

    obs, _ = env.reset()
    ep_rew = 0

    for t in range(steps_per_epoch):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs)
            dist = actor(obs_tensor)
            action = dist.sample()
            logp = dist.log_prob(action)
            val = critic(obs_tensor)
        obs2, rew, terminated, truncated, _ = env.step(action.numpy())

        obs_buff[t] = obs
        act_buff[t] = action.numpy().item()
        rew_buff[t] = rew
        logp_buff[t] = logp
        val_buff[t] = val

        ep_rew += rew
        obs = obs2
    print(f"ep_rew : {ep_rew}\n")
    return obs_buff, act_buff, rew_buff, logp_buff, val_buff


def learn(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim, hidden_sizes)
    critic = Critic(obs_dim, act_dim, hidden_sizes)
    policy_optimizer = torch.optim.Adam(actor.parameters(), lr=policy_learning_rate)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=value_function_learning_rate)

    for epoch in range(epochs):
        print(f"----- {epoch} -----")
        obs_buff, act_buff, rew_buff, logp_buff, val_buff = rollout(env, actor, critic)

        rews = np.append(rew_buff, 0)
        ret_buff = discounted_cumulative_sums(rews, gamma)[:-1].copy()

        # calculate advantage (gae)
        vals = np.append(val_buff, 0)
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        adv_buff = discounted_cumulative_sums(deltas, gamma * lam)
        adv_mean, adv_std = np.mean(adv_buff), np.std(adv_buff)
        adv_buff = (adv_buff - adv_mean) / (adv_std + 1e-12)

        obs_tens = torch.tensor(obs_buff, dtype=torch.float32)
        act_tens = torch.tensor(act_buff, dtype=torch.float32)
        ret_tens = torch.tensor(ret_buff, dtype=torch.float32)
        logp_tens = torch.tensor(logp_buff, dtype=torch.float32)
        adv_tens = torch.tensor(adv_buff, dtype=torch.float32)

        for _ in range(train_policy_iteration):
            dist = actor(obs_tens)
            new_logp = dist.log_prob(act_tens.unsqueeze(-1))

            ratios = torch.exp(new_logp - logp_tens)
            clip_adv = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * adv_tens
            act_loss = -(torch.min(ratios * adv_tens, clip_adv)).mean()

            policy_optimizer.zero_grad()
            act_loss.backward()
            policy_optimizer.step()

        for _ in range(train_value_iteration):
            val = critic(obs_tens)
            cri_loss = torch.mean((ret_tens - val) ** 2)

            value_optimizer.zero_grad()
            cri_loss.backward()
            value_optimizer.step()


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    learn(env)
