import gymnasium as gym
from tqdm import tqdm
import torch

from buffer import Buffer
from neural_network import Actor, Critic
from utils import logprobabilities, sample_action


# hyperparameters
steps_per_epoch = 4000
epochs = 10
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iteration = 80
train_value_iteration = 80
lam = 0.97
hidden_sizes = [64, 64]


def train_policy(actor, policy_optimizer, obs_buffer, act_buffer, logp_buffer, adv_buffer):
    logp = logprobabilities(actor(obs_buffer), act_buffer)
    ratio = torch.exp(logp - logp_buffer)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_buffer
    loss_pi = -(torch.min(ratio * adv_buffer, clip_adv)).mean()

    policy_optimizer.zero_grad()
    loss_pi.backward()
    policy_optimizer.step()

    approx_kl = (logp - logp_buffer).mean().item()
    return approx_kl


def train_value_function(critic, value_optimizer, obs_buffer, ret_buffer):
    value_loss = torch.mean(torch.pow(ret_buffer - critic(obs_buffer), 2))
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()


def train(env, buffer, actor, critic, policy_optimizer, value_optimizer):
    obs, _ = env.reset()
    episode_return, episode_length = 0, 0

    actor.train()
    critic.train()

    loop = tqdm(range(epochs))
    for epoch in loop:
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        for t in range(steps_per_epoch):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs).unsqueeze(0)
                logits, action = sample_action(actor, obs_tensor)

                obs_new, rew, done, _, _ = env.step(action.item())
                episode_return += rew
                episode_length += 1

                value_t = critic(obs_tensor)
                logp_t = logprobabilities(logits, action)

                buffer.store(obs, action, rew, value_t, logp_t)
                obs = obs_new

                if done or (t == steps_per_epoch - 1):
                    last_value = 0 if done else critic(obs_tensor).item()
                    buffer.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    obs, _ = env.reset()
                    episode_return, episode_length = 0, 0

        (
            obs_buffer,
            act_buffer,
            adv_buffer,
            ret_buffer,
            logp_buffer,
        ) = buffer.get()

        for _ in range(train_policy_iteration):
            kl = train_policy(actor, policy_optimizer, obs_buffer, act_buffer, logp_buffer, adv_buffer)
            if kl > 1.5 * 0.01:
                break

        for _ in range(train_value_iteration):
            train_value_function(critic, value_optimizer, obs_buffer, ret_buffer)

        loop.set_postfix(mean_return=sum_return / num_episodes, mean_length=sum_length / num_episodes)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    act_space = env.action_space

    buffer = Buffer(obs_space.shape[0], steps_per_epoch)
    actor = Actor(obs_space.shape[0], act_space.n, hidden_sizes)
    critic = Critic(obs_space.shape[0], hidden_sizes)

    policy_optimizer = torch.optim.Adam(actor.parameters(), lr=policy_learning_rate)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=value_function_learning_rate)

    train(env, buffer, actor, critic, policy_optimizer, value_optimizer)

