import gymnasium as gym
import numpy as np
from tqdm import tqdm


env_str = "Taxi-v3"  # 'CliffWalking-v0', 'FrozenLake-v1'
env = gym.make(env_str)
env_test = gym.make(env_str, render_mode="human")

observation_space = env.observation_space.n
action_space = env.action_space.n

print("observation space: ", observation_space)
print("action space: ", action_space)

# Q-table
Q = np.zeros((observation_space, action_space))

# parameters
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05
gamma = 0.999
alpha = 0.1

epoch = 1000
max_step = 200


def train(env):
    print("training...")
    global epsilon
    for i in tqdm(range(epoch)):
        observation, info = env.reset()
        terminated = False
        truncated = False
        curr_step = max_step
        while True:
            action = 0
            # epsilon greedy
            if np.random.random() > epsilon:
                action = np.argmax(Q[observation, :])
            else:
                action = env.action_space.sample()

            new_observation, reward, terminated, truncated, info = env.step(action)

            # Q_old = Q_old + alpha * (R + gamma * Q_new - Q_old)
            Q[observation, action] = Q[observation, action] \
                + alpha * (reward + gamma * np.max(Q[new_observation, :]) - Q[observation, action])

            observation = new_observation

            if terminated or truncated or curr_step <= 0:
                break

            curr_step -= 1
    epsilon = epsilon_decay * epsilon
    epsilon = max(epsilon_min, epsilon)


def test(env):
    observation, info = env.reset()
    terminated = False
    truncated = False
    curr_step = max_step

    while True:
        env.render()
        action = np.argmax(Q[observation, :])

        new_observation, reward, terminated, truncated, info = env.step(action)
        observation = new_observation

        if terminated or truncated or curr_step <= 0:
            break
        curr_step -= 1
    env.close()


if __name__ == "__main__":
    train(env)
    test(env_test)
