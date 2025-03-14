import numpy as np
import torch

from utils import discounted_cumulative_sums


class Buffer:
    def __init__(self, obs_dim, size, gamma=0.99, lam=0.95):
        self.obs_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros(size, dtype=np.float32)
        self.adv_buffer = np.zeros(size, dtype=np.float32)
        self.rew_buffer = np.zeros(size, dtype=np.float32)
        self.ret_buffer = np.zeros(size, dtype=np.float32)
        self.val_buffer = np.zeros(size, dtype=np.float32)
        self.logp_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.tra_start_idx = 0, 0

    def store(self, obs, act, rew, val, logp):
        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act
        self.rew_buffer[self.ptr] = rew
        self.val_buffer[self.ptr] = val
        self.logp_buffer[self.ptr] = logp
        self.ptr += 1

    def finish_trajectory(self, last_value=0):
        tra_slice = slice(self.tra_start_idx, self.ptr)
        rews = np.append(self.rew_buffer[tra_slice], last_value)
        vals = np.append(self.val_buffer[tra_slice], last_value)

        # A = Q - V
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        # gae
        self.adv_buffer[tra_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.ret_buffer[tra_slice] = discounted_cumulative_sums(rews, self.gamma)[:-1]

        self.tra_start_idx = self.ptr

    def get(self):
        self.ptr, self.tra_start_idx = 0, 0

        # normalize
        adv_mean, adv_std = (np.mean(self.adv_buffer), np.std(self.adv_buffer))
        self.adv_buffer = (self.adv_buffer - adv_mean) / (adv_std + 1e-8)

        return (
            torch.as_tensor(self.obs_buffer),
            torch.as_tensor(self.act_buffer),
            torch.as_tensor(self.adv_buffer),
            torch.as_tensor(self.ret_buffer),
            torch.as_tensor(self.logp_buffer)
        )