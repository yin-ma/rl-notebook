import numpy as np
import torch


class Buffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buff = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buff = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buff = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buff = np.zeros(size, dtype=np.float32)
        self.done_buff = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, obs2, done):
        self.obs_buff[self.ptr] = obs
        self.obs2_buff[self.ptr] = obs2
        self.act_buff[self.ptr] = act
        self.rew_buff[self.ptr] = rew
        self.done_buff[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min((self.size + 1), self.max_size)

    def sample_batch(self, batch_size=32, device="cpu"):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs_buff[idxs]).to(device),
            torch.as_tensor(self.act_buff[idxs]).to(device),
            torch.as_tensor(self.rew_buff[idxs]).to(device),
            torch.as_tensor(self.obs2_buff[idxs]).to(device),
            torch.as_tensor(self.done_buff[idxs]).to(device)
        )