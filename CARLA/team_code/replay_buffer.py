import numpy as np
import torch

class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 obs_space,
                 action_dim: int,
                 device: torch.device):

        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        bev_shape = obs_space['bev_semantics'].shape
        meas_shape = obs_space['measurements'].shape

        self.bev = np.zeros((capacity, *bev_shape), dtype=np.uint8)
        self.meas = np.zeros((capacity, *meas_shape), dtype=np.float32)

        self.next_bev = np.zeros((capacity, *bev_shape), dtype=np.uint8)
        self.next_meas = np.zeros((capacity, *meas_shape), dtype=np.float32)

        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.bev[self.ptr] = obs['bev_semantics']
        self.meas[self.ptr] = obs['measurements']

        self.next_bev[self.ptr] = next_obs['bev_semantics']
        self.next_meas[self.ptr] = next_obs['measurements']

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'obs': {
                'bev_semantics': torch.tensor(
                    self.bev[idxs], device=self.device, dtype=torch.float32
                ) / 255.0,
                'measurements': torch.tensor(
                    self.meas[idxs], device=self.device, dtype=torch.float32
                )
            },
            'actions': torch.tensor(
                self.actions[idxs], device=self.device, dtype=torch.float32
            ),
            'rewards': torch.tensor(
                self.rewards[idxs], device=self.device, dtype=torch.float32
            ),
            'next_obs': {
                'bev_semantics': torch.tensor(
                    self.next_bev[idxs], device=self.device, dtype=torch.float32
                ) / 255.0,
                'measurements': torch.tensor(
                    self.next_meas[idxs], device=self.device, dtype=torch.float32
                )
            },
            # hi vọng chạy được
            'dones': torch.tensor(
                self.dones[idxs], device=self.device, dtype=torch.float32
            )
        }
        return batch #return batch
    # --------- END OF FILE --------- #
