import torch
from torch import nn
from copy import deepcopy
from model import XtMaCNN


class TD3Actor(nn.Module):
    def __init__(self, obs_space, action_dim, config):
        super().__init__()
        self.encoder = XtMaCNN(obs_space, states_neurons=(256, 256), config=config)

        self.policy = nn.Sequential(
            nn.Linear(config.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # TD3 expects [-1, 1]
        )

    def forward(self, obs):
        features = self.encoder(
            obs['bev_semantics'],
            obs['measurements']
        )
        return self.policy(features)


class TD3Critic(nn.Module):
    def __init__(self, obs_space, action_dim, config):
        super().__init__()
        self.encoder = XtMaCNN(obs_space, states_neurons=(256, 256), config=config)

        self.q1 = nn.Sequential(
            nn.Linear(config.features_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(config.features_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        features = self.encoder(
            obs['bev_semantics'],
            obs['measurements']
        )
        x = torch.cat([features, action], dim=1)
        return self.q1(x), self.q2(x)
