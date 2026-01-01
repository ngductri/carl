import torch
from torch import nn
from copy import deepcopy
from model import XtMaCNN


class TD3Actor(nn.Module):
    def __init__(self, obs_space, action_dim, config):
        super().__init__()
        self.actor_encoder = XtMaCNN(obs_space, states_neurons=(256, 256), config=config)

        last = nn.Linear(256, action_dim)
        nn.init.constant_(last.bias[0], 0.0)
        nn.init.constant_(last.bias[1], 0.0)  # an toàn hơn

        self.policy = nn.Sequential(
            nn.Linear(config.features_dim, 256),
            nn.ReLU(),
            last,
            nn.Tanh()
        )

    def forward(self, obs):
        features = self.actor_encoder(obs['bev_semantics'], obs['measurements'])
        return self.policy(features)


class TD3Critic(nn.Module):
    def __init__(self, obs_space, action_dim, config):
        super().__init__()
        self.critic_encoder1 = XtMaCNN(obs_space, states_neurons=(256, 256), config=config)
        self.critic_encoder2 = XtMaCNN(obs_space, states_neurons=(256, 256), config=config)

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
        action = torch.clamp(action, -1.0, 1.0)
        f1 = self.critic_encoder1(obs['bev_semantics'], obs['measurements'])
        f2 = self.critic_encoder2(obs['bev_semantics'], obs['measurements'])
        return self.q1(torch.cat([f1, action], dim=1)), \
               self.q2(torch.cat([f2, action], dim=1))
