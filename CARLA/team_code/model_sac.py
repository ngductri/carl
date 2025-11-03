'''
SAC Actor and Critic networks for CaRL autonomous driving.
Clean implementation - only what SAC needs, matching CaRL's structure.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import gymnasium as gym

# Import existing CaRL encoders
from model import XtMaCNN

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    """
    SAC Actor network - outputs Gaussian policy for continuous control.
    Uses reparameterization trick for differentiable sampling.
    """
    def __init__(self, observation_space, action_space, config, policy_head_arch=(256, 256)):
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.policy_head_arch = list(policy_head_arch)
        
        # Use CaRL's existing feature extractor (same as PPO)
        states_neurons = (256, 256)
        self.features_extractor = XtMaCNN(observation_space, config=config, states_neurons=states_neurons)
        
        # LSTM support (optional, same as PPO)
        if config.use_lstm:
            self.lstm = nn.LSTM(config.features_dim, config.features_dim, num_layers=config.num_lstm_layers)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param, 1.0)
        
        # Build policy head
        self.build_policy_head()
        
        # Action rescaling (same as PPO's action space handling)
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )
    
    def build_policy_head(self):
        """Build policy MLP (matches PPO structure)"""
        layers = []
        last_dim = self.config.features_dim
        
        for layer_size in self.policy_head_arch:
            layers.append(nn.Linear(last_dim, layer_size))
            if self.config.use_layer_norm and self.config.use_layer_norm_policy_head:
                layers.append(nn.LayerNorm(layer_size))
            layers.append(nn.ReLU())
            last_dim = layer_size
        
        self.policy_head = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(last_dim, np.prod(self.action_space.shape))
        self.fc_logstd = nn.Linear(last_dim, np.prod(self.action_space.shape))
    
    def get_features(self, obs):
        """Extract features from BEV + measurements (same as PPO)"""
        bev_semantics = obs['bev_semantics'].to(dtype=torch.float32)  # Cast from uint8
        measurements = obs['measurements']
        birdview = bev_semantics / 255.0  # Normalize to [0, 1]
        features = self.features_extractor(birdview, measurements)
        return features
    
    def lstm_forward(self, features, lstm_state, done):
        """LSTM forward with done masking (same as PPO)"""
        batch_size = lstm_state[0].shape[1]
        hidden = features.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    def get_action(self, obs, action=None, lstm_state=None, done=None):
        """
        Sample action from policy or compute log_prob of given action.
        
        Args:
            obs: observations dict with 'bev_semantics', 'measurements', 'value_measurements'
            action: if provided, compute log_prob for this action; else sample new action
            lstm_state: tuple of (h, c) if using LSTM
            done: done flags for LSTM reset
        
        Returns:
            action: sampled action in environment action space
            log_prob: log probability of action (for SAC loss)
            entropy: entropy of distribution (for logging)
            lstm_state: updated LSTM state (or None)
        """
        # Extract features
        features = self.get_features(obs)
        
        # LSTM (if enabled)
        if self.config.use_lstm:
            features, lstm_state = self.lstm_forward(features, lstm_state, done)
        
        # Policy network
        latent_pi = self.policy_head(features)
        mean = self.fc_mean(latent_pi)
        log_std = self.fc_logstd(latent_pi)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        
        # Create Gaussian distribution
        normal = Normal(mean, std)
        
        if action is None:
            # Sample action with reparameterization trick
            x_t = normal.rsample()  # mean + std * N(0,1) - differentiable!
        else:
            # Given action: compute pre-tanh value for log_prob
            action_scaled = (action - self.action_bias) / self.action_scale
            action_scaled = torch.clamp(action_scaled, -1.0 + 1e-6, 1.0 - 1e-6)
            x_t = torch.atanh(action_scaled)  # Inverse tanh
        
        # Squash through tanh to [-1, 1]
        y_t = torch.tanh(x_t)
        # Scale to action space
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with Jacobian correction for tanh
        log_prob = normal.log_prob(x_t)
        # Subtract log of Jacobian determinant: log|d tanh(x)/dx|
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1)  # Sum over action dimensions
        
        # Entropy (for logging/debugging)
        entropy = normal.entropy().sum(1)
        
        return action, log_prob, entropy, lstm_state


class SoftQNetwork(nn.Module):
    """
    SAC Critic network - estimates Q(s,a).
    Takes both state and action as input (unlike PPO's V(s)).
    """
    def __init__(self, observation_space, action_space, config, value_head_arch=(256, 256)):
        super().__init__()
        self.config = config
        self.value_head_arch = list(value_head_arch)
        
        # Use CaRL's existing feature extractor
        states_neurons = (256, 256)
        self.features_extractor = XtMaCNN(observation_space, config=config, states_neurons=states_neurons)
        
        # LSTM support (optional)
        if config.use_lstm:
            self.lstm = nn.LSTM(config.features_dim, config.features_dim, num_layers=config.num_lstm_layers)
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param, 1.0)
        
        # Build Q-network
        self.build_q_head(action_space)
    
    def build_q_head(self, action_space):
        """
        Build Q-network MLP.
        Input: features + actions (concatenated)
        Output: scalar Q-value
        """
        layers = []
        # Q-network takes features + actions as input
        last_dim = self.config.features_dim + np.prod(action_space.shape)
        
        for layer_size in self.value_head_arch:
            layers.append(nn.Linear(last_dim, layer_size))
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(layer_size))
            layers.append(nn.ReLU())
            last_dim = layer_size
        
        # Output single Q-value
        layers.append(nn.Linear(last_dim, 1))
        self.q_head = nn.Sequential(*layers)
    
    def get_features(self, obs):
        """Extract features from BEV + measurements (same as PPO)"""
        bev_semantics = obs['bev_semantics'].to(dtype=torch.float32)
        measurements = obs['measurements']
        birdview = bev_semantics / 255.0
        features = self.features_extractor(birdview, measurements)
        return features
    
    def lstm_forward(self, features, lstm_state, done):
        """LSTM forward with done masking (same as PPO)"""
        batch_size = lstm_state[0].shape[1]
        hidden = features.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    def forward(self, obs, action, lstm_state=None, done=None):
        """
        Compute Q(s,a).
        
        Args:
            obs: observations dict
            action: actions to evaluate
            lstm_state: LSTM hidden state (if using LSTM)
            done: done flags for LSTM masking
        
        Returns:
            q_value: Q(s,a) estimate
        """
        # Extract features
        features = self.get_features(obs)
        
        # LSTM (if enabled)
        if self.config.use_lstm:
            features, _ = self.lstm_forward(features, lstm_state, done)
        
        # Concatenate features with actions (key difference from PPO!)
        q_input = torch.cat([features, action], dim=1)
        
        # Q-network
        q_value = self.q_head(q_input)
        return q_value


class SACPolicy(nn.Module):
    """
    SAC Policy container - holds actor, 2 critics, and 2 target critics.
    Matches CaRL's PPOPolicy interface for easy integration.
    """
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 policy_head_arch=(256, 256),
                 value_head_arch=(256, 256),
                 states_neurons=(256, 256),  # For compatibility, passed to XtMaCNN
                 config=None):
        super().__init__()
        self.action_space = action_space
        self.config = config
        
        # Create actor and critics
        self.actor = Actor(observation_space, action_space, config, policy_head_arch=policy_head_arch)
        self.qf1 = SoftQNetwork(observation_space, action_space, config, value_head_arch=value_head_arch)
        self.qf2 = SoftQNetwork(observation_space, action_space, config, value_head_arch=value_head_arch)
        
        # Create target networks (frozen copies)
        self.qf1_target = SoftQNetwork(observation_space, action_space, config, value_head_arch=value_head_arch)
        self.qf2_target = SoftQNetwork(observation_space, action_space, config, value_head_arch=value_head_arch)
        
        # Initialize targets to match critics
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # Freeze target networks
        for param in self.qf1_target.parameters():
            param.requires_grad = False
        for param in self.qf2_target.parameters():
            param.requires_grad = False
    
    def get_action(self, obs, action=None, lstm_state=None, done=None):
        """
        Forward to actor's get_action.
        For compatibility with training loop.
        """
        return self.actor.get_action(obs, action, lstm_state, done)
