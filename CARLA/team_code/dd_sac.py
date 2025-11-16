'''
Self-contained SAC training algorithm for CaRL. Adapted from CleanRL https://github.com/vwxyzjn/cleanrl
Refactored for clarity and proper checkpoint saving
'''

import argparse
import os
import random
import time
import pathlib
import re
import datetime
import math
from collections import deque
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import wandb
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from pytictoc import TicToc
import zmq

from model_sac import SACPolicy
from rl_config import GlobalConfig
import rl_utils as rl_u
from sac_config import SACConfig

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
torch.set_num_threads(1)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def strtobool(v):
    return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


def none_or_str(value):
    if value == 'None':
        return None
    return value


def save_checkpoint(agent, actor_optimizer, q_optimizer, config, folder, prefix):
    """
    Save model and optimizers - FIXED to always save optimizers
    
    Args:
        agent: DDP-wrapped SAC policy
        actor_optimizer: actor optimizer (or None)
        q_optimizer: Q-network optimizer (or None)
        config: training config
        folder: save directory
        prefix: filename prefix (e.g., 'model_best', 'model_latest_000010000')
    """
    # Save model weights
    model_file = os.path.join(folder, f'{prefix}.pth')
    torch.save(agent.module.state_dict(), model_file)
    print(f'  ✓ Saved model: {prefix}.pth')
    
    # Save actor optimizer (FIXED: always save if provided)
    if actor_optimizer is not None:
        actor_optimizer_file = os.path.join(folder, f'{prefix}_actor_optimizer.pth')
        torch.save(actor_optimizer.state_dict(), actor_optimizer_file)
        print(f'  ✓ Saved actor optimizer: {prefix}_actor_optimizer.pth')
    
    # Save Q optimizer (FIXED: always save if provided)
    if q_optimizer is not None:
        q_optimizer_file = os.path.join(folder, f'{prefix}_q_optimizer.pth')
        torch.save(q_optimizer.state_dict(), q_optimizer_file)
        print(f'  ✓ Saved Q optimizer: {prefix}_q_optimizer.pth')
    
    # Save config
    json_config = jsonpickle.encode(config)
    config_file = os.path.join(folder, 'config.json')
    with open(config_file, 'wt', encoding='utf-8') as f:
        f.write(json_config)


def cleanup_old_checkpoints(folder, current_checkpoint_name):
    """
    Remove old model_latest_* checkpoints, keeping only the current one
    
    Args:
        folder: checkpoint directory
        current_checkpoint_name: current checkpoint filename (without extension)
    """
    removed_count = 0
    for file in os.listdir(folder):
        if file.startswith('model_latest_') and file.endswith('.pth'):
            if file != f'{current_checkpoint_name}.pth':
                # Remove model
                old_model_file = os.path.join(folder, file)
                if os.path.isfile(old_model_file):
                    os.remove(old_model_file)
                    removed_count += 1
                
                # Remove associated optimizer files
                base_name = file.replace('.pth', '')
                actor_opt_file = os.path.join(folder, f'{base_name}_actor_optimizer.pth')
                q_opt_file = os.path.join(folder, f'{base_name}_q_optimizer.pth')
                
                if os.path.isfile(actor_opt_file):
                    os.remove(actor_opt_file)
                if os.path.isfile(q_opt_file):
                    os.remove(q_opt_file)
    
    if removed_count > 0:
        print(f'  ✓ Cleaned up {removed_count} old checkpoint(s)')


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args(config):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    # Distributed and general args
    parser.add_argument('--rdzv_addr', default='localhost', type=str)
    parser.add_argument('--exp_name', type=str, default=config.exp_name)
    parser.add_argument('--gym_id', type=str, default=config.gym_id)
    parser.add_argument('--tcp_store_port', type=int, required=True)
    parser.add_argument('--seed', type=int, default=config.seed)
    parser.add_argument('--total_timesteps', type=int, default=config.total_timesteps)
    parser.add_argument('--torch_deterministic', type=lambda x: bool(strtobool(x)), 
                       default=config.torch_deterministic, nargs='?', const=True)
    parser.add_argument('--allow_tf32', type=lambda x: bool(strtobool(x)), 
                       default=config.allow_tf32, nargs='?', const=True)
    parser.add_argument('--benchmark', type=lambda x: bool(strtobool(x)), 
                       default=config.benchmark, nargs='?', const=True)
    parser.add_argument('--matmul_precision', type=str, default=config.matmul_precision)
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), 
                       default=config.cuda, nargs='?', const=True)
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), 
                       default=config.track, nargs='?', const=True)
    parser.add_argument('--wandb_project_name', type=str, default=config.wandb_project_name)
    parser.add_argument('--wandb_entity', type=str, default=config.wandb_entity)
    parser.add_argument('--capture_video', type=lambda x: bool(strtobool(x)), 
                       default=config.capture_video, nargs='?', const=True)
    parser.add_argument('--visualize', type=lambda x: bool(strtobool(x)), 
                       default=config.visualize, nargs='?', const=True)
    parser.add_argument('--logdir', type=str, default=config.logdir)
    parser.add_argument('--load_file', type=none_or_str, nargs='?', default=config.load_file)
    
    # Environment args
    parser.add_argument('--ports', nargs='+', default=config.ports, type=int)
    parser.add_argument('--gpu_ids', nargs='+', default=config.gpu_ids, type=int)
    parser.add_argument('--num_envs_per_proc', type=int, default=config.num_envs_per_proc)
    parser.add_argument('--compile_model', type=lambda x: bool(strtobool(x)), 
                       default=config.compile_model, nargs='?', const=True)
    
    # SAC specific arguments
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--buffer_storage', type=str, default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_starts', type=int, default=5000)
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--q_lr', type=float, default=3e-4)
    parser.add_argument('--policy_frequency', type=int, default=2)
    parser.add_argument('--target_network_frequency', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--autotune', type=lambda x: bool(strtobool(x)), 
                       default=True, nargs='?', const=True)
    parser.add_argument('--lr_schedule', default='none', type=str)
    
    # CaRL-specific arguments
    parser.add_argument('--use_new_bev_obs', type=lambda x: bool(strtobool(x)), 
                       default=config.use_new_bev_obs, nargs='?', const=True)
    parser.add_argument('--obs_num_channels', type=int, default=config.obs_num_channels)
    parser.add_argument('--map_folder', type=str, default=config.map_folder)
    parser.add_argument('--pixels_per_meter', type=float, default=config.pixels_per_meter)
    parser.add_argument('--reward_type', type=str, default=config.reward_type)
    parser.add_argument('--image_encoder', type=str, default=config.image_encoder)
    parser.add_argument('--use_layer_norm', type=lambda x: bool(strtobool(x)), 
                       default=config.use_layer_norm, nargs='?', const=True)
    parser.add_argument('--bev_semantics_width', type=int, default=config.bev_semantics_width)
    parser.add_argument('--bev_semantics_height', type=int, default=config.bev_semantics_height)
    parser.add_argument('--obs_num_measurements', type=int, default=config.obs_num_measurements)
    parser.add_argument('--use_value_measurements', type=lambda x: bool(strtobool(x)), 
                       default=config.use_value_measurements, nargs='?', const=True)
    parser.add_argument('--num_value_measurements', type=int, default=config.num_value_measurements)
    parser.add_argument('--normalize_rewards', type=lambda x: bool(strtobool(x)), 
                       default=config.normalize_rewards, nargs='?', const=True)
    parser.add_argument('--adam_eps', type=float, default=config.adam_eps)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)
    parser.add_argument('--beta_1', type=float, default=config.beta_1)
    parser.add_argument('--beta_2', type=float, default=config.beta_2)
    
    args, unknown = parser.parse_known_args()
    print('Unknown Arguments', unknown)
    return args


# =============================================================================
# REPLAY BUFFER (Keep your existing implementation)
# =============================================================================

class ReplayBuffer:
    """Replay buffer for storing transitions - handles CaRL's dict observations"""
    def __init__(self, buffer_size, obs_space, action_shape, device, num_envs=1, storage='cpu'):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.num_envs = num_envs
        self.device = device
        self.storage = storage
        
        # Get shapes
        bev_shape = obs_space.spaces['bev_semantics'].shape
        meas_shape = obs_space.spaces['measurements'].shape
        val_meas_shape = obs_space.spaces['value_measurements'].shape
        
        print(f"\nAllocating replay buffer ({storage.upper()} storage): {buffer_size:,} transitions")
        print(f"  BEV shape: {bev_shape}, Measurements: {meas_shape}")
        
        # Allocate storage based on mode
        if storage == 'cpu':
            self.observations = {
                'bev_semantics': np.zeros((buffer_size,) + bev_shape, dtype=np.uint8),
                'measurements': np.zeros((buffer_size,) + meas_shape, dtype=np.float32),
                'value_measurements': np.zeros((buffer_size,) + val_meas_shape, dtype=np.float32)
            }
            self.next_observations = {
                'bev_semantics': np.zeros((buffer_size,) + bev_shape, dtype=np.uint8),
                'measurements': np.zeros((buffer_size,) + meas_shape, dtype=np.float32),
                'value_measurements': np.zeros((buffer_size,) + val_meas_shape, dtype=np.float32)
            }
            self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.float32)
            self.rewards = np.zeros(buffer_size, dtype=np.float32)
            self.dones = np.zeros(buffer_size, dtype=np.float32)
            
            total_mb = (
                self.observations['bev_semantics'].nbytes + 
                self.next_observations['bev_semantics'].nbytes +
                self.observations['measurements'].nbytes +
                self.next_observations['measurements'].nbytes +
                self.observations['value_measurements'].nbytes +
                self.next_observations['value_measurements'].nbytes +
                self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            ) / 1024 / 1024
            print(f"  CPU memory: ~{total_mb:.1f}MB")
        else:  # GPU storage
            self.observations = {
                'bev_semantics': torch.zeros((buffer_size,) + bev_shape, dtype=torch.uint8, device=device),
                'measurements': torch.zeros((buffer_size,) + meas_shape, dtype=torch.float32, device=device),
                'value_measurements': torch.zeros((buffer_size,) + val_meas_shape, dtype=torch.float32, device=device)
            }
            self.next_observations = {
                'bev_semantics': torch.zeros((buffer_size,) + bev_shape, dtype=torch.uint8, device=device),
                'measurements': torch.zeros((buffer_size,) + meas_shape, dtype=torch.float32, device=device),
                'value_measurements': torch.zeros((buffer_size,) + val_meas_shape, dtype=torch.float32, device=device)
            }
            self.actions = torch.zeros((buffer_size,) + action_shape, dtype=torch.float32, device=device)
            self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
            self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
            
            total_mb = sum([
                t.element_size() * t.nelement() for t in [
                    self.observations['bev_semantics'], self.next_observations['bev_semantics'],
                    self.observations['measurements'], self.next_observations['measurements'],
                    self.observations['value_measurements'], self.next_observations['value_measurements'],
                    self.actions, self.rewards, self.dones
                ]
            ]) / 1024 / 1024
            print(f"  GPU memory: ~{total_mb:.1f}MB")
    
    def add_batch(self, obs, next_obs, actions, rewards, dones):
        """Add transitions (handles both CPU and GPU storage, works with vectorized envs)"""
        def to_storage(x, target_dtype=None):
            """Convert input to storage format (CPU numpy or GPU tensor)"""
            if self.storage == 'cpu':
                if torch.is_tensor(x):
                    return x.cpu().numpy()
                return np.array(x) if not isinstance(x, np.ndarray) else x
            else:  # GPU storage
                if torch.is_tensor(x):
                    tensor = x.to(self.device) if x.device != self.device else x
                    if target_dtype and tensor.dtype != target_dtype:
                        tensor = tensor.to(target_dtype)
                    return tensor
                else:
                    # Convert numpy/list to tensor on GPU
                    return torch.tensor(x, device=self.device, dtype=target_dtype if target_dtype else torch.float32)
        
        # Normalize inputs to consistent format for indexing
        # Convert to numpy temporarily just for determining batch size and indexing
        if torch.is_tensor(actions):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = np.array(actions) if not isinstance(actions, np.ndarray) else actions
        
        if torch.is_tensor(rewards):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = np.array(rewards) if not isinstance(rewards, np.ndarray) else rewards
        
        if torch.is_tensor(dones):
            dones_np = dones.cpu().numpy()
        else:
            dones_np = np.array(dones) if not isinstance(dones, np.ndarray) else dones
        
        # Determine batch size from observations (most reliable)
        # Observations should always have shape (batch_size, ...)
        obs_batch_size = None
        for key in obs.keys():
            if torch.is_tensor(obs[key]):
                obs_batch_size = obs[key].shape[0]
            else:
                obs_batch_size = np.array(obs[key]).shape[0]
            break
        
        batch_size = obs_batch_size if obs_batch_size is not None else actions_np.shape[0]
        
        # Validate shapes with detailed error messages
        if rewards_np.shape[0] != batch_size:
            error_msg = (
                f"\nShape mismatch in replay buffer!\n"
                f"  Actions shape: {actions_np.shape} -> batch_size: {batch_size}\n"
                f"  Rewards shape: {rewards_np.shape}\n"
                f"  Dones shape: {dones_np.shape}\n"
                f"  Obs shapes: {[(k, obs[k].shape) for k in obs.keys()]}\n"
                f"  Next obs shapes: {[(k, next_obs[k].shape) for k in next_obs.keys()]}"
            )
            raise ValueError(error_msg)
        
        if dones_np.shape[0] != batch_size:
            error_msg = (
                f"\nShape mismatch in replay buffer!\n"
                f"  Actions shape: {actions_np.shape} -> batch_size: {batch_size}\n"
                f"  Rewards shape: {rewards_np.shape}\n"
                f"  Dones shape: {dones_np.shape}\n"
                f"  Obs shapes: {[(k, obs[k].shape) for k in obs.keys()]}\n"
                f"  Next obs shapes: {[(k, next_obs[k].shape) for k in next_obs.keys()]}"
            )
            raise ValueError(error_msg)
        
        # Add each transition
        for i in range(batch_size):
            idx = self.ptr
            
            # Index into the data (handle both tensor and numpy)
            def index_data(data, i):
                if torch.is_tensor(data):
                    return data[i]
                else:
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    return data[i]
            
            # Store observations - to_storage handles CPU/GPU conversion
            self.observations['bev_semantics'][idx] = to_storage(index_data(obs['bev_semantics'], i), torch.uint8)
            self.observations['measurements'][idx] = to_storage(index_data(obs['measurements'], i), torch.float32)
            self.observations['value_measurements'][idx] = to_storage(index_data(obs['value_measurements'], i), torch.float32)
            
            self.next_observations['bev_semantics'][idx] = to_storage(index_data(next_obs['bev_semantics'], i), torch.uint8)
            self.next_observations['measurements'][idx] = to_storage(index_data(next_obs['measurements'], i), torch.float32)
            self.next_observations['value_measurements'][idx] = to_storage(index_data(next_obs['value_measurements'], i), torch.float32)
            
            self.actions[idx] = to_storage(index_data(actions, i), torch.float32)
            self.rewards[idx] = to_storage(rewards_np[i], torch.float32)
            self.dones[idx] = to_storage(dones_np[i], torch.float32)
            
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """Sample batch (returns GPU tensors)"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        if self.storage == 'cpu':
            return (
                {k: torch.as_tensor(v[idxs], device=self.device) for k, v in self.observations.items()},
                {k: torch.as_tensor(v[idxs], device=self.device) for k, v in self.next_observations.items()},
                torch.as_tensor(self.actions[idxs], device=self.device),
                torch.as_tensor(self.rewards[idxs], device=self.device).reshape(-1, 1),
                torch.as_tensor(self.dones[idxs], device=self.device).reshape(-1, 1)
            )
        else:
            return (
                {k: v[idxs] for k, v in self.observations.items()},
                {k: v[idxs] for k, v in self.next_observations.items()},
                self.actions[idxs],
                self.rewards[idxs].reshape(-1, 1),
                self.dones[idxs].reshape(-1, 1)
            )


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def make_env(gym_id, args, run_name, port, config):
    def thunk():
        render_mode = 'human' if args.visualize else 'rgb_array'
        env = gym.make(gym_id, port=port, config=config, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.capture_video:
            env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
        env = gym.wrappers.ClipAction(env)
        if config.normalize_rewards:
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


# =============================================================================
# SAC TRAINING FUNCTIONS
# =============================================================================

def update_learning_rate(args, config, update, num_updates, actor_optimizer, q_optimizer, a_optimizer=None):
    """Update learning rates based on schedule"""
    if args.lr_schedule == 'linear':
        frac = 1.0 - (update - 1.0) / num_updates
        config.current_policy_lr = frac * args.policy_lr
        config.current_q_lr = frac * args.q_lr
    elif args.lr_schedule == 'cosine':
        frac = update / num_updates
        config.current_policy_lr = 0.5 * args.policy_lr * (1 + math.cos(frac * math.pi))
        config.current_q_lr = 0.5 * args.q_lr * (1 + math.cos(frac * math.pi))
    # Add other schedules as needed
    
    for param_group in actor_optimizer.param_groups:
        param_group['lr'] = config.current_policy_lr
    for param_group in q_optimizer.param_groups:
        param_group['lr'] = config.current_q_lr
    if a_optimizer:
        for param_group in a_optimizer.param_groups:
            param_group['lr'] = config.current_q_lr


def sac_update(agent, rb, args, alpha, log_alpha, target_entropy,
               actor_optimizer, q_optimizer, a_optimizer, device):
    """
    Perform one SAC update
    
    Returns:
        metrics: dict of losses and values for logging
        alpha: updated alpha value
    """
    # Sample from replay buffer
    obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = rb.sample(args.batch_size)
    
    # Update Q-networks
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _, _ = agent.module.get_action(next_obs_batch)
        
        next_state_log_pi = next_state_log_pi.reshape(-1, 1)
        
        qf1_next_target = agent.module.qf1_target(next_obs_batch, next_state_actions)
        qf2_next_target = agent.module.qf2_target(next_obs_batch, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        
        # Q-value clipping for stability
        min_qf_next_target = torch.clamp(min_qf_next_target, -100, 100)
        next_q_value = rewards_batch + (1 - dones_batch) * args.gamma * min_qf_next_target
        next_q_value = torch.clamp(next_q_value, -100, 100)
    
    qf1_a_values = agent.module.qf1(obs_batch, actions_batch)
    qf2_a_values = agent.module.qf2(obs_batch, actions_batch)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss
    
    q_optimizer.zero_grad()
    qf_loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent.module.qf1.parameters()) + list(agent.module.qf2.parameters()), 
        max_norm=0.5
    )
    q_optimizer.step()
    
    # Update policy network
    pi, log_pi, _, _ = agent.module.get_action(obs_batch)
    qf1_pi = agent.module.qf1(obs_batch, pi)
    qf2_pi = agent.module.qf2(obs_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    actor_loss = (alpha * log_pi - min_qf_pi).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(agent.module.actor.parameters(), max_norm=0.5)
    actor_optimizer.step()
    
    # Update temperature (alpha)
    alpha_loss = None
    if args.autotune:
        alpha_loss = (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()
        a_optimizer.zero_grad()
        alpha_loss.backward()
        a_optimizer.step()
        alpha = log_alpha.exp().item()
    
    # Return metrics
    metrics = {
        'qf1_loss': qf1_loss.item(),
        'qf2_loss': qf2_loss.item(),
        'qf_loss': qf_loss.item(),
        'actor_loss': actor_loss.item(),
        'alpha': alpha,
        'qf1_mean': qf1_a_values.mean().item(),
        'qf2_mean': qf2_a_values.mean().item(),
        'target_q_mean': next_q_value.mean().item(),
    }
    
    if alpha_loss is not None:
        metrics['alpha_loss'] = alpha_loss.item()
    
    return metrics, alpha


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    register(id='CARLAEnv-v0', entry_point='env_gym:CARLAEnv', max_episode_steps=None)
    
    config = SACConfig()
    args = parse_args(config)
    
    # Distributed setup
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f'RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}')
    
    run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}'
    
    # Setup logging (rank 0 only)
    if rank == 0:
        exp_folder = os.path.join(args.logdir, f'{args.exp_name}')
        wandb_folder = os.path.join(exp_folder, 'wandb')
        pathlib.Path(exp_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(wandb_folder).mkdir(parents=True, exist_ok=True)
        
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=False,
                save_code=False,
                mode='online',
                dir=wandb_folder,
            )
        
        writer = SummaryWriter(exp_folder)
        writer.add_text('hyperparameters', 
                       '|param|value|\n|-|-|\n%s' % '\n'.join([f'|{k}|{v}|' for k, v in vars(args).items()]))
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Config setup
    if args.load_file is not None:
        load_folder = pathlib.Path(args.load_file).parent.resolve()
        with open(os.path.join(load_folder, 'config.json'), 'rt') as f:
            loaded_config = jsonpickle.decode(f.read())
        config.__dict__.update(loaded_config.__dict__)
    
    config.initialize(**vars(args))
    
    # Distributed initialization
    torch.distributed.init_process_group(
        backend='gloo' if args.num_envs_per_proc == 1 else 'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=45)
    )
    
    device = torch.device(f'cuda:{args.gpu_ids[rank]}') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    
    # PyTorch settings
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.set_float32_matmul_precision(args.matmul_precision)
    
    if rank == 0:
        with open(os.path.join(exp_folder, 'config.json'), 'w') as f:
            f.write(jsonpickle.encode(config))
    
    # Send config to environments via ZMQ
    context = zmq.Context()
    socket_list = []
    for i in range(args.num_envs_per_proc):
        socket = context.socket(zmq.PAIR)
        socket_list.append(socket)
        current_folder = pathlib.Path(__file__).parent.resolve()
        comm_folder = os.path.join(current_folder, 'comm_files')
        pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
        comm_file = os.path.join(comm_folder, str(args.ports[args.num_envs_per_proc * local_rank + i]))
        socket.bind(f'ipc://{comm_file}.conf_lock')
        socket.send_string(jsonpickle.encode(config))
        _ = socket.recv_string()
        socket.close()
    context.term()
    del socket_list
    
    print(f'Rank {rank}: Sent config files')
    
    # Create environment
    if args.num_envs_per_proc > 1:
        env = gym.vector.AsyncVectorEnv([
            make_env(args.gym_id, args, run_name, args.ports[args.num_envs_per_proc * local_rank + i], config)
            for i in range(args.num_envs_per_proc)
        ])
    else:
        env = gym.vector.SyncVectorEnv([
            make_env(args.gym_id, args, run_name, args.ports[args.num_envs_per_proc * local_rank + i], config)
        ])
    
    assert isinstance(env.single_action_space, gym.spaces.Box), 'only continuous action space is supported'
    
    # Create SAC agent
    agent = SACPolicy(env.single_observation_space, env.single_action_space, config=config).to(device)
    
    if config.compile_model:
        agent = torch.compile(agent)
    
    # Load checkpoint if resuming
    start_step = 0
    if args.load_file is not None:
        load_file_name = os.path.basename(args.load_file)
        algo_step = re.findall(r'\d+', load_file_name)
        if len(algo_step) > 0:
            start_step = int(algo_step[0]) + 1
            print(f'Resuming from step: {start_step}')
        agent.load_state_dict(torch.load(args.load_file, map_location=device), strict=True)
    
    # Wrap with DDP
    agent = torch.nn.parallel.DistributedDataParallel(
        agent, device_ids=None, output_device=None, 
        broadcast_buffers=False, find_unused_parameters=False
    )
    print(f'Rank {rank}: Created DistributedDataParallel')
    
    # Initialize learning rates
    if args.load_file is None:
        config.current_policy_lr = args.policy_lr
        config.current_q_lr = args.q_lr
    
    if rank == 0:
        num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {num_params:,}')
    
    # Create optimizers
    if config.weight_decay > 0.0:
        q_optimizer = optim.AdamW(
            list(agent.module.qf1.parameters()) + list(agent.module.qf2.parameters()),
            lr=config.current_q_lr, eps=config.adam_eps, 
            weight_decay=config.weight_decay, betas=(config.beta_1, config.beta_2)
        )
        actor_optimizer = optim.AdamW(
            agent.module.actor.parameters(),
            lr=config.current_policy_lr, eps=config.adam_eps,
            weight_decay=config.weight_decay, betas=(config.beta_1, config.beta_2)
        )
    else:
        q_optimizer = optim.Adam(
            list(agent.module.qf1.parameters()) + list(agent.module.qf2.parameters()),
            lr=config.current_q_lr, eps=config.adam_eps, betas=(config.beta_1, config.beta_2)
        )
        actor_optimizer = optim.Adam(
            agent.module.actor.parameters(),
            lr=config.current_policy_lr, eps=config.adam_eps, betas=(config.beta_1, config.beta_2)
        )
    
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=config.current_q_lr, eps=config.adam_eps)
    else:
        alpha = args.alpha
        log_alpha = None
        a_optimizer = None
        target_entropy = None
    
    # Load optimizers if resuming
    if args.load_file is not None:
        actor_opt_file = args.load_file.replace('.pth', '_actor_optimizer.pth')
        q_opt_file = args.load_file.replace('.pth', '_q_optimizer.pth')
        if os.path.exists(actor_opt_file):
            actor_optimizer.load_state_dict(torch.load(actor_opt_file, map_location=device))
            print(f'Rank {rank}: Loaded actor optimizer')
        if os.path.exists(q_opt_file):
            q_optimizer.load_state_dict(torch.load(q_opt_file, map_location=device))
            print(f'Rank {rank}: Loaded Q optimizer')
        if rank == 0:
            writer.add_scalar('charts/restart', 1, start_step)
    
    # Create replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        env.single_observation_space,
        env.single_action_space.shape,
        device,
        num_envs=args.num_envs_per_proc,
        storage=args.buffer_storage
    )
    
    # Training setup
    num_updates = args.total_timesteps
    start_time = time.time()
    
    # Reset environment
    reset_obs = env.reset(seed=[args.seed + rank * args.num_envs_per_proc + i for i in range(args.num_envs_per_proc)])
    next_obs = {
        'bev_semantics': torch.tensor(reset_obs[0]['bev_semantics'], device=device, dtype=torch.uint8),
        'measurements': torch.tensor(reset_obs[0]['measurements'], device=device, dtype=torch.float32),
        'value_measurements': torch.tensor(reset_obs[0]['value_measurements'], device=device, dtype=torch.float32)
    }
    next_done = torch.zeros(args.num_envs_per_proc, device=device)
    next_lstm_state = (
        torch.zeros(config.num_lstm_layers, args.num_envs_per_proc, config.features_dim, device=device),
        torch.zeros(config.num_lstm_layers, args.num_envs_per_proc, config.features_dim, device=device),
    ) if config.use_lstm else None
    
    # Tracking variables
    if rank == 0:
        avg_returns = deque(maxlen=100)
        config.max_training_score = -float('inf')
        config.best_iteration = -1
        config.latest_iteration = -1
    
    total_returns = np.zeros(world_size)
    total_lengths = np.zeros(world_size)
    num_total_returns = np.zeros(world_size)
    
    config.global_step = start_step
    
    # Timing tracking
    inference_times = []
    env_times = []
    training_times = []
    
    # Training loop
    agent.train()
    print(f'Rank {rank}: Starting training loop\n')
    
    for update in tqdm(range(start_step, num_updates), disable=rank != 0):
        t0 = TicToc()
        t1 = TicToc()
        t2 = TicToc()
        t4 = TicToc()
        
        t0.tic()
        config.global_step = update
        
        # Update learning rate
        update_learning_rate(args, config, update, num_updates, actor_optimizer, q_optimizer, a_optimizer)
        
        # Collect data
        t1.tic()
        if rb.size < args.learning_starts:
            actions = np.array([env.single_action_space.sample() for _ in range(args.num_envs_per_proc)])
        else:
            with torch.no_grad():
                actions, _, _, next_lstm_state = agent.module.get_action(
                    next_obs, 
                    action=None, 
                    lstm_state=next_lstm_state, 
                    done=next_done
                )
                actions = actions.cpu().numpy()
        inference_times.append(t1.tocvalue())
        
        # Environment step
        t2.tic()
        next_obs_raw, reward, termination, truncation, info = env.step(actions)
        env_times.append(t2.tocvalue())
        done = np.logical_or(termination, truncation)
        
        # Store transition
        rb.add_batch(next_obs, next_obs_raw, actions, reward, done)
        
        # Update next_obs
        next_obs = {
            'bev_semantics': torch.tensor(next_obs_raw['bev_semantics'], device=device, dtype=torch.uint8),
            'measurements': torch.tensor(next_obs_raw['measurements'], device=device, dtype=torch.float32),
            'value_measurements': torch.tensor(next_obs_raw['value_measurements'], device=device, dtype=torch.float32)
        }
        next_done = torch.tensor(done, device=device, dtype=torch.float32)
        
        # LSTM reset on done
        if config.use_lstm:
            for idx in range(args.num_envs_per_proc):
                if done[idx]:
                    next_lstm_state[0][:, idx, :] = 0
                    next_lstm_state[1][:, idx, :] = 0
        
        # Log episode completions
        if 'final_info' in info.keys():
            for idx, single_info in enumerate(info['final_info']):
                if single_info is not None and 'episode' in single_info.keys():
                    ep_return = single_info['episode']['r'].item()
                    ep_length = single_info['episode']['l'].item()
                    print(f'[Rank {rank}] Step {config.global_step:,} | Episode Return: {ep_return:.2f} | Length: {ep_length}')
                    
                    total_returns[rank] += ep_return
                    total_lengths[rank] += ep_length
                    num_total_returns[rank] += 1
        
        # SAC training update
        if rb.size >= args.learning_starts:
            t4.tic()
            
            # Perform SAC update
            metrics, alpha = sac_update(
                agent, rb, args, alpha, log_alpha, target_entropy,
                actor_optimizer, q_optimizer, a_optimizer, device
            )
            
            training_times.append(t4.tocvalue())
            
            # Update target networks
            if update % args.target_network_frequency == 0:
                for param, target_param in zip(agent.module.qf1.parameters(), agent.module.qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(agent.module.qf2.parameters(), agent.module.qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            # Logging
            if update % 100 == 0 and rank == 0:
                sps = int(config.global_step / (time.time() - start_time))
                print(f'[TRAIN] Step {config.global_step:,} | '
                      f'Q1: {metrics["qf1_loss"]:.4f} | Q2: {metrics["qf2_loss"]:.4f} | '
                      f'Actor: {metrics["actor_loss"]:.4f} | Alpha: {metrics["alpha"]:.4f} | SPS: {sps}')
                
                for key, value in metrics.items():
                    writer.add_scalar(f'losses/{key}' if 'loss' in key else f'charts/{key}', value, config.global_step)
                writer.add_scalar('charts/policy_lr', config.current_policy_lr, config.global_step)
                writer.add_scalar('charts/q_lr', config.current_q_lr, config.global_step)
                writer.add_scalar('charts/SPS', sps, config.global_step)
        else:
            # Warmup phase
            if update % 100 == 0 and rank == 0:
                sps = int(config.global_step / (time.time() - start_time))
                warmup_pct = (rb.size / args.learning_starts) * 100
                print(f'[WARMUP] Step {update:,} | Buffer: {rb.size}/{args.learning_starts} ({warmup_pct:.1f}%) | SPS: {sps}')
        
        # Periodic summary (every 1000 steps)
        if update % 1000 == 0 and update > 0:
            # Sync episode statistics
            total_returns_tensor = torch.tensor(total_returns, device=device)
            total_lengths_tensor = torch.tensor(total_lengths, device=device)
            num_total_returns_tensor = torch.tensor(num_total_returns, device=device)
            
            torch.distributed.all_reduce(total_returns_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_lengths_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(num_total_returns_tensor, op=torch.distributed.ReduceOp.SUM)
            
            if rank == 0:
                total_num_episodes = int(num_total_returns_tensor.sum().item())
                if total_num_episodes > 0:
                    avg_return = total_returns_tensor.sum().item() / total_num_episodes
                    avg_length = total_lengths_tensor.sum().item() / total_num_episodes
                    avg_returns.append(avg_return)
                    
                    # Log to tensorboard
                    writer.add_scalar('charts/avg_episodic_return', avg_return, config.global_step)
                    writer.add_scalar('charts/avg_episodic_length', avg_length, config.global_step)
                    writer.add_scalar('charts/num_episodes', total_num_episodes, config.global_step)
                    
                    # Print summary
                    elapsed = time.time() - start_time
                    progress_pct = (config.global_step / args.total_timesteps) * 100
                    eta_seconds = (elapsed / config.global_step) * (args.total_timesteps - config.global_step)
                    
                    print(f'\n{"="*70}')
                    print(f'Progress: {config.global_step:,}/{args.total_timesteps:,} ({progress_pct:.1f}%)')
                    print(f'Episodes: {total_num_episodes} | Avg Return: {avg_return:.2f} | Avg Length: {avg_length:.0f}')
                    print(f'Time: {elapsed/3600:.1f}h elapsed | {eta_seconds/3600:.1f}h remaining')
                    
                    if len(avg_returns) > 0:
                        windowed_avg = sum(avg_returns) / len(avg_returns)
                        print(f'Windowed Avg (last 100): {windowed_avg:.2f} | Best: {config.max_training_score:.2f}')
                        
                        # Save best model
                        if windowed_avg >= config.max_training_score:
                            config.max_training_score = windowed_avg
                            if config.best_iteration != update:
                                config.best_iteration = update
                                print(f'\n🎯 New best model! Saving...')
                                save_checkpoint(agent, actor_optimizer, q_optimizer, config, exp_folder, 'model_best')
                    
                    # Timing breakdown
                    if len(inference_times) > 0 and len(env_times) > 0:
                        print(f'Timing: Inference {np.mean(inference_times[-1000:]):.4f}s | '
                              f'Env {np.mean(env_times[-1000:]):.4f}s', end='')
                        if len(training_times) > 0:
                            print(f' | Training {np.mean(training_times[-1000:]):.4f}s')
                        else:
                            print()
                    
                    print(f'{"="*70}\n')
                
                # Reset counters
                total_returns.fill(0)
                total_lengths.fill(0)
                num_total_returns.fill(0)
        
        # Save checkpoints (FIXED: Always save optimizers)
        config.latest_iteration = update
        if rank == 0 and update % 10000 == 0 and update > 0:
            checkpoint_name = f'model_latest_{update:09d}'
            print(f'\n💾 Saving checkpoint at step {update:,}...')
            save_checkpoint(agent, actor_optimizer, q_optimizer, config, exp_folder, checkpoint_name)
            
            # Cleanup old checkpoints
            cleanup_old_checkpoints(exp_folder, checkpoint_name)
            
            # Eval checkpoints at specific intervals
            frac = update / num_updates
            if hasattr(config, 'current_eval_interval_idx') and hasattr(config, 'eval_intervals'):
                if config.current_eval_interval_idx < len(config.eval_intervals):
                    if frac >= config.eval_intervals[config.current_eval_interval_idx]:
                        eval_checkpoint_name = f'model_eval_{update:09d}'
                        print(f'💾 Saving eval checkpoint: {eval_checkpoint_name}')
                        save_checkpoint(agent, None, None, config, exp_folder, eval_checkpoint_name)
                        config.current_eval_interval_idx += 1
            
            print()  # Newline after checkpoint operations
    
    # Training complete
    env.close()
    if rank == 0:
        print(f'\n{"="*70}')
        print('Training Complete!')
        print(f'{"="*70}\n')
        
        print('💾 Saving final checkpoint...')
        save_checkpoint(agent, actor_optimizer, q_optimizer, config, exp_folder, 'model_final')
        
        writer.close()
        if args.track:
            wandb.finish(exit_code=0, quiet=True)
        
        print(f'\n✅ All done! Results saved to: {exp_folder}')


if __name__ == '__main__':
    main()
