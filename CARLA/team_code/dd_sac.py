'''
Self-contained SAC training algorithm for CaRL. Adapted from CleanRL https://github.com/vwxyzjn/cleanrl
'''

import argparse
import os
import random
import time
import pathlib
import re
import gc
import datetime
import math
from collections import deque

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


def strtobool(v):
    return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


def none_or_str(value):
    if value == 'None':
        return None
    return value


def save(agent, actor_optimizer, q_optimizer, config, folder, prefix):
    """Save model and optimizers - matching CaRL's save pattern"""
    model_file = os.path.join(folder, f'{prefix}.pth')
    torch.save(agent.module.state_dict(), model_file)
    
    if actor_optimizer is not None:
        actor_optimizer_file = os.path.join(folder, f'{prefix}_actor_optimizer.pth')
        torch.save(actor_optimizer.state_dict(), actor_optimizer_file)
    
    if q_optimizer is not None:
        q_optimizer_file = os.path.join(folder, f'{prefix}_q_optimizer.pth')
        torch.save(q_optimizer.state_dict(), q_optimizer_file)
    
    json_config = jsonpickle.encode(config)
    with open(os.path.join(folder, 'config.json'), 'wt', encoding='utf-8') as f2:
        f2.write(json_config)


def parse_args(config):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    # Distributed and general args (matching dd_ppo.py exactly)
    parser.add_argument('--rdzv_addr', default='localhost', type=str, help='Master address for the TCP store.')
    parser.add_argument('--exp_name', type=str, default=config.exp_name, help='the name of this experiment')
    parser.add_argument('--gym_id', type=str, default=config.gym_id, help='the id of the gym environment')
    parser.add_argument('--tcp_store_port', type=int, required=True, help='port for the key value store')
    parser.add_argument('--seed', type=int, default=config.seed, help='seed of the experiment')
    parser.add_argument('--total_timesteps',
                       type=int,
                       default=config.total_timesteps,
                       help='total timesteps of the experiments')
    parser.add_argument('--torch_deterministic',
                       type=lambda x: bool(strtobool(x)),
                       default=config.torch_deterministic,
                       nargs='?',
                       const=True,
                       help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--allow_tf32',
                       type=lambda x: bool(strtobool(x)),
                       default=config.allow_tf32,
                       nargs='?',
                       const=True,
                       help='whether to use tensor cores, lower numeric precision but faster speed')
    parser.add_argument('--benchmark',
                       type=lambda x: bool(strtobool(x)),
                       default=config.benchmark,
                       nargs='?',
                       const=True,
                       help='Whether to benchmark different algorithms on hardware to find the fastest')
    parser.add_argument('--matmul_precision',
                       type=str,
                       default=config.matmul_precision,
                       help=' Options highest=float32, high=tf32, medium=bfloat16')
    parser.add_argument('--cuda',
                       type=lambda x: bool(strtobool(x)),
                       default=config.cuda,
                       nargs='?',
                       const=True,
                       help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track',
                       type=lambda x: bool(strtobool(x)),
                       default=config.track,
                       nargs='?',
                       const=True,
                       help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb_project_name',
                       type=str,
                       default=config.wandb_project_name,
                       help='the wandb project name')
    parser.add_argument('--wandb_entity',
                       type=str,
                       default=config.wandb_entity,
                       help='the entity (team) of wandb project')
    parser.add_argument('--capture_video',
                       type=lambda x: bool(strtobool(x)),
                       default=config.capture_video,
                       nargs='?',
                       const=True,
                       help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--visualize',
                       type=lambda x: bool(strtobool(x)),
                       default=config.visualize,
                       nargs='?',
                       const=True,
                       help='if toggled, Game will render on screen')
    parser.add_argument('--logdir', type=str, default=config.logdir, help='The directory to log the data into.')
    parser.add_argument('--load_file',
                       type=none_or_str,
                       nargs='?',
                       default=config.load_file,
                       help='model weights for initialization')
    
    # Environment args
    parser.add_argument('--ports',
                       nargs='+',
                       default=config.ports,
                       type=int,
                       help='Ports of the carla_gym wrapper to connect to.'
                       'It requires to submit a port for every envs'
                       '#ports == --nproc_per_node')
    parser.add_argument('--gpu_ids',
                       nargs='+',
                       default=config.gpu_ids,
                       type=int,
                       help='Which GPUs to train on. Index 0 indicates GPU for rank 0 etc.')
    parser.add_argument('--num_envs_per_proc',
                       type=int,
                       default=config.num_envs_per_proc,
                       help='Number of environments per process. Only considered for dd_ppo.py')
    parser.add_argument('--compile_model',
                       type=lambda x: bool(strtobool(x)),
                       default=config.compile_model,
                       nargs='?',
                       const=True,
                       help='whether to compile the model with torch.compile.')
    
    # SAC specific arguments
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='replay buffer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor gamma')
    parser.add_argument('--tau', type=float, default=0.005, help='target network update rate')
    parser.add_argument('--batch_size', type=int, default=256, help='minibatch size')
    parser.add_argument('--learning_starts', type=int, default=5000, 
                       help='timestep to start learning (SAC needs initial random exploration to fill buffer)')
    parser.add_argument('--policy_lr', type=float, default=3e-4, help='the learning rate of the policy network')
    parser.add_argument('--q_lr', type=float, default=3e-4, help='the learning rate of the Q network')
    parser.add_argument('--policy_frequency', type=int, default=2, help='policy update frequency')
    parser.add_argument('--target_network_frequency', type=int, default=1, help='target network update frequency')
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy regularization coefficient')
    parser.add_argument('--autotune', type=lambda x: bool(strtobool(x)), 
                       default=True, nargs='?', const=True, help='automatic temperature tuning')
    parser.add_argument('--lr_schedule',
                       default='none',
                       type=str,
                       help='Which lr schedule to use. Options: (linear, kl, none, step, cosine, cosine_restart)')
    
    # All CaRL-specific arguments (matching dd_ppo.py)
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
    parser.add_argument('--adam_eps', type=float, default=config.adam_eps, help='eps parameter of adam optimizer')
    parser.add_argument('--weight_decay',
                       type=float,
                       default=config.weight_decay,
                       help='Weight decay applied to optimizer. AdamW is used when > 0.0')
    parser.add_argument('--beta_1', type=float, default=config.beta_1, help='Beta 1 parameter of adam')
    parser.add_argument('--beta_2', type=float, default=config.beta_2, help='Beta 2 parameter of adam')
    
    args, unknown = parser.parse_known_args()
    print('Unknown Arguments', unknown)
    return args


class ReplayBuffer:
    """Replay buffer for storing transitions - handles CaRL's dict observations"""
    def __init__(self, buffer_size, obs_space, action_shape, device, num_envs=1):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.num_envs = num_envs
        self.device = device
        
        # Allocate memory for observations (store on CPU to save GPU memory)
        bev_shape = obs_space.spaces['bev_semantics'].shape
        meas_shape = obs_space.spaces['measurements'].shape
        val_meas_shape = obs_space.spaces['value_measurements'].shape
        
        print(f"Allocating CPU replay buffer: {buffer_size} transitions")
        print(f"  BEV shape: {bev_shape}, Measurements: {meas_shape}")
        
        # Store in CPU RAM as numpy arrays (saves GPU memory!)
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
        
        # Estimate CPU memory usage
        total_mb = (
            self.observations['bev_semantics'].nbytes + 
            self.next_observations['bev_semantics'].nbytes +
            self.observations['measurements'].nbytes +
            self.next_observations['measurements'].nbytes +
            self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        ) / 1024 / 1024
        
        print(f"  Replay buffer CPU memory: ~{total_mb:.1f}MB")
        print(f"  GPU memory saved: ~{total_mb:.1f}MB ✅")
    
    def add_batch(self, obs, next_obs, actions, rewards, dones):
        """Add a batch of transitions from multiple environments."""
    
        # Helper to convert tensor or array to numpy
        def to_numpy(x):
            if torch.is_tensor(x):
                return x.cpu().numpy()
            return x
        
        # Get batch size - handle both single and multi-env
        if isinstance(actions, np.ndarray):
            batch_size = len(actions) if actions.ndim > 0 else 1
        else:
            batch_size = 1
        
        # Store each environment's transition
        for i in range(batch_size):
            idx = self.ptr
            
            # Handle indexing for both single and multi-env
            def get_item(data, i):
                data_np = to_numpy(data)
                # If first dimension matches batch_size, index it; otherwise use as-is
                if data_np.shape[0] == batch_size:
                    return data_np[i]
                else:
                    # Single item, no batch dimension
                    return data_np
            
            self.observations['bev_semantics'][idx] = get_item(obs['bev_semantics'], i)
            self.observations['measurements'][idx] = get_item(obs['measurements'], i)
            self.observations['value_measurements'][idx] = get_item(obs['value_measurements'], i)
            
            self.next_observations['bev_semantics'][idx] = get_item(next_obs['bev_semantics'], i)
            self.next_observations['measurements'][idx] = get_item(next_obs['measurements'], i)
            self.next_observations['value_measurements'][idx] = get_item(next_obs['value_measurements'], i)
            
            self.actions[idx] = get_item(actions, i)
            self.rewards[idx] = rewards[i] if isinstance(rewards, np.ndarray) and len(rewards) > i else rewards
            self.dones[idx] = dones[i] if isinstance(dones, np.ndarray) and len(dones) > i else dones
            
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        Moves data from CPU to GPU only during sampling.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # Move only the sampled batch to GPU (this is the only GPU memory used by buffer!)
        return (
            {
                'bev_semantics': torch.as_tensor(self.observations['bev_semantics'][idxs], device=self.device),
                'measurements': torch.as_tensor(self.observations['measurements'][idxs], device=self.device),
                'value_measurements': torch.as_tensor(self.observations['value_measurements'][idxs], device=self.device)
            },
            {
                'bev_semantics': torch.as_tensor(self.next_observations['bev_semantics'][idxs], device=self.device),
                'measurements': torch.as_tensor(self.next_observations['measurements'][idxs], device=self.device),
                'value_measurements': torch.as_tensor(self.next_observations['value_measurements'][idxs], device=self.device)
            },
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device).reshape(-1, 1),
            torch.as_tensor(self.dones[idxs], device=self.device).reshape(-1, 1)
        )


def make_env(gym_id, args, run_name, port, config):
    def thunk():
        if args.visualize:
            render_mode = 'human'  # Only works well with num envs 1 right now
        else:
            render_mode = 'rgb_array'
        
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


def main():
    register(
        id='CARLAEnv-v0',
        entry_point='env_gym:CARLAEnv',
        max_episode_steps=None,
    )
    
    config = SACConfig()
    args = parse_args(config)
    
    # Torchrun initialization - matching dd_ppo.py exactly
    rank = int(os.environ['RANK'])  # Rank across all processes
    local_rank = int(os.environ['LOCAL_RANK'])  # Rank on Node
    world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
    print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')
    
    run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}'
    
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
                allow_val_change=True,
                save_code=False,
                mode='online',
                resume='auto',
                dir=wandb_folder,
                settings=wandb.Settings(_disable_stats=True, _disable_meta=True, start_method='fork')
            )
        
        writer = SummaryWriter(exp_folder)
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
        )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print('Is cuda available?:', torch.cuda.is_available())
    
    # Load the config before overwriting values with current arguments
    if args.load_file is not None:
        load_folder = pathlib.Path(args.load_file).parent.resolve()
        with open(os.path.join(load_folder, 'config.json'), 'rt', encoding='utf-8') as f:
            json_config = f.read()
        loaded_config = jsonpickle.decode(json_config)
        # Overwrite all properties that were set in the saved config.
        config.__dict__.update(loaded_config.__dict__)
    
    # Configure config. Converts all arguments into config attributes
    config.initialize(**vars(args))
    
    # Initialize distributed - matching dd_ppo.py
    torch.distributed.init_process_group(
        backend='gloo' if args.num_envs_per_proc == 1 else 'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=45)
    )
    
    device_id = args.gpu_ids[rank]
    print(device_id)
    if device_id < 0:
        print('ERROR! Device id must be positive.')
    
    device = torch.device(f'cuda:{args.gpu_ids[rank]}') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.device(device)
    
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.set_float32_matmul_precision(args.matmul_precision)
    
    if rank == 0:
        json_config = jsonpickle.encode(config)
        with open(os.path.join(exp_folder, 'config.json'), 'w', encoding='utf-8') as f2:
            f2.write(json_config)
    
    # Send config to environments via ZMQ - matching dd_ppo.py exactly
    context = zmq.Context()
    socket_list = []  # So that the garbage collector doesn't clean up sockets before we are done.
    for i in range(args.num_envs_per_proc):
        socket_list.append(context.socket(zmq.PAIR))
        current_folder = pathlib.Path(__file__).parent.resolve()
        comm_folder = os.path.join(current_folder, 'comm_files')
        pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
        communication_file = os.path.join(comm_folder, str(args.ports[args.num_envs_per_proc * local_rank + i]))
        socket_list[i].bind(f'ipc://{communication_file}.conf_lock')
        json_config = jsonpickle.encode(config)
        socket_list[i].send_string(json_config)
        _ = socket_list[i].recv_string()
        socket_list[i].close()
    
    context.term()
    del socket_list
    
    print(f'Rank {rank}, sent CONFIG files')
    
    # Create environment - matching dd_ppo.py exactly
    if args.num_envs_per_proc > 1:
        env = gym.vector.AsyncVectorEnv([
            make_env(args.gym_id, args, run_name, args.ports[args.num_envs_per_proc * local_rank + i], config)
            for i in range(args.num_envs_per_proc)
        ])
    else:
        # We don't want to spawn subprocesses for a single environment, so we use Sync==Sequential Vector env.
        env = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id, args, run_name, args.ports[args.num_envs_per_proc * local_rank + i], config)])
    
    assert isinstance(env.single_action_space, gym.spaces.Box), 'only continuous action space is supported'
    
    # Create SAC agent - matching dd_ppo.py pattern exactly
    agent = SACPolicy(env.single_observation_space, env.single_action_space, config=config).to(device)
    
    if config.compile_model:
        agent = torch.compile(agent)
    
    start_step = 0
    if args.load_file is not None:
        load_file_name = os.path.basename(args.load_file)
        algo_step = re.findall(r'\d+', load_file_name)
        if len(algo_step) > 0:
            start_step = int(algo_step[0]) + 1  # That step was already finished.
            print('Start training from step:', start_step)
        agent.load_state_dict(torch.load(args.load_file, map_location=device), strict=True)
    
    print(f'Rank {rank}, START DistributedDataParallel')
    agent = torch.nn.parallel.DistributedDataParallel(agent,
                                                      device_ids=None,
                                                      output_device=None,
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=False)
    print(f'Rank:{rank}, Created DistributedDataParallel')
    
    # If we are resuming training use last learning rate from config.
    # If we start a fresh training set the current learning rate according to arguments.
    if args.load_file is None:
        config.current_policy_lr = args.policy_lr
        config.current_q_lr = args.q_lr
    
    if rank == 0:
        model_parameters = filter(lambda p: p.requires_grad, agent.parameters())
        num_params = sum(np.prod(p.size()) for p in model_parameters)
        print('Total trainable parameters: ', num_params)
    
    # Optimizers - matching dd_ppo.py pattern
    if config.weight_decay > 0.0:
        q_optimizer = optim.AdamW(list(agent.module.qf1.parameters()) + list(agent.module.qf2.parameters()),
                                 lr=config.current_q_lr,
                                 eps=config.adam_eps,
                                 weight_decay=config.weight_decay,
                                 betas=(config.beta_1, config.beta_2))
        actor_optimizer = optim.AdamW(list(agent.module.actor.parameters()),
                                     lr=config.current_policy_lr,
                                     eps=config.adam_eps,
                                     weight_decay=config.weight_decay,
                                     betas=(config.beta_1, config.beta_2))
    else:
        q_optimizer = optim.Adam(list(agent.module.qf1.parameters()) + list(agent.module.qf2.parameters()),
                                lr=config.current_q_lr,
                                eps=config.adam_eps,
                                betas=(config.beta_1, config.beta_2))
        actor_optimizer = optim.Adam(list(agent.module.actor.parameters()),
                                    lr=config.current_policy_lr,
                                    eps=config.adam_eps,
                                    betas=(config.beta_1, config.beta_2))
    
    print(f'Rank:{rank}, Created Optimizer')
    
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
    
    # Load optimizers if resuming
    if args.load_file is not None:
        actor_optimizer.load_state_dict(torch.load(args.load_file.replace('.pth', '_actor_optimizer.pth'), map_location=device))
        q_optimizer.load_state_dict(torch.load(args.load_file.replace('.pth', '_q_optimizer.pth'), map_location=device))
        print(f'Rank:{rank}, Load model')
        if rank == 0:
            writer.add_scalar('charts/restart', 1, config.global_step)  # Log that a restart happened
    
    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        env.single_observation_space,
        env.single_action_space.shape,
        device,
        num_envs=args.num_envs_per_proc
    )
    
    # Training loop setup
    num_updates = args.total_timesteps  # SAC updates every timestep
    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
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
    
    if rank == 0:
        avg_returns = deque(maxlen=100)
        config.max_training_score = -float('inf')
        config.best_iteration = -1
        config.latest_iteration = -1
    
    # Episode tracking (all ranks)
    total_returns = np.zeros(world_size)
    total_lengths = np.zeros(world_size)
    num_total_returns = np.zeros(world_size)
    
    config.global_step = start_step
    
    # Timing tracking - PPO style
    inference_times = []
    env_times = []
    training_times = []
    
    # Training loop
    agent.train()
    
    print(f'Rank:{rank}, Start training loop', flush=True)
    
    # SAC vs PPO Training Loop Differences:
    # 
    # PPO (On-Policy):
    # - Collects large batch (e.g., 2048 steps) using current policy
    # - Computes advantages/returns (heavy preprocessing in t3)
    # - Trains for multiple epochs on this batch (t4)
    # - Discards data and repeats
    # - No learning_starts needed (always uses fresh data)
    #
    # SAC (Off-Policy):
    # - Collects 1 transition per step, stores in replay buffer
    # - Light preprocessing (just add to buffer in t3)
    # - Samples random batch from buffer and trains once (t4)
    # - Reuses old data indefinitely
    # - Needs learning_starts to fill buffer with diverse exploration before training
    #
    # Timing structure matches PPO for consistency:
    # t0: Data collection (action selection + env step)
    # t1: Forward pass
    # t2: Env step
    # t3: Pre-processing (buffer operations vs GAE computation)
    # t4: Training (single update vs multiple epochs)
    # t5: Logging
    
    for update in tqdm(range(start_step, num_updates), disable=rank != 0):
        config.global_step = update
        
        # Timing similar to PPO structure
        t0 = TicToc()  # Data collection (action + env step)
        t1 = TicToc()  # Forward pass / action selection
        t2 = TicToc()  # Env step
        t3 = TicToc()  # Pre-processing (buffer operations, data prep)
        t4 = TicToc()  # Training (Q-networks, actor, targets)
        t5 = TicToc()  # Logging
        
        t0.tic()  # Start data collection timing
        
        # Annealing the learning rate - matching dd_ppo.py exactly
        if args.lr_schedule == 'linear':
            frac = 1.0 - (update - 1.0) / num_updates
            config.current_policy_lr = frac * args.policy_lr
            config.current_q_lr = frac * args.q_lr
        elif args.lr_schedule == 'step':
            frac = update / num_updates
            lr_multiplier = 1.0
            if hasattr(config, 'lr_schedule_step_perc'):
                for change_percentage in config.lr_schedule_step_perc:
                    if frac > change_percentage:
                        lr_multiplier *= config.lr_schedule_step_factor
            config.current_policy_lr = lr_multiplier * args.policy_lr
            config.current_q_lr = lr_multiplier * args.q_lr
        elif args.lr_schedule == 'cosine':
            frac = update / num_updates
            config.current_policy_lr = 0.5 * args.policy_lr * (1 + math.cos(frac * math.pi))
            config.current_q_lr = 0.5 * args.q_lr * (1 + math.cos(frac * math.pi))
        elif args.lr_schedule == 'cosine_restart':
            frac = update / (num_updates + 1)  # + 1 so it doesn't become 100 %
            if hasattr(config, 'lr_schedule_cosine_restarts'):
                for idx, frac_restart in enumerate(config.lr_schedule_cosine_restarts):
                    if frac >= frac_restart:
                        current_idx = idx
                base_frac = config.lr_schedule_cosine_restarts[current_idx]
                length_current_interval = (config.lr_schedule_cosine_restarts[current_idx + 1] -
                                           config.lr_schedule_cosine_restarts[current_idx])
                frac_current_iter = (frac - base_frac) / length_current_interval
                config.current_policy_lr = 0.5 * args.policy_lr * (1 + math.cos(frac_current_iter * math.pi))
                config.current_q_lr = 0.5 * args.q_lr * (1 + math.cos(frac_current_iter * math.pi))
        
        # Update learning rates
        for param_group in actor_optimizer.param_groups:
            param_group['lr'] = config.current_policy_lr
        for param_group in q_optimizer.param_groups:
            param_group['lr'] = config.current_q_lr
        if args.autotune:
            for param_group in a_optimizer.param_groups:
                param_group['lr'] = config.current_q_lr
        
        # ALGO LOGIC: Collect data
        t1.tic()
        if update < args.learning_starts:
            actions = np.array([env.single_action_space.sample() for _ in range(args.num_envs_per_proc)])
        else:
            with torch.no_grad():
                actions, _, _, next_lstm_state = agent.module.get_action(next_obs, next_lstm_state, next_done)
                actions = actions.cpu().numpy()
        inference_times.append(t1.tocvalue())
        
        # TRY NOT TO MODIFY: execute the game and log data
        t3.tic()
        obs_for_buffer = next_obs
        t3.tocvalue()
        
        t2.tic()
        next_obs_raw, reward, termination, truncation, info = env.step(actions)
        env_times.append(t2.tocvalue())
        done = np.logical_or(termination, truncation)
        
        next_obs_for_buffer = next_obs_raw
        
        # Store transitions as a batch
        rb.add_batch(obs_for_buffer, next_obs_for_buffer, actions, reward, done)
        
        # Update next_obs
        next_obs = {
            'bev_semantics': torch.tensor(next_obs_raw['bev_semantics'], device=device, dtype=torch.uint8),
            'measurements': torch.tensor(next_obs_raw['measurements'], device=device, dtype=torch.float32),
            'value_measurements': torch.tensor(next_obs_raw['value_measurements'], device=device, dtype=torch.float32)
        }
        next_done = torch.tensor(done, device=device, dtype=torch.float32)
        
        # Handle LSTM state reset on done
        if config.use_lstm:
            for idx in range(args.num_envs_per_proc):
                if done[idx]:
                    next_lstm_state[0][:, idx, :] = 0
                    next_lstm_state[1][:, idx, :] = 0
        
        # Log episode statistics - PPO style
        if 'final_info' in info.keys():
            for idx, single_info in enumerate(info['final_info']):
                if single_info is not None:
                    if 'episode' in single_info.keys():
                        # Print immediately (same as PPO)
                        print(f'rank: {rank}, config.global_step={config.global_step}, '
                              f'episodic_return={single_info["episode"]["r"]}')
                        
                        # Track totals (same as PPO)
                        total_returns[rank] += single_info['episode']['r'].item()
                        total_lengths[rank] += single_info['episode']['l'].item()
                        num_total_returns[rank] += 1
        
        # Train the agent (similar to PPO's training loop structure)
        if update > args.learning_starts:
            t4.tic()
            
            # Sample from replay buffer
            data = rb.sample(args.batch_size)
            obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = data
            
            # Update Q-networks
            with torch.no_grad():
                # Sample next actions from current policy
                next_state_actions, next_state_log_pi, _, _ = agent.module.get_action(next_obs_batch)
                
                # Compute target Q-values using target networks
                qf1_next_target = agent.module.qf1_target(next_obs_batch, next_state_actions)
                qf2_next_target = agent.module.qf2_target(next_obs_batch, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi.unsqueeze(-1)
                next_q_value = rewards_batch + (1 - dones_batch) * args.gamma * min_qf_next_target
            
            qf1_a_values = agent.module.qf1(obs_batch, actions_batch)
            qf2_a_values = agent.module.qf2(obs_batch, actions_batch)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            q_optimizer.zero_grad()
            qf_loss.backward()
            nn.utils.clip_grad_norm_(list(agent.module.qf1.parameters()) + list(agent.module.qf2.parameters()), 
                                    max_norm=1.0)
            q_optimizer.step()
            
            # Update policy network
            if update % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _, _ = agent.module.get_action(obs_batch)
                    qf1_pi = agent.module.qf1(obs_batch, pi)
                    qf2_pi = agent.module.qf2(obs_batch, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi.unsqueeze(-1)) - min_qf_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(agent.module.actor.parameters(), max_norm=1.0)
                    actor_optimizer.step()
                    
                    # Update temperature (alpha)
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _, _ = agent.module.get_action(obs_batch)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
            
            # Update target networks
            if update % args.target_network_frequency == 0:
                for param, target_param in zip(agent.module.qf1.parameters(), agent.module.qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(agent.module.qf2.parameters(), agent.module.qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            training_times.append(t4.tocvalue())
            
            # Logging
            t5.tic()
            if update % 100 == 0 and rank == 0:
                elapsed_time = time.time() - start_time
                sps = int(config.global_step / elapsed_time) if elapsed_time > 0 else 0
                
                print(f'[TRAIN] Step {config.global_step:,} | '
                      f'Q1: {qf1_loss.item():.4f} | Q2: {qf2_loss.item():.4f} | '
                      f'Actor: {actor_loss.item():.4f} | Alpha: {alpha:.4f} | '
                      f'SPS: {sps}')
                
                writer.add_scalar('losses/qf1_loss', qf1_loss.item(), config.global_step)
                writer.add_scalar('losses/qf2_loss', qf2_loss.item(), config.global_step)
                writer.add_scalar('losses/qf_loss', qf_loss.item(), config.global_step)
                writer.add_scalar('losses/actor_loss', actor_loss.item(), config.global_step)
                writer.add_scalar('losses/alpha', alpha, config.global_step)
                writer.add_scalar('charts/policy_lr', config.current_policy_lr, config.global_step)
                writer.add_scalar('charts/q_lr', config.current_q_lr, config.global_step)
                writer.add_scalar('charts/SPS', sps, config.global_step)
                writer.add_scalar('charts/restart', 0, config.global_step)
                if args.autotune:
                    writer.add_scalar('losses/alpha_loss', alpha_loss.item(), config.global_step)
            t5.tocvalue()
        else:
            # Before learning starts
            t4.tic()
            training_times.append(t4.tocvalue())
            t5.tic()
            if update % 100 == 0 and rank == 0:
                elapsed_time = time.time() - start_time
                sps = int(config.global_step / elapsed_time) if elapsed_time > 0 else 0
                warmup_pct = (rb.size / args.learning_starts) * 100
                print(f'[WARMUP] Step {update:,} | Buffer: {rb.size}/{args.learning_starts} '
                      f'({warmup_pct:.1f}%) | SPS: {sps}')
            t5.tocvalue()
        
        # Periodic summary - PPO style (every 1000 steps)
        if update % 1000 == 0 and update > 0:
            # Synchronize episode statistics across ranks
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
                    
                    # Update tracking
                    avg_returns.append(avg_return)
                    
                    # Log to tensorboard
                    writer.add_scalar('charts/avg_episodic_return', avg_return, config.global_step)
                    writer.add_scalar('charts/avg_episodic_length', avg_length, config.global_step)
                    writer.add_scalar('charts/num_episodes', total_num_episodes, config.global_step)
                    
                    # Print summary
                    elapsed = time.time() - start_time
                    progress_pct = (config.global_step / args.total_timesteps) * 100
                    eta_seconds = (elapsed / config.global_step) * (args.total_timesteps - config.global_step) if config.global_step > 0 else 0
                    
                    print(f'\n{"="*70}')
                    print(f'Step {config.global_step:,}/{args.total_timesteps:,} ({progress_pct:.1f}%)')
                    print(f'Episodes completed: {total_num_episodes}')
                    print(f'Avg Return: {avg_return:.2f} | Avg Length: {avg_length:.0f}')
                    print(f'Elapsed: {elapsed/3600:.1f}h | ETA: {eta_seconds/3600:.1f}h')
                    
                    if len(avg_returns) > 0:
                        windowed_avg = sum(avg_returns) / len(avg_returns)
                        print(f'Windowed Avg (last 100): {windowed_avg:.2f}')
                        print(f'Best Score: {config.max_training_score:.2f}')
                        
                        # Save best model
                        if windowed_avg >= config.max_training_score:
                            config.max_training_score = windowed_avg
                            if config.best_iteration != update:
                                config.best_iteration = update
                                save(agent, None, None, config, exp_folder, 'model_best')
                                print(f'✓ New best model saved!')
                    
                    # Timing info
                    if len(inference_times) > 0:
                        print(f'Avg Inference: {np.mean(inference_times[-1000:]):.4f}s | '
                              f'Env Step: {np.mean(env_times[-1000:]):.4f}s', end='')
                        if len(training_times) > 0 and update > args.learning_starts:
                            print(f' | Training: {np.mean(training_times[-1000:]):.4f}s')
                        else:
                            print()
                    
                    print(f'{"="*70}\n')
                
                # Reset counters
                total_returns.fill(0)
                total_lengths.fill(0)
                num_total_returns.fill(0)
        
        # Save checkpoint - matching dd_ppo.py pattern
        config.latest_iteration = update
        if rank == 0 and update % 10000 == 0 and update > 0:
            save(agent, actor_optimizer, q_optimizer, config, exp_folder, f'model_latest_{update:09d}')
            
            # Eval checkpoints at specific intervals
            frac = update / num_updates
            if hasattr(config, 'current_eval_interval_idx') and hasattr(config, 'eval_intervals'):
                if config.current_eval_interval_idx < len(config.eval_intervals):
                    if frac >= config.eval_intervals[config.current_eval_interval_idx]:
                        save(agent, None, None, config, exp_folder, f'model_eval_{update:09d}')
                        config.current_eval_interval_idx += 1
            
            # Cleanup old checkpoints
            for file in os.listdir(exp_folder):
                if file.startswith('model_latest_') and file.endswith('.pth'):
                    if file != f'model_latest_{update:09d}.pth':
                        old_model_file = os.path.join(exp_folder, file)
                        if os.path.isfile(old_model_file):
                            os.remove(old_model_file)
                        # Remove associated optimizer files
                        actor_opt_file = os.path.join(exp_folder, file.replace('.pth', '_actor_optimizer.pth'))
                        q_opt_file = os.path.join(exp_folder, file.replace('.pth', '_q_optimizer.pth'))
                        if os.path.isfile(actor_opt_file):
                            os.remove(actor_opt_file)
                        if os.path.isfile(q_opt_file):
                            os.remove(q_opt_file)
    
    env.close()
    if rank == 0:
        writer.close()
        save(agent, actor_optimizer, q_optimizer, config, exp_folder, 'model_final')
        if args.track:
            wandb.finish(exit_code=0, quiet=True)
        print('Done training.')


if __name__ == '__main__':
    main()