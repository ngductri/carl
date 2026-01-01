'''
Self-contained TD3 training algorithm. Adapted from CleanRL https://github.com/vwxyzjn/cleanrl
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

# from model import td3
from model_td3 import TD3Actor, TD3Critic
from replay_buffer import ReplayBuffer
from rl_config import GlobalConfig
from birds_eye_view.bev_observation import image_to_class_labels
import rl_utils as rl_u

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
torch.set_num_threads(1)


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


def none_or_str(value):
  if value == 'None':
    return None
  return value


def save(model, optimizer, config, folder, model_file, optimizer_file):
  model_file = os.path.join(folder, model_file)
  torch.save(model.module.state_dict(), model_file)

  if optimizer is not None:
    optimizer_file = os.path.join(folder, optimizer_file)
    torch.save(optimizer.state_dict(), optimizer_file)

  json_config = jsonpickle.encode(config)
  with open(os.path.join(folder, 'config.json'), 'wt', encoding='utf-8') as f2:
    f2.write(json_config)


def parse_args(config):
  # fmt: off
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument('--rdzv_addr', default='localhost', type=str, help='Master address for the TCP store.')
  parser.add_argument('--exp_name', type=str, default=config.exp_name, help='the name of this experiment')
  parser.add_argument('--gym_id', type=str, default=config.gym_id, help='the id of the gym environment')
  parser.add_argument('--algo', type=str, default='td3', help='algorithm name, e.g., ppo or td3')
  parser.add_argument('--tcp_store_port', type=int, required=True, help='port for the key value store')
  parser.add_argument('--learning_rate',
                      type=float,
                      default=config.learning_rate,
                      help='the learning rate of the optimizer')
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

  # Algorithm specific arguments
  parser.add_argument('--total_batch_size',
                      type=int,
                      default=config.total_batch_size,
                      help='the total amount of data collected at every step across all environments')
  parser.add_argument('--total_minibatch_size',
                      type=int,
                      default=config.total_minibatch_size,
                      help='the total minibatch sized used for training (across all environments)')
  parser.add_argument('--lr_schedule',
                      default=config.lr_schedule,
                      type=str,
                      help='Which lr schedule to use. Options: (linear, kl, none, step, cosine, cosine_restart)')
  parser.add_argument('--gae',
                      type=lambda x: bool(strtobool(x)),
                      default=config.gae,
                      nargs='?',
                      const=True,
                      help='Use GAE for advantage computation')
  parser.add_argument('--gamma', type=float, default=config.gamma, help='the discount factor gamma')
  parser.add_argument('--gae_lambda',
                      type=float,
                      default=config.gae_lambda,
                      help='the lambda for the general advantage estimation')
  parser.add_argument('--update_epochs',
                      type=int,
                      default=config.update_epochs,
                      help='the K epochs to update the policy')
  parser.add_argument('--norm_adv',
                      type=lambda x: bool(strtobool(x)),
                      default=config.norm_adv,
                      nargs='?',
                      const=True,
                      help='Toggles advantages normalization')
  parser.add_argument('--clip_coef', type=float, default=config.clip_coef, help='the surrogate clipping coefficient')
  parser.add_argument('--clip_vloss',
                      type=lambda x: bool(strtobool(x)),
                      default=config.clip_vloss,
                      nargs='?',
                      const=True,
                      help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
  parser.add_argument('--ent_coef', type=float, default=config.ent_coef, help='coefficient of the entropy')
  parser.add_argument('--vf_coef', type=float, default=config.vf_coef, help='coefficient of the value function')
  parser.add_argument('--max_grad_norm',
                      type=float,
                      default=config.max_grad_norm,
                      help='the maximum norm for the gradient clipping')
  parser.add_argument('--target_kl', type=float, default=config.target_kl, help='the target KL divergence threshold')
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
  parser.add_argument('--expl_coef', type=float, default=config.expl_coef, help='coefficient of the exploration')
  parser.add_argument('--adam_eps', type=float, default=config.adam_eps, help='eps parameter of adam optimizer')
  parser.add_argument('--cpu_collect',
                      type=lambda x: bool(strtobool(x)),
                      default=config.cpu_collect,
                      nargs='?',
                      const=True,
                      help='If true than the agent will be run on the cpu during data collection. Can be faster.')
  parser.add_argument('--use_exploration_suggest',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_exploration_suggest,
                      nargs='?',
                      const=True,
                      help='Whether to use the exploration loss from roach.')
  parser.add_argument('--use_speed_limit_as_max_speed',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_speed_limit_as_max_speed,
                      nargs='?',
                      const=True,
                      help='If true rr_maximum_speed will be overwritten to the current speed limit affecting the ego '
                      'vehicle.')
  parser.add_argument('--beta_min_a_b_value',
                      type=float,
                      default=config.beta_min_a_b_value,
                      help='Nugget that gets added to the softplus output of the network.'
                      'Aims to prevent degenerate distributions with the Beta.')
  parser.add_argument('--use_rpo',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_rpo,
                      nargs='?',
                      const=True,
                      help='Whether to use robust policy optimization (add noise to mean during training)')
  parser.add_argument('--rpo_alpha',
                      type=float,
                      default=config.rpo_alpha,
                      help='Noise added during training is of uniform shape [-rpo_alpha, rpo_alpha]')
  parser.add_argument('--use_new_bev_obs',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_new_bev_obs,
                      nargs='?',
                      const=True,
                      help='Whether to use the new bev observation or the old roach one.')
  parser.add_argument('--obs_num_channels',
                      type=int,
                      default=config.obs_num_channels,
                      help='Number of channels is the BEV image.')
  parser.add_argument('--map_folder',
                      type=str,
                      default=config.map_folder,
                      help='The map folder to use with the new representation. Needs to align with pixels_per_meter')
  parser.add_argument('--pixels_per_meter',
                      type=float,
                      default=config.pixels_per_meter,
                      help='Pixels per meter in the new BEV image.')
  parser.add_argument('--route_width',
                      type=int,
                      default=config.route_width,
                      help='Width of the rendered route in pixel. Only affects new obs.')
  parser.add_argument('--reward_type',
                      type=str,
                      default=config.reward_type,
                      help='Reward function to be used during training. Options: roach, simple_reward')
  parser.add_argument('--consider_tl',
                      type=lambda x: bool(strtobool(x)),
                      default=config.consider_tl,
                      nargs='?',
                      const=True,
                      help='If set to false traffic light infractions are turned off. Used in simple reward')
  parser.add_argument('--eval_time', type=float, default=config.eval_time, help='Time until the model times out.')
  parser.add_argument('--terminal_reward',
                      type=float,
                      default=config.terminal_reward,
                      help='Time until the model times out.')
  parser.add_argument('--normalize_rewards',
                      type=lambda x: bool(strtobool(x)),
                      default=config.normalize_rewards,
                      nargs='?',
                      const=True,
                      help=' Whether to use gymnasiums reward normalization.')
  parser.add_argument('--speeding_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.speeding_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route if the agent drives too fast. Only simple reward.')
  parser.add_argument('--min_thresh_lat_dist',
                      type=float,
                      default=config.min_thresh_lat_dist,
                      help='Time until the model times out.')
  parser.add_argument('--num_route_points_rendered',
                      type=int,
                      default=config.num_route_points_rendered,
                      help='Number of route points rendered into the BEV seg observation.')
  parser.add_argument('--use_green_wave',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_green_wave,
                      nargs='?',
                      const=True,
                      help='If True in some routes all TL that the agent encounters are set to green.')
  parser.add_argument('--image_encoder',
                      type=str,
                      default=config.image_encoder,
                      help='Which image cnn encoder to use. Either roach, roach_ln, or timm model name')
  parser.add_argument('--use_comfort_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_comfort_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to apply a soft penalty if comfort limits are exceeded.')
  parser.add_argument('--use_layer_norm',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_layer_norm,
                      nargs='?',
                      const=True,
                      help='Whether to use LayerNorm before ReLU in MLPs.')
  parser.add_argument('--use_vehicle_close_penalty',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_vehicle_close_penalty,
                      nargs='?',
                      const=True,
                      help='Whether to use a penalty for being too close to the front vehicle.')
  parser.add_argument('--render_green_tl',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_green_tl,
                      nargs='?',
                      const=True,
                      help='Whether to render green traffic lights into the observation.')
  parser.add_argument('--distribution',
                      type=str,
                      default=config.distribution,
                      help='Distribution used for the action space. Options beta, normal, beta_uni_mix')
  parser.add_argument('--weight_decay',
                      type=float,
                      default=config.weight_decay,
                      help='Weight decay applied to optimizer. AdamW is used when > 0.0')
  parser.add_argument('--use_dd_ppo_preempt',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_dd_ppo_preempt,
                      nargs='?',
                      const=True,
                      help='Whether to use the dd-ppo preemption technique to early stop stragglers.')
  parser.add_argument('--use_termination_hint',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_termination_hint,
                      nargs='?',
                      const=True,
                      help='Whether to give a penalty depending on vehicle speed when crashing or running red light.')
  parser.add_argument('--use_perc_progress',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_perc_progress,
                      nargs='?',
                      const=True,
                      help='Whether to multiply RC reward by percentage away from lane center.')
  parser.add_argument('--use_min_speed_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_min_speed_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to penalize the agent for driving slower than other agents on avg.')
  parser.add_argument('--use_leave_route_done',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_leave_route_done,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route when leaving the precomputed path.')
  parser.add_argument('--use_temperature',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_temperature,
                      nargs='?',
                      const=True,
                      help='Whether to scale the output distribution values with a predicted temperature.')
  parser.add_argument('--obs_num_measurements',
                      type=int,
                      default=config.obs_num_measurements,
                      help='Number of scalar measurements in observation.')
  parser.add_argument('--use_extra_control_inputs',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_extra_control_inputs,
                      nargs='?',
                      const=True,
                      help='Whether to use extra control inputs such as integral of past steering.')
  parser.add_argument('--condition_outside_junction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.condition_outside_junction,
                      nargs='?',
                      const=True,
                      help=' Whether to render the route outside junctions.')
  parser.add_argument('--use_layer_norm_policy_head',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_layer_norm_policy_head,
                      nargs='?',
                      const=True,
                      help='Applicable if use_layer_norm=True, whether to also apply layernorm to the policy head.'
                      'Can be useful to remove to allow the policy to predict large values (for a, b of Beta).')
  parser.add_argument('--use_hl_gauss_value_loss',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_hl_gauss_value_loss,
                      nargs='?',
                      const=True,
                      help='Whether to use the histogram loss gauss to train the value head via classification '
                      '(instead of regression + L2)')
  parser.add_argument('--use_outside_route_lanes',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_outside_route_lanes,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route when invading opposing lanes or sidewalks.')
  parser.add_argument('--use_max_change_penalty',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_max_change_penalty,
                      nargs='?',
                      const=True,
                      help='Whether to apply a soft penalty when the action changes too fast.')
  parser.add_argument('--use_lstm',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_lstm,
                      nargs='?',
                      const=True,
                      help=' Whether to use an LSTM after the feature encoder.')
  parser.add_argument('--terminal_hint',
                      type=float,
                      default=config.terminal_hint,
                      help='Reward at the end of the episode when colliding, the number will be subtracted.')
  parser.add_argument('--penalize_yellow_light',
                      type=lambda x: bool(strtobool(x)),
                      default=config.penalize_yellow_light,
                      nargs='?',
                      const=True,
                      help='Whether to penalize running a yellow light.')
  parser.add_argument('--use_target_point',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_target_point,
                      nargs='?',
                      const=True,
                      help='Whether to input a target point in the measurements.')
  parser.add_argument('--use_value_measurements',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_value_measurements,
                      nargs='?',
                      const=True,
                      help='Whether to use value measurements (otherwise all are set to 0)')
  parser.add_argument('--bev_semantics_width',
                      type=int,
                      default=config.bev_semantics_width,
                      help='Numer of pixels the bev_semantics is wide')
  parser.add_argument('--bev_semantics_height',
                      type=int,
                      default=config.bev_semantics_height,
                      help='Numer of pixels the bev_semantics is high')
  parser.add_argument('--pixels_ev_to_bottom',
                      type=int,
                      default=config.pixels_ev_to_bottom,
                      help='Numer of pixels from the vehicle to the bottom.')
  parser.add_argument('--use_history',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_history,
                      nargs='?',
                      const=True,
                      help='Whether to use historic vehicle and pedestrian observations in BEV obs')
  parser.add_argument('--use_off_road_term',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_off_road_term,
                      nargs='?',
                      const=True,
                      help='Whether to terminate when the agent drives off the drivable area')
  parser.add_argument('--off_road_term_perc',
                      type=float,
                      default=config.off_road_term_perc,
                      help='Percentage of agent overlap with off-road, that triggers the termination')
  parser.add_argument('--beta_1', type=float, default=config.beta_1, help='Beta 1 parameter of adam')
  parser.add_argument('--beta_2', type=float, default=config.beta_2, help='Beta 2 parameter of adam')
  parser.add_argument('--render_speed_lines',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_speed_lines,
                      nargs='?',
                      const=True,
                      help='Whether to render the speed lines for moving objects')
  parser.add_argument('--use_new_stop_sign_detector',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_new_stop_sign_detector,
                      nargs='?',
                      const=True,
                      help='Whether to use a different stop sign detector that prevents the policy from cheating by '
                      'changing lanes.')
  parser.add_argument('--use_positional_encoding',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_positional_encoding,
                      nargs='?',
                      const=True,
                      help='Whether to add positional encoding to the image')
  parser.add_argument('--use_ttc',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_ttc,
                      nargs='?',
                      const=True,
                      help='Whether to use TTC in the reward.')
  parser.add_argument('--num_value_measurements',
                      type=int,
                      default=config.num_value_measurements,
                      help='Number of measurements exclusive to the value head.')
  parser.add_argument('--render_yellow_time',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_yellow_time,
                      nargs='?',
                      const=True,
                      help='Whether to indicate the remaining time to red in yellow light rendering.')
  parser.add_argument('--use_single_reward',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_single_reward,
                      nargs='?',
                      const=True,
                      help='Whether to only use RC als reward source in simple reward')
  parser.add_argument('--use_rl_termination_hint',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_rl_termination_hint,
                      nargs='?',
                      const=True,
                      help='Whether to include rl infraction for termination hints')
  parser.add_argument('--render_shoulder',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_shoulder,
                      nargs='?',
                      const=True,
                      help='Whether to render shoulder lanes as roads.')
  parser.add_argument('--use_shoulder_channel',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_shoulder_channel,
                      nargs='?',
                      const=True,
                      help='Whether to use an extra channel for shoulder lanes.')
  parser.add_argument('--lane_distance_violation_threshold',
                      type=float,
                      default=config.lane_distance_violation_threshold,
                      help='Grace distance in m at which no lane perc penalty is applied')
  parser.add_argument('--lane_dist_penalty_softener',
                      type=float,
                      default=config.lane_dist_penalty_softener,
                      help='If smaller than 1 reduces lane distance penalty')
  parser.add_argument('--comfort_penalty_factor',
                      type=float,
                      default=config.comfort_penalty_factor,
                      help='Max comfort penalty if all comfort metrics are violated.')
  parser.add_argument('--use_survival_reward',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_survival_reward,
                      nargs='?',
                      const=True,
                      help='Whether to add a constant reward every frame')
  parser.add_argument('--td3_reward_scale', type=float, default=config.td3_reward_scale,
                      help='Scaling factor applied to rewards for TD3')
  # TD3-specific hyperparameters
  parser.add_argument('--buffer_size', type=int, default=int(1e6), help='Replay buffer capacity for TD3')
  parser.add_argument('--batch_size', type=int, default=256, help='TD3 batch size for updates')
  parser.add_argument('--learning_starts', type=int, default=5000, help='Timesteps collected before updates start')
  parser.add_argument('--exploration_noise', type=float, default=0.1, help='Std of Gaussian exploration noise')
  parser.add_argument('--policy_noise', type=float, default=0.2, help='Noise added to target policy during backup')
  parser.add_argument('--noise_clip', type=float, default=0.5, help='Clamping range for target policy noise')
  parser.add_argument('--tau', type=float, default=0.005, help='Target network smoothing coefficient')
  parser.add_argument('--policy_delay', type=int, default=2, help='How many critic updates per actor update')

  args, unknown = parser.parse_known_args()
  print('Unkown Arguments', unknown)
  # fmt: on
  return args


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
      env = gym.wrappers.NormalizeReward(env, gamma=config.gamma)
      env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

  return thunk


def soft_update(source, target, tau):
  for src_param, tgt_param in zip(source.parameters(), target.parameters()):
    tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)


def obs_to_tensor(obs, device):
  return {
      'bev_semantics': torch.tensor(obs['bev_semantics'], device=device, dtype=torch.float32) / 255.0,
      'measurements': torch.tensor(obs['measurements'], device=device, dtype=torch.float32),
  }


def build_spaces(config):
  obs_space = gym.spaces.Dict({
      'bev_semantics':
          gym.spaces.Box(
              0,
              255,
              shape=(config.obs_num_channels, config.bev_semantics_height, config.bev_semantics_width),
              dtype=np.uint8),
      'measurements':
          gym.spaces.Box(-np.inf, np.inf, shape=(config.obs_num_measurements,), dtype=np.float32),
  })

  action_space = gym.spaces.Box(config.action_space_min,
                                 config.action_space_max,
                                 shape=(config.action_space_dim,),
                                 dtype=np.float32)
  return obs_space, action_space


def init_rollout_sockets(args, config, local_rank, rank):
  context = zmq.Context()
  sockets = []
  recv_counters = []
  current_folder = pathlib.Path(__file__).parent.resolve()
  comm_folder = os.path.join(current_folder, 'comm_files')
  pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)

  for env_idx in range(args.num_envs_per_proc):
    sock = context.socket(zmq.PAIR)
    communication_file = os.path.join(comm_folder, str(args.ports[args.num_envs_per_proc * local_rank + env_idx]))
    sock.bind(f'ipc://{communication_file}.lock')
    hello_msg = sock.recv_string()
    print(f'Rank {rank}, env {env_idx} handshake: {hello_msg}')
    sockets.append(sock)
    recv_counters.append(0)

  return context, sockets, recv_counters


def recv_env_message(sock, env_idx, recv_counters, config):
  data = sock.recv_multipart(copy=False)
  recv_counters[env_idx] += 1

  bev = np.frombuffer(data[0], dtype=np.uint8).reshape(config.obs_num_channels, config.bev_semantics_height,
                                                        config.bev_semantics_width)
  measurements = np.frombuffer(data[1], dtype=np.float32)
  _ = data[2]  # consume action mask / padding to keep alignment with leaderboard protocol
  reward = float(np.frombuffer(data[3], dtype=np.float32).item())
  termination = bool(np.frombuffer(data[4], dtype=np.bool_).item())
  truncation = bool(np.frombuffer(data[5], dtype=np.bool_).item())
  info = {
      'n_steps': int(np.frombuffer(data[6], dtype=np.int32).item()),
      'suggest': int(np.frombuffer(data[7], dtype=np.int32).item()),
  }
  num_sent = np.frombuffer(data[8], dtype=np.uint64).item()

  if recv_counters[env_idx] != num_sent:
    raise ValueError(
        f'Communication breakdown, Leaderboard sent {num_sent} frames but learner consumed {recv_counters[env_idx]}')

  obs = {'bev_semantics': bev, 'measurements': measurements}
  return obs, reward, termination, truncation, info


def stack_obs_list(obs_list):
  return {
      'bev_semantics': np.stack([obs['bev_semantics'] for obs in obs_list], axis=0),
      'measurements': np.stack([obs['measurements'] for obs in obs_list], axis=0),
  }


def save_checkpoint(folder, prefix, actor, critic, actor_opt, critic_opt, config, global_step):
  checkpoint = {
      'actor': actor.module.state_dict(),
      'critic': critic.module.state_dict(),
      'actor_optimizer': actor_opt.state_dict(),
      'critic_optimizer': critic_opt.state_dict(),
      'config': config.__dict__,
      'global_step': global_step,
  }
  torch.save(checkpoint, os.path.join(folder, f'{prefix}.pth'))
  json_config = jsonpickle.encode(config)
  with open(os.path.join(folder, 'config.json'), 'wt', encoding='utf-8') as f2:
    f2.write(json_config)


def main():
  register(
      id='CARLAEnv-v0',
      entry_point='env_gym:CARLAEnv',
      max_episode_steps=None,
  )
  config = GlobalConfig()
  args = parse_args(config)
  # Torchrun initialization
  rank = int(os.environ['RANK'])
  local_rank = int(os.environ['LOCAL_RANK'])
  world_size = int(os.environ['WORLD_SIZE'])
  print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')

  run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}'
  writer = None
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
          settings=wandb.Settings(_disable_stats=True, _disable_meta=True, start_method='fork'),
      )

    writer = SummaryWriter(exp_folder)
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  print('Is cuda available?:', torch.cuda.is_available())

  if args.load_file is not None:
    load_folder = pathlib.Path(args.load_file).parent.resolve()
    config_path = os.path.join(load_folder, 'config.json')
    if os.path.exists(config_path):
      with open(config_path, 'rt', encoding='utf-8') as f:
        json_config = f.read()
      loaded_config = jsonpickle.decode(json_config)
      config.__dict__.update(loaded_config.__dict__)

  config.initialize(**vars(args))

  if config.use_dd_ppo_preempt:
    num_rollouts_done_store = torch.distributed.TCPStore(args.rdzv_addr, args.tcp_store_port, world_size, rank == 0)
    torch.distributed.init_process_group(backend='gloo' if
                                         (args.cpu_collect or args.num_envs_per_proc == 1) else 'nccl',
                                         store=num_rollouts_done_store,
                                         world_size=world_size,
                                         rank=rank,
                                         timeout=datetime.timedelta(minutes=45))
  else:
    torch.distributed.init_process_group(backend='gloo' if
                                         (args.cpu_collect or args.num_envs_per_proc == 1) else 'nccl',
                                         init_method='env://',
                                         world_size=world_size,
                                         rank=rank,
                                         timeout=datetime.timedelta(minutes=45))

  device_id = args.gpu_ids[rank]
  if device_id < 0:
    print('ERROR! Device id must be positive.')

  device = torch.device(f'cuda:{device_id}') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
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

  # Send config to leaderboard workers via ZMQ (unchanged setup)
  context = zmq.Context()
  socket_list = []
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

  obs_space, action_space = build_spaces(config)
  assert isinstance(action_space, gym.spaces.Box), 'only continuous action space is supported'

  num_envs = args.num_envs_per_proc
  action_dim = action_space.shape[0]

  actor = TD3Actor(obs_space, action_dim, config).to(device)
  critic = TD3Critic(obs_space, action_dim, config).to(device)
  actor_target = TD3Actor(obs_space, action_dim, config).to(device)
  critic_target = TD3Critic(obs_space, action_dim, config).to(device)
  actor_target.load_state_dict(actor.state_dict())
  critic_target.load_state_dict(critic.state_dict())

  action_low = torch.as_tensor(action_space.low, device=device, dtype=torch.float32)
  action_high = torch.as_tensor(action_space.high, device=device, dtype=torch.float32)
  action_scale = action_high  # Assume symmetric bounds

  if config.compile_model:
    actor = torch.compile(actor)
    critic = torch.compile(critic)

  actor = torch.nn.parallel.DistributedDataParallel(actor,
                                                    device_ids=None,
                                                    output_device=None,
                                                    broadcast_buffers=False,
                                                    find_unused_parameters=False)
  critic = torch.nn.parallel.DistributedDataParallel(critic,
                                                     device_ids=None,
                                                     output_device=None,
                                                     broadcast_buffers=False,
                                                     find_unused_parameters=False)

  actor_optimizer = optim.Adam(actor.parameters(),
                               lr=config.learning_rate,
                               eps=config.adam_eps,
                               betas=(config.beta_1, config.beta_2))
  critic_optimizer = optim.Adam(critic.parameters(),
                                lr=config.learning_rate,
                                eps=config.adam_eps,
                                betas=(config.beta_1, config.beta_2))

  start_step = 0
  if args.load_file is not None:
    checkpoint = torch.load(args.load_file, map_location=device)
    if isinstance(checkpoint, dict):
      if 'actor' in checkpoint:
        actor.module.load_state_dict(checkpoint['actor'])
      if 'critic' in checkpoint:
        critic.module.load_state_dict(checkpoint['critic'])
      if 'actor_optimizer' in checkpoint and checkpoint['actor_optimizer'] is not None:
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
      if 'critic_optimizer' in checkpoint and checkpoint['critic_optimizer'] is not None:
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
      start_step = checkpoint.get('global_step', 0)
      if 'config' in checkpoint:
        config.__dict__.update(checkpoint['config'])
    actor_target.load_state_dict(actor.module.state_dict())
    critic_target.load_state_dict(critic.module.state_dict())
    if writer is not None:
      writer.add_scalar('charts/restart', 1, start_step)

  replay_buffer = ReplayBuffer(capacity=args.buffer_size,
                               obs_space=obs_space,
                               action_dim=action_dim,
                               device=device)

  rollout_context, rollout_sockets, recv_counters = init_rollout_sockets(args, config, local_rank, rank)

  initial_obs_list = []
  for env_idx, sock in enumerate(rollout_sockets):
    init_obs, _, _, _, _ = recv_env_message(sock, env_idx, recv_counters, config)
    initial_obs_list.append(init_obs)

  obs = stack_obs_list(initial_obs_list)

  start_time = time.time()
  episode_returns = deque(maxlen=100)
  episode_lengths = deque(maxlen=100)
  episode_returns_env = np.zeros(num_envs, dtype=np.float32)
  episode_lengths_env = np.zeros(num_envs, dtype=np.int32)

  for global_step in range(start_step, args.total_timesteps):
    config.global_step = global_step

    if global_step < args.learning_starts:
      actions = np.stack([action_space.sample() for _ in range(num_envs)])
    else:
      obs_tensor = obs_to_tensor(obs, device)
      with torch.no_grad():
        action_tensor = actor(obs_tensor)
      noise = torch.zeros_like(action_tensor)
      sigma = max(0.05, 0.2 * (1 - global_step / args.total_timesteps))  # Decay noise over time
      noise[:, 0] = torch.randn_like(action_tensor[:, 0]) * sigma  # steer
      noise[:, 1] = torch.randn_like(action_tensor[:, 1]) * sigma * 0.5  # throttle

      action_tensor = torch.clamp(
          action_tensor + noise,
          action_low,
          action_high
      )
      actions = action_tensor.cpu().numpy()

    for env_idx, sock in enumerate(rollout_sockets):
      sock.send(np.asarray(actions[env_idx], dtype=np.float32).tobytes(), copy=False)

    next_obs_list = []
    rewards = np.zeros(num_envs, dtype=np.float32)
    dones = np.zeros(num_envs, dtype=bool)

    for env_idx, sock in enumerate(rollout_sockets):
      obs_i, reward_i, termination_i, truncation_i, _ = recv_env_message(sock, env_idx, recv_counters, config)
      next_obs_list.append(obs_i)
      rewards[env_idx] = reward_i
      dones[env_idx] = termination_i  # Only consider termination as done

    next_obs = stack_obs_list(next_obs_list)

    for env_idx in range(num_envs):
      replay_buffer.add(
          {
              'bev_semantics': obs['bev_semantics'][env_idx],
              'measurements': obs['measurements'][env_idx],
          },
          actions[env_idx],
          rewards[env_idx],
          {
              'bev_semantics': next_obs['bev_semantics'][env_idx],
              'measurements': next_obs['measurements'][env_idx],
          },
          float(dones[env_idx]),
      )

      episode_returns_env[env_idx] += rewards[env_idx]
      episode_lengths_env[env_idx] += 1

      if dones[env_idx]:
        episode_returns.append(episode_returns_env[env_idx])
        episode_lengths.append(int(episode_lengths_env[env_idx]))
        if writer is not None:
          writer.add_scalar('charts/episodic_return', episode_returns_env[env_idx], global_step)
          writer.add_scalar('charts/episodic_length', episode_lengths_env[env_idx], global_step)
        episode_returns_env[env_idx] = 0.0
        episode_lengths_env[env_idx] = 0

    obs = next_obs

    actor_loss = None
    critic_loss = None
    if replay_buffer.size >= args.batch_size and global_step >= args.learning_starts:
      batch = replay_buffer.sample(args.batch_size)
      assert isinstance(batch['obs'], dict), 'ReplayBuffer must return dict observations'
      assert batch['obs']['bev_semantics'].dtype == torch.float32, 'Obs tensors must be float32'

      with torch.no_grad():
        noise = torch.randn_like(batch['actions']) * args.policy_noise  # Symmetric noise
        noise = noise.clamp(-args.noise_clip, args.noise_clip)  # Clip noise
        next_actions = (actor_target(batch['next_obs']) + noise).clamp(-1, 1)  # Clamp actions to [-1, 1]
        target_q1, target_q2 = critic_target(batch['next_obs'], next_actions)
        target_q = torch.min(target_q1, target_q2)
        batch['rewards'] = torch.clamp(batch['rewards'], -10.0, 10.0)  # Clamp rewards to stabilize training
        target = batch['rewards'] + args.gamma * (1 - batch['dones']) * target_q  # Adjusted target calculation

      current_q1, current_q2 = critic(batch['obs'], batch['actions'])
      critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
      critic_optimizer.zero_grad()
      critic_loss.backward()
      nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
      critic_optimizer.step()

      if (global_step % args.policy_delay) == 0:
        actor_actions = actor(batch['obs'])
        actor_loss = -critic(batch['obs'], actor_actions)[0].mean()
        # Add smooth penalty term to the actor loss
        smooth_loss = 0.001 * torch.mean(actor_actions[:, 0]**2)  # Reduced smooth penalty for steering
        actor_loss = actor_loss + smooth_loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_optimizer.step()
        soft_update(actor.module, actor_target, args.tau)
        soft_update(critic.module, critic_target, args.tau)

    if writer is not None and global_step % 100 == 0:
      if critic_loss is not None:
        writer.add_scalar('losses/critic', critic_loss.item(), global_step)
        writer.add_scalar('debug/q1_mean', current_q1.mean().item(), global_step)
        writer.add_scalar('debug/q2_mean', current_q2.mean().item(), global_step)
      if actor_loss is not None:
        writer.add_scalar('losses/actor', actor_loss.item(), global_step)
      if len(episode_returns) > 0:
        writer.add_scalar('charts/windowed_avg_return', sum(episode_returns) / len(episode_returns), global_step)
      steps_per_second = int(((global_step + 1) * num_envs) / (time.time() - start_time))
      writer.add_scalar('charts/SPS', steps_per_second, global_step)

    if rank == 0 and global_step % 5000 == 0 and global_step > start_step:
      save_checkpoint(exp_folder, f'model_latest_{global_step:09d}', actor, critic, actor_optimizer,
                      critic_optimizer, config, global_step)

  for sock in rollout_sockets:
    sock.close()
  rollout_context.term()
  if writer is not None:
    save_checkpoint(exp_folder, 'model_final', actor, critic, actor_optimizer, critic_optimizer, config,
                    args.total_timesteps)
    writer.close()
    if rank == 0:
      wandb.finish(exit_code=0, quiet=True)
      print('Done training.')


if __name__ == '__main__':
  main()
