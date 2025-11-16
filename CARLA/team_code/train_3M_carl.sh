#!/bin/bash
# train_sac_3M.sh - Main training script for SAC
# Usage: bash train_sac_3M.sh

start=`date +%s`
echo "START TIME: $(date)"

# ============================================================================
# ENVIRONMENT SETUP (Update paths for your system)
# ============================================================================
export SCENARIO_RUNNER_ROOT=/home/trung/CaRL/CARLA/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/home/trung/CaRL/CARLA/custom_leaderboard/leaderboard
export CARLA_ROOT=/home/trung/CaRL/CARLA/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# Weights & Biases login (optional, comment out if not using)
wandb login --relogin '3fe2c59de02d17028b44262900b4b0940afb05f4'

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
repetition=2
program_seed=$((000 + 100 * repetition))
start_port=$((1024 + 1000 * repetition))
ex_name=$(printf "SAC_3M_%02d" ${repetition})

# ============================================================================
# LAUNCH TRAINING
# ============================================================================
python -u train_parallel.py \
  --debug False \
  --team_code_folder /home/trung/CaRL/CARLA/team_code \
  --git_root /home/trung/CaRL/CARLA \
  --carla_root /home/trung/CaRL/CARLA/carla \
  \
  `# Experiment settings` \
  --exp_name "${ex_name}" \
  --seed ${program_seed} \
  --start_port ${start_port} \
  --algo sac \
  \
  `# Hardware settings` \
  --gpu_ids 0 \
  --num_envs_per_gpu 1 \
  --num_envs_per_node 1 \
  --num_nodes 1 \
  --node_id 0 \
  --rdzv_addr 127.0.0.1 \
  --rdzv_port 0 \
  \
  `# Training towns - start conservative with 4 towns` \
  --train_towns 1 \
  \
  `# SAC Algorithm Parameters` \
  --total_timesteps 1000000 \
  --buffer_size 100 \
  --batch_size 32 \
  --learning_starts 10 \
  --policy_lr 0.0003 \
  --q_lr 0.0003 \
  --gamma 0.99 \
  --tau 0.005 \
  --alpha 0.2 \
  --autotune_alpha False \
  --policy_frequency 2 \
  --utd_ratio 1 \
  --buffer_storage gpu \
  \
  `# Observation settings (matching CaRL)` \
  --use_new_bev_obs 1 \
  --obs_num_channels 10 \
  --bev_semantics_width 256 \
  --bev_semantics_height 256 \
  --pixels_ev_to_bottom 100 \
  --map_folder maps_2ppm_cv \
  --pixels_per_meter 2 \
  --obs_num_measurements 8 \
  --use_value_measurements 1 \
  --num_value_measurements 3 \
  \
  `# Network architecture` \
  --image_encoder roach_ln2 \
  --use_layer_norm 1 \
  --use_layer_norm_policy_head 1 \
  \
  `# Reward settings (matching CaRL)` \
  --reward_type simple_reward \
  --normalize_rewards 0 \
  --consider_tl 1 \
  --terminal_reward 0.0 \
  --terminal_hint 1.0 \
  --use_termination_hint 1 \
  --use_perc_progress 1 \
  --use_single_reward 1 \
  --use_comfort_infraction 1 \
  --comfort_penalty_factor 0.5 \
  --use_ttc 1 \
  \
  `# Route and scenario settings` \
  --routes_folder 1000_meters_old_scenarios_01 \
  --route_repetitions 20 \
  --eval_time 1200 \
  --route_width 6 \
  --num_route_points_rendered 150 \
  \
  `# Safety settings` \
  --speeding_infraction 1 \
  --min_thresh_lat_dist 2.0 \
  --use_outside_route_lanes 1 \
  --use_off_road_term 1 \
  --off_road_term_perc 0.95 \
  --penalize_yellow_light 0 \
  --use_rl_termination_hint 1 \
  \
  `# Rendering settings` \
  --render_green_tl 1 \
  --render_speed_lines 1 \
  --render_yellow_time 1 \
  --render_shoulder 0 \
  --use_shoulder_channel 1 \
  \
  `# Other settings` \
  --use_green_wave 0 \
  --use_history 0 \
  --use_new_stop_sign_detector 1 \
  --use_positional_encoding 0 \
  --use_target_point 0 \
  --use_extra_control_inputs 0 \
  --condition_outside_junction 0 \
  --use_max_change_penalty 0 \
  --use_min_speed_infraction 0 \
  --use_leave_route_done 0 \
  --use_vehicle_close_penalty 0 \
  --lane_distance_violation_threshold 0.0 \
  --lane_dist_penalty_softener 1.0 \
  --use_survival_reward 0 \
  \
  `# Tracking` \
  --track 1 \
  \
  `# SAC-specific: Disable PPO-only features` \
  --use_lstm False \
  --use_temperature False \
  --use_rpo False \
  --use_hl_gauss_value_loss False \
  --use_exploration_suggest 0 &

wait

# ============================================================================
# COMPLETION
# ============================================================================
end=`date +%s`
runtime=$((end-start))

echo "END TIME: $(date)"
printf 'Runtime: %dd:%dh:%dm:%ds\n' \
  $((${runtime}/86400)) \
  $((${runtime}%86400/3600)) \
  $((${runtime}%3600/60)) \
  $((${runtime}%60)) \
  2>&1 | tee /home/trung/CaRL/CARLA/results/"${ex_name}"/train_time.txt

echo "Training complete! Results saved to: /home/trung/CaRL/CARLA/results/${ex_name}"
