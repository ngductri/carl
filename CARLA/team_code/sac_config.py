'''
Configuration class for SAC training in CaRL.
Separates SAC-specific hyperparameters from environment/model config.
'''
import numpy as np

from rl_config import GlobalConfig
class SACConfig(GlobalConfig):
    '''
    Configuration class for SAC (Soft Actor-Critic) training.
    Contains all hyperparameters for SAC algorithm and CaRL environment.
    '''

    def __init__(self):
        self.algo = 'ppo'
        # ============================================================================
        # CARLA Environment Parameters
        # ============================================================================
        self.frame_rate = 10.0  # Frames per second of the CARLA simulator
        self.original_frame_rate = 20.0  # Frames per second of the CARLA evaluation server
        self.time_interval = 1.0 / self.frame_rate  # ms per step in CARLA time
        
        # ============================================================================
        # BEV Observation Parameters
        # ============================================================================
        self.pixels_per_meter = 5.0  # 1 / pixels_per_meter = size of pixel in meters
        self.bev_semantics_width = 192  # Number of pixels the bev_semantics is wide
        self.bev_semantics_height = 192  # Number of pixels the bev_semantics is high
        self.pixels_ev_to_bottom = 40  # Number of pixels from the vehicle to the bottom
        self.obs_num_channels = 15  # Number of channels in the bev observation
        self.obs_num_measurements = 8  # Number of scalar measurements in observation
        
        self.use_new_bev_obs = False  # Whether to use bev_observation.py instead of chauffeurnet.py
        self.route_width = 16  # Width of the rendered route in pixel
        self.red_light_thickness = 3  # Width of the red light line
        self.num_route_points_rendered = 80  # Number of route points rendered into BEV
        self.condition_outside_junction = True  # Whether to render the route outside junctions
        self.render_speed_lines = False  # Whether to render speed lines for moving objects
        self.render_yellow_time = False  # Whether to indicate remaining time to red in yellow light
        self.render_shoulder = False # Whether to render shoulder lanes as roads
        self.use_shoulder_channel = True  # Whether to use an extra channel for shoulder lanes
        self.render_green_tl = True  # Whether to render green traffic lights into observation
        
        # BEV rendering parameters
        self.scale_bbox = True  # Whether to scale up the bounding boxes
        self.scale_factor_vehicle = 1.0
        self.scale_factor_walker = 2.0
        self.min_ext_bounding_box = 0.8
        self.scale_mask_col = 1.0  # Scaling factor for ego vehicle bounding box
        self.map_folder = 'maps_low_res'  # Map folder for preprocessed map data
        
        # ============================================================================
        # Value Measurements (for critic input)
        # ============================================================================
        self.use_value_measurements = True  # Whether to use value measurements
        self.num_value_measurements = 10  # Number of measurements exclusive to value head
        
        # ============================================================================
        # Reward Function Parameters
        # ============================================================================
        self.reward_type = 'roach'  # Reward function to use. Options: roach, simple_reward
        self.normalize_rewards = False  # Whether to use gymnasium's reward normalization
        
        # Roach reward hyperparameters
        self.rr_maximum_speed = 6.0  # Maximum speed in m/s encouraged by roach reward
        self.vehicle_distance_threshold = 15  # Distance threshold for vehicles (meters)
        self.max_vehicle_detection_number = 10  # Max number of vehicles for reward
        self.rr_vehicle_proximity_threshold = 9.5  # Threshold for vehicle hazard
        self.pedestrian_distance_threshold = 15  # Distance threshold for pedestrians
        self.max_pedestrian_detection_number = 10  # Max number of pedestrians for reward
        self.rr_pedestrian_proximity_threshold = 9.5  # Threshold for pedestrian hazard
        self.rr_tl_dist_threshold = 18.0  # Distance at which traffic lights are considered
        self.min_thresh_lat_dist = 3.5  # Lateral distance threshold for route deviation
        self.use_speed_limit_as_max_speed = False  # Use current speed limit as max speed
        
        # Ego vehicle extent (for collision detection)
        self.ego_extent_x = 2.44619083404541
        self.ego_extent_y = 0.9183566570281982
        self.ego_extent_z = 0.7451388239860535
        
        # ============================================================================
        # SAC Algorithm Hyperparameters
        # ============================================================================
        # Core SAC parameters
        self.buffer_size = int(1e6)  # Replay buffer size
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Target network soft update rate (Polyak averaging)
        self.batch_size = 256  # Minibatch size for training
        self.learning_starts = 5000  # Timesteps before training starts (initial exploration)
        
        # Learning rates
        self.policy_lr = 3e-4  # Actor learning rate
        self.q_lr = 3e-4  # Critic learning rate
        self.current_policy_lr = 3e-4  # Current actor learning rate (for scheduling)
        self.current_q_lr = 3e-4  # Current critic learning rate (for scheduling)
        
        # Update frequencies
        self.policy_frequency = 2  # Actor update frequency (updates per critic update)
        self.target_network_frequency = 1  # Target network update frequency
        
        # Entropy regularization
        self.alpha = 0.2  # Temperature parameter (entropy coefficient)
        self.autotune = True  # Automatic temperature tuning
        
        # Learning rate scheduling
        self.lr_schedule = 'none'  # Options: linear, kl, none, step, cosine, cosine_restart
        self.lr_schedule_step_factor = 0.1  # Multiplier for step decrease
        self.lr_schedule_step_perc = (0.5, 0.75)  # Training percentage for lr decay
        self.lr_schedule_cosine_restarts = (0.0, 0.25, 0.50, 0.75, 1.0)  # Cosine restart points
        
        # ============================================================================
        # Neural Network Architecture
        # ============================================================================
        self.image_encoder = 'roach'  # Image encoder: roach, roach_ln, or timm model
        self.features_dim = 256  # Dimension of features from encoder
        self.use_layer_norm = False  # Whether to use LayerNorm in MLPs
        self.use_layer_norm_policy_head = True  # LayerNorm in policy head
        
        # LSTM parameters (optional recurrence)
        self.use_lstm = False  # Whether to use LSTM after feature encoder
        self.num_lstm_layers = 1  # Number of LSTM layers
        
        # Actor and critic head architectures
        self.policy_head_arch = (256, 256)  # Actor MLP hidden layers
        self.value_head_arch = (256, 256)  # Critic MLP hidden layers
        
        # ============================================================================
        # Optimizer Parameters
        # ============================================================================
        self.adam_eps = 1e-5  # Adam epsilon parameter
        self.weight_decay = 0.0  # Weight decay (use AdamW if > 0.0)
        self.beta_1 = 0.9  # Adam beta1
        self.beta_2 = 0.999  # Adam beta2
        
        # ============================================================================
        # Action Space
        # ============================================================================
        self.action_space_dim = 2  # [steering, acceleration]
        self.action_space_min = -1.0  # Minimum action value
        self.action_space_max = 1.0  # Maximum action value
        self.action_repeat = 1  # How often an action is repeated
        
        # ============================================================================
        # Training Parameters
        # ============================================================================
        self.exp_name = 'SAC_001'  # Experiment name
        self.gym_id = 'CARLAEnv-v0'  # Gym environment ID
        self.seed = 1  # Random seed
        self.total_timesteps = 10000000  # Total training timesteps
        
        # Hardware settings
        self.cuda = True  # Enable CUDA
        self.torch_deterministic = True  # Deterministic CUDA operations
        self.allow_tf32 = False  # Use TF32 format (faster but less precise)
        self.benchmark = False  # CUDNN benchmarking
        self.matmul_precision = 'highest'  # Options: highest, high, medium
        self.compile_model = False  # Use torch.compile
        
        # Multi-environment and distributed settings
        self.num_envs_per_proc = 1  # Environments per process
        self.ports = (5555,)  # CARLA ports for each environment
        self.gpu_ids = (0,)  # GPU IDs for each rank
        
        # ============================================================================
        # Logging and Evaluation
        # ============================================================================
        self.track = False  # Track with Weights & Biases
        self.wandb_project_name = 'sac-carl'  # W&B project name
        self.wandb_entity = None  # W&B entity (team)
        self.logdir = ''  # Log directory
        self.capture_video = False  # Capture episode videos
        self.visualize = False  # Render environment on screen
        
        # Evaluation and checkpointing
        self.eval_intervals = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        self.current_eval_interval_idx = 0  # Current eval checkpoint index
        
        # Model loading
        self.load_file = None  # Path to pretrained model weights
        
        # ============================================================================
        # Training State (updated during training)
        # ============================================================================
        self.global_step = 0  # Current training step
        self.max_training_score = -np.inf  # Best episodic return
        self.best_iteration = 0  # Iteration of best model
        self.latest_iteration = 0  # Latest saved iteration
        
        # ============================================================================
        # Legacy/Unused Parameters (kept for compatibility)
        # ============================================================================
        # These are from PPO but kept to avoid breaking environment/model code
        self.distribution = 'beta'  # Action distribution (not used in SAC)
        self.debug = False  # Debug mode
        self.logging_freq = 10  # Logging frequency
        self.logger_region_of_interest = 30.0  # Logging region size
        self.light_radius = 15.0  # Traffic light consideration radius
        self.route_points = 10  # Number of route points in logger
        
        # History parameters (if needed by environment)
        half_second = int(self.frame_rate * 0.5)
        self.history_idx = [-3 * half_second - 1, -2 * half_second - 1, 
                           -1 * half_second - 1, -0 * half_second - 1]
        self.history_idx_2 = [-3 * half_second - 1, -2 * half_second - 1, 
                             -1 * half_second - 1]
        self.use_history = False
        
        # BEV classes for rendering (BGR format)
        self.bev_classes_list = (
            (0, 0, 0),  # unlabeled
            (150, 150, 150),  # road
            (255, 255, 255),  # route
            (255, 255, 0),  # lane marking
            (0, 0, 255),  # vehicle
            (0, 255, 255),  # pedestrian
            (255, 255, 0),  # traffic light
            (160, 160, 0),  # stop sign
            (0, 255, 0),  # speed sign
        )
        
        # Additional environment parameters
        self.max_speed_actor = 33.33  # Max speed for other actors (m/s) = 120 km/h
        self.min_speed_actor = -2.67  # Min speed for other actors (m/s) = -10 km/h
        self.use_exploration_suggest = True  # Roach exploration loss (not used in SAC)
        self.use_extra_control_inputs = False  # Extra control inputs
        self.max_avg_steer_angle = 60.0  # Maximum average steering angle
        self.use_target_point = False  # Target point in measurements
        self.use_positional_encoding = False  # Positional encoding in image
        self.start_delay_frames = int(2.0 / self.time_interval + 0.5)  # Initial brake frames
        
        # Reward shaping parameters (simple_reward)
        self.consider_tl = True  # Consider traffic lights
        self.terminal_reward = 0.0  # Terminal reward
        self.terminal_hint = 10.0  # Penalty for collision
        self.speeding_infraction = False  # Terminate on speeding
        self.use_comfort_infraction = False  # Comfort penalty
        self.use_vehicle_close_penalty = False  # Penalty for close vehicles
        self.use_termination_hint = False  # Speed-dependent termination penalty
        self.use_perc_progress = False  # Multiply reward by lane progress
        self.use_min_speed_infraction = False  # Penalize slow driving
        self.use_leave_route_done = True  # Terminate when leaving route
        self.use_outside_route_lanes = False  # Terminate on wrong lanes
        self.use_max_change_penalty = False  # Penalty for rapid action changes
        self.penalize_yellow_light = True  # Penalize running yellow lights
        self.use_off_road_term = False  # Terminate when off-road
        self.use_new_stop_sign_detector = False  # Alternative stop sign detection
        self.use_ttc = False  # Use time-to-collision in reward
        self.use_single_reward = True  # Use only route compliance reward
        self.use_rl_termination_hint = False  # Red light termination hints
        self.use_survival_reward = False  # Constant per-frame reward
        self.use_green_wave = False  # All traffic lights green
        
        # Comfort thresholds (nuPlan dataset values)
        self.max_abs_lon_jerk = 30.0  # m/s^3
        self.max_abs_mag_jerk = 30.0  # m/s^3
        self.min_lon_accel = -20.0  # m/s^2
        self.max_lon_accel = 10.0  # m/s^2
        self.max_abs_lat_accel = 9.0  # m/s^2
        self.max_abs_yaw_rate = 1.0  # rad/s
        self.max_abs_yaw_accel = 3.0  # rad/s^2
        self.comfort_penalty_ticks = 500
        self.comfort_penalty_factor = 0.5
        
        # Other thresholds
        self.eval_time = 1200.0  # Episode timeout (seconds)
        self.n_step_exploration = 100  # Frames for exploration loss
        self.lane_distance_violation_threshold = 0.0
        self.lane_dist_penalty_softener = 1.0
        self.max_change = 0.25  # Max action change
        self.off_road_term_perc = 0.5  # Off-road overlap threshold
        self.ttc_resolution = 2  # TTC evaluation interval
        self.ttc_penalty_ticks = 500
        self.max_overspeed_value_threshold = 2.23  # m/s
        self.survival_reward_magnitude = 0.0001
        self.green_wave_prob = 0.05
        self.ego_forecast_time = 1.0  # Ego forecast seconds
        self.ego_forecast_min_speed = 2.5  # m/s
        self.rr_tl_offset = -0.8 * self.ego_extent_x

    def initialize(self, **kwargs):
        '''
        Update config attributes from kwargs (typically from argparse).
        '''
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    
    
    
    
