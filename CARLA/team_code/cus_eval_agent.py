import os
import math
from collections import deque

import carla
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

# ====== YOUR CODEBASE IMPORTS ======
from rl_config import GlobalConfig
import rl_utils as rl_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.bev_observation import ObsManager as ObsManager2
from birds_eye_view.run_stop_sign import RunStopSign
from nav_planner import RoutePlanner
from model import PPOPolicy


class CustomEvalAgent:
    """
    Standalone RL inference agent (NO leaderboard, NO scenario_runner)
    """

    def __init__(self):
        self.initialized = False
        self.step = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # 1. LOAD CONFIG + MODEL
    # =====================================================
    def setup(self, conf_dir: str):
        """
        conf_dir:
            - config.json
            - model_final.pth (or model_*.pth)
        """
        self.conf_dir = conf_dir
        self.config = GlobalConfig()

        # ---- load config.json ----
        with open(os.path.join(conf_dir, "config.json"), "r", encoding="utf-8") as f:
            import jsonpickle
            loaded_cfg = jsonpickle.decode(f.read())
        self.config.__dict__.update(loaded_cfg.__dict__)

        # ---- env flags ----
        self.sample_type = os.environ.get("SAMPLE_TYPE", "mean")
        self.high_freq_inference = int(os.environ.get("HIGH_FREQ_INFERENCE", 0))

        # ---- observation / action space ----
        self.observation_space = spaces.Dict({
            "bev_semantics": spaces.Box(
                0, 255,
                shape=(
                    self.config.obs_num_channels,
                    self.config.bev_semantics_height,
                    self.config.bev_semantics_width
                ),
                dtype=np.uint8
            ),
            "measurements": spaces.Box(
                -math.inf, math.inf,
                shape=(self.config.obs_num_measurements,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Box(
            self.config.action_space_min,
            self.config.action_space_max,
            shape=(self.config.action_space_dim,),
            dtype=np.float32
        )

        # ---- load models ----
        self.agents = []
        for file in os.listdir(conf_dir):
            if file.startswith("model") and file.endswith(".pth"):
                model = PPOPolicy(
                    self.observation_space,
                    self.action_space,
                    config=self.config
                ).to(self.device)

                state = torch.load(
                    os.path.join(conf_dir, file),
                    map_location=self.device
                )
                model.load_state_dict(state, strict=True)
                model.eval()
                self.agents.append(model)

        assert len(self.agents) > 0, "❌ No model*.pth found"

        self.model_count = len(self.agents)

        # ---- LSTM state (if used) ----
        self.last_lstm_states = [
            (
                torch.zeros(
                    self.config.num_lstm_layers, 1,
                    self.config.features_dim, device=self.device
                ),
                torch.zeros(
                    self.config.num_lstm_layers, 1,
                    self.config.features_dim, device=self.device
                )
            )
            for _ in range(self.model_count)
        ]
        self.done = torch.zeros(1, device=self.device)

        # ---- action repeat ----
        if self.high_freq_inference:
            self.total_action_repeat = int(self.config.action_repeat)
        else:
            self.total_action_repeat = int(
                self.config.action_repeat *
                (self.config.original_frame_rate // self.config.frame_rate)
            )

        print(f"[Agent] Loaded {self.model_count} model(s)")
        self.initialized = False

    # =====================================================
    # 2. ATTACH TO WORLD + VEHICLE + ROUTE
    # =====================================================
    def attach(self, world: carla.World, vehicle: carla.Vehicle, dense_route):
        """
        dense_route: list of (carla.Transform, RoadOption)
        """
        self.world = world
        self.vehicle = vehicle
        self.world_map = world.get_map()
        self.dense_global_plan_world_coord = dense_route

        # ---- stop sign & BEV ----
        self.stop_sign_criteria = RunStopSign(self.world, self.world_map)

        if self.config.use_new_bev_obs:
            self.bev_manager = ObsManager2(self.config)
            self.bev_manager.attach_ego_vehicle(
                self.vehicle,
                self.stop_sign_criteria,
                self.world_map,
                dense_route
            )
        else:
            self.bev_manager = ObsManager(self.config)
            self.bev_manager.attach_ego_vehicle(
                self.vehicle,
                self.stop_sign_criteria,
                self.world_map
            )

        # ---- route planner ----
        self.route_planner = RoutePlanner()
        self.route_planner.set_route(dense_route)

        self.initialized = True
        print("[Agent] Attached to vehicle")

    # =====================================================
    # 3. OBSERVATION
    # =====================================================
    def _get_waypoint_route(self):
        loc = self.vehicle.get_location()
        pos = np.array([loc.x, loc.y])
        return self.route_planner.run_step(pos)

    def _preprocess_obs(self, waypoint_route):
        self.stop_sign_criteria.tick(self.vehicle)

        actors = self.world.get_actors()
        vehicles = actors.filter("*vehicle*")
        walkers = actors.filter("*walker*")
        static = actors.filter("*static*")

        bev = self.bev_manager.get_observation(
            waypoint_route,
            vehicles,
            walkers,
            static,
            debug=False
        )

        # ---- measurements ----
        control = self.vehicle.get_control()
        vel = self.vehicle.get_velocity()
        tf = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()

        np_vel = np.array([vel.x, vel.y, vel.z])
        np_fwd = np.array([fwd.x, fwd.y, fwd.z])
        forward_speed = np.dot(np_vel, np_fwd)

        vel_ego = rl_u.inverse_conversion_2d(
            np.array([vel.x, vel.y]),
            np.zeros(2),
            np.deg2rad(tf.rotation.yaw)
        )

        speed_limit = self.vehicle.get_speed_limit()
        if isinstance(speed_limit, float):
            max_speed = speed_limit / 3.6
        else:
            max_speed = self.config.rr_maximum_speed

        measurements = np.array([
            control.steer,
            control.throttle,
            control.brake,
            float(control.gear),
            float(vel_ego[0]),
            float(vel_ego[1]),
            float(forward_speed),
            max_speed
        ], dtype=np.float32)

        obs = {
            "bev_semantics": bev["bev_semantic_classes"],
            "measurements": measurements,
            "value_measurements": np.zeros(
                (self.config.num_value_measurements,),
                dtype=np.float32
            )
        }
        return obs

    # =====================================================
    # 4. MAIN INFERENCE STEP
    # =====================================================
    @torch.inference_mode()
    def run_step(self, timestamp: int):
        if not self.initialized:
            return carla.VehicleControl(brake=1.0)

        self.step += 1

        if self.step % self.total_action_repeat != 0:
            return self.last_control

        waypoint_route = self._get_waypoint_route()
        obs = self._preprocess_obs(waypoint_route)

        obs_tensor = {
            "bev_semantics": torch.tensor(
                obs["bev_semantics"][None],
                device=self.device,
                dtype=torch.float32
            ),
            "measurements": torch.tensor(
                obs["measurements"][None],
                device=self.device
            ),
            "value_measurements": torch.tensor(
                obs["value_measurements"][None],
                device=self.device
            )
        }

        actions = []
        for i, agent in enumerate(self.agents):
            act, *_ , self.last_lstm_states[i] = agent.forward(
                obs_tensor,
                sample_type=self.sample_type,
                lstm_state=self.last_lstm_states[i],
                done=self.done
            )
            actions.append(act)

        action = torch.stack(actions).mean(dim=0)[0].cpu().numpy()
        control = self._action_to_control(action)
        self.last_control = control
        return control

    # =====================================================
    # 5. ACTION → CONTROL
    # =====================================================
    def _action_to_control(self, action):
        steer = float(action[0])
        acc = float(action[1])

        if acc >= 0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = -acc

        return carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake
        )

    # =====================================================
    # 6. CLEANUP
    # =====================================================
    def destroy(self):
        self.agents.clear()
        self.last_lstm_states.clear()
        print("[Agent] Destroyed")
