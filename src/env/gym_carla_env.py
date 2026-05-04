import os
import sys
import math
import random
import colorsys
import torch
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

import carla
import gym
from gym import spaces

ws_root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."
))

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src"
    )))
    sys.path.append(os.path.abspath(os.path.join(
        ws_root_path, "src", "env"
    )))
    
    from reward import decomposed_reward, baseline_reward
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from carla_client.connection import connect_carla, configure_simulation
    from managers.actors import VehicleManager, NPCManager
    from managers.sensors import SensorManager
    from managers.utils.config_manager import load_config
    from managers.utils.logger import Log
    from risk.module import RiskModule, AGENT_FEAT_DIM
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

@dataclass
class FrameRecordState:
    """
    @brief Tracks frame recording state for logging or saving images.

    @details
    Stores the current index, the start time of the recording session,
    and the directory path where frames will be saved.
    """
    
    index: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    file_path: str = None

class CarlaEnv(gym.Env):
    """
    @brief Initialise the CARLA environment.

    @details
    Loads configuration, connects to the CARLA simulator, and
    initialises observation and action spaces.

    Rendering behaviour is controlled by two independent flags from config:
        - is_display: whether to show a live OpenCV window during training
        - is_save_frames: whether to save camera frames to disk

    Frame save path is organised as:
        <render_root>/frames/<run_tag>/<datestamp>/<timestamp>_<index>.png

    @param run_tag str Identifier for the current run, used to organise
                        saved frames under results/<run_tag>/frames/.
                        Defaults to 'default'.
    @throws RuntimeError If configuration loading or CARLA connection fails.
    """

    def __init__(self, run_tag: str = "default"):
        """
        @brief Initialise the CARLA environment.

        Loads configuration, connects to the CARLA simulator, and
        initialises observation and action spaces.

        @param run_tag str Identifier for the current run, used for logging.
        @throws RuntimeError If configuration loading or CARLA connection fails.
        """

        super().__init__()

        # --- Load configuration ---
        self.config = load_config()
        if self.config is None:
            raise RuntimeError("Failed to load CARLA config")

        sim_config = self.config['simulation']
        cam_config = self.config['sensors']['camera']
        lidar_config = self.config['sensors']['lidar']
        act_config = self.config['actions']

        self.is_display = self.config['render']['is_display']
        self.is_save_frames = self.config['render']['is_save_frames']

        # --- Connect to CARLA ---
        self.client, self.world = connect_carla(port=sim_config['port'])
        if self.client is None or self.world is None:
            raise RuntimeError("Failed to connect to CARLA simulator")

        configure_simulation(
            self.client,
            self.world,
            sync_mode=sim_config['sync_mode'],
            dt=sim_config['dt']
        )

        # --- Observation space ---
        camera_dim = cam_config['train_resolution']['x'] * cam_config['train_resolution']['y'] * cam_config['channels']
        lidar_dim = lidar_config['points'] * lidar_config['features_per_point']
        
        self.risk_module = RiskModule(self.config)
        risk_dim = self.risk_module.feature_dim

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(camera_dim + lidar_dim + risk_dim,),
            dtype=np.float32
        )

        # --- Action space ---
        self.action_space = spaces.Box(
            low=np.array([
                act_config['steer'][0],
                act_config['throttle'][0],
                act_config['brake'][0]
            ]),
            high=np.array([
                act_config['steer'][1],
                act_config['throttle'][1],
                act_config['brake'][1]
            ]),
            dtype=np.float32
        )

        # --- Managers ---
        self.max_episode_length = sim_config['max_episode_length']
        self.vehicle_manager = VehicleManager(self.world)
        self.sensor_manager = SensorManager()
        self.npc_manager = NPCManager(self.world, self.client)

        # --- Reproducibility ---
        self._base_seed = self.config.get('training', {}).get('seed', 42)
        self._episode_count = 0

        # --- Internal state ---
        self.vehicle = None
        self.collision_history = []
        self.episode_length = 0
        self._route_dists = []
        self._wheelbase = 2.5  # updated after spawn via get_wheelbase()
        self._route_dists = []  # cumulative arc distances, rebuilt each episode
        self._risk_vec = None
        self._ego = None
        self._agents = None
        self._stopping = False
        self._prev_control = np.zeros(3)  # [steer, throttle, brake]
        self._stuck_steps = 0

        self._destination = None
        self._prev_dist_to_goal = None

        # --- Frame recording setup (only if saving is enabled) ---
        if self.is_save_frames:
            render_root = self.config['render']['root']

            self.frame_record_state = FrameRecordState()
            
            frames_root = os.path.join(
                ws_root_path, render_root, "frames", run_tag
            )
            os.makedirs(frames_root, exist_ok=True)
                
            datestamp = self.frame_record_state.start_time.strftime("%Y%m%d")
            frames_dir = os.path.join(
                frames_root, f"{datestamp}"
            )
            os.makedirs(frames_dir, exist_ok=True)

            timestamp = self.frame_record_state.start_time.strftime("%Y%m%d_%H%M%S")
            self.frame_record_state.file_path = os.path.join(
                frames_dir, f"{timestamp}"
            )

    def reset(self) -> dict:
        """
        @brief Reset the environment.
 
        @details
        Spawns a new ego vehicle, attaches sensors, and advances the
        simulation to produce the initial observation.
 
        @return dict Initial observation containing:
            - 'camera': flattened RGB camera data [H*W*3]
            - 'lidar': flattened xyz LiDAR data [points*3]
            - 'ego_state': ego vehicle state [6 + waypoints_ahead*3 + 8]
            - 'risk_features': risk feature vector [top_k*11 + 6]
        """

        self._cleanup_actors()

        # Per-episode RNG: episode N always gets the same environment regardless
        # of which algorithm is running or how many steps prior episodes took.
        episode_seed = self._base_seed + self._episode_count
        episode_rng = random.Random(episode_seed)
        self._episode_count += 1

        # Spawn at a pre-validated CARLA spawn point (always accepted), then
        # snap _waypoint_idx to the closest route waypoint so navigation and
        # visualisation are aligned with the vehicle's actual spawn location.
        spawn_points = self.world.get_map().get_spawn_points()
        min_dist = self.config['simulation'].get('min_route_distance', 50.0)
        grp = GlobalRoutePlanner(self.world.get_map(), 2.0)

        min_waypoints = self.config['simulation'].get('min_route_waypoints', 50)

        self.vehicle = None
        for _ in range(20):
            start_sp = episode_rng.choice(spawn_points)
            candidates = [sp for sp in spawn_points if start_sp.location.distance(sp.location) >= min_dist]
            end_loc = episode_rng.choice(candidates if candidates else spawn_points).location

            self._waypoints = grp.trace_route(start_sp.location, end_loc)
            self._dist_to_wp = 0.0

            # Reject routes that are too short to be meaningful
            if len(self._waypoints) < min_waypoints:
                continue

            # Anchor the goal marker to the actual last waypoint, not the raw
            # spawn point used as the routing target (they can differ by metres).
            self._destination = self._waypoints[-1][0].transform.location

            # Reject routes whose first waypoint heads the opposite way to the
            # spawn point. This prevents the vehicle from spawning in the
            # oncoming lane relative to the route it is meant to follow.
            first_wp_fwd = self._waypoints[0][0].transform.get_forward_vector()
            spawn_fwd = start_sp.get_forward_vector()
            if first_wp_fwd.x * spawn_fwd.x + first_wp_fwd.y * spawn_fwd.y < 0.0:
                continue

            # Find _waypoint_idx from start_sp before spawning so the vehicle
            # lands at the same position as the navigation target.
            # require_road_alignment ensures we snap to a same-direction waypoint.
            self._waypoint_idx, self._prev_dist_to_goal = self._find_closest_waypoint_ahead(
                transform=start_sp,
                require_road_alignment=True,
            )

            # Reject if the snap puts us too close to the end of the route
            if len(self._waypoints) - self._waypoint_idx < min_waypoints:
                continue

            self.vehicle = self.vehicle_manager.spawn_ego_vehicle(
                set_physics=True, transform=start_sp
            )
            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError("Failed to find a valid spawnable route after 20 attempts")

        self._wheelbase = self.vehicle_manager.get_wheelbase(self.vehicle)

        self._route_dists = [0.0]
        for _i in range(1, len(self._waypoints)):
            _a = self._waypoints[_i - 1][0].transform.location
            _b = self._waypoints[_i][0].transform.location
            self._route_dists.append(
                self._route_dists[-1] + math.hypot(_b.x - _a.x, _b.y - _a.y)
            )

        self.episode_length = 0
        self.collision_history = []

        self._risk_vec = None
        self._ego = None
        self._agents = None
        self._stopping = False
        self._prev_control = np.zeros(3)
        self._stuck_steps = 0

        self._setup_sensors()

        sim = self.config['simulation']
        self.npc_manager.spawn(
            num_vehicles=sim.get('num_npc_vehicles', 5),
            ego_location=self.vehicle.get_location(),
            min_distance=sim.get('min_npc_distance', 15.0),
            tm_port=sim.get('tm_port', 8000),
            sync_mode=sim['sync_mode'],
            rng=episode_rng,
            seed=episode_seed,
        )
        self.npc_manager.spawn_static_on_route(
            waypoints=self._waypoints,
            from_idx=self._waypoint_idx,
            num_obstacles=sim.get('num_static_obstacles', 3),
            min_gap=sim.get('static_obstacle_min_gap', 20),
            rng=episode_rng,
        )

        self.world.tick()

        # Recompute from actual post-tick vehicle location to guarantee a finite
        # value — _find_closest_waypoint_ahead may return inf if no waypoint found.
        self._prev_dist_to_goal = self.vehicle.get_location().distance(self._get_next_waypoint())
        self._dist_to_wp = self._prev_dist_to_goal

        self._draw_waypoints()
        self.world.tick()  # flush drawings so they appear before the first step

        return self._get_observation()

    def step(self, action, log=True):
        """
        @brief Execute a single environment step.
 
        @details
        Applies the control action, advances the simulation, and computes
        the resulting observation, reward, and termination condition.
 
        @param action numpy.ndarray Control input [steer, throttle, brake]
        @param log bool Whether to log diagnostic information
        @return tuple (obs, reward, done, info)
            - obs: observation dictionary
            - reward: np.ndarray Decomposed reward [nav, safety, risk]
            - done: termination flag
            - info: additional diagnostic information
        """

        self.episode_length += 1

        # Trigger stopping sequence once the route is complete
        if self._goal_reached() and not self._stopping:
            self._stopping = True

        if self._stopping:
            # Brake to a halt — override RL and controller
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            steer, throttle, brake = 0.0, 0.0, 1.0
        else:
            steer, throttle, brake = self._apply_control(action)

        self.world.tick()

        self._draw_waypoints()

        # State updates
        self._ego = self._get_ego_state()
        self._agents = self._get_nearby_actors()

        # Route forward direction at the current waypoint.
        # Using the waypoint's own forward vector (lane direction) rather than
        # the bearing to the waypoint location gives a stable turn signal: it is
        # +1 when the vehicle faces along the lane and becomes negative as soon as
        # it diverges — even before it has moved away from the waypoint position.
        route_fwd = self._waypoints[self._waypoint_idx][0].transform.get_forward_vector()
        wp_dx, wp_dy = float(route_fwd.x), float(route_fwd.y)

        # Risk and Reward logic
        wrong_way_risk = self._compute_wrong_way_risk()

        self._risk_vec = self.risk_module.compute(
            self._ego,
            self._agents,
            wrong_way_risk
        )

        progress = self._compute_progress()

        wp_boundary = self.config['simulation']['off_route_distance']
        normalised_dist = min(self._dist_to_wp / wp_boundary, 1.0)

        # Gate progress reward on route adherence: drifting off-route linearly
        # reduces the progress signal to zero, so the vehicle is never rewarded
        # for closing distance to the destination while off the road.
        adherence = 1.0 - normalised_dist
        gated_progress = progress * adherence

        baseline_rewards = baseline_reward(
            self.config['risk']['reward'],
            self._ego,
            self._check_collision(),
            gated_progress,
            top_k=self.risk_module.top_k,
            wrong_way_risk=wrong_way_risk,
            normalised_dist=normalised_dist,
        )

        rewards = decomposed_reward(
            self.config['risk']['reward'],
            self._ego,
            self._check_collision(),
            gated_progress,
            self._risk_vec,
            self.risk_module.top_k,
            wp_dx=wp_dx,
            wp_dy=wp_dy,
            wrong_way_risk=wrong_way_risk,
            normalised_dist=normalised_dist,
        )

        # Exponential lateral penalty: replaces the separate linear cross-track and
        # quadratic off-route penalties with a single convex curve.
        # The gradient pulling the vehicle back grows with distance, so small
        # deviations are cheap but large ones become rapidly more expensive.
        #   penalty(d) = scale * (exp(k*d) - 1) / (exp(k) - 1)
        # At d=0: 0.  At d=1: scale.  Gradient at d=1 is k× the gradient at d=0.
        lat_scale = self.config['risk']['reward'].get('lateral_scale', 25.0)
        lat_exp   = self.config['risk']['reward'].get('lateral_exponent', 3.0)
        lateral_penalty = lat_scale * (math.expm1(lat_exp * normalised_dist) / math.expm1(lat_exp))
        rewards[0] -= lateral_penalty

        reward = rewards

        # Construct observation
        obs = self._get_observation()

        if self._check_collision():
            Log.warning(__file__, f"Collision detected with {len(self.collision_history)} events")

        velocity = self.vehicle.get_velocity()
        speed = math.hypot(velocity.x, velocity.y)
        fully_stopped = self._stopping and speed < 0.5

        # Extract minimum TTC for risk-stop detection and info logging.
        ttc_min = float('inf')
        if self._risk_vec is not None:
            ttc_min = min(
                (self._risk_vec[i * AGENT_FEAT_DIM + 4]
                 for i in range(self.risk_module.top_k)
                 if self._risk_vec[i * AGENT_FEAT_DIM] > 0),
                default=float('inf')
            )

        at_light, light_is_green = self._get_traffic_light_obs()

        # Stopped at red/yellow: legitimate wait, same treatment as risk_stop.
        red_light_stop = at_light and not light_is_green and speed < 0.5
        # Moving through a red/yellow light: penalise per step.
        red_light_running = at_light and not light_is_green and speed >= 0.5
        if red_light_running:
            rewards[0] -= self.config['risk']['reward'].get('red_light_penalty', 10.0)

        # A vehicle stopped when a collision is genuinely imminent is waiting for
        # a gap — not stuck. Suspend the stuck counter and refund the time penalty
        # so the RL can learn that waiting is valid.
        risk_stop = ttc_min < self.config['simulation'].get('risk_stop_ttc', 1.5) and speed < 0.5
        valid_stop = risk_stop or red_light_stop
        if not self._stopping and not valid_stop and speed < 0.5:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        stuck = self._stuck_steps >= self.config['simulation'].get('stuck_timeout', 100)

        if valid_stop:
            rewards[0] += self.config['risk']['reward']['time_penalty']

        off_road = self._off_road()
        if off_road:
            rewards[0] -= self.config['risk']['reward'].get('off_road_penalty', 50.0)

        done = (
            self._check_collision() or
            fully_stopped or
            self._timeout() or
            self._off_route() or
            off_road or
            stuck
        )

        info = {
            'collision': self._check_collision(),
            'goal_reached': fully_stopped,
            'stuck': stuck,
            'off_route': self._off_route(),
            'off_road': off_road,
            'ttc_min': float(ttc_min),
            'goal_dist': self._prev_dist_to_goal, # route completion proxy
            'baseline_reward': float(baseline_rewards.sum()),
            'reward_navigation': rewards[0],
            'reward_safety': rewards[1],
            'reward_risk': rewards[2],
            'wp_idx': self._waypoint_idx,
            'wp_total': len(self._waypoints), # route completion = wp_idx / wp_total
        }

        speed = np.sqrt(self._ego['vx']**2 + self._ego['vy']**2)

        if log:
            Log.info(__file__, f"""SNAPSHOT
        Status: {'ALIVE 🟢' if self.vehicle.is_alive else 'DEAD 🔴'}
        Pose: x {self._ego['x']:.2f}, y {self._ego['y']:.2f}, z {self._ego['z']:.2f}, yaw {self._ego['yaw']}
        Speed: {speed} m/s
        Control: steer {steer}, throttle {throttle}, brake {brake}
        Waypoint: {self._waypoint_idx}/{len(self._waypoints)}""")

        return obs, reward, done, info
        
    def _apply_control(self, action) -> tuple[float, float, float]:
        """
        @brief Decode the RL action and apply it directly to the vehicle.

        @param action torch.Tensor | numpy.ndarray
            2-D action [steer, combined_tb]: combined_tb > 0 → throttle, < 0 → brake.
            3-D action [steer, throttle, brake] decoded from tanh space (PPO legacy).
        @return tuple[float, float, float] Applied (steer, throttle, brake)
        """

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.ndim > 1:
            action = action[0]

        steer_smooth    = self.config['simulation'].get('steer_smoothing', 0.3)
        throttle_smooth = self.config['simulation'].get('throttle_smoothing', 0.7)

        steer = float(np.clip(action[0], -1.0, 1.0))
        steer = steer_smooth * self._prev_control[0] + (1.0 - steer_smooth) * steer

        if len(action) == 2:
            # 2-D combined throttle-brake: positive = throttle, negative = brake.
            # "No brake, moderate throttle" maps to ~0.3 — well inside tanh interior,
            # so SAC entropy does not conflict with the dominant driving state.
            combined = float(action[1])
            combined = throttle_smooth * self._prev_control[1] + (1.0 - throttle_smooth) * combined
            throttle = float(np.clip( combined, 0.0, 1.0)) if combined > 0.0 else 0.0
            brake    = float(np.clip(-combined, 0.0, 1.0)) if combined < 0.0 else 0.0
            self._prev_control = np.array([steer, combined, 0.0])
        else:
            # 3-D legacy path (PPO): [steer, throttle, brake] in tanh space.
            throttle = float(np.clip((action[1] + 1) / 2, 0.0, 1.0))
            brake    = float(np.clip((action[2] + 1) / 2, 0.0, 1.0))
            throttle = throttle_smooth * self._prev_control[1] + (1.0 - throttle_smooth) * throttle
            if brake > 0.05:
                throttle = 0.0
            elif throttle > 0.05:
                brake = 0.0
            self._prev_control = np.array([steer, throttle, brake])

        self.vehicle.apply_control(carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake
        ))

        return steer, throttle, brake

    def _compute_controller_action(self) -> tuple[float, float, float]:
        """
        @brief Pure-pursuit waypoint follower returning a physical control action.

        @details
        Steering is computed as the signed angle from the vehicle heading to
        the current target waypoint, normalised by the maximum CARLA steer
        angle (0.7 rad ≈ 40°).

        Throttle and brake are set by a proportional speed controller that
        targets speed_cap (m/s) from the reward config.

        @return tuple[float, float, float] (steer, throttle, brake) in [−1,1] / [0,1] / [0,1]
        """

        if not self._waypoints or self._waypoint_idx >= len(self._waypoints):
            return 0.0, 0.0, 1.0   # no route: hold brakes

        tf = self.vehicle.get_transform()
        vehicle_loc = tf.location

        velocity = self.vehicle.get_velocity()
        speed = math.hypot(velocity.x, velocity.y)

        # Speed-adaptive lookahead distance along the route.
        min_lookahead = self.config['simulation'].get('min_lookahead', 8.0)
        lookahead_gain = self.config['simulation'].get('lookahead_gain', 1.5)
        lookahead = max(min_lookahead, lookahead_gain * speed)

        # Project the vehicle onto the current route segment to get its arc position,
        # then pick the first waypoint at least `lookahead` arc-metres ahead.
        route_wp = self._waypoints[self._waypoint_idx][0]
        next_seg_idx = min(self._waypoint_idx + 1, len(self._waypoints) - 1)
        wp_a = route_wp.transform.location
        wp_b = self._waypoints[next_seg_idx][0].transform.location
        ax, ay = wp_b.x - wp_a.x, wp_b.y - wp_a.y
        bx, by = vehicle_loc.x - wp_a.x, vehicle_loc.y - wp_a.y
        seg_sq = ax * ax + ay * ay
        t = max(0.0, min(1.0, (bx * ax + by * ay) / max(seg_sq, 1e-6)))
        proj_arc = self._route_dists[self._waypoint_idx] + t * math.sqrt(seg_sq)

        target_wp = self._waypoints[-1][0]
        for i in range(self._waypoint_idx, len(self._waypoints)):
            if self._route_dists[i] - proj_arc >= lookahead:
                target_wp = self._waypoints[i][0]
                break

        # Geometric pure-pursuit: δ = arctan(2L sin(α) / ld).
        # The ld denominator scales down steer for distant targets.
        wp_loc = target_wp.transform.location
        d_to_target = math.hypot(wp_loc.x - vehicle_loc.x, wp_loc.y - vehicle_loc.y)
        bearing_error = self.vehicle_manager.calculate_steering_to_waypoint(tf, target_wp)
        pp_angle = math.atan2(2.0 * self._wheelbase * math.sin(bearing_error),
                              max(d_to_target, 1.0))
        steer = float(np.clip(pp_angle / 0.7, -1.0, 1.0))

        # Proportional speed controller.
        target_speed = self.config['risk']['reward']['speed_cap']
        error = target_speed - speed
        if error > 0:
            throttle = float(min(1.0, error / target_speed))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(min(1.0, -error / target_speed))

        return steer, throttle, brake

    def _check_collision(self) -> bool:
        """
        @brief Check if a collision has occurred in the current episode.
 
        @details
        Returns True if any collision has been recorded by the
        collision sensor attached to the ego vehicle during the episode.
 
        @return bool True if a collision has been detected, False otherwise.
        """

        return len(self.collision_history) > 0

    def _compute_wrong_way_risk(self) -> float:
        """
        @brief Compute a risk score for travelling against lane direction.
 
        @details
        The score is normalised to the range [0, 1], where:
            - 0 indicates correct alignment with the lane
            - 1 indicates full reversal
 
        @return float Wrong-way risk value in [0, 1].
        """

        if self._at_turning_junction():
            return 0.0

        vehicle_loc = self.vehicle.get_location()
        vehicle_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)

        waypoint = self.world.get_map().get_waypoint(vehicle_loc)
        lane_vector = waypoint.transform.get_forward_vector()
        lane_yaw = math.atan2(lane_vector.y, lane_vector.x)

        angle_diff = abs((vehicle_yaw - lane_yaw + math.pi) % (2*math.pi) - math.pi)
        risk = min(1.0, angle_diff / math.pi)
        return risk

    def _find_closest_waypoint_ahead(self, from_idx: int = 0, search_limit: int = None, transform: 'carla.Transform' = None, require_road_alignment: bool = False) -> tuple[int, float]:
        """
        @brief Find the closest waypoint ahead of the vehicle within a search window.

        @param from_idx  Start index in self._waypoints to search from.
        @param search_limit  Max number of waypoints to scan. None means entire route.
        @param transform  Transform to use as reference. Defaults to the vehicle's current transform.
        @param require_road_alignment  If True, also require the waypoint's own road forward
               vector to align with the reference forward (dot > 0). Used at spawn time to
               avoid snapping to waypoints on the opposite-direction lane.
        @return (index, distance) of the nearest ahead waypoint.
        """
        tf = transform if transform is not None else self.vehicle.get_transform()
        vehicle_loc = tf.location
        vehicle_fwd = tf.get_forward_vector()

        end = len(self._waypoints) if search_limit is None else min(from_idx + search_limit, len(self._waypoints))
        best_idx, best_dist = from_idx, float('inf')
        fallback_idx, fallback_dist = from_idx, float('inf')
        for i in range(from_idx, end):
            wp_loc = self._waypoints[i][0].transform.location
            dx = wp_loc.x - vehicle_loc.x
            dy = wp_loc.y - vehicle_loc.y
            d = math.hypot(dx, dy)  # √(dx²+dy²)
            if d < fallback_dist:
                fallback_dist, fallback_idx = d, i

            # Skip waypoints whose road direction opposes the reference heading.
            if require_road_alignment:
                wp_fwd = self._waypoints[i][0].transform.get_forward_vector()
                if wp_fwd.x * vehicle_fwd.x + wp_fwd.y * vehicle_fwd.y < 0.0:
                    continue

            if d > 0 and (dx / d * vehicle_fwd.x + dy / d * vehicle_fwd.y) >= 0:
                if d < best_dist:
                    best_dist, best_idx = d, i

        # No waypoint found ahead — fall back to closest regardless of direction
        if best_dist == float('inf'):
            return fallback_idx, fallback_dist
        return best_idx, best_dist

    def _compute_progress(self) -> float:
        """
        @brief Compute the progress made towards the goal during this step.
 
        @details
        Progress is the change in Euclidean distance to the next waypoint.
        Positive values indicate the vehicle is approaching the goal.
        Returns a waypoint bonus of 5.0 when a waypoint is reached.
 
        @return float Distance closed towards the next waypoint in metres,
                      or 5.0 if a waypoint was reached this step.
        """

        best_idx, best_dist = self._find_closest_waypoint_ahead(
            from_idx=self._waypoint_idx, search_limit=10
        )

        waypoint_advanced = best_idx > self._waypoint_idx
        self._waypoint_idx = best_idx
        self._dist_to_wp = best_dist if best_dist < float('inf') else 0.0

        progress = self._prev_dist_to_goal - self._dist_to_wp
        self._prev_dist_to_goal = self._dist_to_wp

        if waypoint_advanced:
            return 5.0  # waypoint reached bonus

        return float(progress)

    def _goal_reached(self) -> bool:
        """
        @brief Determine if the ego vehicle has reached the destination.

        @details
        Goal is reached when the vehicle is within goal_threshold metres of
        _destination, which is anchored to the last route waypoint. The
        waypoint index is not checked — it lags one step behind and would
        prevent detection when the vehicle physically arrives but the progress
        tracker has not yet caught up.

        @return bool True if within goal_threshold of the destination.
        """

        if self._destination is None:
            return False

        dist = self.vehicle.get_location().distance(self._destination)
        return dist < self.config['simulation'].get('goal_threshold', 5.0)

    def _timeout(self) -> bool:
        """
        @brief Check if the episode has exceeded the maximum length.

        @return bool True if the episode has timed out.
        """

        return self.episode_length >= self.max_episode_length

    def _get_traffic_light_obs(self) -> tuple[bool, bool]:
        """
        @brief Query the traffic light state affecting the ego vehicle.

        @return (at_light, is_green) where:
            at_light  — True if the vehicle is currently influenced by a traffic light.
            is_green  — True if the light is green (or no light present → safe to proceed).
        """
        try:
            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                state = tl.get_state()
                is_green = state == carla.TrafficLightState.Green
                return True, is_green
        except Exception:
            pass
        return False, True  # no light → treat as green

    def _at_turning_junction(self) -> bool:
        """
        @brief Return True if the vehicle is at a junction where the route turns.

        @details
        Used to gate checks and rewards that are unreliable inside intersection
        geometry (wrong-way risk, off-road detection, blend cap, box colour).

        Two conditions must both hold:
          1. CARLA reports the vehicle is on junction geometry (is_junction=True).
          2. The planned route turns here — the heading change from the current
             route waypoint to one ~20 m ahead exceeds 25°. This excludes
             tunnels and straight-through sections that CARLA also marks as
             junctions.

        @return bool True if the vehicle is at a real turning intersection.
        """
        wp = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if wp is None or not wp.is_junction:
            return False
        if not self._waypoints or self._waypoint_idx >= len(self._waypoints):
            return False
        # Check a window of ≈20 m in both directions so the junction is
        # detected consistently from entry through exit:
        #   - approaching: forward heading change is large, backward is small
        #   - mid-turn:    both are large
        #   - near exit:   forward is small, backward heading change is large
        lookahead  = 10  # waypoints ≈ 20 m at 2 m spacing
        cur_idx    = self._waypoint_idx
        ahead_idx  = min(cur_idx + lookahead, len(self._waypoints) - 1)
        behind_idx = max(cur_idx - lookahead, 0)
        cur_fwd    = self._waypoints[cur_idx][0].transform.get_forward_vector()
        ahead_fwd  = self._waypoints[ahead_idx][0].transform.get_forward_vector()
        behind_fwd = self._waypoints[behind_idx][0].transform.get_forward_vector()
        dot_fwd  = cur_fwd.x * ahead_fwd.x  + cur_fwd.y * ahead_fwd.y
        dot_back = cur_fwd.x * behind_fwd.x + cur_fwd.y * behind_fwd.y
        return dot_fwd < 0.9 or dot_back < 0.9  # heading change > ~25° in either window

    def _off_road(self) -> bool:
        """
        @brief Check if the vehicle has left the drivable road surface.

        @details
        Uses CARLA's map API to query the lane type at the vehicle's current
        location without snapping to the nearest road. Returns True when no
        Driving-type lane is found, indicating the vehicle is on a sidewalk,
        grass, or other non-drivable surface.

        Junction interiors are excluded because CARLA does not classify
        intersection geometry as LaneType.Driving.

        @return bool True if the vehicle is off the drivable surface.
        """
        if self._at_turning_junction():
            return False
        wp = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        return wp is None

    def _off_route(self) -> bool:
        """
        @brief Check if the vehicle has strayed beyond the observable waypoint range.

        @details
        Terminates the episode when the distance to the next waypoint reaches
        the normalisation boundary (waypoints_ahead * 2.0 metres), ensuring the
        full [0, 1] observation range is used before termination triggers.

        @return bool True if the vehicle is beyond the off-route boundary.
        """

        return self._dist_to_wp >= self.config['simulation']['off_route_distance']

    def _get_next_waypoint(self):
        """
        @brief Retrieve the next target waypoint along the planned route.
 
        @return carla.Location Location of the next waypoint, or the
                                destination if the route is exhausted.
        """

        if self._waypoints and self._waypoint_idx < len(self._waypoints):
            return self._waypoints[self._waypoint_idx][0].transform.location
        return self._destination

    def _get_lane_context(self) -> np.ndarray:
        """
        @brief Query CARLA map API for adjacent lane availability and change permissions.

        @return np.ndarray of shape (6,):
            [left_valid, left_offset_norm, right_valid, right_offset_norm,
             change_left_allowed, change_right_allowed]
        """
        if self.vehicle is None or self._ego is None:
            return np.zeros(6, dtype=np.float32)
        try:
            carla_map = self.world.get_map()
            wp = carla_map.get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                return np.zeros(6, dtype=np.float32)

            ego_fwd = np.array([math.cos(self._ego['yaw']), math.sin(self._ego['yaw'])])
            lane_width_norm = 5.0  # normalise offset by this distance in metres

            def _adjacent(adj_wp):
                if adj_wp is None or adj_wp.lane_type != carla.LaneType.Driving:
                    return 0.0, 0.0
                fwd = adj_wp.transform.get_forward_vector()
                if np.dot(ego_fwd, np.array([fwd.x, fwd.y])) < 0.0:
                    return 0.0, 0.0  # oncoming direction
                offset = float(np.clip(adj_wp.lane_width / lane_width_norm, 0.0, 1.0))
                return 1.0, offset

            left_valid,  left_offset  = _adjacent(wp.get_left_lane())
            right_valid, right_offset = _adjacent(wp.get_right_lane())

            lc = wp.lane_change
            change_left  = 1.0 if lc in (carla.LaneChange.Left,  carla.LaneChange.Both) else 0.0
            change_right = 1.0 if lc in (carla.LaneChange.Right, carla.LaneChange.Both) else 0.0

            return np.array(
                [left_valid, left_offset, right_valid, right_offset, change_left, change_right],
                dtype=np.float32,
            )
        except Exception:
            return np.zeros(6, dtype=np.float32)

    def _get_ego_state(self) -> dict:
        """
        @brief Extract the current state of the ego vehicle.
 
        @return dict Dictionary containing:
            - 'x', 'y', 'z': position coordinates in metres
            - 'vx', 'vy': velocity components in metres per second
            - 'yaw': heading angle in radians
            - 'length', 'width': bounding box dimensions in metres
        """
        
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        bbox = self.vehicle.bounding_box

        return {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
            'vx': velocity.x,
            'vy': velocity.y,
            'yaw': np.deg2rad(transform.rotation.yaw),
            'length': bbox.extent.x * 2,
            'width': bbox.extent.y * 2,
        }

    def _get_nearby_actors(self, radius: float = 50.0) -> list[dict]:
        """
        @brief Retrieve nearby actors within a specified radius.
 
        @details
        Queries the CARLA world for vehicles and pedestrians within
        the given radius. Pedestrians are approximated with a fixed
        bounding box of 0.5m × 0.5m.
 
        @param radius float Search radius in metres (default 50.0).
        @return list[dict] List of nearby actor dicts with keys:
            - 'id', 'x', 'y', 'vx', 'vy', 'yaw', 'length', 'width', 'category'
        """

        ego_loc = self.vehicle.get_location()
        actors  = []

        actor_list = self.world.get_actors()

        for actor in actor_list.filter('vehicle.*'):
            if actor.id == self.vehicle.id: continue
            if actor.get_location().distance(ego_loc) > radius: continue

            t = actor.get_transform()
            v = actor.get_velocity()
            bb = actor.bounding_box
            actors.append({
                'id': actor.id,
                'x': t.location.x,
                'y': t.location.y,
                'vx': v.x,
                'vy': v.y,
                'yaw': np.deg2rad(t.rotation.yaw),
                'length': bb.extent.x * 2,
                'width': bb.extent.y * 2,
                'category': 'vehicle',
            })

        for actor in actor_list.filter('walker.pedestrian.*'):
            if actor.get_location().distance(ego_loc) > radius: continue

            t = actor.get_transform()
            v = actor.get_velocity()
            actors.append({
                'id': actor.id,
                'x': t.location.x,
                'y': t.location.y,
                'vx': v.x,
                'vy': v.y,
                'yaw': np.deg2rad(t.rotation.yaw),
                'length': 0.5,
                'width': 0.5,
                'category': 'walker',
            })

        return actors

    def _get_observation(self):
        """
        @brief Construct the current observation dictionary.
 
        @details
        Combines sensor data, ego state, and risk features into a structured
        observation dict. Sensor outputs are returned as flat float32 arrays.
 
        Camera data is returned as flat RGB (channels=3) by the sensor manager.
        LiDAR data is returned with 4 features per point (xyzI) from the sensor
        buffer - the intensity channel is dropped here to match the configured
        features_per_point=3.
 
        @return dict Observation containing:
            - 'camera': flat RGB array [H*W*3]
            - 'lidar': flat xyz array [points*3] (intensity dropped)
            - 'ego_state': [x, y, z, vx, vy, yaw, wp_dx0, wp_dy0, wp_d0, ..., ct_norm, he_norm,
                            left_valid, left_offset, right_valid, right_offset, change_left, change_right]
                           (6 + waypoints_ahead*3 + 2 + 6 values)
            - 'risk_features': risk vector [top_k*11 + 6] = [61]
        """
        
        # Ensure ego state exists
        if self._ego is None:
            self._ego = self._get_ego_state()  # <-- compute ego state if missing

        # Sensor configuration
        lidar_config = self.config['sensors']['lidar']
        cam_config = self.config['sensors']['camera']
        cam_res = cam_config['train_resolution']
        cam_channels = cam_config['channels']

        # Get sensor data
        sensor_obs = self.sensor_manager.get_observation(
            camera_resolution=(cam_res['x'], cam_res['y']),
            lidar_points=lidar_config['points']
        ).astype(np.float32)

        cam_dim = cam_res['x'] * cam_res['y'] * cam_channels # 21168 (RGB)
        lidar_raw_dim = lidar_config['points'] * 4 # 4096 (xyzI from sensor)
 
        camera_obs = sensor_obs[:cam_dim]

        # Slice full lidar block then keep only xyz features per point
        lidar_raw = sensor_obs[cam_dim:cam_dim + lidar_raw_dim].reshape(lidar_config['points'], 4)
        lidar_obs = lidar_raw[:, :3].flatten() # drop intensity channel → [1024*3]

        if self._risk_vec is None:
            self._agents = self._get_nearby_actors()
            self._risk_vec = self.risk_module.compute(self._ego, self._agents)

        risk_obs = self._risk_vec.astype(np.float32)

        # Next N waypoints in the vehicle's local (ego) frame.
        # Rotating world-frame (dx, dy) by -ego_yaw makes the representation
        # orientation-invariant: the same turn geometry always produces the same
        # features regardless of which cardinal direction the vehicle faces.
        N_WAYPOINTS = self.config['risk']['waypoints_ahead']
        wp_dist_norm = self.config['risk']['waypoints_ahead'] * 2.0
        cos_yaw = math.cos(self._ego['yaw'])
        sin_yaw = math.sin(self._ego['yaw'])
        waypoint_obs = []
        for i in range(N_WAYPOINTS):
            idx = self._waypoint_idx + i
            if idx < len(self._waypoints):
                wp_loc = self._waypoints[idx][0].transform.location
                dx = wp_loc.x - self._ego['x']
                dy = wp_loc.y - self._ego['y']
                dist = max(np.sqrt(dx**2 + dy**2), 1e-6)
                # Rotate into ego frame: forward = +x_ego, left = +y_ego
                dx_ego =  dx * cos_yaw + dy * sin_yaw
                dy_ego = -dx * sin_yaw + dy * cos_yaw
                waypoint_obs.extend([dx_ego / dist, dy_ego / dist, min(dist / wp_dist_norm, 1.0)])
            else:
                waypoint_obs.extend([0.0, 0.0, 0.0])  # pad if route ends

        # Cross-track error and heading error relative to the current route segment.
        # Both are signed and normalised to [-1, 1] so the RL has direct geometric
        # state without needing to infer it from waypoint deltas.
        ct_norm = 0.0
        he_norm = 0.0
        if self._waypoint_idx < len(self._waypoints):
            # Walk forward from the current waypoint index until we reach one
            # that is at least lookahead_dist metres ahead of the vehicle.
            # Using a look-ahead target (rather than the immediate next waypoint)
            # gives a smoother, more stable cross-track / heading signal.
            lookahead_dist = self.config['simulation'].get('lookahead_dist', 8.0)
            la_idx = self._waypoint_idx
            for i in range(self._waypoint_idx, len(self._waypoints)):
                loc = self._waypoints[i][0].transform.location
                if math.sqrt((loc.x - self._ego['x']) ** 2 + (loc.y - self._ego['y']) ** 2) >= lookahead_dist:
                    la_idx = i
                    break
            else:
                la_idx = len(self._waypoints) - 1

            wp_cur_loc  = self._waypoints[la_idx][0].transform.location
            wp_prev_loc = self._waypoints[max(0, la_idx - 1)][0].transform.location
            seg_x = wp_cur_loc.x - wp_prev_loc.x
            seg_y = wp_cur_loc.y - wp_prev_loc.y
            seg_len = max(math.sqrt(seg_x ** 2 + seg_y ** 2), 1e-6)
            seg_dx, seg_dy = seg_x / seg_len, seg_y / seg_len
            rel_x = self._ego['x'] - wp_prev_loc.x
            rel_y = self._ego['y'] - wp_prev_loc.y
            # Signed cross-track: positive = left of route direction
            cross_track = rel_x * seg_dy - rel_y * seg_dx
            ct_norm = float(np.clip(cross_track / self.config['simulation']['off_route_distance'], -1.0, 1.0))
            # Heading error: ego yaw minus route direction, normalised to [-1, 1]
            heading_err = self._ego['yaw'] - math.atan2(seg_y, seg_x)
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
            he_norm = float(heading_err / math.pi)

        lane_ctx = self._get_lane_context()
        at_light, light_is_green = self._get_traffic_light_obs()

        ego_state = np.nan_to_num(np.array([
            self._ego['x'], self._ego['y'], self._ego['z'],
            self._ego['vx'], self._ego['vy'], self._ego['yaw'],
            *waypoint_obs,  # N waypoints × 3 = N*3 values
            ct_norm, he_norm,  # cross-track error, heading error
            *lane_ctx,  # left/right lane validity, offset, change permission
            float(at_light),        # 1.0 if a traffic light governs this vehicle
            float(light_is_green),  # 1.0 if green (or no light), 0.0 if red/yellow
        ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "camera": camera_obs,
            "lidar": lidar_obs,
            "ego_state": ego_state,
            "risk_features": risk_obs
        }

    def _is_done_baseline(self):
        """
        @brief Determine whether the episode has terminated (baseline check).
 
        @details
        Termination occurs on collision or episode timeout.
        Does not include goal_reached - used for baseline comparison only.
 
        @return bool True if the episode is complete.
        """

        return (len(self.collision_history) > 0 or self.episode_length >= self.max_episode_length)

    def _setup_sensors(self):
        """
        @brief Attach and initialise sensors on the ego vehicle.
 
        @details
        Attaches camera, LiDAR, and collision sensors via SensorManager.
        The camera is spawned at the view_resolution (800×600) for display,
        while training uses the train_resolution (84×84) via get_observation().
 
        @note Must be called after the ego vehicle has been spawned.
        """

        lidar_config = self.config['sensors']['lidar']
        cam_config = self.config['sensors']['camera']['view_resolution']

        self.sensor_manager.attach_sensors(
            self.world,
            self.vehicle,
            camera_resolution=(cam_config['x'], cam_config['y']),
            lidar_range=lidar_config['range']
        )

        self.collision_history = self.sensor_manager.collision_history

    def _cleanup_actors(self):
        """
        @brief Destroy all active actors in the environment.
 
        @details
        Destroys the ego vehicle and all attached sensors, and resets
        collision history. Called on reset() and close().
        """

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        self.npc_manager.destroy_all()

        self.sensor_manager.destroy_sensors()
        self.collision_history = []

    def _draw_waypoints(self):
        """
        @brief Visualise upcoming waypoints in the CARLA simulator.

        @details
        Draws the next waypoints_ahead waypoints from config using a
        colour gradient (red to blue via HSV), and marks the destination
        with a larger red dot. The number of waypoints drawn and their
        colours are both derived from risk.waypoints_ahead in config.

        Colour scheme:
            - Waypoints -N to -1 (behind) : grey — path already travelled
            - Waypoints +0 to +(N-1)      : HSV gradient red to blue — fed into model
            - Destination                 : red (larger marker)
        """

        debug = self.world.debug
        n = self.config['risk']['waypoints_ahead']

        # Draw a line from the vehicle to the lookahead waypoint — the same point
        # used for cross-track and heading error in the observation.
        if self._waypoint_idx < len(self._waypoints):
            vehicle_loc   = self.vehicle.get_location()
            lookahead_dist = self.config['simulation'].get('lookahead_dist', 8.0)
            la_idx = self._waypoint_idx
            for i in range(self._waypoint_idx, len(self._waypoints)):
                loc = self._waypoints[i][0].transform.location
                if math.sqrt((loc.x - vehicle_loc.x) ** 2 + (loc.y - vehicle_loc.y) ** 2) >= lookahead_dist:
                    la_idx = i
                    break
            else:
                la_idx = len(self._waypoints) - 1
            target_loc = self._waypoints[la_idx][0].transform.location + carla.Location(z=0.5)
            debug.draw_line(vehicle_loc + carla.Location(z=0.5), target_loc, thickness=0.03, color=carla.Color(255, 255, 0), life_time=0.1)

        # Draw N waypoints behind the current index as grey trail
        for i in range(1, n + 1):
            idx = self._waypoint_idx - i
            if idx < 0:
                break
            wp, _ = self._waypoints[idx]
            debug.draw_point(
                wp.transform.location + carla.Location(z=0.5),
                size=0.08,
                color=carla.Color(180, 180, 180),
                life_time=0.1,
            )

        # Draw the exact N waypoints fed into the model observation
        for i in range(n):
            idx = self._waypoint_idx + i
            if idx >= len(self._waypoints):
                break
            wp, _ = self._waypoints[idx]
            debug.draw_point(
                wp.transform.location + carla.Location(z=0.5),
                size=0.08,
                color=carla.Color(255, 255, 0),
                life_time=0.1,
            )

        # Draw destination as a bigger red marker
        if self._destination is not None:
            debug.draw_point(
                self._destination + carla.Location(z=1.0),
                size=0.2,
                color=carla.Color(255, 0, 0),
                life_time=0.1,
            )

        self._draw_lane_context()

    def _draw_lane_context(self):
        """
        @brief Draw adjacent lane availability around the vehicle.

        Green points = valid same-direction lane.
        Orange points = lane exists but is oncoming (blocked).
        Covers ±4 m longitudinally so the indicator stays compact.
        """
        if self.vehicle is None or self._ego is None:
            return
        try:
            carla_map = self.world.get_map()
            wp = carla_map.get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Any,
            )
            if wp is None:
                return

            debug = self.world.debug
            ego_fwd = np.array([math.cos(self._ego['yaw']), math.sin(self._ego['yaw'])])

            colour_valid   = carla.Color(0, 220, 0)
            colour_invalid = carla.Color(255, 0, 0)
            colour_junction = carla.Color(139, 69, 19)

            def _draw_current_lane(cur_wp):
                # Brown inside a junction: directional check is unreliable there.
                # Green if current lane runs in the same direction as the route waypoints,
                # red if the vehicle is facing against the flow (oncoming / wrong-way).
                if self._at_turning_junction():
                    colour = colour_junction
                else:
                    same_dir = False
                    if self._waypoints and self._waypoint_idx < len(self._waypoints):
                        route_fwd = self._waypoints[self._waypoint_idx][0].transform.get_forward_vector()
                        cur_fwd   = cur_wp.transform.get_forward_vector()
                        same_dir  = (route_fwd.x * cur_fwd.x + route_fwd.y * cur_fwd.y) > 0.0
                    colour = colour_valid if same_dir else colour_invalid

                bb = self.vehicle.bounding_box
                bb.location = self.vehicle.get_location()
                bb.extent = carla.Vector3D(bb.extent.x, bb.extent.y, 0.01)
                bb.location.z += 0.1

                debug.draw_box(
                    bb, self.vehicle.get_transform().rotation, thickness=0.01, color=colour, life_time=0.1)

            route_fwd = np.array([0.0, 0.0])
            if self._waypoints and self._waypoint_idx < len(self._waypoints):
                rf = self._waypoints[self._waypoint_idx][0].transform.get_forward_vector()
                route_fwd = np.array([rf.x, rf.y])

            def _draw_adjacent(adj_wp):
                if adj_wp is None or adj_wp.lane_type != carla.LaneType.Driving:
                    return
                fwd = adj_wp.transform.get_forward_vector()
                same_dir = np.dot(route_fwd, np.array([fwd.x, fwd.y])) >= 0.0
                colour = colour_valid if same_dir else colour_invalid
                base = adj_wp.transform.location

                if True: # draw a line indicating the lane direction
                    length = 8.0
                    begin = carla.Location(x=base.x - fwd.x * length, y=base.y - fwd.y * length, z=base.z + 0.3)
                    end   = carla.Location(x=base.x + fwd.x * length, y=base.y + fwd.y * length, z=base.z + 0.3)
                    debug.draw_line(begin, end, thickness=0.01, color=colour, life_time=0.1)
                else: # draw a box representing the lane area
                    length = 6.0
                    centre = carla.Location(x=base.x, y=base.y, z=base.z + 0.1)
                    extent = carla.Vector3D(x=length, y=adj_wp.lane_width * 0.5, z=0.05)
                    box = carla.BoundingBox(centre, extent)
                    debug.draw_box(box, adj_wp.transform.rotation, thickness=0.015, color=colour, life_time=0.1)

            _draw_adjacent(wp.get_left_lane())
            _draw_adjacent(wp.get_right_lane())
            _draw_current_lane(wp)
        except Exception:
            pass

    def render(self):
        """
        @brief Render the environment based on config flags.
 
        @details
        Two independent rendering modes controlled by config:
            - is_display: renders a live OpenCV window via SensorManager.
                          Used during development to observe agent behaviour.
            - is_save_frames: saves individual camera frames as PNG files.
                              Used for recording demo videos of trained policies.
 
        Both modes can be active simultaneously. Frame filenames follow the
        pattern: <timestamp>_<index>.png
 
        @note Frame saving requires is_save_frames=True in config and
              initialises the FrameRecordState in __init__.
        """

        if self.is_display:
            self.sensor_manager.render_all() # existing OpenCV render
        if self.is_save_frames:
            filename = f"{self.frame_record_state.file_path}_{self.frame_record_state.index}.png"
            if self.sensor_manager.save_camera_frame(file_path=filename):
                self.frame_record_state.index += 1

    def close(self):
        """
        @brief Release all environment resources.
 
        @details
        Safely destroys the ego vehicle and all attached sensors.
        """
        
        try:
            self._cleanup_actors()
        except Exception as e:
            Log.error(__file__, e)