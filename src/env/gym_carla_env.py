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
    from managers.actors import VehicleManager
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

        # --- Internal state ---
        self.vehicle = None
        self.collision_history = []
        self.episode_length = 0
        
        self._risk_vec = None
        self._ego = None
        self._agents = None

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
            - 'ego_state': ego vehicle state [16]
            - 'risk_features': risk feature vector [top_k*11 + 6]
        """

        self._cleanup_actors()

        # Spawn at a pre-validated CARLA spawn point (always accepted), then
        # snap _waypoint_idx to the closest route waypoint so navigation and
        # visualisation are aligned with the vehicle's actual spawn location.
        spawn_points = self.world.get_map().get_spawn_points()
        min_dist = self.config['simulation'].get('min_route_distance', 50.0)
        grp = GlobalRoutePlanner(self.world.get_map(), 2.0)

        min_waypoints = self.config['simulation'].get('min_route_waypoints', 50)

        self.vehicle = None
        for _ in range(20):
            start_sp = random.choice(spawn_points)
            candidates = [sp for sp in spawn_points if start_sp.location.distance(sp.location) >= min_dist]
            end_loc = random.choice(candidates if candidates else spawn_points).location

            self._waypoints = grp.trace_route(start_sp.location, end_loc)
            self._destination = end_loc
            self._dist_to_wp = 0.0

            # Reject routes that are too short to be meaningful
            if len(self._waypoints) < min_waypoints:
                continue

            # Find _waypoint_idx from start_sp before spawning so the vehicle
            # lands at the same position as the navigation target.
            self._waypoint_idx, self._prev_dist_to_goal = self._find_closest_waypoint_ahead(
                transform=start_sp
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

        self.episode_length = 0
        self.collision_history = []

        self._risk_vec = None
        self._ego = None
        self._agents = None

        self._setup_sensors()

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

        steer, throttle, brake = self._apply_control(action)
        self.world.tick()

        self._draw_waypoints()

        # State updates
        self._ego = self._get_ego_state()
        self._agents = self._get_nearby_actors()

        # Waypoint direction
        target = self._get_next_waypoint()
        dx = target.x - self._ego['x']
        dy = target.y - self._ego['y']
        dist = max(np.sqrt(dx**2 + dy**2), 1e-6)
        wp_dx, wp_dy = dx / dist, dy / dist

        # Risk and Reward logic
        WRONG_WAY_THRESHOLD = 0.5
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
        )

        # Quadratic off-route penalty: grows as (dist/boundary)^2 * scale,
        # reaching maximum at off_route_distance where the episode also terminates.
        off_route_penalty = (normalised_dist ** 2) * self.config['risk']['reward']['off_route_penalty_scale']
        rewards[0] -= off_route_penalty

        reward = rewards

        # Construct observation
        obs = self._get_observation()

        if self._check_collision():
            Log.warning(__file__, f"Collision detected with {len(self.collision_history)} events")

        done = (
            self._check_collision() or
            self._goal_reached() or
            self._timeout() or
            self._off_route()
        )

        info = {
            'collision': self._check_collision(),
            'goal_reached': self._goal_reached(),
            'off_route': self._off_route(),
            'ttc_min': float(min(
                (self._risk_vec[i * AGENT_FEAT_DIM + 4] for i in range(self.risk_module.top_k)
                 if self._risk_vec[i * AGENT_FEAT_DIM] > 0),
                default=float('inf')
            )) if self._risk_vec is not None else float('inf'),
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
        
    def _apply_control(self, action) -> list[float]:
        """
        @brief Convert the policy action into CARLA control commands.
 
        @details
        Accepts both NumPy arrays and torch tensors. Actions are squashed
        using tanh to enforce valid ranges:
            - steer ∈ [-1, 1]
            - throttle ∈ [0, 1]
            - brake ∈ [0, 1]
 
        Throttle and brake are treated as mutually exclusive.
 
        @param action torch.Tensor | numpy.ndarray Raw action input
        @return list[float] Applied control values [steer, throttle, brake]
        """

        # --- Convert to numpy ---
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # If batch dimension exists → take first element
        if action.ndim > 1:
            action = action[0]

        # --- Squash to valid range ---
        action = np.tanh(action)

        steer = float(action[0])                # [-1, 1]
        throttle = float((action[1] + 1) / 2)   # [0, 1]
        brake = float((action[2] + 1) / 2)      # [0, 1]

        # --- Enforce mutual exclusivity ---
        if throttle > 0.05:
            brake = 0.0
        elif brake > 0.05:
            throttle = 0.0

        # --- Apply control ---
        self.vehicle.apply_control(carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake
        ))

        # Log.info(__file__, f"raw action: {action}, throttle: {throttle:.3f}, brake: {brake:.3f}")

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

        vehicle_loc = self.vehicle.get_location()
        vehicle_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        
        waypoint = self.world.get_map().get_waypoint(vehicle_loc)
        lane_vector = waypoint.transform.get_forward_vector()
        lane_yaw = math.atan2(lane_vector.y, lane_vector.x)

        angle_diff = abs((vehicle_yaw - lane_yaw + math.pi) % (2*math.pi) - math.pi)
        risk = min(1.0, angle_diff / math.pi) # normalised [0, 1]
        return risk

    def _find_closest_waypoint_ahead(self, from_idx: int = 0, search_limit: int = None, transform: 'carla.Transform' = None) -> tuple[int, float]:
        """
        @brief Find the closest waypoint ahead of the vehicle within a search window.

        @param from_idx  Start index in self._waypoints to search from.
        @param search_limit  Max number of waypoints to scan. None means entire route.
        @param transform  Transform to use as reference. Defaults to the vehicle's current transform.
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
        @brief Determine if the ego vehicle has completed the waypoint route.

        @details
        Goal is reached when the vehicle has followed the planned waypoints
        to the final waypoint and is within goal_threshold metres of it.
        This ensures the vehicle must follow the route rather than shortcut
        directly to the destination.

        @return bool True if at the last waypoint and within goal_threshold.
        """

        if not self._waypoints:
            return False

        last_idx = len(self._waypoints) - 1
        if self._waypoint_idx < last_idx:
            return False

        last_wp_loc = self._waypoints[last_idx][0].transform.location
        dist = self.vehicle.get_location().distance(last_wp_loc)
        return dist < self.config['simulation'].get('goal_threshold', 5.0)

    def _timeout(self) -> bool:
        """
        @brief Check if the episode has exceeded the maximum length.

        @return bool True if the episode has timed out.
        """

        return self.episode_length >= self.max_episode_length

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
            - 'ego_state': [x, y, z, vx, vy, yaw, wp_dx0, wp_dy0, wp_d0, ...] (6 + waypoints_ahead*3 values)
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

        # Next N waypoints as relative (dx, dy) from ego position
        N_WAYPOINTS = self.config['risk']['waypoints_ahead']
        wp_dist_norm = self.config['risk']['waypoints_ahead'] * 2.0
        waypoint_obs = []
        for i in range(N_WAYPOINTS):
            idx = self._waypoint_idx + i
            if idx < len(self._waypoints):
                wp_loc = self._waypoints[idx][0].transform.location
                dx = wp_loc.x - self._ego['x']
                dy = wp_loc.y - self._ego['y']
                dist = max(np.sqrt(dx**2 + dy**2), 1e-6)
                waypoint_obs.extend([dx / dist, dy / dist, min(dist / wp_dist_norm, 1.0)])
            else:
                waypoint_obs.extend([0.0, 0.0, 0.0])  # pad if route ends

        ego_state = np.nan_to_num(np.array([
            self._ego['x'], self._ego['y'], self._ego['z'],
            self._ego['vx'], self._ego['vy'], self._ego['yaw'],
            *waypoint_obs # N waypoints x 3 = N*3 values
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

        # Draw a line from the vehicle to the current target waypoint
        if self._waypoint_idx < len(self._waypoints):
            vehicle_loc = self.vehicle.get_location() + carla.Location(z=0.5)
            target_loc = self._waypoints[self._waypoint_idx][0].transform.location + carla.Location(z=0.5)
            debug.draw_line(vehicle_loc, target_loc, thickness=0.02, color=carla.Color(255, 255, 0), life_time=0.1)

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
        colours = []
        for i in range(n):
            hue = (i / max(n - 1, 1)) * 0.8  # span red → blue, avoid wrapping back to red
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colours.append(carla.Color(int(r * 255), int(g * 255), int(b * 255)))

        for i in range(n):
            idx = self._waypoint_idx + i
            if idx >= len(self._waypoints):
                break
            wp, _ = self._waypoints[idx]
            debug.draw_point(
                wp.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=colours[i],
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