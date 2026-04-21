import os
import sys
import math
import random
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

        self.vehicle = self.vehicle_manager.spawn_ego_vehicle(set_physics=True)
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        self.episode_length = 0
        self.collision_history = []

        self._risk_vec = None
        self._ego = None
        self._agents = None

        self._setup_sensors()

        self.world.tick()

        spawn_points = self.world.get_map().get_spawn_points()
        start = self.vehicle.get_location()
        end = random.choice(spawn_points).location
        self._destination = end
        self._waypoint_idx = 0

        grp = GlobalRoutePlanner(self.world.get_map(), 2.0)
        self._waypoints = grp.trace_route(start, end)
        self._prev_dist_to_goal = start.distance(self._get_next_waypoint())

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

        baseline_rewards = baseline_reward(
            self.config['risk']['reward'],
            self._ego,
            self._check_collision(),
            self._compute_progress(),
            top_k=self.risk_module.top_k,
        )

        rewards = decomposed_reward(
            self.config['risk']['reward'],
            self._ego,
            self._check_collision(),
            self._compute_progress(),
            self._risk_vec,
            self.risk_module.top_k,
            wp_dx=wp_dx,
            wp_dy=wp_dy,
            wrong_way_risk=wrong_way_risk,
        )

        reward = rewards

        # Construct observation
        obs = self._get_observation()

        if self._check_collision():
            Log.warning(__file__, f"Collision detected with {len(self.collision_history)} events")

        done = (
            self._check_collision() or
            self._goal_reached() or
            self._timeout()
        )

        info = {
            'collision': self._check_collision(),
            'goal_reached': self._goal_reached(),
            'ttc_min': float(self._risk_vec[4::AGENT_FEAT_DIM].min()) if self._risk_vec is not None else 0.0, # near-miss rate (TTC threshold)
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

        target = self._get_next_waypoint()
        current_dist = self.vehicle.get_location().distance(target)
        progress = self._prev_dist_to_goal - current_dist
        self._prev_dist_to_goal = current_dist

        # Advance waypoint if close enough
        if current_dist < 2.0 and self._waypoint_idx < len(self._waypoints) - 1:
            self._waypoint_idx += 1
            self._prev_dist_to_goal = self.vehicle.get_location().distance(
                self._get_next_waypoint()
            )
            return 5.0 # waypoint reached bonus, overrides step progress

        return float(progress)

    def _goal_reached(self) -> bool:
        """
        @brief Determine if the ego vehicle has reached its destination.
 
        @details
        Checks whether the Euclidean distance from the ego vehicle to
        the destination is less than the configured goal threshold.
 
        @return bool True if within goal_threshold metres of destination.
        """

        if self._destination is None: return False

        dist = self.vehicle.get_location().distance(self._destination)

        return dist < self.config['simulation'].get('goal_threshold', 5.0)

    def _timeout(self) -> bool:
        """
        @brief Check if the episode has exceeded the maximum length.
 
        @return bool True if the episode has timed out.
        """

        return self.episode_length >= self.max_episode_length

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
            - 'ego_state': [x, y, z, vx, vy, yaw, wp_dx0, wp_dy0, ..., wp_dx4, wp_dy4] (16 values)
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
        waypoint_obs = []
        for i in range(N_WAYPOINTS):
            idx = self._waypoint_idx + i
            if idx < len(self._waypoints):
                wp_loc = self._waypoints[idx][0].transform.location
                dx = wp_loc.x - self._ego['x']
                dy = wp_loc.y - self._ego['y']
                dist = max(np.sqrt(dx**2 + dy**2), 1e-6)
                waypoint_obs.extend([dx / dist, dy / dist])
            else:
                waypoint_obs.extend([0.0, 0.0])  # pad if route ends

        ego_state = np.array([
            self._ego['x'], self._ego['y'], self._ego['z'],
            self._ego['vx'], self._ego['vy'], self._ego['yaw'],
            *waypoint_obs # 5 waypoints × 2 = 10 values
        ], dtype=np.float32)

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
        Draws the next 10 waypoints ahead of the current position using
        a rainbow colour scheme, and marks the destination with a larger
        red dot.
 
        Colour scheme:
            - Waypoints +0 to +5  : rainbow (red → orange → yellow → green → blue → indigo)
            - Remaining waypoints : violet
            - Destination         : red (larger marker)
        """

        debug = self.world.debug

        rainbow = [
            carla.Color(255, 0,   0  ), # red
            carla.Color(255, 127, 0  ), # orange
            carla.Color(255, 255, 0  ), # yellow
            carla.Color(0,   255, 0  ), # green
            carla.Color(0,   0,   255), # blue
            carla.Color(75,  0,   130), # indigo
        ]
        violet = carla.Color(148, 0, 211)

        total_waypoints_to_draw = 10

        # Find the closest waypoint to the vehicle's current position
        ego_loc = self.vehicle.get_location()
        closest_idx = min(
            range(self._waypoint_idx, min(self._waypoint_idx + 50, len(self._waypoints))),
            key=lambda i: self._waypoints[i][0].transform.location.distance(ego_loc),
            default=self._waypoint_idx
        )

        # Draw next N waypoints from closest
        for i in range(total_waypoints_to_draw):
            idx = closest_idx + i
            if idx >= len(self._waypoints):
                break
            wp, _ = self._waypoints[idx]
            colour = rainbow[i] if i < len(rainbow) else violet
            debug.draw_point(
                wp.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=colour,
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