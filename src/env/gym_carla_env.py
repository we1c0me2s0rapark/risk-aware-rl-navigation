import os
import sys
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

import carla
import gym
from gym import spaces

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."
    )))
    
    from carla_client.connection import connect_carla, configure_simulation
    from managers.actors import VehicleManager
    from managers.sensors import SensorManager
    from managers.utils.config_manager import load_config
    from managers.utils.logger import Log
    from risk.module import RiskModule, AGENT_FEAT_DIM
    from env.reward import risk_aware_reward
except ImportError as e:
    Log.error(__file__, e)

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
    @brief OpenAI Gym environment for CARLA-based autonomous driving.

    @details
    Provides a reinforcement learning interface integrating:
        - Vehicle control
        - Multi-modal sensor observations (camera and LiDAR)
        - Simulation management

    Observations:
        Flattened vector combining:
        - RGB camera image
        - LiDAR point cloud
        - Risk features

    Actions:
        Continuous control vector [steer, throttle, brake].
    """

    metadata = {'render.modes': ['human', 'rgb_array']}
    """
    @brief Valid rendering modes.

    @note If a requested mode is not listed, Gym wrappers may skip rendering
          or raise a warning/error.
    """

    def __init__(self):
        """
        @brief Initialise the CARLA environment.

        Loads configuration, connects to the CARLA simulator, and
        initialises observation and action spaces.

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

    def reset(self):
        """
        @brief Reset the environment.

        Spawns a new ego vehicle, attaches sensors, and advances
        the simulation to generate the initial observation.

        @return dict Initial observation dictionary containing:
            - 'camera': Camera vector
            - 'lidar': LiDAR vector
            - 'ego_state': Ego vehicle state
            - 'risk_features': Risk module features
        """

        self._cleanup_actors()

        self.frame_record_state = FrameRecordState()
        
        frames_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frames"))
        os.makedirs(frames_dir, exist_ok=True)
            
        datestamp = self.frame_record_state.start_time.strftime("%Y%m%d")
        file_dir = os.path.join(frames_dir, f"{datestamp}")
        os.makedirs(file_dir, exist_ok=True)

        timestamp = self.frame_record_state.start_time.strftime("%Y%m%d_%H%M%S")
        self.frame_record_state.file_path = os.path.join(file_dir, f"{timestamp}")

        self.vehicle = self.vehicle_manager.spawn_ego_vehicle(set_physics=True)
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        self._setup_sensors()

        self.episode_length = 0
        self.collision_history = []

        self._risk_vec = None
        self._ego = None
        self._agents = None

        self.world.tick()

        spawn_points = self.world.get_map().get_spawn_points()
        self._destination = random.choice(spawn_points).location
        self._prev_dist_to_goal = self.vehicle.get_location().distance(self._destination)

        return self._get_observation()

    def step(self, action):
        """
        @brief Execute one environment step.

        Applies the given control to the vehicle, advances the simulation,
        and returns observation, reward, and termination status.

        @param action numpy.ndarray Control vector [steer, throttle, brake]
        @return tuple (obs, reward, done, info)
            - obs: Current observation dictionary
            - reward: Reward for this step
            - done: Episode termination flag
            - info: Additional info (e.g., collision count, distance to goal)
        """

        self.episode_length += 1

        steer, throttle, brake = self._apply_control(action)
        self.world.tick()

        self._ego = self._get_ego_state()
        self._agents = self._get_nearby_actors()

        self._risk_vec = self.risk_module.compute(self._ego, self._agents)

        baseline_reward = self._compute_reward_baseline()
        risk_reward = risk_aware_reward(
            self._ego,
            self._check_collision(),
            self._compute_progress(),
            self._risk_vec,
            self.risk_module.top_k
        )

        reward = risk_reward # keep risk_reward as the main reward

        obs = self._get_observation()

        done = self._check_collision() or self._goal_reached() or self._timeout()

        info = {
            'collision': self._check_collision(),
            'ttc_min': float(self._risk_vec[4::AGENT_FEAT_DIM].min()) if self._risk_vec is not None else 0.0,
            'goal_dist': self._prev_dist_to_goal,
            'baseline_reward': baseline_reward,
            'risk_reward': risk_reward,
        }

        speed = np.sqrt(self._ego['vx']**2 + self._ego['vy']**2)

        Log.info(__file__, f"""SNAPSHOT
    Status: {'ALIVE 🟢' if self.vehicle.is_alive else 'DEAD 🔴'}
    Pose: x {self._ego['x']:.2f}, y {self._ego['y']:.2f}, z {self._ego['z']:.2f}, yaw {self._ego['yaw']}
    Speed: {speed} m/s
    Control: steer {steer}, throttle {throttle}, brake {brake}""")

        return obs, reward, done, info

    def _apply_control(self, action) -> list[float]:
        """
        @brief Clip and apply a control action to the ego vehicle.
        
        @details
        The action is first clipped to the valid ranges defined by the
        action_space. Throttle and brake are mutually exclusive: if throttle
        exceeds 0.05, brake is set to 0, and vice versa. The resulting control
        is applied to the CARLA vehicle actor.

        @param action numpy.ndarray Control vector [steer, throttle, brake].
        @return list[float] Applied control values [steer, throttle, brake].
        """
        steer, throttle, brake = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        if throttle > 0.05: brake = 0.0
        elif brake > 0.05: throttle = 0.0

        self.vehicle.apply_control(carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake)
        ))

        return steer, throttle, brake

    def _check_collision(self) -> bool:
        """
        @brief Check if a collision has occurred in the current episode.

        @details
        Returns True if any collision has been recorded by the
        collision sensor attached to the ego vehicle during the episode.
        Useful for terminating episodes or modifying rewards.

        @return bool True if a collision has been detected, False otherwise.
        """

        return len(self.collision_history) > 0

    def _compute_progress(self) -> float:
        """
        @brief Compute the progress made towards the goal during this step.

        @details
        The progress is defined as the change in Euclidean distance
        from the ego vehicle to its destination between the previous
        step and the current step. Positive values indicate that the
        vehicle has approached the goal, negative values indicate moving
        away from it. If no destination is set, the forward distance
        travelled in metres over this timestep is returned as an approximation.

        @return float Distance closed towards the goal in metres.
            Positive if approaching the destination, negative if moving away.
        """

        if self._destination is None:
            v = self.vehicle.get_velocity()
            return float(np.sqrt(v.x**2 + v.y**2 + v.z**2) * self.config['simulation']['dt'])

        current_dist = self.vehicle.get_location().distance(self._destination)
        progress = self._prev_dist_to_goal - current_dist
        self._prev_dist_to_goal = current_dist

        return float(progress)

    def _goal_reached(self) -> bool:
        """
        @brief Determine if the ego vehicle has reached its destination.

        @details
        Checks whether the Euclidean distance from the ego vehicle to
        the destination is less than the configured goal threshold
        in metres. If no destination is defined, the goal is considered
        unreached.

        @return bool True if the vehicle is within goal_threshold metres
                    of the destination, False otherwise.
        """

        if self._destination is None: return False

        dist = self.vehicle.get_location().distance(self._destination)

        return dist < self.config['simulation'].get('goal_threshold', 5.0)

    def _timeout(self) -> bool:
        """
        @brief Check if the episode has exceeded the maximum length.

        @details
        Returns True if the number of steps executed in the current
        episode has reached or exceeded the maximum allowed by the
        simulation configuration.

        @return bool True if the episode is timed out, False otherwise.
        """

        return self.episode_length >= self.max_episode_length

    def _get_ego_state(self) -> dict:
        """
        @brief Extract the current state of the ego vehicle.

        @details
        The state includes the vehicle's 3D position, velocity components,
        yaw angle in radians, and bounding box dimensions. This information
        is primarily used for risk computation and observation construction.

        @return dict Dictionary containing the ego state:
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
        Queries the CARLA world to find all vehicles and pedestrians
        within a given Euclidean distance from the ego vehicle.
        Each actor is represented as a dictionary containing its
        position, velocity, heading, dimensions, and category.
        Pedestrians are approximated with a fixed bounding box size.

        @param radius float Search radius in metres around the ego vehicle.
        @return list[dict] List of dictionaries, each representing a nearby actor:
            - 'id': CARLA actor ID
            - 'x', 'y': position coordinates in metres
            - 'vx', 'vy': velocity components in metres per second
            - 'yaw': heading in radians
            - 'length', 'width': bounding box dimensions in metres
            - 'category': 'vehicle' or 'walker'
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
        @brief Retrieve the current observation from the environment.

        @details
        Constructs a comprehensive observation dictionary combining:
            - Camera data
            - LiDAR data
            - Ego vehicle state
            - Risk module features

        The ego vehicle state is computed if it is not already cached.
        If risk features have not yet been calculated, the method queries
        nearby actors and computes the risk vector using the RiskModule.

        Sensor data is flattened and returned as float32 NumPy arrays.

        @return dict Observation dictionary containing:
            - 'camera' : Flattened NumPy array of camera data
            - 'lidar' : Flattened NumPy array of LiDAR data
            - 'ego_state' : NumPy array of ego vehicle state [x, y, z, vx, vy, yaw]
            - 'risk_features' : NumPy array of computed risk features
        """
        
        # Ensure ego state exists
        if self._ego is None:
            self._ego = self._get_ego_state()  # <-- compute ego state if missing

        ego = self._ego

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

        cam_dim = cam_res['x'] * cam_res['y'] * cam_channels
        lidar_dim = lidar_config['points'] * lidar_config['features_per_point']
        risk_dim = self.risk_module.feature_dim

        camera_obs = sensor_obs[:cam_dim]
        lidar_obs = sensor_obs[cam_dim:cam_dim + lidar_dim]

        if self._risk_vec is None:
            self._agents = self._get_nearby_actors()
            self._risk_vec = self.risk_module.compute(self._ego, self._agents)
        risk_obs = self._risk_vec.astype(np.float32)

        # Construct ego_state array
        ego_state = np.array([
            ego['x'], ego['y'], ego['z'],
            ego['vx'], ego['vy'], ego['yaw']
        ], dtype=np.float32)

        return {
            "camera": camera_obs,
            "lidar": lidar_obs,
            "ego_state": ego_state,
            "risk_features": risk_obs
        }

    def _compute_reward_baseline(self):
        """
        @brief Compute the reward for the current timestep.

        Encourages forward motion whilst penalising:
        - collisions
        - excessive steering

        @return float Reward value.
        """

        v = self.vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        reward = speed * 0.05
        if len(self.collision_history) > 0:
            reward -= 10.0
        reward -= abs(self.vehicle.get_control().steer) * 0.01
        return reward

    def _is_done_baseline(self):
        """
        @brief Check whether the episode has terminated.

        Termination occurs if:
        - a collision is recorded
        - the maximum episode length is reached

        @return bool True if the episode is done, False otherwise.
        """

        return (len(self.collision_history) > 0 or self.episode_length >= self.max_episode_length)

    def _setup_sensors(self):
        """
        @brief Attach and initialise sensors on the ego vehicle.

        @details
        Configures and attaches all required sensors (camera and LiDAR) to
        the ego vehicle using the SensorManager. Camera resolution and LiDAR
        range are read from the loaded configuration. Collision history is
        initialised and linked to the sensor manager for later queries.

        @note The sensors must be attached after the ego vehicle has been spawned.
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
        Ensures proper cleanup of the ego vehicle and all attached sensors.
        Resets collision history to an empty list. This function should be
        called when resetting the environment or closing it to prevent
        resource leaks.

        @note Calling this method does not remove non-ego actors from the world.
        """

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        self.sensor_manager.destroy_sensors()
        self.collision_history = []

    def render(self, mode='human', save=False):
        """
        @brief Render the environment.

        @details
        Modes:
            - 'human': display GUI via sensor manager
            - 'rgb_array': return the latest camera frame as a NumPy array

        @param mode str Render mode ('human' or 'rgb_array').
        @param save bool Whether to save the frame to disk (only relevant for 'rgb_array').
        @return numpy.ndarray|None Camera frame if mode='rgb_array', otherwise None.
        """

        render_mode = self.config.get('render', {}).get('mode', 'human')

        if render_mode == 'human':
            self.sensor_manager.render_all() # existing OpenCV render
        elif render_mode == 'rgb_array':
            parent_dir = self.frame_record_state.file_path
            os.makedirs(parent_dir, exist_ok=True)
            if not os.path.exists(parent_dir): return

            filename = f"{self.frame_record_state.start_time.strftime('%Y%m%d_%H%M%S')}_{self.frame_record_state.index}.png"
            file_path = os.path.join(parent_dir, filename)
            if self.sensor_manager.save_camera_frame(file_path=file_path):
                self.frame_record_state.index += 1
        else:
            Log.warning(__file__, f"Render mode {mode} not supported")

    def close(self):
        """
        @brief Close the environment and release all resources.

        Safely destroys vehicle and sensor actors.
        """
        
        try:
            self._cleanup_actors()
        except Exception as e:
            Log.error(__file__, e)