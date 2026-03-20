import os
import sys
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

import carla
import gym
from gym import spaces

try:
    from carla_client.connection import connect_carla, configure_simulation
    from managers.actors import VehicleManager
    from managers.sensors import SensorManager
    from managers.utils.config_manager import load_config
except ImportError as e:
    print(f"[{__name__}] Error: {e}")

@dataclass
class FrameRecordState:
    index: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    file_path: str = None

class CarlaEnv(gym.Env):
    """
    @brief OpenAI Gym environment for CARLA-based autonomous driving.

    Provides a reinforcement learning interface integrating vehicle
    control, multi-modal sensor observations (camera and LiDAR), and
    simulation management.

    @details
    Observations:
        Flattened vector combining:
        - RGB camera image
        - LiDAR point cloud

    Actions:
        Continuous control vector:
        [steer, throttle, brake]
    """

    metadata = {'render.modes': ['human', 'rgb_array']}
    """ 
    @brief Valid rendering modes for this Gym environment.

    Gym checks this metadata to verify supported render modes:
    - 'human'    : display GUI via sensor manager / OpenCV
    - 'rgb_array': return camera frame as a NumPy array

    @note If a requested mode is not listed here, Gym wrappers
        may skip rendering or raise a warning/error.
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
        train_cam = cam_config['train_resolution']
        camera_dim = train_cam['x'] * train_cam['y'] * cam_config['channels']
        lidar_dim = lidar_config['points'] * lidar_config['features_per_point']

        obs_dim = camera_dim + lidar_dim

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
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

    def reset(self):
        """
        @brief Reset the environment.

        Spawns a new ego vehicle, attaches sensors, and advances
        the simulation to generate the initial observation.

        @return numpy.ndarray Initial observation vector.
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

        self.vehicle = self.vehicle_manager.spawn_ego_vehicle()
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        self._setup_sensors()

        self.episode_length = 0
        self.collision_history = []

        self.world.tick()

        return self._get_observation()

    def step(self, action):
        """
        @brief Execute one environment step.

        Applies the given control to the vehicle, advances the
        simulation, and returns observation, reward, and termination status.

        @param action numpy.ndarray Control vector [steer, throttle, brake].
        @return tuple (obs, reward, done, info) where:
            - obs: numpy.ndarray Current observation vector.
            - reward: float Reward for this step.
            - done: bool Whether the episode has terminated.
            - info: dict Additional information (e.g., collision count).
        """

        self.episode_length += 1

        steer, throttle, brake = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )

        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done()
        info = {'collision': len(self.collision_history)}

        return obs, reward, done, info

    def _get_observation(self):
        """
        @brief Retrieve the current observation vector.

        Combines processed camera and LiDAR data into a
        flattened and normalised vector suitable for ML pipelines.

        @return numpy.ndarray Observation vector.
        """

        lidar_config = self.config['sensors']['lidar']
        cam_config = self.config['sensors']['camera']['train_resolution']

        return self.sensor_manager.get_observation(
            camera_resolution=(cam_config['x'], cam_config['y']),
            lidar_points=lidar_config['points']
        )

    def _compute_reward(self):
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

    def _is_done(self):
        """
        @brief Check whether the episode has terminated.

        Termination occurs if:
        - a collision is recorded
        - the maximum episode length is reached

        @return bool True if the episode is done, False otherwise.
        """

        return (
            len(self.collision_history) > 0 or
            self.episode_length >= self.max_episode_length
        )

    def _setup_sensors(self):
        """
        @brief Attach sensors to the ego vehicle.

        Configures camera resolution and LiDAR range based on configuration.
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

        Ensures proper cleanup of ego vehicle and attached sensors.
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
            print(f"Render mode {mode} not supported")

    def close(self):
        """
        @brief Close the environment and release all resources.

        Safely destroys vehicle and sensor actors.
        """
        
        try:
            self._cleanup_actors()
        except Exception as e:
            print(f"[{__name__}] Warning during cleanup: {e}")