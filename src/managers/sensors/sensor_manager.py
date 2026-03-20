import os
import sys
import pygame
import cv2
import carla
import numpy as np
from collections import deque

try:
    from cpp import sensor_utils
    from managers.utils.config_manager import load_config
except ImportError as e:
    print(f"[{__name__}] Error: {e}")

class SensorManager:
    """
    @brief Lightweight sensor data handler for processing via a C++ back-end.

    Captures raw camera and LiDAR data from CARLA sensor callbacks,
    stores the most recent observations, and delegates computationally
    intensive operations (e.g. image conversion and LiDAR projection)
    to optimised C++ routines via Pybind11 bindings.

    Intended for scenarios where rendering is external or minimal
    visualisation is required.
    """

    def __init__(self):
        """
        @brief Initialise internal buffers and state.

        Allocates default buffers for camera and LiDAR data, loads
        configuration parameters, and enables processing via an
        active execution flag.

        @throws RuntimeError if configuration loading fails.
        """

        self.config = load_config() 
        if self.config is None: raise RuntimeError("Failed to load CARLA config")

        # Extract sub-sections for cleaner access
        lidar_config = self.config['sensors']['lidar']
        cam_config = self.config['sensors']['camera']
        train_cam_config = cam_config['train_resolution']

        self.camera_data = np.zeros(
            (train_cam_config['x'], train_cam_config['y'], 4), 
            dtype=np.uint8
        )
        self.lidar_range = lidar_config['range']
        self.lidar_data = np.zeros((1, 4), dtype=np.float32)
        self.active = True

        # Hold references to CARLA sensor actors
        self.camera_sensor = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.collision_history = []

    def attach_sensors(self, world, vehicle, camera_resolution=(800, 600), lidar_range=50):
        """
        @brief Attach CARLA sensors to the specified vehicle.

        Spawns and configures RGB camera, LiDAR, and collision sensors,
        and registers asynchronous callback handlers for each stream.

        @param world CARLA simulation world instance.
        @param vehicle Vehicle actor to which sensors are attached.
        @param camera_resolution Tuple defining camera width and height.
        @param lidar_range Maximum LiDAR sensing range in metres.
        """

        self.lidar_range = lidar_range
        
        cam_width, cam_height = camera_resolution
        bp_lib = world.get_blueprint_library()

        # Collision sensor
        col_bp = bp_lib.find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        self.collision_sensor.listen(lambda event: self.collision_history.append(event))

        # Camera sensor
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(cam_width))
        cam_bp.set_attribute('image_size_y', str(cam_height))
        self.camera_sensor = world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle
        )
        self.camera_sensor.listen(lambda image: self.camera_callback(image))

        # LiDAR sensor
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(self.lidar_range))
        self.lidar_sensor = world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(z=2.5)), attach_to=vehicle
        )
        self.lidar_sensor.listen(lambda data: self.lidar_callback(data))

    def get_collisions(self):
        """
        @brief Retrieve the total number of collision events recorded.

        @return Integer count of collision events.
        """

        return len(self.collision_history)
        
    def get_observation(self, camera_resolution=(84,84), lidar_points=1024):
        """
        @brief Construct a normalised observation vector.

        Processes the latest camera frame and LiDAR point cloud by:
        - resizing the camera image,
        - downsampling LiDAR points,
        - normalising both modalities.

        The result is a flattened vector suitable for machine learning
        pipelines (e.g. reinforcement learning agents).

        @param camera_resolution Target resolution for the camera image.
        @param lidar_points Maximum number of LiDAR points retained.

        @return Concatenated NumPy array of camera and LiDAR features.
        """

        cam = self.camera_data[..., :3] # RGB only
        lidar = self.lidar_data

        # Resize camera
        cam_resized = cv2.resize(cam, camera_resolution)

        # Downsample LiDAR
        if lidar.shape[0] > lidar_points:
            idx = np.random.choice(lidar.shape[0], lidar_points, replace=False)
            lidar = lidar[idx]

        cam_norm = cam_resized.astype(np.float32) / 255.0
        lidar_norm = np.clip(lidar / self.lidar_range, -1.0, 1.0) # clip to [-1,1] to avoid outliers

        obs = np.concatenate([cam_norm.flatten(), lidar_norm.flatten()])
        return obs

    def save_camera_frame(self, file_path):
        """
        Save latest camera frame as a PNG (BGR for OpenCV).

        Returns True if saved successfully, False otherwise.
        """
        if self.camera_data is None: return False

        try:
            bgr_frame = self.camera_data[..., :3] # convert BGRA -> BGR
            bgr_frame = bgr_frame[:, :, ::-1] # RGB -> BGR

            cv2.imwrite(file_path, bgr_frame)
            return True

        except Exception as e:
            print(f"[save_camera_frame] Error: {e}")
            return False

    def camera_callback(self, image):
        """
        @brief Process incoming camera frames from CARLA.

        Converts raw BGRA image data into a NumPy array and forwards it
        to a C++ utility for resizing and colour-space conversion.

        @param image CARLA image object containing raw BGRA data.
        """

        if not self.active: return

        # Shape: (H, W, 4) - BGRA
        self.camera_data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        
        if self.camera_data is not None:
            sensor_utils.process_camera_buffer(self.camera_data, 224, 224)

    def lidar_callback(self, data):
        """
        @brief Process incoming LiDAR point cloud data.

        Converts the raw LiDAR buffer into a structured NumPy array and
        maintains a rolling buffer of recent points for downstream use.

        @param data CARLA LiDAR measurement containing raw point cloud data.
        """

        if not self.active: return
        
        raw_points = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
        
        # Maintain rolling buffer of recent points
        if self.lidar_data is None:
            self.lidar_data = raw_points
        else:
            # Concatenate and keep last N points
            self.lidar_data = np.vstack((self.lidar_data, raw_points))[-30000:]

    def render_all(self):
        """
        @brief Render combined sensor data using C++ visualisation utilities.

        Delegates rendering to an OpenCV-based backend, producing a
        split-screen view of camera and LiDAR data.
        """

        if self.camera_data is not None and self.lidar_data is not None:
            sensor_utils.draw_sensors(self.camera_data, self.lidar_data, 50.0)

    def destroy_sensors(self):
        """
        @brief Safely destroy all sensor actors.

        Releases CARLA actor resources and clears internal references.
        """

        for sensor in [self.camera_sensor, self.lidar_sensor, self.collision_sensor]:
            if sensor is not None: sensor.destroy()

        self.camera_sensor = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.collision_history.clear()

    def close(self):
        """
        @brief Disable further sensor processing.

        Stops callbacks from modifying internal state and ensures
        all sensor actors are properly destroyed.
        """

        self.destroy_sensors()
        self.active = False

class SensorVisualiser:
    """
    @brief Real-time sensor visualisation using Pygame.

    Provides a dual-view interface displaying:
    - RGB camera feed (upper region)
    - LiDAR bird's-eye view (lower region)

    LiDAR data is accumulated over time to produce a denser spatial
    representation, with colour encoding based on height.
    """

    def __init__(self, width=800, height=1000):
        """
        @brief Initialise the visualisation window and layout.

        Sets up the Pygame display surface, rendering regions, and
        internal buffers for LiDAR point accumulation.

        @param width Width of the display window in pixels.
        @param height Height of the display window in pixels.
        """

        pygame.init()

        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CARLA Dual View: Camera & Lidar")
        
        self.point_history = deque(maxlen=30000) 
        self.active = True # Safety flag for threaded callbacks
        
        self.width = width
        self.height = height

        # Define layout regions
        self.cam_rect = pygame.Rect(0, 0, width, 600)
        self.lidar_rect = pygame.Rect(0, 600, width, 400)

        # Scaling factor for LiDAR visualisation
        self.lidar_scale = 15

    def camera_callback(self, image):
        """
        @brief Render camera frames in the upper display region.

        Converts CARLA BGRA image data into RGB format and displays
        it using Pygame surfaces.

        @param image CARLA image object containing raw BGRA data.
        """

        if not self.active: return
            
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = array.reshape((image.height, image.width, 4))
            rgb_array = array[:, :, [2, 1, 0]]
            
            surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(surface, (self.cam_rect.width, self.cam_rect.height))
            
            self.display.blit(scaled_surface, (0, 0))
        except pygame.error:
            pass # Handle window closure during callback

    def lidar_callback(self, lidar_data):
        """
        @brief Render LiDAR data in a bird's-eye view.

        Processes incoming point cloud data, filters invalid values,
        and accumulates points over time. Points are rendered with
        height-based colour encoding.

        @param lidar_data CARLA LiDAR measurement containing raw point cloud data.
        """

        if not self.active: return

        # Convert raw buffer to structured array
        raw_points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(raw_points, (int(raw_points.shape[0] / 4), 4))
        
        # Remove invalid (NaN/Inf) points
        points = points[np.all(np.isfinite(points), axis=1)]

        for pt in points:
            self.point_history.append(pt)

        try:
            # Clear LiDAR display region
            pygame.draw.rect(self.display, (10, 10, 10), self.lidar_rect)
            
            center_x = self.lidar_rect.centerx
            center_y = self.lidar_rect.centery
            
            for pt in self.point_history:
                # Extra safety check before math
                if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                    continue
                    
                draw_x = int(center_x - pt[1] * self.lidar_scale)
                draw_y = int(center_y - pt[0] * self.lidar_scale)

                if self.lidar_rect.collidepoint(draw_x, draw_y):
                    # Height-based colour mapping
                    z_val = pt[2] if np.isfinite(pt[2]) else 0
                    color_val = min(255, max(0, int((z_val + 2) * 50))) 
                    self.display.set_at((draw_x, draw_y), (color_val, 255, 255 - color_val))

            pygame.display.flip()
        except (pygame.error, ValueError, TypeError):
            pass

    def check_exit(self):
        """
        @brief Check for window close events.

        @return True if the user has requested to close the window;
                otherwise False.
        """

        if not self.active: return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return True
        return False

    def close(self):
        """
        @brief Shut down the visualisation system.

        Disables further rendering and safely terminates the Pygame context.
        """

        self.active = False
        pygame.quit()