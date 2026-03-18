import os
import sys
import pygame
import numpy as np
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(current_dir)

helper_path = os.path.join(src_path, 'helper')

if helper_path not in sys.path:
    sys.path.append(helper_path)
    
import sensor_utils

class SensorManager:
    """
    @brief Lightweight sensor data handler for processing via C++ back-end utilities.

    This class captures raw camera and LiDAR data from CARLA sensor callbacks,
    stores the most recent observations, and delegates computationally intensive
    processing (e.g. image conversion and LiDAR projection) to optimised C++
    routines via Pybind11 bindings.

    It is designed for scenarios where rendering is handled externally or
    minimal visualisation is required.
    """

    def __init__(self):
        """
        @brief Initialises internal buffers and state flags.

        Allocates default buffers for camera and LiDAR data and enables
        processing via the active flag.
        """
        self.camera_data = np.zeros((600, 800, 4), dtype=np.uint8)
        self.lidar_data = np.zeros((1, 4), dtype=np.float32)
        self.active = True

    def camera_callback(self, image):
        """
        @brief Processes incoming camera frames from CARLA.

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
        @brief Processes incoming LiDAR point cloud data.

        Converts raw LiDAR buffer into a structured NumPy array and
        maintains a rolling history of recent points for downstream use.

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
        @brief Renders combined sensor data using C++ visualisation utilities.

        Delegates rendering to an OpenCV-based backend, which produces a
        split-screen view of camera and LiDAR data.
        """
        if self.camera_data is not None and self.lidar_data is not None:
            sensor_utils.draw_sensors(self.camera_data, self.lidar_data, 50.0)

    def close(self):
        """
        @brief Disables further sensor processing.

        Sets the active flag to False, preventing callbacks from modifying
        internal state or invoking processing routines.
        """
        self.active = False
        
class SensorVisualiser:
    """
    @brief Real-time sensor visualisation using Pygame.

    Provides a dual-view interface displaying:
    - RGB camera feed (top section)
    - LiDAR bird’s-eye-view (bottom section)

    LiDAR data is accumulated over time to produce a denser spatial
    representation, with colour encoding based on height.
    """

    def __init__(self, width=800, height=1000):
        """
        @brief Initialises the visualisation window and layout.

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
        @brief Renders camera frames in the upper display region.

        Converts CARLA BGRA image data into RGB format and displays it
        using Pygame surfaces.

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
        @brief Renders LiDAR data in a bird’s-eye-view representation.

        Processes incoming point cloud data, filters invalid values,
        and accumulates points over time. Points are rendered in the
        lower display region with colour encoding based on height.

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
        @brief Checks for window close events.

        @return True if the user has requested to close the window,
                otherwise False.
        """
        if not self.active: return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return True
        return False

    def close(self):
        """
        @brief Shuts down the visualisation system.

        Disables further rendering and safely terminates the Pygame context.
        """
        self.active = False
        pygame.quit()