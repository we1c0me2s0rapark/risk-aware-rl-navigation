import os
import sys
import tty
import termios
import carla
import random
import time
import numpy as np

try:    
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src"
    )))

    from carla_client.connection import connect_carla, configure_simulation
    from carla_client.utilities import is_q_pressed
    from managers.utils import SpectatorManager
    from managers.actors import VehicleManager
    from managers.sensors import SensorManager
except ImportError as e:
    print(f"[{__name__}] Error: {e}")

def main():
    """
    @brief Entry point for testing ego vehicle control with sensor integration.

    Establishes a connection to the CARLA simulator, spawns a single ego vehicle,
    and attaches multiple sensors (RGB camera and LiDAR). The vehicle is advanced
    using a kinematic bicycle model while sensor data is processed and rendered.

    The simulation operates in synchronous mode to ensure deterministic execution.
    A spectator camera follows the ego vehicle, and all resources are safely
    released upon termination.
    """

    # Preserve terminal configuration for later restoration
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    sensors = []

    try:
        print("Connecting to CARLA...")
        client, world = connect_carla(timeout=15.0)

        if client is None or world is None:
            raise RuntimeError("Could not connect to CARLA. Please ensure the server is running and accessible.")

        # Simulation parameters
        DT = 0.05 # time step for kinematic updates (s)
        LOOK_AHEAD = 1.0 # distance to find next waypoint

        # Enable synchronous mode for deterministic simulation stepping
        configure_simulation(client, world, sync_mode=True, dt=DT)

        # Initialise management components
        vehicle_manager = VehicleManager(world)
        spec_manager = SpectatorManager(world)
        sensor_manager = SensorManager()

        if vehicle_manager is None:
            raise RuntimeError("Failed to initialise VehicleManager.")
        if spec_manager is None:
            raise RuntimeError("Failed to initialise SpectatorManager.")
        if sensor_manager is None:
            raise RuntimeError("Failed to initialise SensorManager.")
        
        # Ensure no residual vehicles remain from previous sessions
        vehicle_manager.destroy_all()

        # Spawn the ego vehicle
        ego_vehicle = vehicle_manager.spawn_ego_vehicle()

        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")
        
        # Retrieve and report the vehicle wheelbase
        L = vehicle_manager.get_wheelbase(ego_vehicle)
        print(f"Calculated wheelbase for {ego_vehicle.type_id}: {L:.2f}m")

        # Disable physics for pure kinematic control
        if ego_vehicle.is_alive:
            ego_vehicle.set_simulate_physics(False)

        # Tick the world to register the spawned vehicle and the physics change
        world.tick()

        print("Tracking started... Press 'q' to stop.")

        # Enable non-blocking keyboard input
        tty.setcbreak(fd)

        if not ego_vehicle.is_alive:
            raise RuntimeError(f"Vehicle {ego_vehicle.id} no longer alive.")

        # ---------------------------------------------------------------------
        # Sensor Configuration
        # ---------------------------------------------------------------------

        blueprint_library = world.get_blueprint_library()

        # Configure and attach an RGB camera sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Mount at the front of the vehicle, slightly elevated
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        sensors.append(camera)

        # Configure and attach a LiDAR sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '100000')
        
        # Mount centrally on the vehicle roof
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        sensors.append(lidar)

        # Register sensor callbacks
        camera.listen(lambda image: sensor_manager.camera_callback(image))
        lidar.listen(lambda data: sensor_manager.lidar_callback(data))

        # ---------------------------------------------------------------------
        # Main Simulation Loop
        # ---------------------------------------------------------------------
        while True:
            # Exit condition via keyboard input
            if is_q_pressed():
                print("\n'q' pressed - Bye! 👋\n")
                break

            # Retrieve current pose and corresponding map waypoint
            current_transform = ego_vehicle.get_transform()
            map_waypoint = world.get_map().get_waypoint(current_transform.location)

            # Determine candidate waypoints ahead of the vehicle
            candidates = map_waypoint.next(LOOK_AHEAD)
            if candidates:
                target_wp = candidates[0] # pick the first candidate for simplicity
                target_v = 0.75 # set a constant speed of 0.75 m/s for testing

                # Compute steering command towards the target waypoint
                steer_angle = vehicle_manager.calculate_steering_to_waypoint(current_transform, target_wp)

                # Predict next pose using kinematic bicycle model
                new_transform = vehicle_manager.get_next_kinematic_pose(
                    current_transform, 
                    target_v, 
                    steer_angle, 
                    L, 
                    DT
                )
                
                # Apply the updated transform to the vehicle
                ego_vehicle.set_transform(new_transform)

            # Update spectator to follow the ego vehicle
            spec_manager.set_chase_view(ego_vehicle)

            # Advance the simulation
            world.tick()

            # Render sensor outputs
            sensor_manager.render_all()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Restore terminal configuration
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        # Reset simulation to asynchronous mode
        if 'world' in locals() and world is not None and 'client' in locals() and client is not None:
            print("Resetting simulation settings...")
            configure_simulation(client, world, sync_mode=False)

            print("Cleaning up sensors...")
            client.apply_batch([carla.command.DestroyActor(x) for x in sensors if x is not None and x.is_alive])

        # Clean up all managed vehicles
        if 'vehicle_manager' in locals() and vehicle_manager is not None:
            print("Cleaning up vehicle...")
            vehicle_manager.destroy_all()

        print("\nSuccessfully exited 🎯✨")

if __name__ == '__main__':
    main()