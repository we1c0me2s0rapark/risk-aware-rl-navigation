import os
import sys
import tty
import termios
import carla

try:
    # Allow importing from the src directory
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "src"
    )))

    from carla_client.connection import connect_carla, configure_simulation
    from carla_client.utilities import is_q_pressed
    from managers.utils import SpectatorManager
    from managers.actors import VehicleManager
    from managers.sensors import SensorVisualiser
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

def main():
    """
    @brief Entry point for ego vehicle control with integrated sensor visualisation.

    Connects to the CARLA simulator, spawns a single ego vehicle, and attaches
    RGB camera and LiDAR sensors. Sensor data is streamed to a visualisation
    module for real-time rendering, while the vehicle is advanced using a
    kinematic bicycle model.

    The simulation runs in synchronous mode to ensure deterministic stepping.
    A spectator camera follows the ego vehicle, and all allocated resources
    (sensors, vehicles, and visualiser) are safely released upon termination.
    """

    # Preserve terminal configuration for later restoration
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # Container for tracking spawned sensor actors
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

        if vehicle_manager is None:
            raise RuntimeError("Failed to initialise VehicleManager.")
        if spec_manager is None:
            raise RuntimeError("Failed to initialise SpectatorManager.")

        # Remove any pre-existing vehicles from the simulation
        vehicle_manager.destroy_all()

        # Spawn the ego vehicle
        ego_vehicle = vehicle_manager.spawn_ego_vehicle()

        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")
        
        # Retrieve and report vehicle wheelbase
        L = vehicle_manager.get_wheelbase(ego_vehicle)
        print(f"Calculated wheelbase for {ego_vehicle.type_id}: {L:.2f}m")

        # Disable physics to allow direct kinematic control
        if ego_vehicle.is_alive:
            ego_vehicle.set_simulate_physics(False)

        # Tick the world to register spawned actors and configuration changes
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

        # Configure RGB camera sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Mount at the front of the vehicle with slight elevation
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        sensors.append(camera)

        # Configure LiDAR sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '100000')
        
        # Mount centrally on the vehicle roof
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        sensors.append(lidar)

        # Initialise sensor visualisation module
        visualiser = SensorVisualiser(width=800, height=1000)
        
        # Register sensor callbacks
        camera.listen(lambda image: visualiser.camera_callback(image))
        lidar.listen(lambda data: visualiser.lidar_callback(data))

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
            if not candidates:
                print("No waypoints found ahead of the vehicle.")
                break
            
            # Select target waypoint and define constant velocity
            target_wp = candidates[0] # pick the first candidate for simplicity
            target_v = 0.75 # set a constant speed of 0.75 m/s for testing

            # Compute steering command towards the waypoint
            steer_angle = vehicle_manager.calculate_steering_to_waypoint(current_transform, target_wp)

            # Predict next pose using kinematic bicycle model
            new_transform = vehicle_manager.get_next_kinematic_pose(
                current_transform, 
                target_v, 
                steer_angle, 
                L, 
                DT
            )
            
            # Apply updated transform
            ego_vehicle.set_transform(new_transform)

            # Update spectator to follow the ego vehicle
            spec_manager.set_chase_view(ego_vehicle)

            # Advance simulation
            world.tick()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Restore terminal configuration
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Close visualisation window if initialised
        if 'visualiser' in locals() and visualiser:
            visualiser.close()
        
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