import os
import sys
import tty
import termios
import carla
import numpy as np

# Allow importing from the src directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carla_client.connection import connect_carla, configure_simulation
from carla_client.vehicle_manager import VehicleManager
from carla_client.spectator_manager import SpectatorManager
from carla_client.utilities import is_q_pressed

def main() -> None:
    """
    @brief Test ego vehicle spawning and kinematic tracking in CARLA.

    Connects to the CARLA server, spawns an ego vehicle, disables physics,
    and updates the pose using a kinematic bicycle model. Press 'q' to quit.
    """

    # Save terminal settings to restore them later.
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        print("Connecting to CARLA...")
        client, world = connect_carla(timeout=15.0)

        if client is None or world is None:
            raise RuntimeError("Could not connect to CARLA. Please ensure the server is running and accessible.")

        # Simulation parameters
        DT = 0.05 # time step for kinematic updates (s)
        LOOK_AHEAD = 1.0 # distance to find next waypoint

        # Set synchronous mode for deterministic behaviour and to control the simulation step manually.
        configure_simulation(client, world, sync_mode=True, dt=DT)

        # Instantiate the vehicle and spectator managers
        vehicle_manager = VehicleManager(world)
        spec_manager = SpectatorManager(world)

        if vehicle_manager is None:
            raise RuntimeError("💥 Failed to initialise VehicleManager.")
        if spec_manager is None:
            raise RuntimeError("💥 Failed to initialise SpectatorManager.")

        # Spawn the ego vehicle
        ego_vehicle = vehicle_manager.spawn_ego_vehicle()

        if ego_vehicle is None:
            raise RuntimeError("💥 Failed to spawn ego vehicle.")
        
        L = vehicle_manager.get_wheelbase(ego_vehicle)
        print(f"Calculated wheelbase for {ego_vehicle.type_id}: {L:.2f}m")

        # Disable physics for pure kinematic control
        ego_vehicle.set_simulate_physics(False)

        # Tick the world to register the spawned vehicle and the physics change
        world.tick()

        print("Tracking started... Press 'q' to stop.")

        tty.setcbreak(fd)

        if not ego_vehicle.is_alive:
            raise RuntimeError(f"💥 Vehicle {ego_vehicle.id} no longer alive.")

        while True:
            if is_q_pressed():
                print("\n'q' pressed - Bye! 👋\n")
                break

            # Get current transform and waypoint
            current_transform = ego_vehicle.get_transform()
            map_waypoint = world.get_map().get_waypoint(current_transform.location)

            # Find next waypoints ahead
            candidates = map_waypoint.next(LOOK_AHEAD)
            if not candidates:
                print("No waypoints found ahead of the vehicle.")
                break
            
            target_wp = candidates[0] # pick the first candidate for simplicity
            target_v = 0.75 # set a constant speed of 0.75 m/s for testing

            steer_angle = vehicle_manager.calculate_steering_to_waypoint(current_transform, target_wp)

            new_transform = vehicle_manager.get_next_kinematic_pose(
                current_transform, 
                target_v, 
                steer_angle, 
                L, 
                DT
            )
            
            ego_vehicle.set_transform(new_transform)

             # Extract the updated yaw from the new transform
            current_new_yaw = new_transform.rotation.yaw

            # Move spectator to follow the ego vehicle
            spec_manager.set_chase_view(ego_vehicle)

            # Tick the world to apply changes
            world.tick()

    except Exception as e:
        print(f"An error occurred during execution: {e}")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        if 'world' in locals() and world is not None and 'client' in locals() and client is not None:
            print("Resetting simulation settings...")
            configure_simulation(client, world, sync_mode=False)

        if 'vehicle_manager' in locals() and vehicle_manager is not None:
            print("Cleaning up vehicle...")
            vehicle_manager.destroy_all()

        print("\nSuccessfully exited 🎯✨")

if __name__ == "__main__":
    main()