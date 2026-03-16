import os
import sys
import tty
import termios
import select
import carla
import numpy as np

# Allow importing from the src directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carla_client.connection import connect_carla
from carla_client.vehicle_manager import VehicleManager

def is_q_pressed() -> bool:
    """
    @brief Check whether the 'q' key has been pressed.

    Uses non-blocking input to detect a single key press without requiring Enter.
    
    @return True if 'q' was pressed; False otherwise.
    """
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        ch = sys.stdin.read(1)
        return ch.lower() == "q"
    return False

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

        # Set synchronous mode for deterministic behaviour
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        # Spawn ego vehicle
        vehicle_manager = VehicleManager(world)
        vehicle_manager.destroy_all()
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
            local_offset = carla.Location(x=-5, y=0, z=3) # 5m behind and 3m above the vehicle
            rotation = carla.Rotation(yaw=current_new_yaw) # rotate the offset to match the vehicle's orientation

            rotated_offset = \
                rotation.get_forward_vector() * local_offset.x \
                + rotation.get_right_vector() * local_offset.y \
                + rotation.get_up_vector() * local_offset.z # convert the offset into global coordinates relative to the car's facing direction

            spec_location = new_transform.location + rotated_offset # apply the rotated offset to the vehicle's location to get the spectator's location

            spec_transform = carla.Transform(
                spec_location, 
                carla.Rotation(pitch=-15, yaw=current_new_yaw, roll=0)
            )

            spectator = world.get_spectator()
            spectator.set_transform(spec_transform)

            # Tick the world to apply changes
            world.tick()

    except Exception as e:
        print(f"An error occurred during execution: {e}")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        if 'world' in locals():
            print("Resetting simulation settings...")
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        if 'vehicle_manager' in locals():
            print("Cleaning up vehicle...")
            vehicle_manager.destroy_all()

        print("\nSuccessfully exited 🎯✨")

if __name__ == "__main__":
    main()