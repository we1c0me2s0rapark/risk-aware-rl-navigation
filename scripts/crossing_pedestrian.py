import os
import sys
import tty
import termios
import select
import carla

# Allow importing from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carla_client.connection import connect_carla, configure_simulation
from carla_client.vehicle_manager import VehicleManager
from carla_client.pedestrian_manager import PedestrianManager

def is_q_pressed() -> bool:
    """
    @brief Detects whether the 'q' key has been pressed.

    Utilises non-blocking standard input to capture a single key press
    without requiring the user to press Enter.

    @return True if the 'q' key is pressed; otherwise False.
    """
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        ch = sys.stdin.read(1)
        return ch.lower() == "q"
    return False

def main() -> None:
    """
    @brief Entry point for testing coordinated vehicle and pedestrian interaction.

    Establishes a CARLA simulation session, spawns an ego vehicle and a pedestrian,
    and continuously updates pedestrian motion until user termination.

    The simulation runs in synchronous mode with a fixed time step.
    """

    # Preserve terminal configuration for later restoration
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        print("Connecting to CARLA...")
        client, world = connect_carla(timeout=15.0)

        if client is None or world is None:
            return

        DT = 0.02

        configure_simulation(client, world, sync_mode=True, dt=DT)

        # Instantiate management components
        v_manager = VehicleManager(world)
        p_manager = PedestrianManager(world)

        # Spawn ego vehicle
        ego_vehicle = v_manager.spawn_ego_vehicle()
        if ego_vehicle is None:
            return

        # Apply handbrake to ensure the vehicle remains stationary during the test
        ego_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        world.tick()

        carla_map = world.get_map()

        # Derive reference vectors from vehicle transform
        v_transform = ego_vehicle.get_transform()
        v_forward = v_transform.get_forward_vector()
        v_right = v_transform.get_right_vector()

        # Compute pedestrian spawn position (8 m ahead, 4 m to the right)
        raw_spawn = v_transform.location + (v_forward * 8.0) + (v_right * 4.0)

        # Project spawn location onto nearest navigable waypoint
        spawn_loc = carla_map.get_waypoint(raw_spawn).transform.location

        # Compute pedestrian target position (crossing trajectory)
        raw_target = v_transform.location + (v_forward * 8.0) - (v_right * 6.0)
        target_loc = carla_map.get_waypoint(raw_target).transform.location

        # Spawn pedestrian actor
        p_manager.spawn_pedestrian(spawn_loc, target_loc, speed=1.2, use_ai=False)

        # Allow simulation ticks for proper initialisation
        for _ in range(20):
            world.tick()

        print("\nTracking started... Press 'q' to stop.")
        tty.setcbreak(fd)

        while True:
            if is_q_pressed():
                print("\n'q' pressed - terminating simulation; Bye 👋\n")
                break
            
            # Update all pedestrian states and optionally render debug visuals
            p_manager.update_all(debug=world.debug)
            
            if False:
                # Overhead spectator view centered on the vehicle
                spec_loc = v_transform.location + carla.Location(z=15)
                world.get_spectator().set_transform(carla.Transform(spec_loc, carla.Rotation(pitch=-90)))
            else:
                # Third-person chase camera positioned relative to the vehicle
                v_rot = v_transform.rotation
                local_offset = carla.Location(x=-5, y=0, z=3) # 5m behind and 3m above the vehicle

                rotation = carla.Rotation(yaw=v_rot.yaw)

                # Transform local offset into world-space coordinates
                rotated_offset = \
                    rotation.get_forward_vector() * local_offset.x \
                    + rotation.get_right_vector() * local_offset.y \
                    + rotation.get_up_vector() * local_offset.z # convert the offset into global coordinates relative to the car's facing direction

                spec_loc = v_transform.location + rotated_offset # apply the rotated offset to the vehicle's location to get the spectator's location

                spec_transform = carla.Transform(
                    spec_loc, 
                    carla.Rotation(pitch=-15, yaw=v_rot.yaw, roll=0)
                )

                spectator = world.get_spectator()
                spectator.set_transform(spec_transform)

            world.tick()

    finally:
        # Restore terminal configuration
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        # Reset simulation to asynchronous mode
        if 'world' in locals() and world is not None and 'client' in locals() and client is not None:
            configure_simulation(client, world, sync_mode=False)

        # Perform resource cleanup
        if 'v_manager' in locals() and v_manager is not None: v_manager.destroy_all()
        if 'p_manager' in locals() and p_manager is not None: p_manager.destroy_all()

if __name__ == "__main__":
    main()