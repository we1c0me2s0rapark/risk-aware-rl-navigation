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
    from managers.actors import VehicleManager, PedestrianManager
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")


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
        spec_manager = SpectatorManager(world)

        if v_manager is None:
            raise RuntimeError("💥 Failed to initialise VehicleManager.")
        if p_manager is None:
            raise RuntimeError("💥 Failed to initialise PedestrianManager.")
        if spec_manager is None:
            raise RuntimeError("💥 Failed to initialise SpectatorManager.")

        # Spawn ego vehicle
        ego_vehicle = v_manager.spawn_ego_vehicle()
        if ego_vehicle is None:
            raise RuntimeError("💥 Failed to spawn ego vehicle.")

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
            
            if True:
                spec_manager.set_overhead_view(ego_vehicle)
            else:
                spec_manager.set_chase_view(ego_vehicle)

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