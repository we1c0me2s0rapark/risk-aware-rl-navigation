import sys
import os

# Allow importing from the src directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carla_client.connection import connect_carla
from carla_client.vehicle_manager import VehicleManager

def main() -> None:
    """
    @brief Test ego vehicle spawning in the CARLA environment.

    Connects to the simulator, spawns a vehicle via the VehicleManager,
    and ensures clean destruction upon completion.
    """
    print("Connecting to CARLA...")
    client, world = connect_carla(timeout=15.0)

    if client is None or world is None:
        print("Failed to connect to CARLA.")
        return

    # Initialise the manager with the current world instance
    vehicle_manager = VehicleManager(world)
    
    # Attempt to spawn the default ego vehicle
    ego_vehicle = vehicle_manager.spawn_ego_vehicle()

    if ego_vehicle is not None:
        print(f"Ego vehicle spawned successfully; vehicle ID: {ego_vehicle.id}")
    else:
        print("Ego vehicle spawn failed.")

    # Prevent immediate script termination to allow for visual verification
    if vehicle_manager.vehicles:
        input("Press Enter to destroy vehicles and exit...")
        vehicle_manager.destroy_all()

if __name__ == "__main__":
    main()