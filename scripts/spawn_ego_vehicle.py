import sys
import os

# allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carla_client.connection import connect_carla
from carla_client.actor_manager import ActorManager

def main() -> None:
    """
    @brief Test ego vehicle spawning in CARLA.
    """
    print("Connecting to CARLA...")
    world = connect_carla(timeout=15.0)

    if world is None:
        print("Failed to connect to CARLA.")
        return

    actor_manager = ActorManager(world)
    ego_vehicle = actor_manager.spawn_ego_vehicle()

    if ego_vehicle is not None:
        print(f"👏 Ego vehicle spawned successfully; vehicle ID: {ego_vehicle.id}")
    else:
        print("💥 Ego vehicle spawn failed.")

    if actor_manager.spawned_actors:
        input("Press Enter to destroy actors and exit...")
        actor_manager.destroy_all()

if __name__ == "__main__":
    main()
