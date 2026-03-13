import carla
from typing import List, Optional

class ActorManager:
    """
    @brief Manage CARLA actors created during experiments.

    Responsible for spawning, tracking, and destroying actors.
    Currently handles ego vehicle; can be extended for NPCs, pedestrians, sensors, etc.
    """

    def __init__(self, world: carla.World) -> None:
        """
        @brief Initialise the actor manager.

        @param world
            CARLA world instance used for actor spawning and management.
        """
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        self.spawned_actors: List[carla.Actor] = []

    def spawn_ego_vehicle(
        self,
        vehicle_filter: str = "vehicle.tesla.model3",
        spawn_index: int = 0
    ) -> Optional[carla.Vehicle]:
        """
        @brief Spawn an ego vehicle in the CARLA world.

        @param vehicle_filter
            Blueprint filter to select the ego vehicle model.
        @param spawn_index
            Index of the spawn point to use.

        @return Optional[carla.Vehicle]
            The spawned ego vehicle if successful, otherwise None.
        """
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points available in the current map.")
            return None

        if spawn_index >= len(spawn_points):
            print(f"Spawn index {spawn_index} is out of range. Using spawn index 0 instead.")
            spawn_index = 0

        blueprints = self.blueprint_library.filter(vehicle_filter)
        if not blueprints:
            print(f"No vehicle blueprint found for filter: {vehicle_filter}")
            return None

        blueprint = blueprints[0]
        transform = spawn_points[spawn_index]
        actor = self.world.try_spawn_actor(blueprint, transform)

        if actor is None:
            print("Failed to spawn ego vehicle.")
            return None

        self.spawned_actors.append(actor)
        print(f"Spawned ego vehicle: {actor.type_id}")
        return actor

    def destroy_all(self) -> None:
        """
        @brief Destroy all actors managed by this instance.
        """
        for actor in self.spawned_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError as exc:
                print(f"Failed to destroy actor {actor.id}: {exc}")

        self.spawned_actors.clear()
        print("Destroyed all managed actors.")
