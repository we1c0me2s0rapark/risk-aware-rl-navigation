import carla

def get_map_name(world: carla.World) -> str:
    """
    @brief Retrieve the name of the current CARLA map.

    @param world
        A carla.World object representing the simulation world.

    @return str
        The name of the map currently loaded in the world.
    """
    return world.get_map().name

def get_actor_count(world: carla.World) -> int:
    """
    @brief Return the total number of actors in the CARLA world.

    @param world
        A carla.World object representing the simulation world.

    @return int
        The total number of actors currently present in the world.
    """
    return len(world.get_actors())

def print_world_summary(world: carla.World) -> None:
    """
    @brief Print basic diagnostic information about the CARLA world.

    Displays the current map name and total number of actors
    in the world for quick inspection and debugging.

    @param world
        A carla.World object representing the simulation world.
    """

    map_name = world.get_map().name
    actor_count = len(world.get_actors())

    print()
    print("World diagnostics")
    print(f"Map: {map_name}")
    print(f"Actors: {actor_count}")
    print()