import random as _random
import carla
from typing import List, Optional

class NPCManager:
    """
    @class NPCManager
    @brief Manages NPC vehicles spawned under CARLA Traffic Manager autopilot.

    @details
    Spawns 4-wheeled vehicles at random map spawn points that satisfy a
    minimum distance from the ego vehicle, then hands each one to the
    Traffic Manager so it drives, changes lanes, and obeys traffic rules
    automatically. This creates realistic TTC events for the ego agent
    to respond to without requiring manual NPC controllers.

    Bikes and motorcycles are excluded because the TM handles them poorly
    at urban / highway speeds.
    """

    def __init__(self, world: carla.World, client: carla.Client) -> None:
        """
        @brief Initialise the NPC manager.

        @param world  CARLA world instance.
        @param client CARLA client (needed to obtain the Traffic Manager).
        """
        self.world = world
        self.client = client
        self._vehicles: List[carla.Vehicle] = []

    def spawn(
        self,
        num_vehicles: int,
        ego_location: carla.Location,
        min_distance: float = 15.0,
        tm_port: int = 8000,
        sync_mode: bool = True,
        rng: Optional[_random.Random] = None,
        seed: Optional[int] = None,
    ) -> int:
        """
        @brief Spawn NPC vehicles and register them with the Traffic Manager.

        @param num_vehicles  Number of vehicles to attempt to spawn.
        @param ego_location  Ego vehicle location; spawn points closer than
                             min_distance are skipped.
        @param min_distance  Minimum distance (metres) from ego for a valid spawn.
        @param tm_port       Traffic Manager port.
        @param sync_mode     Whether the simulation is in synchronous mode.
        @param rng           Seeded Random instance for reproducibility.
        @param seed          If provided, also seeds the Traffic Manager's internal RNG.
        @return              Number of vehicles actually spawned.
        """
        if num_vehicles <= 0:
            return 0

        rng = rng or _random

        tm = self.client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(sync_mode)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.global_percentage_speed_difference(-20.0)  # slightly under speed limit
        if seed is not None:
            tm.set_random_device_seed(seed)

        blueprints = [
            bp for bp in self.world.get_blueprint_library().filter('vehicle.*')
            if int(bp.get_attribute('number_of_wheels')) == 4
        ]

        spawn_points = self.world.get_map().get_spawn_points()
        rng.shuffle(spawn_points)

        for sp in spawn_points:
            if len(self._vehicles) >= num_vehicles:
                break
            if sp.location.distance(ego_location) < min_distance:
                continue
            bp = rng.choice(blueprints)
            if bp.has_attribute('color'):
                bp.set_attribute('color', rng.choice(bp.get_attribute('color').recommended_values))
            npc = self.world.try_spawn_actor(bp, sp)
            if npc is not None:
                npc.set_autopilot(True, tm_port)
                self._vehicles.append(npc)

        return len(self._vehicles)

    def spawn_static_on_route(
        self,
        waypoints: list,
        from_idx: int,
        num_obstacles: int,
        min_gap: int = 20,
        rng: Optional[_random.Random] = None,
    ) -> int:
        """
        @brief Spawn stationary NPC vehicles at random positions along the route.

        @details
        Selects num_obstacles waypoints at random from the route, starting at
        least min_gap waypoints ahead of from_idx so the ego has room to
        accelerate before encountering the first obstacle. Vehicles are spawned
        with the handbrake engaged and no autopilot, so they remain stationary.
        Physics stays enabled so collisions register on the ego's sensor.

        @param waypoints      Full route waypoint list from GlobalRoutePlanner.
        @param from_idx       Current ego waypoint index (start of search window).
        @param num_obstacles  Number of static obstacles to place.
        @param min_gap        Minimum waypoints ahead of from_idx before placing
                              an obstacle (~2 m per waypoint, default 20 ≈ 40 m).
        @return               Number of obstacles actually spawned.
        """
        if num_obstacles <= 0:
            return 0

        start = from_idx + min_gap
        end = len(waypoints) - 10   # leave room at the end
        if start >= end:
            return 0

        rng = rng or _random

        blueprints = [
            bp for bp in self.world.get_blueprint_library().filter('vehicle.*')
            if int(bp.get_attribute('number_of_wheels')) == 4
        ]

        positions = rng.sample(
            range(start, end),
            min(num_obstacles, end - start)
        )

        for idx in positions:
            wp = waypoints[idx][0]
            bp = rng.choice(blueprints)
            if bp.has_attribute('color'):
                bp.set_attribute('color', rng.choice(bp.get_attribute('color').recommended_values))
            npc = self.world.try_spawn_actor(bp, wp.transform)
            if npc is not None:
                npc.apply_control(carla.VehicleControl(hand_brake=True))
                self._vehicles.append(npc)

        return len(self._vehicles)

    def destroy_all(self) -> None:
        """@brief Destroy all managed NPC vehicles and clear the internal list."""
        for npc in self._vehicles:
            if npc.is_alive:
                npc.destroy()
        self._vehicles.clear()