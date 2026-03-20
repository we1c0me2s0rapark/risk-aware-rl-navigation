import carla
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PedestrianState:
    """
    @brief Represents the runtime state of a pedestrian actor.

    @param actor The CARLA walker actor instance.
    @param target_loc The destination location the pedestrian should move towards.
    @param speed Desired walking speed in metres per second.
    @param is_active Indicates whether the pedestrian is still moving.
    @param arrival_threshold Distance threshold at which the pedestrian is considered to have arrived.
    @param controller Optional AI controller associated with the pedestrian.
    """
    actor: carla.Actor
    target_loc: carla.Location
    speed: float
    is_active: bool = True
    arrival_threshold: float = 0.5
    controller: Optional[carla.Actor] = None
    
class PedestrianManager:
    """
    @class PedestrianManager
    @brief Manages pedestrian actors and their navigation behaviour within the CARLA simulation.
    """

    def __init__(self, world: carla.World) -> None:
        """
        @brief Initialises the pedestrian manager.

        @param world The CARLA world instance used for actor spawning and management.
        """
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        # Track states which now contain everything we need
        self.pedestrians: List[PedestrianState] = []

    def spawn_pedestrian(self, spawn_loc: carla.Location, target_loc: carla.Location, speed: float = 1.2, use_ai: bool = False) -> Optional[PedestrianState]:
        """
        @brief Spawns a pedestrian actor and registers it for management.

        @param spawn_loc World location at which the pedestrian should be spawned.
        @param target_loc Target destination the pedestrian should navigate towards.
        @param speed Desired walking speed in metres per second.
        @param use_ai If true, attaches a CARLA AI walker controller to the actor.
        @return A PedestrianState instance if spawning succeeds; otherwise None.
        """
        # Select a pedestrian blueprint
        walker_bp = self.blueprint_library.filter("walker.pedestrian.*")
        blueprint = walker_bp[0]
        
        # Apply a slight vertical offset to mitigate ground collision artefacts
        spawn_transform = carla.Transform(spawn_loc + carla.Location(z=0.5))
        walker = self.world.try_spawn_actor(blueprint, spawn_transform)

        if walker is None:
            print("Failed to spawn pedestrian.")
            return None
            
        ai_controller = None
        if use_ai:
            controller_bp = self.blueprint_library.find('controller.ai.walker')
            ai_controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), walker)
            if ai_controller:
                ai_controller.start()

        # Create the state object
        ped_state = PedestrianState(
            actor=walker,
            target_loc=target_loc,
            speed=speed,
            controller=ai_controller
        )
        
        self.pedestrians.append(ped_state)
        return ped_state
        
    def update_all(self, debug: Optional[carla.DebugHelper] = None) -> None:
        """
        @brief Updates all managed pedestrians, advancing them towards their targets.

        This method should be invoked once per simulation tick.

        @param debug Optional CARLA debug helper for visualisation.
        """
        for p in self.pedestrians:
            if not p.is_active:
                continue
                
            curr_loc = p.actor.get_location()
            direction = p.target_loc - curr_loc
            distance = direction.length()
            
            # Movement control logic
            control = carla.WalkerControl()
            if distance > p.arrival_threshold:
                control.direction = direction.make_unit_vector()
                control.speed = p.speed
            else:
                control.direction = carla.Vector3D(0, 0, 0)
                control.speed = 0.0
                p.is_active = False
            
            p.actor.apply_control(control)

            # Debug visualisation
            if debug:
                z_offset = carla.Location(z=0.0)

                # Draw line from pedestrian to target location
                debug.draw_line(
                    curr_loc + z_offset,
                    p.target_loc + z_offset, 
                    thickness=0.02, 
                    color=carla.Color(255, 255, 0), 
                    life_time=0.1
                )
                
                # Draw marker at target location
                debug.draw_point(
                    p.target_loc, 
                    size=0.1, 
                    color=carla.Color(255, 0, 0), 
                    life_time=0.1
                )

    def destroy_all(self) -> None:
        """
        @brief Destroys all managed pedestrian actors and associated controllers.

        Ensures proper cleanup of simulation resources.
        """
        for p in self.pedestrians:
            if p.controller and p.controller.is_alive:
                p.controller.stop()
                p.controller.destroy()
            if p.actor and p.actor.is_alive:
                p.actor.destroy()
        
        self.pedestrians.clear()
        print("Cleanup: All pedestrians removed.")