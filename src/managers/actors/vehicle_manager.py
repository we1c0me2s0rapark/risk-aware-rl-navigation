import carla
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VehicleState:
    """
    @brief Runtime state of a vehicle in the simulation.

    @param actor The CARLA Vehicle actor.
    @param target_waypoint Optional waypoint the vehicle is currently heading towards.
    @param desired_speed Desired speed in m/s.
    @param is_active Whether the vehicle is currently active/moving.
    @param controller Optional AI or custom controller associated with the vehicle.
    """
    actor: carla.Vehicle
    target_waypoint: Optional[carla.Waypoint] = None
    wheelbase: Optional[float] = None
    desired_speed: float = 0.0
    is_active: bool = True
    controller: Optional[carla.Actor] = None

class VehicleManager:
    """
    @class VehicleManager
    @brief Manage CARLA vehicles created during experiments.

    Responsible for spawning, tracking, and destroying vehicles.
    """

    def __init__(self, world: carla.World) -> None:
        """
        @brief Initialise the vehicle manager.

        @param world CARLA world instance used for vehicle spawning and management.
        """
        self.world = world
        self.blueprint_library = world.get_blueprint_library()
        self.vehicles: List[VehicleState] = []

    def spawn_ego_vehicle(self, vehicle_filter: str = "vehicle.tesla.model3", spawn_index: int = 0, set_physics: bool = False) -> Optional[carla.Vehicle]:
        """
        @brief Spawn an ego vehicle in the CARLA world.

        @param vehicle_filter Blueprint filter string for the desired model.
        @param spawn_index The index of the map spawn point to utilise.
        @return The spawned carla.Vehicle instance, or None if the operation fails.
        """
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points available in the current map.")
            return None

        if spawn_index >= len(spawn_points):
            print(f"Spawn index {spawn_index} is out of range. Defaulting to index 0.")
            spawn_index = 0

        blueprints = self.blueprint_library.filter(vehicle_filter)
        if not blueprints:
            print(f"No vehicle blueprint found for filter: {vehicle_filter}")
            return None

        blueprint = blueprints[0]
        transform = spawn_points[spawn_index]
        
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle:
            vehicle_state = VehicleState(
                actor=vehicle,
                wheelbase=self.get_wheelbase(vehicle),
                desired_speed=0.0
            )
            self.vehicles.append(vehicle_state)
            print(f"Spawned ego vehicle: {vehicle.type_id}")
            return vehicle

        return None

    def get_wheelbase(self, vehicle: carla.Vehicle) -> float:
        """
        @brief Calculate the wheelbase of a given CARLA vehicle.
        
        @param vehicle The carla.Vehicle instance.
        @return The wheelbase in metres.
        """
        physics_control = vehicle.get_physics_control()
        
        # Wheels are indexed; 0: Front Left, 1: Front Right, 2: Rear Left, 3: Rear Right.
        # Calculate the average X position for front and rear axles.
        # CARLA wheel positions are in centimetres; divide by 100 to convert to metres.
        front_wheels_x = (physics_control.wheels[0].position.x + physics_control.wheels[1].position.x) / 200.0
        rear_wheels_x = (physics_control.wheels[2].position.x + physics_control.wheels[3].position.x) / 200.0
        
        # The longitudinal distance between axles is the wheelbase.
        return abs(front_wheels_x - rear_wheels_x)

    def calculate_steering_to_waypoint(self, current_transform: carla.Transform, target_waypoint: carla.Waypoint) -> float:
        """
        @brief Calculate the steering angle needed to head towards a waypoint.

        @param current_transform The current pose of the vehicle.
        @param target_waypoint The target waypoint to reach.
        @return The required steering angle in radians.
        """
        target_loc = target_waypoint.transform.location
        current_loc = current_transform.location
        
        # Compute the heading angle to the target in world space.
        dy = target_loc.y - current_loc.y
        dx = target_loc.x - current_loc.x
        angle_to_target = np.arctan2(dy, dx)
        
        # Obtain current yaw in radians.
        current_yaw = np.radians(current_transform.rotation.yaw)
        
        # Determine the steering error.
        steer_angle = angle_to_target - current_yaw
        
        # Normalise the angle to the interval [-pi, pi].
        steer_angle = (steer_angle + np.pi) % (2 * np.pi) - np.pi
        
        return steer_angle

    def get_next_kinematic_pose(
            self, 
            current_transform: carla.Transform, 
            velocity: float, 
            steer_angle: float,
            L: float,
            dt: float) -> carla.Transform:
        """
        @brief Update pose using the Kinematic Bicycle Model.

        @param current_transform The current vehicle transform.
        @param velocity The scalar velocity in m/s.
        @param steer_angle The steering angle in radians.
        @param L The vehicle wheelbase in metres.
        @param dt The time increment in seconds.
        @return The predicted carla.Transform for the next time step.
        """
        # Constrain steering angle to physical limits to ensure numerical stability.
        steer_angle = np.clip(steer_angle, -0.7, 0.7)
        
        x = current_transform.location.x
        y = current_transform.location.y
        theta = np.radians(current_transform.rotation.yaw)
        
        # Apply kinematic update equations.
        new_x = x + velocity * np.cos(theta) * dt
        new_y = y + velocity * np.sin(theta) * dt
        new_theta = theta + (velocity / L) * np.tan(steer_angle) * dt
        
        return carla.Transform(
            carla.Location(x=new_x, y=new_y, z=current_transform.location.z),
            carla.Rotation(yaw=np.degrees(new_theta))
        )

    def destroy_all(self) -> None:
        """
        @brief Destroy all vehicle actors managed by this instance.
        """
        for vehicle in self.vehicles:
            try:
                if vehicle.actor.is_alive:
                    vehicle.actor.destroy()
            except RuntimeError as exc:
                print(f"Failed to destroy vehicle {vehicle.id}: {exc}")

        self.vehicles.clear()
        print("Destroyed all managed vehicles.")