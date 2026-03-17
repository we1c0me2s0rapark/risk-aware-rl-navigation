import carla
from typing import Optional

class SpectatorManager:
    """
    @class SpectatorManager
    @brief Manages the CARLA spectator camera.

    Provides methods to set the camera to an overhead view or a chase camera
    following a specified vehicle.
    """

    def __init__(self, world: carla.World):
        """
        @brief Initialises the spectator manager.

        @param world The CARLA world instance used to access the spectator actor.
        """
        self.world = world
        self.spectator = world.get_spectator()

    def set_overhead_view(self, vehicle: carla.Vehicle, height: float = 15.0):
        """
        @brief Positions the spectator directly above the vehicle.

        Sets a top-down view centred on the given vehicle at a specified height.

        @param vehicle The vehicle to centre the camera upon.
        @param height Height above the vehicle in metres (default: 15.0 m).
        """
        transform = vehicle.get_transform()
        spec_loc = transform.location + carla.Location(z=height)
        self.spectator.set_transform(
            carla.Transform(spec_loc, carla.Rotation(pitch=-90, yaw=0, roll=0))
        )

    def set_chase_view(self, vehicle: carla.Vehicle, offset: Optional[carla.Location] = None, pitch: float = -15.0):
        """
        @brief Positions the spectator behind and above the vehicle (third-person view).

        Transforms a local offset relative to the vehicle into world-space coordinates
        and positions the spectator accordingly.

        @param vehicle The vehicle to follow.
        @param offset Local offset relative to the vehicle (x: behind, y: lateral, z: above).
                      Defaults to 5 m behind and 3 m above.
        @param pitch Pitch angle of the camera in degrees (default: -15.0°).
        """
        if offset is None:
            offset = carla.Location(x=-5, y=0, z=3)

        v_transform = vehicle.get_transform()
        v_rot = v_transform.rotation

        # Convert local offset into world-space coordinates
        rotated_offset = (
            v_transform.get_forward_vector() * offset.x +
            v_transform.get_right_vector() * offset.y +
            v_transform.get_up_vector() * offset.z
        )
        spec_loc = v_transform.location + rotated_offset

        spec_transform = carla.Transform(
            spec_loc,
            carla.Rotation(pitch=pitch, yaw=v_rot.yaw, roll=0)
        )
        self.spectator.set_transform(spec_transform)

    def destroy(self):
        """
        @brief Cleans up the spectator manager.

        Currently a placeholder for compatibility with other managers.
        Can be extended if additional cleanup is needed in future.
        """
        self.spectator = None
        print("Spectator manager cleaned up.")