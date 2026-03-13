import sys
import os

# Allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carla_client.connection import connect_carla
from carla_client.diagnostics import print_world_summary

def main() -> None:
    """
    @brief Run a basic CARLA connection test.

    Attempts to connect to CARLA using connect_carla(). If the connection
    is successful, prints confirmation along with the current map name
    and the total number of actors in the world. Otherwise, prints a
    failure message.
    """

    print("Connecting to CARLA...")

    world = connect_carla()

    if world is not None:
        print_world_summary(world)
    else:
        print("Failed to connect to CARLA.")

if __name__ == "__main__":
    main()