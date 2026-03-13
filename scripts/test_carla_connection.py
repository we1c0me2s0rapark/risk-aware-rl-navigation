import carla
import subprocess
from typing import Optional, List


def get_windows_host_ip() -> Optional[str]:
    """
    @brief Retrieve the Windows host IP address from WSL.

    In WSL2 networking the Windows host typically appears as the default
    gateway of the Linux network interface. This function queries the
    routing table and extracts the gateway address.

    @return Optional[str]
        The detected Windows host IP address if available, otherwise None.
    """

    try:
        result = subprocess.check_output(
            ["ip", "route"],
            encoding="utf-8"
        )

        for line in result.splitlines():
            if line.startswith("default"):
                parts = line.split()

                if len(parts) >= 3:
                    return parts[2]

    except Exception as exc:
        print(f"Exception: {exc}")
        pass

    return None


def attempt_connection(host: str, port: int, timeout: float) -> Optional[carla.World]:
    """
    @brief Attempt to connect to CARLA using a single host.

    @param host
        Host address to try.

    @param port
        CARLA TCP port.

    @param timeout
        Connection timeout in seconds.

    @return Optional[carla.World]
        A CARLA world object if the connection succeeds, otherwise None.
    """
    try:
        print(f"Trying {host}...")
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        print(f"Connected via {host}")
        return world
    except RuntimeError as exc:
        print(f"Connection via {host} failed: {exc}")
    except Exception as exc:
        print(f"Unexpected error via {host}: {exc}")

    return None


def connect_carla(port: int = 2000, timeout: float = 5.0) -> carla.World:
    """
    @brief Establish a connection to the CARLA simulator.

    The function first tries localhost. If that fails, it tries a valid
    Windows host IP extracted from /etc/resolv.conf, if available.

    @param port
        TCP port used by the CARLA simulator.

    @param timeout
        Connection timeout in seconds.

    @return carla.World
        The CARLA world object obtained from the simulator.

    @throws RuntimeError
        Raised when the simulator cannot be reached.
    """
    hosts: List[str] = ["localhost"]

    windows_host_ip = get_windows_host_ip()
    if windows_host_ip and windows_host_ip not in hosts:
        # hosts.append(windows_host_ip)
        hosts.insert(0, windows_host_ip) # insert at the start to try it first

    for host in hosts:
        world = attempt_connection(host, port, timeout)
        if world:
            return world
    return None

    raise RuntimeError(
        "Could not connect to CARLA. "
        "Ensure that the simulator is running and listening on port 2000."
    )


def main() -> None:
    """
    @brief Run a basic CARLA connection test.
    """
    print("Connecting to CARLA...")
    world = connect_carla()
    if world is None:
        print("Failed to connect to CARLA.")
        return
    print("Connected successfully")
    print(f"Map: {world.get_map().name}")
    print(f"Actors in world: {len(world.get_actors())}")


if __name__ == "__main__":
    main()