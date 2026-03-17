import carla
import random
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

    Tries to establish a connection to CARLA at the given host and port.
    Prints debug messages indicating success or failure.

    @param host
        Host address to attempt.
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
        print(f"🚀 Connected via {host}")
        return client, world
    except RuntimeError as exc:
        print(f"💥 Connection via {host} failed: {exc}")
    except Exception as exc:
        print(f"💥 Unexpected error via {host}: {exc}")

    return None, None

def connect_carla(port: int = 2000, timeout: float = 5.0) -> Optional[carla.World]:
    """
    @brief Establish a connection to the CARLA simulator.

    The function first attempts to connect to a valid Windows host IP 
    (if available in WSL2), then falls back to localhost. If the 
    connection succeeds, a carla.World object is returned. If all 
    attempts fail, an error message is printed and None is returned.

    @param port
        TCP port used by the CARLA simulator (default 2000).
    @param timeout
        Connection timeout in seconds (default 5.0).

    @return Optional[carla.World]
        The CARLA world object if the connection succeeds, otherwise None.
    """

    hosts: List[str] = ["localhost"]

    windows_host_ip = get_windows_host_ip()
    if windows_host_ip and windows_host_ip not in hosts:
        # hosts.append(windows_host_ip)
        hosts.insert(0, windows_host_ip) # insert at the start to try it first

    for host in hosts:
        client, world = attempt_connection(host, port, timeout)
        if world: return client, world

    print(f"Could not connect to CARLA on hosts: {hosts}")
    return None, None

def configure_simulation(client: carla.Client, world: carla.World, sync_mode: bool = True, dt: float = 0.05, seed: Optional[int] = None) -> None:
    """
    @brief Configure the CARLA simulation settings.

    @param client The carla.Client instance.
    @param world The carla.World instance to configure.
    @param sync_mode Whether to enable synchronous mode.
    @param dt The fixed delta seconds for the simulation step.
    """
    settings = world.get_settings()
    settings.synchronous_mode = sync_mode
    settings.fixed_delta_seconds = dt if sync_mode else None
    world.apply_settings(settings)
    
    # If enabling sync mode, we must also configure the Traffic Manager
    # to be synchronous to prevent vehicle jitters.
    if not sync_mode: return

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    if seed is not None: random_seed = seed
    else: random_seed = random.randint(0, 999999)
    traffic_manager.set_random_device_seed(random_seed)