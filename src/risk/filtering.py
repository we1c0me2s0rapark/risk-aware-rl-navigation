import numpy as np

# Maps agent category string to one-hot index used in the feature vector.
#   vehicle → index 0
#   walker  → index 1
#   cyclist → index 2
CATEGORY_MAP = {'vehicle': 0, 'walker': 1, 'cyclist': 2}

def filter_nearest_agents(ego: dict, agents: list[dict], top_k: int = 5, max_radius: float = 50.0, per_category: bool = False) -> list[dict]:
    """
    @brief Filter and rank nearby agents by Euclidean distance from ego.

    @details
    First discards all agents beyond max_radius, then returns the closest
    top_k agents. Optionally returns top_k per category independently,
    which ensures the policy always sees representatives of each agent type
    even if one category dominates the local scene.

    @param ego         Dict with keys: x, y (ego position).
    @param agents      List of agent dicts with keys: x, y, vx, vy, yaw,
                       category, length, width.
    @param top_k       Maximum number of agents to return (per category if
                       per_category=True, otherwise globally).
    @param max_radius  Search radius in metres. Agents beyond this are ignored.
    @param per_category If True, returns top_k per category (up to top_k * 3
                        agents total). If False, returns top_k globally.
    @return Sorted list of agent dicts, closest first.
    """

    # -------------------------------------------------------------------------
    # Step 1: Distance helper.
    #
    #   Computes the Euclidean distance from ego to a given agent.
    #   Used both for radius filtering and for sorting.
    #
    #   distance = √((a.x - ego.x)² + (a.y - ego.y)²)
    # -------------------------------------------------------------------------
    def dist(a):
        return np.hypot(a['x'] - ego['x'], a['y'] - ego['y'])

    # -------------------------------------------------------------------------
    # Step 2: Radius filter.
    #
    #   Discard all agents beyond max_radius to limit the search space.
    #   This mirrors the LiDAR range in the sensor config (default 50m),
    #   so the risk module never reasons about agents it cannot perceive.
    # -------------------------------------------------------------------------
    in_range = [a for a in agents if dist(a) < max_radius]

    # -------------------------------------------------------------------------
    # Step 3a: Global top-k (per_category=False, default).
    #
    #   Sort all in-range agents by distance and return the closest top_k.
    #   This is the standard mode used by the risk feature vector.
    #
    #   Example (top_k=3):
    #     distances: [2m, 5m, 8m, 12m, 30m] → returns [2m, 5m, 8m]
    # -------------------------------------------------------------------------
    if not per_category:
        return sorted(in_range, key=dist)[:top_k]

    # -------------------------------------------------------------------------
    # Step 3b: Per-category top-k (per_category=True).
    #
    #   For each category (vehicle, walker, cyclist), independently select
    #   the closest top_k agents and combine the results.
    #
    #   This prevents a dense cluster of vehicles from crowding out all
    #   walker and cyclist slots, which is important in urban scenes where
    #   pedestrian risk is disproportionately high relative to their count.
    #
    #   Example (top_k=2):
    #     vehicles: [3m, 7m, 15m] → [3m, 7m]
    #     walkers:  [6m, 20m]     → [6m, 20m]
    #     cyclists: []            → []
    #     result:   [3m, 7m, 6m, 20m]  (up to top_k * 3 = 6 agents)
    # -------------------------------------------------------------------------
    result = []
    for cat in CATEGORY_MAP:
        subset = [a for a in in_range if a.get('category') == cat]
        result.extend(sorted(subset, key=dist)[:top_k])

    return result