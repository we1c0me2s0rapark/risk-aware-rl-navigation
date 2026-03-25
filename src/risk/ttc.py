import numpy as np

def compute_ttc(ego_state: dict, agent_state: dict, min_ttc: float = 0.01) -> float:
    """
    @brief Compute Time-To-Collision (TTC) between ego and a single agent.

    @details
    Assumes constant velocities and point-projected 1D closing dynamics
    along the line connecting the two agents. Uses bounding box extents
    to compute the actual edge-to-edge gap rather than centre-to-centre.

    @param ego_state   Dict with keys: x, y, vx, vy, length.
    @param agent_state Dict with keys: x, y, vx, vy, length (optional).
    @param min_ttc     Floor value returned when agents are already overlapping.
    @return TTC in seconds. Returns inf if diverging, min_ttc if overlapping.
    """

    # -------------------------------------------------------------------------
    # Step 1: Displacement vector from ego centre to agent centre (metres).
    #
    #   dx, dy form the direction vector used both for distance
    #   and for the closing speed projection in Step 3.
    # -------------------------------------------------------------------------
    dx = agent_state['x'] - ego_state['x']
    dy = agent_state['y'] - ego_state['y']

    # Euclidean distance between the two centres; distance = √(dx² + dy²)
    distance = np.sqrt(dx**2 + dy**2)

    # -------------------------------------------------------------------------
    # Step 2: Relative velocity of ego w.r.t. agent (m/s).
    #
    #   This is the velocity of ego as seen from the agent's reference frame.
    #   If both move identically, rel_v = 0 → no closing motion.
    #
    #   Example:
    #     ego:   vx=5, agent: vx=3  →  rel_vx = 2  (ego closing at 2 m/s)
    #     ego:   vx=3, agent: vx=5  →  rel_vx = -2 (ego retreating)
    # -------------------------------------------------------------------------
    rel_vx = ego_state['vx'] - agent_state['vx']
    rel_vy = ego_state['vy'] - agent_state['vy']

    # -------------------------------------------------------------------------
    # Step 3: Guard - agents are already coincident (distance ≈ 0).
    #
    #   Avoids division by zero in the projection step below.
    #   Returns min_ttc to signal imminent or already-occurring collision.
    # -------------------------------------------------------------------------
    if distance < 1e-3:
        return min_ttc

    # -------------------------------------------------------------------------
    # Step 4: Project relative velocity onto the ego→agent unit vector.
    #
    #   The full relative velocity may have a lateral component (sideways).
    #   Only the longitudinal component - along the line joining the two centres -
    #   contributes to closing. We extract it via dot product:
    #
    #     unit vector d̂ = (dx, dy) / distance
    #
    #     closing_speed = rel_v · d̂
    #                   = (rel_vx * dx + rel_vy * dy) / distance
    #
    #   Positive → agents approaching one another.
    #   Zero or negative → parallel or diverging.
    # -------------------------------------------------------------------------
    closing_speed = (rel_vx * dx + rel_vy * dy) / distance

    # -------------------------------------------------------------------------
    # Step 5: Guard - agents are diverging or moving in parallel.
    #
    #   If closing_speed ≤ 0 there is no future collision under constant
    #   velocity assumptions. Returns inf to signal no risk.
    # -------------------------------------------------------------------------
    if closing_speed <= 0: # diverging or parallel → no collision
        return float('inf')

    # -------------------------------------------------------------------------
    # Step 6: Edge-to-edge safety gap (metres).
    #
    #   Raw distance is centre-to-centre. Collision occurs when edges meet,
    #   so we subtract half the length of each agent:
    #
    #   safety_gap = distance - (ego.length/2 + agent.length/2)
    #
    #   Falls back to 1.0m if agent has no length key (e.g. walkers).
    # -------------------------------------------------------------------------
    safety_gap = distance - 0.5 * (ego_state['length'] + agent_state.get('length', 1.0))

    # -------------------------------------------------------------------------
    # Step 7: TTC = safety_gap / closing_speed
    #
    #   Time until the edges of the two bounding boxes meet, assuming
    #   both agents maintain their current velocities.
    #
    #   Example:
    #     safety_gap = 10m, closing_speed = 5 m/s → TTC = 2.0s
    #
    #   Floored at min_ttc to avoid returning zero or negative values
    #   when agents are already partially overlapping (safety_gap < 0).
    # -------------------------------------------------------------------------
    return max(min_ttc, safety_gap / closing_speed)