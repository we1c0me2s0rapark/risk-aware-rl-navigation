import numpy as np

def extract_relative_velocity_features(ego: dict, agent: dict) -> np.ndarray:
    """
    @brief Extract relative velocity features between ego and a single agent.

    @details
    Decomposes the agent's velocity relative to ego into longitudinal
    and lateral components aligned with ego's heading, plus a heading
    difference encoded as (cos, sin) to preserve angular continuity.

    @param ego   Dict with keys: vx, vy, yaw.
    @param agent Dict with keys: vx, vy, yaw.
    @return np.ndarray of shape (5,):
        [rel_speed, closing_speed, lateral_speed, cos(heading_diff), sin(heading_diff)]
    """

    # -------------------------------------------------------------------------
    # Step 1: Relative velocity vector of agent w.r.t. ego (m/s).
    #
    #   This is the velocity of the agent as seen from ego's reference frame.
    #   If both move identically, rel_v = (0, 0) → no relative motion.
    #
    #   Example:
    #     ego:   vx=5, agent: vx=3  →  rel_vx = -2 (agent moving away)
    #     ego:   vx=3, agent: vx=5  →  rel_vx =  2 (agent approaching)
    # -------------------------------------------------------------------------
    rel_vx = agent['vx'] - ego['vx']
    rel_vy = agent['vy'] - ego['vy']

    # Magnitude of the relative velocity vector (scalar speed, always ≥ 0).
    #   rel_speed = √(rel_vx² + rel_vy²)
    rel_speed = np.hypot(rel_vx, rel_vy)

    # -------------------------------------------------------------------------
    # Step 2: Ego's local coordinate axes.
    #
    #   ego_heading: unit vector pointing in the direction ego is facing.
    #     ego_heading = (cos(yaw), sin(yaw))
    #
    #   ego_lateral: unit vector pointing 90° left of ego's heading.
    #     ego_lateral = (-sin(yaw), cos(yaw))
    #
    #   These two axes form ego's local frame, used to decompose rel_v
    #   into meaningful longitudinal and lateral components below.
    #
    #                    ego_heading →
    #                         ↑
    #   ego_lateral ←  [ ego ]
    # -------------------------------------------------------------------------
    ego_heading = np.array([np.cos(ego['yaw']), np.sin(ego['yaw'])])
    ego_lateral = np.array([-ego_heading[1], ego_heading[0]])

    rel_v = np.array([rel_vx, rel_vy])

    # -------------------------------------------------------------------------
    # Step 3: Project relative velocity onto ego's longitudinal axis.
    #
    #   closing_speed = rel_v · ego_heading
    #
    #   Measures how fast the agent is moving along ego's forward direction.
    #   Negative → agent moving in the same direction as ego (approaching
    #   from behind or being overtaken).
    #   Positive → agent moving against ego's heading (head-on closing).
    #
    #   Note: sign convention here is opposite to compute_ttc, where
    #   closing_speed > 0 means approaching. Here the raw dot product
    #   is preserved so the policy can reason about direction.
    # -------------------------------------------------------------------------
    closing_speed = np.dot(rel_v, ego_heading) # negative = approaching

    # -------------------------------------------------------------------------
    # Step 4: Project relative velocity onto ego's lateral axis.
    #
    #   lateral_speed = rel_v · ego_lateral
    #
    #   Measures how fast the agent is moving sideways relative to ego.
    #   Positive → agent drifting to ego's left.
    #   Negative → agent drifting to ego's right.
    # -------------------------------------------------------------------------
    lateral_speed = np.dot(rel_v, ego_lateral)

    # -------------------------------------------------------------------------
    # Step 5: Heading difference encoded as (cos, sin).
    #
    #   heading_diff = agent.yaw - ego.yaw
    #
    #   Encoded as (cos(heading_diff), sin(heading_diff)) rather than the
    #   raw angle to avoid the discontinuity at ±π (i.e. 179° and -179°
    #   are nearly identical headings but numerically far apart as scalars).
    #
    #   Examples:
    #     heading_diff = 0    → (1,  0)  same direction
    #     heading_diff = π/2  → (0,  1)  agent facing 90° left of ego
    #     heading_diff = π    → (-1, 0)  agent facing opposite direction
    # -------------------------------------------------------------------------
    heading_diff = agent['yaw'] - ego['yaw']

    return np.array([
        rel_speed,              # index 0: scalar relative speed (m/s)
        closing_speed,          # index 1: longitudinal closing component
        lateral_speed,          # index 2: lateral drift component
        np.cos(heading_diff),   # index 3: heading alignment (1 = same, -1 = opposite)
        np.sin(heading_diff),   # index 4: heading rotation direction
    ])