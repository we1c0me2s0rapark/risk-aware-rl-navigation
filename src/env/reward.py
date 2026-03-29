import numpy as np

# Feature slot width — must match AGENT_FEAT_DIM in risk/module.py.
#
# Layout per agent:
#   [ dist | category_one_hot | ttc | risk | relative_velocity ]
#     (1)         (3)           (1)   (1)          (5)          = 11
#
# @note
# Indices below refer to positions within a single agent feature block.

_FEAT_DIM = 11
_TTC_IDX  = 4       # @brief Index of time-to-collision (TTC) within one agent feature block.
_RISK_IDX = 5       # @brief Index of risk score within one agent feature block.

def baseline_reward(
    reward_config: dict,
    ego: dict,
    collision: bool,
    goal_progress: float,
    wp_dx: float = 0.0,
    wp_dy: float = 0.0,
    wrong_way_risk: float = 0.0,
) -> float:
    """
    Compute the baseline reward signal.
    
    Components:
        - Progress towards the goal (metres)
        - Heading alignment with waypoint vector
        - Wrong-way penalty (scaled by reward_config['wrong_way_penalty'])
        - Forward speed reward (capped, scaled)
        - Collision penalty (reward_config['collision_penalty'])
        - Small per-step time penalty (reward_config['time_penalty'])

    Args:
        reward_config: dict containing all reward scaling constants
        ego: dict with ego vehicle state ('vx', 'vy', 'yaw')
        collision: bool, True if collision occurred
        goal_progress: float, distance progressed toward goal (metres)
        wp_dx: float, x-component of normalized waypoint direction
        wp_dy: float, y-component of normalized waypoint direction
        wrong_way_risk: float, in [0,1], 1 = fully against lane

    Returns:
        Baseline reward (float)
    """

    # Progress reward
    r = goal_progress * reward_config['goal_progress_scale']

    # Heading alignment
    heading_vec = np.array([np.cos(ego['yaw']), np.sin(ego['yaw'])])
    wp_vec = np.array([wp_dx, wp_dy])
    alignment = float(np.dot(heading_vec, wp_vec))
    r += alignment * reward_config['heading_alignment_scale']

    # -------------------------------------------------------------------------
    # Wrong-way penalty.
    #
    #   Applies a continuous penalty for travelling against the lane direction.
    #   The value is scaled by wrong_way_risk ∈ [0, 1], where:
    #       0 → correct direction
    #       1 → fully reversed
    #
    #   This provides a persistent corrective signal rather than a single
    #   terminal penalty.
    # -------------------------------------------------------------------------
    r -= wrong_way_risk * reward_config['wrong_way_penalty']

    # -------------------------------------------------------------------------
    # Speed reward (m/s).
    #
    #   Provides a small incentive for maintaining forward motion.
    #   The reward is capped to discourage excessive or unsafe speeds.
    # -------------------------------------------------------------------------
    speed = np.sqrt(ego['vx']**2 + ego['vy']**2)
    r += min(speed, reward_config['speed_cap']) * reward_config['speed_reward_scale']

    # Collision penalty
    if collision: r -= reward_config['collision_penalty']

    # Time penalty per step
    r -= reward_config['time_penalty']

    return r


def risk_aware_reward(
    reward_config: dict,
    ego: dict,
    collision: bool,
    goal_progress: float,
    risk_features: np.ndarray,
    top_k: int = 5,
    wp_dx: float = 0.0,
    wp_dy: float = 0.0,
    wrong_way_risk: float = 0.0,
) -> float:
    """
    Compute risk-aware reward signal.

    Extends baseline reward by penalizing risky interactions with nearby agents:
        - For each of the top-k nearest agents:
            - If TTC < reward_config['ttc_threshold'], apply penalty
            - Penalty = reward_config['ttc_penalty_scale'] * agent risk score

    Args:
        reward_config: dict of reward constants
        ego: dict with ego state
        collision: bool, True if collision occurred
        goal_progress: float, distance progressed toward goal
        risk_features: flattened np.ndarray of shape [num_agents * _FEAT_DIM]
        top_k: int, number of nearest agents to consider
        wp_dx: float, waypoint x-component
        wp_dy: float, waypoint y-component
        wrong_way_risk: float, in [0,1], 1 = fully against lane

    Returns:
        Risk-aware reward (float)
    """
    
    r = baseline_reward(reward_config, ego, collision, goal_progress, wp_dx, wp_dy, wrong_way_risk)

    max_agents = risk_features.size // _FEAT_DIM
    for i in range(min(top_k, max_agents)):
        ttc = risk_features[i * _FEAT_DIM + _TTC_IDX]
        risk_score = risk_features[i * _FEAT_DIM + _RISK_IDX]

        if ttc < reward_config['ttc_threshold']:
            r -= reward_config['ttc_penalty_scale'] * float(risk_score)

    return r