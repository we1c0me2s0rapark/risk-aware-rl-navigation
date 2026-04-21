import numpy as np

# Feature slot width - must match AGENT_FEAT_DIM in risk/module.py.
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

def navigation_reward(reward_config: dict, ego: dict, goal_progress: float, wp_dx: float = 0.0, wp_dy: float = 0.0) -> float:
    """
    @brief Compute the navigation reward component.

    @details
    Rewards forward progress, heading alignment with the next waypoint,
    and maintaining a safe forward speed. Applies a small time penalty
    to discourage idling.

    @param reward_config dict Reward scaling constants.
    @param ego dict Ego vehicle state.
    @param goal_progress float Distance closed towards next waypoint (metres).
    @param wp_dx float Normalised x-component of waypoint direction.
    @param wp_dy float Normalised y-component of waypoint direction.
    @return float Navigation reward.
    """
    heading_vec = np.array([np.cos(ego['yaw']), np.sin(ego['yaw'])])
    wp_vec = np.array([wp_dx, wp_dy])
    alignment = float(np.dot(heading_vec, wp_vec))
    speed = np.sqrt(ego['vx']**2 + ego['vy']**2)

    r  = goal_progress * reward_config['goal_progress_scale']
    r += alignment * reward_config['heading_alignment_scale']
    r += min(speed, reward_config['speed_cap']) * max(alignment, 0.0) * reward_config['speed_reward_scale']
    r -= reward_config['time_penalty']
    return r


def safety_reward(reward_config: dict, collision: bool, wrong_way_risk: float) -> float:
    """
    @brief Compute the safety reward component.

    @details
    Penalises collisions and wrong-way driving. Applied continuously
    each step to provide a persistent corrective signal.

    @param reward_config dict Reward scaling constants.
    @param collision bool True if a collision was detected this step.
    @param wrong_way_risk float Wrong-way risk score in [0, 1].
    @return float Safety reward (non-positive).
    """
    r = 0.0
    if collision:
        r -= reward_config['collision_penalty']
    r -= wrong_way_risk * reward_config['wrong_way_penalty']
    return r


def risk_reward(reward_config: dict, risk_features: np.ndarray, top_k: int) -> float:
    """
    @brief Compute the risk-awareness reward component.

    @details
    Penalises proximity to nearby agents with imminent time-to-collision.
    Only agents with TTC below the configured threshold are penalised.

    @param reward_config dict Reward scaling constants.
    @param risk_features np.ndarray Flat risk vector of shape (top_k * _FEAT_DIM,).
    @param top_k int Number of agent slots in the risk vector.
    @return float Risk penalty (non-positive).
    """
    r = 0.0
    max_agents = risk_features.size // _FEAT_DIM
    for i in range(min(top_k, max_agents)):
        ttc = risk_features[i * _FEAT_DIM + _TTC_IDX]
        risk_score = risk_features[i * _FEAT_DIM + _RISK_IDX]
        if ttc < reward_config['ttc_threshold']:
            r -= reward_config['ttc_penalty_scale'] * float(risk_score)
    return r


def decomposed_reward(
    reward_config: dict,
    ego: dict,
    collision: bool,
    goal_progress: float,
    risk_features: np.ndarray,
    top_k: int,
    wp_dx: float = 0.0,
    wp_dy: float = 0.0,
    wrong_way_risk: float = 0.0,
) -> np.ndarray:
    """
    @brief Compute all three reward components independently.

    @details
    Returns a decomposed reward vector for use with multi-head PPO.
    Each component is computed independently so the critic can learn
    a separate value function per objective.

    Components:
        [0] navigation : progress, alignment, speed
        [1] safety     : collision, wrong-way
        [2] risk       : TTC-based proximity penalty

    @param reward_config dict Reward scaling constants.
    @param ego dict Ego vehicle state.
    @param collision bool True if a collision was detected.
    @param goal_progress float Distance closed towards next waypoint.
    @param risk_features np.ndarray Flat risk vector.
    @param top_k int Number of agent slots in the risk vector.
    @param wp_dx float Normalised x-component of waypoint direction.
    @param wp_dy float Normalised y-component of waypoint direction.
    @param wrong_way_risk float Wrong-way risk score in [0, 1].
    @return np.ndarray of shape (3,) - [r_nav, r_safe, r_risk].
    """
    r_nav  = navigation_reward(reward_config, ego, goal_progress, wp_dx, wp_dy)
    r_safe = safety_reward(reward_config, collision, wrong_way_risk)
    r_risk = risk_reward(reward_config, risk_features, top_k)
    return np.array([r_nav, r_safe, r_risk], dtype=np.float32)


def baseline_reward(
    reward_config: dict,
    ego: dict,
    collision: bool,
    goal_progress: float,
    top_k: int = 5,
    wp_dx: float = 0.0,
    wp_dy: float = 0.0,
    wrong_way_risk: float = 0.0,
) -> float:
    """
    @brief Compute a baseline reward ignoring risk features.

    @details
    Useful for standard PPO training when risk-awareness is not considered.
    
    @param reward_config dict Reward scaling constants.
    @param ego dict Ego vehicle state.
    @param collision bool True if a collision was detected.
    @param goal_progress float Distance closed towards next waypoint.
    @param top_k int Number of agent slots in the risk vector.
    @param wp_dx float Normalised x-component of waypoint direction.
    @param wp_dy float Normalised y-component of waypoint direction.
    @param wrong_way_risk float Wrong-way risk score in [0, 1].
    @return float Scalar reward.
    """
    return decomposed_reward(
        reward_config, ego, collision, goal_progress,
        np.zeros(top_k * _FEAT_DIM), # no risk features for baseline
        top_k, wp_dx, wp_dy, wrong_way_risk
    )