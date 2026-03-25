import numpy as np

# Feature slot width — must match AGENT_FEAT_DIM in risk/module.py.
#   [ dist | cat_onehot | ttc | risk | rel_v ]
#     (1)      (3)       (1)   (1)    (5)     = 11
_FEAT_DIM = 11
_TTC_IDX  = 4   # index of TTC within one agent slot
_RISK_IDX = 5   # index of risk score within one agent slot

def baseline_reward(ego: dict, collision: bool, goal_progress: float) -> float:
    """
    @brief Compute the baseline reward for the current timestep.

    @details
    A simple reward signal that encourages goal-directed progress
    whilst penalising collisions and discouraging lingering.

    Components:
        + goal_progress   : metres closed toward the destination this step.
                            Positive if approaching, negative if retreating.
        - 200.0           : large one-off collision penalty.
        - 0.01            : small time penalty per step to discourage
                            the agent from idling or taking unnecessarily
                            long routes.

    @param ego           Dict with ego state (unused here, reserved for
                         future shaping terms such as speed or heading error).
    @param collision     True if a collision was recorded this step.
    @param goal_progress Distance closed toward goal this step (metres).
    @return float Baseline reward value.
    """

    # -------------------------------------------------------------------------
    # Progress reward.
    #
    #   Directly rewards closing the distance to the destination.
    #   Scale factor 1.0 keeps units in metres per step, which is
    #   interpretable and easy to tune against the penalty terms.
    # -------------------------------------------------------------------------
    r = goal_progress * 1.0

    # -------------------------------------------------------------------------
    # Collision penalty.
    #
    #   Large negative reward to strongly discourage any collision.
    #   Applied once per step whilst the collision flag is set — the
    #   episode typically terminates immediately after, so this fires once.
    # -------------------------------------------------------------------------
    if collision:
        r -= 200.0

    # -------------------------------------------------------------------------
    # Time penalty.
    #
    #   Small per-step cost to discourage the agent from idling or taking
    #   unnecessarily long paths to the goal. Acts as a soft pressure
    #   toward efficiency without overwhelming the progress signal.
    # -------------------------------------------------------------------------
    r -= 0.01

    return r

def risk_aware_reward(ego: dict, collision: bool, goal_progress: float, risk_features: np.ndarray, top_k: int = 5) -> float:
    """
    @brief Compute the risk-aware reward for the current timestep.

    @details
    Extends the baseline reward with a per-agent TTC penalty that
    discourages the agent from entering high-risk proximity to nearby
    agents. Only agents with imminent TTC (< 3s) incur a penalty,
    avoiding noise from distant, low-risk agents.

    Risk penalty per agent:
        penalty = 5.0 × risk_score   (if ttc < 3.0s, else 0)

    Where risk_score ∈ [0, 1] is the category-scaled exponential
    decay score from category_scaled_risk().

    @param ego           Dict with ego state.
    @param collision     True if a collision was recorded this step.
    @param goal_progress Distance closed toward goal this step (metres).
    @param risk_features Flat risk vector of shape (top_k * _FEAT_DIM,)
                         produced by RiskModule.compute().
    @param top_k         Number of agent slots in the risk vector.
    @return float Risk-aware reward value.
    """

    # Start from the baseline reward — progress, collision, time penalty.
    r = baseline_reward(ego, collision, goal_progress)

    # -------------------------------------------------------------------------
    # Per-agent TTC penalty.
    #
    #   The risk vector is a flat array of top_k agent slots, each of
    #   width _FEAT_DIM. Within each slot:
    #     index 4 (_TTC_IDX)  : time-to-collision in seconds
    #     index 5 (_RISK_IDX) : category-scaled risk score in [0, 1]
    #
    #   Layout for slot i:
    #     risk_features[i * _FEAT_DIM + _TTC_IDX]   → ttc
    #     risk_features[i * _FEAT_DIM + _RISK_IDX]  → risk_score
    #
    #   Only penalise agents with ttc < 3.0s (imminent risk threshold).
    #   Agents beyond this are ignored to avoid penalising safe, distant
    #   interactions and to keep the gradient signal focused.
    #
    #   Penalty scale 5.0 is tunable — it controls how strongly the agent
    #   is pushed away from risky proximity relative to the progress reward.
    #
    #   Example:
    #     walker at ttc=1.5s, risk_score=0.92 → penalty = 5.0 × 0.92 = 4.6
    #     vehicle at ttc=4.0s, risk_score=0.26 → no penalty (ttc ≥ 3.0s)
    # -------------------------------------------------------------------------

    # Per-agent TTC penalty: penalise being close to any high-risk agent
    for i in range(top_k):
        ttc = risk_features[i * _FEAT_DIM + _TTC_IDX]
        risk_score = risk_features[i * _FEAT_DIM + _RISK_IDX]

        if ttc < 3.0: # only penalise imminent risk
            r -= 5.0 * float(risk_score) # scaled penalty

    return r