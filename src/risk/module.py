import numpy as np
from dataclasses import dataclass

from risk.ttc import compute_ttc
from risk.features import extract_relative_velocity_features
from risk.filtering import filter_nearest_agents, CATEGORY_MAP
from risk.scaling import category_scaled_risk


@dataclass(frozen=True)
class AgentFeatureLayout:
    """
    @brief Defines the per-agent feature slot layout used in the risk vector.

    @details
    Each nearby agent is described by a fixed-size block of AGENT_FEAT_DIM values.
    The layout is fixed at construction time and immutable (frozen=True), ensuring
    the observation space dimension declared in the gym env always matches the
    vector actually produced by RiskModule.compute().

    Slot layout:
        [ dist | cat_onehot | ttc | risk | rel_v ]
           (1)      (3)       (1)    (1)    (5)     = 11 features per agent

    Index reference within one slot:
        0        : Euclidean distance from ego (metres)
        1, 2, 3  : One-hot category [vehicle, walker, cyclist]
        4        : Time-to-collision in seconds (capped at ttc_max)
        5        : Category-scaled risk score in [0, 1]
        6        : Relative speed (m/s)
        7        : Closing speed along ego heading (positive = approaching)
        8        : Lateral speed across ego heading
        9        : cos(heading difference)
        10       : sin(heading difference)
    """
    dist_dim        : int = 1  # Euclidean distance to agent (metres)
    cat_dim         : int = 3  # one-hot category: vehicle / walker / cyclist
    ttc_dim         : int = 1  # time-to-collision (seconds)
    risk_dim        : int = 1  # category-scaled risk score in [0, 1]
    rel_v_dim       : int = 5  # relative velocity features: speed, closing, lateral, cos(Δyaw), sin(Δyaw)

    @property
    def total(self) -> int:
        """@brief Total number of features per agent slot."""
        return self.dist_dim + self.cat_dim + self.ttc_dim + self.risk_dim + self.rel_v_dim


# Module-level constants derived from the layout.
# All other modules should import AGENT_FEAT_DIM from here rather than
# hardcoding 11, so any future layout change propagates automatically.
AGENT_FEAT_LAYOUT = AgentFeatureLayout()
AGENT_FEAT_DIM    = AGENT_FEAT_LAYOUT.total


class RiskModule:
    """
    @brief Coordinator for all risk estimation logic.

    @details
    Owns the risk configuration (top_k, radius, sigma, ttc_max, category weights)
    and exposes a single compute() entry point used by the gym environment.
    Internally delegates to the pure functions in risk.ttc, risk.features,
    risk.filtering, and risk.scaling, which remain independently unit-testable.
    """

    def __init__(self, config: dict):
        """
        @brief Initialise RiskModule from the simulation config.

        @details
        All parameters are read from the 'risk' section of config.yaml.
        Sensible defaults are provided so the module works even if the
        risk section is absent from the config.

        @param config Full simulation config dict (from load_config()).
        """
        risk_cfg = config.get('risk', {})

        # Number of agent slots in the output risk vector.
        # Determines the observation space dimension alongside AGENT_FEAT_DIM.
        self.top_k = risk_cfg.get('top_k', 5)

        # Maximum distance (metres) beyond which agents are ignored.
        # Should match the LiDAR range in the sensor config.
        self.radius = risk_cfg.get('search_radius', 50.0)

        # Decay rate for the TTC → risk exponential: risk = exp(-ttc / sigma).
        # Smaller sigma → risk falls off more steeply with increasing TTC.
        self.sigma = risk_cfg.get('ttc_sigma', 3.0)

        # TTC beyond which risk is clamped to zero (negligible threat).
        self.ttc_max = risk_cfg.get('ttc_max', 10.0)

        # Per-category multipliers applied to the base risk score.
        # Reflects relative danger: walkers are unprotected and fragile.
        self.cat_weights = risk_cfg.get('category_weights', {
            'vehicle': 1.0,
            'walker':  2.5,
            'cyclist': 1.8,
        })

    # def compute(self, ego: dict, agents: list[dict]) -> np.ndarray:
    #     """
    #     @brief Build a fixed-size risk feature vector from nearby agents.

    #     @details
    #     Pipeline per agent slot:
    #         1. Filter agents within self.radius, keep closest self.top_k.
    #         2. For each agent, compute:
    #             - Euclidean distance from ego.
    #             - One-hot category vector.
    #             - Time-to-collision (compute_ttc).
    #             - Category-scaled risk score (category_scaled_risk).
    #             - Relative velocity features (extract_relative_velocity_features).
    #         3. Concatenate into a single slot of AGENT_FEAT_DIM values.
    #         4. Pad with zero slots if fewer than top_k agents are present.

    #     @param ego    Dict with keys: x, y, vx, vy, yaw, length, width.
    #     @param agents List of agent dicts with same keys plus 'category'.
    #     @return np.ndarray of shape (top_k * AGENT_FEAT_DIM,).
    #     """

    #     # Step 1: Filter to the top_k nearest agents within self.radius.
    #     nearest = filter_nearest_agents(
    #         ego, agents,
    #         top_k=self.top_k,
    #         max_radius=self.radius,
    #     )

    #     slots = []
    #     for agent in nearest:

    #         # Step 2a: Euclidean distance (metres).
    #         dist = np.hypot(agent['x'] - ego['x'], agent['y'] - ego['y'])
    #         dist_norm = dist / self.radius # [0, 1]

    #         # Step 2b: One-hot category vector [vehicle, walker, cyclist].
    #         # Unknown categories default to vehicle (index 0).
    #         cat = np.zeros(AGENT_FEAT_LAYOUT.cat_dim)
    #         cat[CATEGORY_MAP.get(agent.get('category', 'vehicle'), 0)] = 1.0

    #         # Step 2c: Time-to-collision in seconds.
    #         ttc = compute_ttc(ego, agent)

    #         # Step 2d: Category-scaled risk score in [0, 1].
    #         risk = category_scaled_risk(
    #             ttc,
    #             agent.get('category', 'vehicle'),
    #             sigma=self.sigma,
    #             ttc_max=self.ttc_max,
    #             cat_weights=self.cat_weights,
    #         )

    #         # Step 2e: Relative velocity features (5-dimensional).
    #         rel_v = extract_relative_velocity_features(ego, agent)

    #         # Step 3: Concatenate into one agent slot of AGENT_FEAT_DIM values.
    #         slots.append(np.concatenate([[dist], cat, [ttc], [risk], rel_v]))

    #     # Step 4: Pad with zero slots for absent agents.
    #     # Ensures the output shape is always (top_k * AGENT_FEAT_DIM,)
    #     # regardless of how many agents are actually in range.
    #     while len(slots) < self.top_k:
    #         slots.append(np.zeros(AGENT_FEAT_DIM))

    #     return np.concatenate(slots[:self.top_k])

    def compute(self, ego: dict, agents: list[dict], env_risk: float = 0.0) -> np.ndarray:
        """
        @brief Build a fixed-size risk feature vector.
        @details Returns (top_k * 11) + 6 = 61 dimensions total.
        """
        nearest = filter_nearest_agents(ego, agents, top_k=self.top_k, max_radius=self.radius)

        slots = []
        for agent in nearest:
            # Standard 11 features (dist, cat, ttc, risk, rel_v)
            dist = np.hypot(agent['x'] - ego['x'], agent['y'] - ego['y']) / self.radius
            cat = np.zeros(3)
            cat[CATEGORY_MAP.get(agent.get('category', 'vehicle'), 0)] = 1.0
            ttc = compute_ttc(ego, agent)
            risk = category_scaled_risk(ttc, agent.get('category', 'vehicle'))
            rel_v = extract_relative_velocity_features(ego, agent)

            slots.append(np.concatenate([[dist], cat, [ttc], [risk], rel_v]))

        # Pad with zeros (width 11)
        while len(slots) < self.top_k:
            slots.append(np.zeros(11))

        # Flatten agents: 5 agents * 11 = 55 features
        agent_vec = np.concatenate(slots[:self.top_k])

        # Add 6 global features to reach 61
        # [env_risk, ego_v, ego_steer, ... padding]
        global_vec = np.zeros(6)
        global_vec[0] = env_risk
        
        return np.concatenate([agent_vec, global_vec])

    @property
    def feature_dim(self) -> int:
        """
        @brief Total dimension of the risk vector produced by compute().
        @return int top_k * AGENT_FEAT_DIM.
        """
        return self.top_k * AGENT_FEAT_DIM + 6