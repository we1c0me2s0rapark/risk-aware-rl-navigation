import numpy as np

# Category risk weights reflect both physical danger and moral consideration.
#   vehicle → baseline (heaviest, but occupants are protected)
#   walker  → highest weight (unprotected, fragile, higher moral weight)
#   cyclist → intermediate (unprotected but faster, more predictable than walkers
CATEGORY_RISK_WEIGHTS = {
    'vehicle': 1.0,
    'walker': 2.5, # higher moral weight, fragile
    'cyclist': 1.8,
}

# Agents beyond this TTC threshold contribute zero risk.
# At 10s, the exponential decay is already negligible (~0.036 at sigma=3).
TTC_MAX = 10.0 # seconds beyond which risk ≈ 0

def category_scaled_risk(ttc: float, category: str, sigma: float = 3.0, ttc_max: float = TTC_MAX, cat_weights: dict = None) -> float:
    """
    @brief Map TTC to a risk score in [0, 1], scaled by agent category.

    @details
    Uses exponential decay to convert TTC into a base risk score,
    then multiplies by a category-specific weight to reflect the
    relative danger posed by different agent types.

    The decay function is:
        base_risk = exp(-ttc / sigma)

    Which gives:
        ttc = 0s  → base_risk = 1.0  (imminent collision)
        ttc = σ   → base_risk ≈ 0.37 (one risk half-life)
        ttc = 2σ  → base_risk ≈ 0.14
        ttc ≥ ttc_max → base_risk = 0.0 (negligible risk, clamped to zero)

    The final score is clipped to [0, 1] after category scaling,
    since weights > 1.0 (e.g. walker=2.5) can push the product above 1.

    @param ttc         Time-to-collision in seconds (from compute_ttc).
    @param category    Agent category string: 'vehicle', 'walker', or 'cyclist'.
    @param sigma       Decay rate in seconds. Smaller sigma → risk falls off
                       faster with increasing TTC. Tunable via config.
    @param ttc_max     TTC beyond which risk is treated as zero.
    @param cat_weights Dict mapping category → weight.
    @return float Risk score in [0, 1].
    """

    if cat_weights is None:
        cat_weights = CATEGORY_RISK_WEIGHTS

    # -------------------------------------------------------------------------
    # Step 1: Base risk via exponential decay.
    #
    #   Converts TTC into a continuous risk score using:
    #     base_risk = exp(-ttc / sigma)
    #
    #   Intuition: the closer the collision, the higher the risk.
    #   The decay rate sigma controls how steeply risk drops off with time.
    #
    #   Clamped to 0.0 for ttc ≥ ttc_max to avoid near-zero but non-zero
    #   penalties from very distant agents polluting the reward signal.
    #
    #   Example (sigma=3.0):
    #     ttc = 0s  → exp(0)        = 1.00
    #     ttc = 3s  → exp(-1)       ≈ 0.37
    #     ttc = 6s  → exp(-2)       ≈ 0.14
    #     ttc = 10s → clamped to    = 0.00
    # -------------------------------------------------------------------------
    base_risk = np.exp(-ttc / sigma) if ttc < ttc_max else 0.0

    # -------------------------------------------------------------------------
    # Step 2: Scale by category weight.
    #
    #   Multiplies base_risk by a category-specific weight to reflect
    #   the relative danger of each agent type:
    #
    #     vehicle:  1.0 × base_risk  (baseline)
    #     cyclist:  1.8 × base_risk  (unprotected road user)
    #     walker:   2.5 × base_risk  (fragile, unpredictable, moral weight)
    #
    #   Falls back to 1.0 for unknown categories.
    # -------------------------------------------------------------------------
    weight = cat_weights.get(category, 1.0)

    # -------------------------------------------------------------------------
    # Step 3: Clip to [0, 1].
    #
    #   Category weights > 1.0 can push the product above 1.0.
    #   Clip ensures the output is always a valid normalised score.
    #
    #   Example:
    #     walker, ttc=0s → 1.0 * 2.5 = 2.5 → clipped to 1.0
    #     walker, ttc=3s → 0.37 * 2.5 = 0.92 → within range
    # -------------------------------------------------------------------------
    return float(np.clip(base_risk * weight, 0.0, 1.0))