import torch
import numpy as np

class RunningNormaliser:
    """
    @class RunningNormaliser
    @brief Online running mean and variance normaliser for observation features.

    @details
    Implements Welford's online algorithm for computing running mean and
    variance without storing all past values. Used to normalise ego state
    and risk features before they enter the encoder MLP.

    Normalisation stabilises training by keeping input magnitudes consistent
    regardless of world coordinates (e.g. ego x/y in metres can vary widely).

    Usage:
        normaliser = RunningNormaliser(shape=(ego_state_dim,))

        # During rollout - update stats and normalise:
        normaliser.update(ego_state_np)
        normalised = normaliser.normalise(ego_state_tensor)

        # During evaluation - normalise only (no update):
        normalised = normaliser.normalise(ego_state_tensor)

    @note Call update() only during training, not during evaluation,
          to prevent the running stats from drifting on test data.
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-8, clip: float = 10.0):
        """
        @brief Initialise the running normaliser.

        @param shape tuple Shape of the feature vector to normalise (e.g. (16,) for ego_state).
        @param epsilon float Small constant for numerical stability in std division.
        @param clip float Clip normalised values to [-clip, clip] to prevent outliers.
        """
        self.shape = shape
        self.epsilon = epsilon
        self.clip = clip

        # Welford's algorithm state
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64) # sum of squared deviations

    def update(self, x: np.ndarray):
        """
        @brief Update running statistics with a new observation.

        @details
        Uses Welford's online algorithm:
            delta  = x - mean
            mean  += delta / count
            delta2 = x - mean
            M2    += delta * delta2

        @param x np.ndarray New observation of shape matching self.shape.
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        """@brief Running variance estimate."""
        if self.count < 2:
            return np.ones(self.shape, dtype=np.float64)
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> np.ndarray:
        """@brief Running standard deviation estimate."""
        return np.sqrt(self.var)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Normalise a tensor using the running mean and std.

        @details
        z = clip((x - mean) / (std + epsilon), -clip, clip)

        Operates on the last dimension to support batched inputs [B, dim].

        @param x torch.Tensor Input tensor of shape [..., dim].
        @return torch.Tensor Normalised tensor of the same shape.
        """
        device = x.device
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std + self.epsilon, dtype=torch.float32, device=device)

        normalised = (x - mean) / std
        return torch.clamp(normalised, -self.clip, self.clip)

    def state_dict(self) -> dict:
        """
        @brief Serialise normaliser state for checkpointing.

        @return dict Contains count, mean, and M2.
        """
        return {
            'count': self.count,
            'mean': self.mean.copy(),
            'M2': self.M2.copy(),
        }

    def load_state_dict(self, state: dict):
        """
        @brief Restore normaliser state from a checkpoint.

        @param state dict Previously saved state from state_dict().
        """
        self.count = state['count']
        self.mean = state['mean'].copy()
        self.M2 = state['M2'].copy()

class ObservationNormaliser:
    """
    @class ObservationNormaliser
    @brief Wraps per-field normalisers for the full multimodal observation dict.

    @details
    Maintains separate RunningNormaliser instances for ego_state and
    risk_features. Camera and LiDAR are not normalised here since camera
    is already scaled to [0, 1] via /255 in preprocess_obs, and LiDAR
    is a binary BEV grid.

    Usage:
        obs_normaliser = ObservationNormaliser(ego_state_dim=16, risk_feature_dim=61)

        # During training step:
        obs_normaliser.update(obs)           # update running stats
        obs_norm = obs_normaliser.normalise(obs_tensor)  # normalise tensors
    """

    def __init__(self, ego_state_dim: int, risk_feature_dim: int):
        """
        @brief Initialise normalisers for ego state and risk features.

        @param ego_state_dim int Dimension of the ego state vector.
        @param risk_feature_dim int Dimension of the risk feature vector.
        """
        self.ego_normaliser = RunningNormaliser(shape=(ego_state_dim,))
        self.risk_normaliser = RunningNormaliser(shape=(risk_feature_dim,))

    def update(self, obs: dict):
        """
        @brief Update running stats from a raw numpy observation dict.

        @details
        Should be called with the raw numpy observation before preprocessing,
        or with the squeezed numpy values after preprocessing.

        @param obs dict Raw observation with 'ego_state' and 'risk_features' keys.
        """
        if 'ego_state' in obs:
            self.ego_normaliser.update(np.asarray(obs['ego_state']).flatten())
        if 'risk_features' in obs:
            self.risk_normaliser.update(np.asarray(obs['risk_features']).flatten())

    def normalise(self, obs: dict) -> dict:
        """
        @brief Normalise ego_state and risk_features tensors in an observation dict.

        @param obs dict Observation dict with batched torch tensors [1, dim].
        @return dict Same dict with ego_state and risk_features normalised in-place.
        """
        normalised = dict(obs) # shallow copy - don't modify the original

        if 'ego_state' in obs:
            normalised['ego_state'] = self.ego_normaliser.normalise(obs['ego_state'])
        if 'risk_features' in obs:
            normalised['risk_features'] = self.risk_normaliser.normalise(obs['risk_features'])

        return normalised

    def state_dict(self) -> dict:
        """@brief Serialise both normalisers for checkpointing."""
        return {
            'ego': self.ego_normaliser.state_dict(),
            'risk': self.risk_normaliser.state_dict(),
        }

    def load_state_dict(self, state: dict):
        """@brief Restore both normalisers from a checkpoint."""
        self.ego_normaliser.load_state_dict(state['ego'])
        self.risk_normaliser.load_state_dict(state['risk'])