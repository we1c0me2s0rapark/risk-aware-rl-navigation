import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEncoder(nn.Module):
    """
    @class ObservationEncoder
    @brief Encodes multi-modal observations into a latent vector.

    This module takes camera images, LiDAR data, ego state, and optional risk features,
    processes them through separate branches (CNNs for visual and LiDAR inputs, MLP for
    ego/risk features), and projects the concatenated features into a latent representation.

    @note LiDAR and risk features are optional and can be enabled or disabled via flags.
    """

    def __init__(
        self,
        camera_shape=(3, 84, 84),
        lidar_shape=(1, 64, 64),
        ego_state_dim=6,
        risk_feature_dim=3,
        latent_dim=256,
        use_lidar=True,
        use_risk=True
    ):
        """
        @brief Initialises the ObservationEncoder module.

        @param camera_shape Tuple[int, int, int] Shape of camera input (C, H, W)
        @param lidar_shape Tuple[int, int, int] Shape of LiDAR input (C, H, W)
        @param ego_state_dim int Dimension of ego state features
        @param risk_feature_dim int Dimension of risk features
        @param latent_dim int Dimension of output latent vector
        @param use_lidar bool Whether to include LiDAR input
        @param use_risk bool Whether to include risk features
        """
        super().__init__()

        self.use_lidar = use_lidar
        self.use_risk = use_risk

        # ---------------- Camera CNN branch ----------------
        c, h, w = camera_shape
        self.camera_cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened size dynamically
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy = torch.zeros(1, c, h, w, device=device)
            camera_flat_size = self.camera_cnn(dummy).shape[1]

        self.camera_fc = nn.Sequential(
            nn.Linear(camera_flat_size, 128),
            nn.ReLU()
        )

        # ---------------- LiDAR CNN branch ----------------
        if self.use_lidar:
            lc, lh, lw = lidar_shape
            self.lidar_cnn = nn.Sequential(
                nn.Conv2d(lc, 16, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten()
            )

            with torch.no_grad():
                dummy = torch.zeros(1, lc, lh, lw, device=device)
                lidar_flat_size = self.lidar_cnn(dummy).shape[1]

            self.lidar_fc = nn.Sequential(
                nn.Linear(lidar_flat_size, 64),
                nn.ReLU()
            )

        # ---------------- MLP for ego + risk features ----------------
        mlp_input_dim = ego_state_dim + (risk_feature_dim if self.use_risk else 0)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # ---------------- Final latent projection ----------------
        latent_input_dim = 128 + (64 if self.use_lidar else 0) + 64
        self.fc_latent = nn.Sequential(
            nn.Linear(latent_input_dim, latent_dim),
            nn.ReLU()
        )

    def forward(
            self,
            camera: torch.Tensor,
            ego_state: torch.Tensor,
            lidar: torch.Tensor = None,
            risk_features: torch.Tensor = None
        ) -> torch.Tensor:
        
        # 1. Camera Branch (e.g. [B, 128])
        cam_feat = self.camera_fc(self.camera_cnn(camera))
        feats = [cam_feat]

        # 2. LiDAR Branch (e.g. [B, 64])
        if self.use_lidar and lidar is not None:
            lidar_feat = self.lidar_fc(self.lidar_cnn(lidar))
            feats.append(lidar_feat)

        # 3. MLP Branch (Ego + Risk) 
        # CRITICAL: Force flatten to [Batch, Features] to avoid dimension creep
        ego_state = ego_state.view(ego_state.size(0), -1)
        
        if self.use_risk and risk_features is not None:
            risk_features = risk_features.view(risk_features.size(0), -1)
            mlp_input = torch.cat([ego_state, risk_features], dim=-1)
        else:
            mlp_input = ego_state
            
        # Push through MLP [B, 9] -> [B, 64]
        mlp_feat = self.mlp(mlp_input)
        feats.append(mlp_feat)

        # 4. Final Projection
        combined = torch.cat(feats, dim=-1)
        return self.fc_latent(combined)