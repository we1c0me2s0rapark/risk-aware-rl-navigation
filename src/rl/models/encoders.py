import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEncoder(nn.Module):
    """
    @brief Encodes multi-modal observations (camera, LiDAR, ego state, risk features) into a latent vector.
    
    @details
    This module processes camera images through a CNN, optional LiDAR input through a separate CNN,
    and ego-state and risk features through an MLP, before combining them into a fixed-size latent representation.
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
        @brief Initialise the ObservationEncoder.
        @param camera_shape Tuple specifying the camera input shape (C, H, W).
        @param lidar_shape Tuple specifying the LiDAR input shape (C, H, W).
        @param ego_state_dim Dimension of the ego state vector.
        @param risk_feature_dim Dimension of optional risk feature vector.
        @param latent_dim Dimension of the final latent embedding.
        @param use_lidar Whether to use LiDAR input.
        @param use_risk Whether to include risk features.
        """

        super().__init__()

        self.use_lidar = use_lidar
        self.use_risk = use_risk

        # Camera CNN branch
        c, h, w = camera_shape
        self.camera_cnn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernal_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernal_size=4, stride=2), # downsample spatially
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), # feature extraction
            nn.ReLU(),
            nn.Flatten(), # 64 × 7 × 7 = 3136 features
            nn.Linear(64 * 7 * 7, 128), # project 3136 → 128; transform raw features into a compact, task-relevant representation
            nn.ReLU()
        )
        
        # LiDAR CNN branch
        if self.use_lidar:
            lc, lh, lw = lidar_shape
            self.lidar_cnn = nn.Sequential(
                nn.Conv2d(in_channels=lc, out_channels=16, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 14 * 14, 64), # reduce to lower-dimensional vector
                nn.ReLU()
            )
        
        # MLP for ego + risk features
        input_dim = ego_state_dim + (risk_feature_dim if self.use_risk else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Final latent projection
        latent_input_dim = 128 + (64 if self.use_lidar else 0) + 64
        self.fc_latent = nn.Sequential(
            nn.Linear(latent_input_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, camera, ego_state, lidar=None, risk_features=None):
        """
        @brief Forward pass through the encoder.

        @param camera Camera image tensor of shape (B, C, H, W).
        @param ego_state Ego state tensor of shape (B, ego_state_dim).
        @param lidar Optional LiDAR tensor of shape (B, C, H, W).
        @param risk_features Optional risk features tensor of shape (B, risk_feature_dim).

        @return Latent embedding tensor of shape (B, latent_dim).
        """

        # Encode camera observations
        camera_feat = self.camera_cnn(camera)
        feats = [camera_feat]

        # Encode LiDAR observations if provided
        if self.use_lidar and lidar is not None:
            lidar_feat = self.lidar_cnn(lidar)
            feats.append(lidar_feat)
        
        # Prepare MLP input: ego state and optional risk features
        mlp_input = ego_state
        if self.use_risk and risk_features is not None:
            mlp_input = torch.cat([ego_state, risk_features], dim=-1)
        
        # Encode ego + risk features
        mlp_feat = self.mlp(mlp_input)
        feats.append(mlp_feat)

        # Concatenate all features and project to latent space
        latent = torch.cat(feats, dim=-1)
        latent = self.fc_latent(latent)

        return latent