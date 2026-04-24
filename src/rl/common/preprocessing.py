import numpy as np
import torch

def preprocess_obs(obs: dict, config: dict, device: torch.device) -> dict:
    """
    @brief Preprocess raw CARLA observations into batched torch tensors.

    @details
    Converts camera, LiDAR, ego state, and risk features from raw numpy
    arrays into normalised float32 tensors. Camera pixels are scaled to
    [0, 1]. LiDAR points are projected into a 2D bird's-eye-view grid.

    @param obs dict Raw observation dictionary from the CARLA environment.
    @param config dict Environment configuration specifying sensor properties.
    @param device torch.device Target device for the output tensors.
    @return dict Batched tensors with keys: camera, lidar, ego_state, risk_features.
    """
    
    cam_cfg = config['sensors']['camera']
    cam_res = cam_cfg['train_resolution']
    channels = cam_cfg['channels']
    camera = np.ascontiguousarray(
        np.array(obs["camera"], dtype=np.float32).reshape(channels, cam_res['y'], cam_res['x']) / 255.0
    )

    lidar_cfg = config['sensors']['lidar']
    lidar_res = lidar_cfg['train_resolution']
    lidar_range = lidar_cfg['range']
    pts = np.array(obs["lidar"], dtype=np.float32).reshape(-1, 3)
    grid_h, grid_w = lidar_res['y'], lidar_res['x']
    lidar_bev = np.zeros((grid_h, grid_w), dtype=np.float32)
    x_px = (((pts[:, 0] / lidar_range) + 1) / 2 * (grid_w - 1)).astype(np.int32)
    y_px = (((pts[:, 1] / lidar_range) + 1) / 2 * (grid_h - 1)).astype(np.int32)
    mask = (x_px >= 0) & (x_px < grid_w) & (y_px >= 0) & (y_px < grid_h)
    lidar_bev[y_px[mask], x_px[mask]] = 1.0
    lidar = np.ascontiguousarray(lidar_bev[np.newaxis, :, :])

    ego_state = np.ascontiguousarray(np.array(obs["ego_state"], dtype=np.float32).flatten())
    risk_features = np.ascontiguousarray(np.array(obs["risk_features"], dtype=np.float32).flatten())

    # from_numpy shares memory (no extra copy); non_blocking pipelines the H→D transfer
    # behind other CPU work instead of stalling here.
    def to_gpu(arr):
        return torch.from_numpy(arr).to(device, non_blocking=True)

    return {
        "camera":        to_gpu(camera).unsqueeze(0),
        "lidar":         to_gpu(lidar).unsqueeze(0),
        "ego_state":     to_gpu(ego_state).unsqueeze(0),
        "risk_features": to_gpu(risk_features).unsqueeze(0),
    }