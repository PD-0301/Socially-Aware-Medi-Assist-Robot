import numpy as np

# Rotation
R = np.array([
    [0,  -1,  0],
    [0,  0, -1],
    [0, 0,  0]
])

# Translation
T = np.array([[0.0],
              [-0.05],
              [-0.03]])

# 4x4 Transformation Matrix
T_radar_to_cam = np.eye(4)
T_radar_to_cam[:3, :3] = R
T_radar_to_cam[:3, 3:] = T

print("Radar to Camera Transformation Matrix:\n", T_radar_to_cam)
