# auto_calibrate_radar_camera.py

import os
import csv
import numpy as np
import cv2
from glob import glob

CAM_DIR = 'calib_data'
RADAR_LOG = 'radar_log.csv'
OUTPUT_FILE = 'extrinsics_radar_to_camera.npz'
TIME_WINDOW = 15  # seconds
POINT_INDEX = 24   # central checkerboard point

# Load radar log data
radar_timestamps = []
radar_points = []

with open(RADAR_LOG, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        t, x, y, z = map(float, row)
        radar_timestamps.append(t)
        radar_points.append(np.array([x, y, z]))

# Match each camera file with nearest radar point in TIME_WINDOW
camera_points = []
radar_matches = []

pose_files = sorted(glob(os.path.join(CAM_DIR, 'camera_pose_*.npz')))
print(f"[INFO] Found {len(pose_files)} camera pose files")

for pose_file in pose_files:
    data = np.load(pose_file)
    if 'timestamp' not in data:
        print(f"[WARN] Missing timestamp in {pose_file}")
        continue
    cam_time = data['timestamp'].item()
    cam_point = data['cam_points'][POINT_INDEX]

    # Get radar points within time window
    candidates = [radar_points[i] for i, t in enumerate(radar_timestamps) if 0 <= t - cam_time <= TIME_WINDOW]
    if not candidates:
        print(f"[WARN] No radar match for {pose_file}")
        continue

    best_match = min(candidates, key=lambda r: np.linalg.norm(r - cam_point))
    camera_points.append(cam_point)
    radar_matches.append(best_match)
    print(f"[✓] Paired {pose_file} with radar point {best_match}")

# Estimate transformation
if len(camera_points) >= 4:
    camera_points = np.array(camera_points)
    radar_matches = np.array(radar_matches)
    retval, affine, inliers = cv2.estimateAffine3D(radar_matches, camera_points)
    if retval:
        R = affine[:, :3]
        T = affine[:, 3]
        np.savez(OUTPUT_FILE, R=R, T=T)
        print("\n✅ Calibration Success!")
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)
        print(f"💾 Saved calibration to {OUTPUT_FILE}")
    else:
        print("❌ Failed to compute transformation using estimateAffine3D.")
else:
    print("❌ Not enough matched pairs for calibration.")
