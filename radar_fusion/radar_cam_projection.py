import numpy as np
import cv2
import os
import csv
from glob import glob

CAM_DIR = 'calib_data'
RADAR_LOG = 'radar_log.csv'
OUT_FILE = 'extrinsics_radar_to_camera.npz'
POINT_INDEX = 24  # center point in 7x7 checkerboard
TIME_WINDOW = 1.0  # seconds

# Load radar log
timestamps = []
radar_pts = []
with open(RADAR_LOG, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        ts, x, y, z = map(float, row)
        timestamps.append(ts)
        radar_pts.append(np.array([x, y, z]))

# Load camera poses
radar_matches = []
camera_matches = []
pose_files = sorted(glob(os.path.join(CAM_DIR, 'camera_pose_*.npz')))
print(f"[INFO] Found {len(pose_files)} camera pose files.")

for path in pose_files:
    data = np.load(path)
    cam_time = data['timestamp'].item()
    cam_pts = data['cam_points']
    center_cam = cam_pts[POINT_INDEX]  # 3D point in camera frame

    # Match radar timestamp
    close_idxs = [i for i, t in enumerate(timestamps) if abs(t - cam_time) <= TIME_WINDOW]
    if not close_idxs:
        print(f"[WARN] No radar match for {path}")
        continue

    # Choose radar point closest to chessboard center
    radar_subset = np.array([radar_pts[i] for i in close_idxs])
    best_radar = min(radar_subset, key=lambda pt: np.linalg.norm(pt - center_cam))

    camera_matches.append(center_cam)
    radar_matches.append(best_radar)
    print(f"[✓] Paired {os.path.basename(path)}")

# Estimate transformation
if len(camera_matches) >= 4:
    radar_np = np.array(radar_matches)
    camera_np = np.array(camera_matches)

    retval, affine, inliers = cv2.estimateAffine3D(radar_np, camera_np)
    if retval:
        R = affine[:, :3]
        T = affine[:, 3].reshape(3, 1)
        np.savez(OUT_FILE, R=R, T=T)
        print(f"\n✅ Calibration complete!")
        print("Rotation (R):\n", R)
        print("Translation (T):\n", T)
        print(f"💾 Saved to {OUT_FILE}")
    else:
        print("❌ cv2.estimateAffine3D failed.")
else:
    print("❌ Not enough matched pairs for calibration.")
