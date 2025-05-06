import numpy as np
import cv2
import csv

CSV_FILE = 'radar_log.csv'

radar_points = []
camera_points = []

with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for idx, row in enumerate(reader):
        if len(row) < 6:
            print(f"[WARN] Skipping line {idx+2}: Not enough values ({len(row)})")
            continue
        try:
            radar_x, radar_y, radar_z = map(float, row[:3])
            cam_x, cam_y, cam_z = map(float, row[3:6])
            radar_points.append([radar_x, radar_y, radar_z])
            camera_points.append([cam_x, cam_y, cam_z])
        except Exception as e:
            print(f"[ERROR] Line {idx+2}: {e}")

radar_points = np.array(radar_points)
camera_points = np.array(camera_points)

retval, affine_matrix, inliers = cv2.estimateAffine3D(radar_points, camera_points)

if retval:
    R = affine_matrix[:, :3]
    T = affine_matrix[:, 3].reshape(3, 1)
    print("\n✅ Extrinsic Calibration Successful!")
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (T):\n", T)
    np.savetxt("R_matrix.csv", R, delimiter=',')
    np.savetxt("T_vector.csv", T, delimiter=',')
else:
    print("❌ Failed to estimate extrinsic transformation. Check if enough matched points.")
