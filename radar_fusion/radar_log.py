# camera_checkerboard_capture.py (manual capture, one at a time)

import cv2
import numpy as np
import os
import time

CHECKERBOARD = (7, 7)
SQUARE_SIZE = 0.025
SAVE_DIR = "calib_data"
os.makedirs(SAVE_DIR, exist_ok=True)

data = np.load('camera_calibration_output.npz')
K, D = data['K'], data['D']

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.indices(CHECKERBOARD).T.reshape(-1, 2)
objp *= SQUARE_SIZE

cap = cv2.VideoCapture(2)
pose_id = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    vis = frame.copy()
    if ret_cb:
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ret_cb)

    cv2.imshow("Calibration Pose Capture", vis)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and ret_cb:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        _, rvec, tvec = cv2.solvePnP(objp, corners2, K, D)
        R, _ = cv2.Rodrigues(rvec)
        cam_points = (R @ objp.T + tvec).T
        ts = time.time()
        np.savez(os.path.join(SAVE_DIR, f'camera_pose_{pose_id}.npz'), cam_points=cam_points, timestamp=ts)
        print(f"[INFO] Saved camera pose {pose_id} at timestamp {ts}")
        pose_id += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
