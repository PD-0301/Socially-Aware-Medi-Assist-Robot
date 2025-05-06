import cv2
import numpy as np
import os

# === CONFIG ===
CHECKERBOARD = (7, 7)  # inner corners = squares - 1
SQUARE_SIZE = 0.025  # meters
SAVE_DIR = "calib_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load camera intrinsics ===
data = np.load('camera_calibration_output.npz')
K = data['K']
D = data['D']

# === 3D world points of checkerboard
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.indices(CHECKERBOARD).T.reshape(-1, 2)
objp *= SQUARE_SIZE

# === Start camera
cap = cv2.VideoCapture(2)
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    vis = frame.copy()
    if ret_cb:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret_cb)

    cv2.imshow("Calibration", vis)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('s') and ret_cb:
        # Solve pose
        retval, rvec, tvec = cv2.solvePnP(objp, corners2, K, D)
        R, _ = cv2.Rodrigues(rvec)
        cam_points = (R @ objp.T + tvec).T  # shape: (N, 3)

        np.savez(os.path.join(SAVE_DIR, f'camera_pose_{i}.npz'),
                 cam_points=cam_points)
        print(f"[INFO] Saved camera pose for frame {i}")
        i += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
