import cv2
import numpy as np
import os

# === CONFIGURATION ===
CHECKERBOARD = (7, 7)  # Inner corners
SQUARE_SIZE = 0.025  # In meters
SAVE_PATH = "cameras_calibration_output.npz"

# Termination criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), ..., (6,6,0) scaled by square size
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

cap = cv2.VideoCapture(2)
print("[INFO] Press 's' to save a valid frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret_cb:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret_cb)
        cv2.imshow("Checkerboard", vis)
    else:
        cv2.imshow("Checkerboard", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('s') and ret_cb:
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"[INFO] Frame saved ({len(objpoints)} total)")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Calibrate ===
if len(objpoints) >= 5:
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("\n✅ Calibration successful.")
    print("Reprojection error:", ret)
    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients (D):\n", D)

    np.savez(SAVE_PATH, K=K, D=D, rvecs=rvecs, tvecs=tvecs)
    print(f"[INFO] Saved calibration to {SAVE_PATH}")
else:
    print("❌ Not enough valid captures for calibration.")
