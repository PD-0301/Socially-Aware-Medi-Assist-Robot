import cv2
import numpy as np

# === Load Calibration Results ===
K = np.array([
    [383.23896058, 0.0, 345.9891869],
    [0.0, 384.85163322, 223.7339771],
    [0.0, 0.0, 1.0]
])
D = np.array([-0.17272455, 0.05733935, -0.00293362, -0.00081243, -0.04610971])

# === Open Video Feed ===
cap = cv2.VideoCapture(2)
ret, frame = cap.read()
h, w = frame.shape[:2]

# === Compute New Camera Matrix & Map ===
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), 5)

# === Display Live Undistorted Feed ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    
    # Optional: Show original and corrected side by side
    combined = np.hstack((frame, undistorted))
    cv2.imshow('Original (Left) vs Undistorted (Right)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
