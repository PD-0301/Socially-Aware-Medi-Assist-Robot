import serial
import struct
import math
import numpy as np
import cv2
import time

# === CONFIGURATION ===
RADAR_PORT = '/dev/ttyACM1'
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Camera Intrinsics ===
K = np.array([[386.59464185, 0., 385.10781334],
              [0., 379.00255236, 274.90390815],
              [0., 0., 1.]])

# === Radar to Camera Transformation Matrix ===
T_radar_to_cam = np.array([[1., 0.,  0.,   0.],
                           [0., 0., -1., -0.048],
                           [0., 1.,  0., -0.03],
                           [0., 0.,  0.,  1.]])

# === Webcam Setup ===
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

# === Radar Setup ===
radar_ser = serial.Serial(RADAR_PORT, BAUD, timeout=0.5)

def get_radar_points():
    data = radar_ser.read(8192)
    idx = data.find(MAGIC_WORD)
    points = []
    if idx != -1 and len(data) > idx + 48:
        try:
            pkt_len = struct.unpack('<I', data[idx+12:idx+16])[0]
            num_obj = struct.unpack('<H', data[idx+28:idx+30])[0]
            obj_start = idx + 48
            for i in range(num_obj):
                start = obj_start + i * 16
                if len(data) >= start + 16:
                    x, y, z, v = struct.unpack('<ffff', data[start:start+16])
                    if y<0.5:
                        points.append([x, y, z])
        except Exception as e:
            print(f"[ERROR] Radar parsing: {e}")
    return np.array(points)

# === Main Loop ===
try:
    while True:
        radar_points = get_radar_points()

        ret, frame = cap.read()
        if not ret:
            continue

        if radar_points.shape[0] > 0:
            # Convert to homogeneous and transform to camera frame
            ones = np.ones((radar_points.shape[0], 1))
            radar_hom = np.hstack((radar_points, ones))
            cam_points = (T_radar_to_cam @ radar_hom.T).T[:, :3]

            # Filter in front of camera
            cam_points = cam_points[cam_points[:, 2] > 0]

            # Project to image plane
            pixel_coords = (K @ cam_points.T).T
            pixels = pixel_coords[:, :2] / pixel_coords[:, 2:]

            for (u, v), (x, y, z) in zip(pixels.astype(int), cam_points):
                if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                    distance = np.linalg.norm([x, y, z])
                    cv2.circle(frame, (u, v), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"{distance:.2f}m", (u+5, v-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Radar Points on Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    radar_ser.close()
    cv2.destroyAllWindows()
