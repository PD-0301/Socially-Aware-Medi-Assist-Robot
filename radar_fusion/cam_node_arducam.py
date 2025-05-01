import serial
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter

# === Radar serial config ===
PORT = '/dev/ttyACM1'
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Radar filtering parameters ===
FOV_LEFT = -60
FOV_RIGHT = 60
DIST_MIN = 0.2
DIST_MAX = 10.0
VEL_MIN = 0.05

# === Kalman Tracker ===
class RadarKalmanTracker:
    def __init__(self, initial_position):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 0.1
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        x, y = initial_position
        self.kf.x = np.array([[x], [y], [0], [0]])
        self.kf.P *= 20
        self.kf.Q = np.eye(4) * 0.1
        self.kf.R = np.eye(2) * 0.4

    def update(self, position):
        self.kf.update(np.array(position))

    def predict(self):
        self.kf.predict()
        return self.kf.x[0, 0], self.kf.x[1, 0]

# === Radar serial setup ===
ser = serial.Serial(PORT, BAUD, timeout=0.5)

# === Webcam setup ===
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam")

# === Matplotlib setup ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.set_xlim(-8, 8)
ax1.set_ylim(0, 16)
ax1.set_title("Radar Clusters (DBSCAN + Kalman)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
scatter = ax1.scatter([], [])

img_plot = ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax2.axis('off')
ax2.set_title("Webcam Feed")

# === Read + filter radar points ===
def get_radar_points():
    data = ser.read(8192)
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
                    theta = math.degrees(math.atan2(x, y))
                    dist = math.hypot(x, y)
                    if not (FOV_LEFT <= theta <= FOV_RIGHT):
                        continue
                    if dist < DIST_MIN or dist > DIST_MAX:
                        continue
                    if abs(v) < VEL_MIN:
                        continue
                    points.append((x, y))
        except:
            pass
    return points

# === Kalman trackers per cluster ===
trackers = []

# === Main loop ===
try:
    while True:
        raw_points = get_radar_points()
        new_centroids = []

        if raw_points:
            points = np.array(raw_points)

            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(points)
            labels = clustering.labels_

            for label in set(labels):
                if label == -1:
                    continue
                cluster = points[labels == label]
                if len(cluster) < 3:
                    continue

                # Use cluster centroid
                cx = np.mean(cluster[:, 0])
                cy = np.mean(cluster[:, 1])
                new_centroids.append((cx, cy))

        # Track with Kalman
        if len(new_centroids) != len(trackers):
            trackers = [RadarKalmanTracker(pos) for pos in new_centroids]
        else:
            for tracker, pos in zip(trackers, new_centroids):
                tracker.update(pos)

        predicted = [tracker.predict() for tracker in trackers]

        if predicted:
            scatter.set_offsets(np.array(predicted))
        else:
            scatter.set_offsets(np.empty((0, 2)))

        # Webcam frame
        ret, frame = cap.read()
        if ret:
            img_plot.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Update
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
    cap.release()
    ser.close()
    plt.ioff()
    plt.show()
