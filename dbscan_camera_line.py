import serial
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter

# === CONFIGURATION ===
RADAR_PORT = '/dev/ttyACM1'
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

CIRCLE_RADIUS = 0.8
CENTER_LINE_X = 0
FOV_LEFT = -60
FOV_RIGHT = 60
DIST_MIN = 0
DIST_MAX = 10.0
VEL_MIN = 1

# === Kalman Tracker Class ===
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
        self.kf.P *= 10
        self.kf.Q = np.eye(4) * 0.05
        self.kf.R = np.eye(2) * 0.3

    def update(self, position):
        self.kf.update(np.array(position))

    def predict(self):
        self.kf.predict()
        return self.kf.x[0, 0], self.kf.x[1, 0]

# === Radar Serial Setup ===
radar_ser = serial.Serial(RADAR_PORT, BAUD, timeout=0.5)

# === Arducam Webcam Setup ===
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access Arducam on /dev/video2")

# === Matplotlib Setup ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_xlim(-8, 8)
ax1.set_ylim(0, 16)
ax1.set_title("Radar Clusters + Kalman")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
scatter = ax1.scatter([], [], c='red')
ax1.plot(0, 0, marker='o', color='black', markersize=6)
ax1.text(0, 0.5, "Radar", color='black', fontsize=9, ha='center')

theta_left = math.radians(FOV_LEFT)
theta_right = math.radians(FOV_RIGHT)
x_left = DIST_MAX * math.sin(theta_left)
y_left = DIST_MAX * math.cos(theta_left)
x_right = DIST_MAX * math.sin(theta_right)
y_right = DIST_MAX * math.cos(theta_right)
ax1.plot([0, x_left], [0, y_left], '--', color='blue')
ax1.plot([0, x_right], [0, y_right], '--', color='blue')
ax1.fill([0, x_left, x_right], [0, y_left, y_right], color='blue', alpha=0.1, label="Radar FoV")
ax1.axvline(x=CENTER_LINE_X, color='green', linestyle='-.', label="Center Line")
ax1.legend(loc='upper right')

img_plot = ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax2.axis('off')
ax2.set_title("Arducam Feed")

# === Radar Data Parser ===
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
                    theta = math.degrees(math.atan2(x, y))
                    dist = math.hypot(x, y)
                    if FOV_LEFT <= theta <= FOV_RIGHT and DIST_MIN <= dist <= DIST_MAX and abs(v) >= VEL_MIN:
                        points.append((x, y))
        except Exception as e:
            print(f"[ERROR] Radar parsing: {e}")
    return points

# === Kalman Trackers ===
trackers = []
prev_time = time.time()

import gc  # Garbage collection to reduce frame drop

try:
    while True:
        radar_points = get_radar_points()
        new_centroids = []

        if radar_points:
            points = np.array(radar_points)
            clustering = DBSCAN(eps=0.6, min_samples=4).fit(points)
            labels = clustering.labels_
            for label in set(labels):
                if label == -1:
                    continue
                cluster = points[labels == label]
                if len(cluster) < 3:
                    continue
                cx = np.mean(cluster[:, 0])
                cy = np.mean(cluster[:, 1])
                new_centroids.append((cx, cy))

        if len(new_centroids) != len(trackers):
            trackers = [RadarKalmanTracker(pos) for pos in new_centroids]
        else:
            for tracker, centroid in zip(trackers, new_centroids):
                tracker.update(centroid)

        predicted_positions = [tracker.predict() for tracker in trackers]
        scatter.set_offsets(np.array(predicted_positions) if predicted_positions else np.empty((0, 2)))

        for patch in ax1.patches[:]:
            patch.remove()

        stop_flag = False
        for x, y in predicted_positions:
            circle = plt.Circle((x, y), CIRCLE_RADIUS, color='orange', fill=False, linestyle='--')
            ax1.add_patch(circle)
            if abs(x - CENTER_LINE_X) <= CIRCLE_RADIUS:
                stop_flag = True

        print("🚫 STOP: Object intersecting center line!" if stop_flag else "✅ CLEAR")

        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_plot.set_data(frame_rgb)

        curr_time = time.time()
        fps = 1.0 / max(0.01, curr_time - prev_time)
        prev_time = curr_time
        ax1.text(-7.5, 15.5, f"FPS: {fps:.2f}", fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Frame drop mitigation
        gc.collect()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("🔻 Exiting...")
    cap.release()
    radar_ser.close()
    plt.ioff()
    plt.show()
