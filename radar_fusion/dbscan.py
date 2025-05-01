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
PORT = '/dev/ttyACM1'  # Update if needed
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Radar filtering parameters ===
FOV_LEFT = -60
FOV_RIGHT = 60
DIST_MIN = 0
DIST_MAX = 10.0
VEL_MIN = 3

# === Kalman tracker class ===
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

# === Radar UART setup ===
ser = serial.Serial(PORT, BAUD, timeout=0.5)

# === Webcam setup ===
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError(" Cannot access webcam")

# === Matplotlib setup ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Radar view
ax1.set_xlim(-8, 8)
ax1.set_ylim(0, 16)
ax1.set_title("Radar Clusters (DBSCAN + Kalman)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
scatter = ax1.scatter([], [], c='red')

# Mark radar origin
ax1.plot(0, 0, marker='o', color='black', markersize=6)
ax1.text(0, 0.5, "Radar", color='black', fontsize=9, ha='center')

# Draw FoV cone
fov_range = DIST_MAX
theta_left = math.radians(FOV_LEFT)
theta_right = math.radians(FOV_RIGHT)

x_left = fov_range * math.sin(theta_left)
y_left = fov_range * math.cos(theta_left)
x_right = fov_range * math.sin(theta_right)
y_right = fov_range * math.cos(theta_right)

# FoV lines and shaded region
ax1.plot([0, x_left], [0, y_left], linestyle='--', color='blue')
ax1.plot([0, x_right], [0, y_right], linestyle='--', color='blue')
ax1.fill([0, x_left, x_right], [0, y_left, y_right], color='blue', alpha=0.1, label="Radar FoV")
ax1.legend(loc='upper right')

# Webcam feed plot
img_plot = ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax2.axis('off')
ax2.set_title("Webcam Feed")

# === Radar data parsing ===
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
        radar_points = get_radar_points()
        new_centroids = []

        if radar_points:
            points = np.array(radar_points)

            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(points)
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

        # Kalman update/predict per target
        if len(new_centroids) != len(trackers):
            trackers = [RadarKalmanTracker(pos) for pos in new_centroids]
        else:
            for tracker, centroid in zip(trackers, new_centroids):
                tracker.update(centroid)

        predicted_positions = [tracker.predict() for tracker in trackers]

        # Update radar scatter
        if predicted_positions:
            scatter.set_offsets(np.array(predicted_positions))
        else:
            scatter.set_offsets(np.empty((0, 2)))

        # Webcam frame
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_plot.set_data(frame_rgb)

        # Plot refresh
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n Exiting...")
    cap.release()
    ser.close()
    plt.ioff()
    plt.show()
