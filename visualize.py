import serial
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# === Radar serial config ===
PORT = '/dev/ttyACM1'  # Replace with your radar's data port
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Radar filtering parameters ===
FOV_LEFT = -60         # degrees
FOV_RIGHT = 60         # degrees
DIST_MIN = 0.2         # meters
DIST_MAX = 10.0        # meters
VEL_MIN = 0.05         # m/s

# === Setup radar serial ===
ser = serial.Serial(PORT, BAUD, timeout=0.5)

# === Setup webcam ===
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam")

# === Matplotlib setup ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Radar plot
ax1.set_xlim(-8, 8)
ax1.set_ylim(0, 16)
ax1.set_title("Radar Points")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
scatter = ax1.scatter([], [])

# Webcam plot placeholder (match 1280x720, note shape is height x width)
img_plot = ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax2.axis('off')
ax2.set_title("Webcam Feed ()")

# === Radar frame parser ===
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

# === Main loop ===
try:
    while True:
        # Radar points
        radar_pts = get_radar_points()
        if radar_pts:
            radar_pts = np.array(radar_pts)
            scatter.set_offsets(radar_pts)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        # Webcam feed
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_plot.set_data(frame_rgb)

        # Refresh plots
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
    cap.release()
    ser.close()
    plt.ioff()
    plt.show()
