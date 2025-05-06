import serial
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# === CONFIGURATION ===
RADAR_PORT = '/dev/ttyACM1'
BAUD = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

FOV_LEFT = -60
FOV_RIGHT = 60
DIST_MIN = 0.0
DIST_MAX = 10.0
VEL_MIN = 0.0

# === Serial Setup ===
radar_ser = serial.Serial(RADAR_PORT, BAUD, timeout=0.5)

# === Radar Point Reader ===
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
                    if not (FOV_LEFT <= theta <= FOV_RIGHT):
                        continue
                    if dist < DIST_MIN or dist > DIST_MAX:
                        continue
                    if abs(v) < VEL_MIN:
                        continue
                    points.append((x, y))
        except Exception as e:
            print(f"[ERROR] Radar parsing: {e}")
    return np.array(points)

# === Main Logic ===
print("[INFO] Collecting radar data for DBSCAN tuning...")
all_points = []

# Collect frames until we have at least 50 points
while len(all_points) < 50:
    new_points = get_radar_points()
    if len(new_points) > 0:
        all_points.extend(new_points)

X = np.array(all_points)
print(f"[INFO] Total radar points collected: {len(X)}")

# === K-Distance Plot ===
min_samples = 4
k = min_samples - 1
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, k-1])

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(k_distances, marker='o')
plt.ylabel(f"{k}th Nearest Neighbor Distance")
plt.xlabel("Points sorted by distance")
plt.title(f"DBSCAN k-distance Graph (min_samples={min_samples})")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cleanup
radar_ser.close()
