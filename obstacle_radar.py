import serial
import struct
import numpy as np

# Radar setup
data_serial = serial.Serial('COM9', baudrate=921600, timeout=0.5)
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# Distance threshold for obstacle detection
THRESHOLD_DISTANCE = 0.6  # meters

def find_magic_word(data):
    return data.find(MAGIC_WORD)

def detect_obstacle(points):
    # Filter points within threshold distance
    close_pts = [p for p in points if np.linalg.norm([p[0], p[1]]) < THRESHOLD_DISTANCE]
    if not close_pts:
        return "✅ Path Clear"

    # Determine average X position of obstacles
    avg_x = np.mean([p[0] for p in close_pts])

    if avg_x > 0.2:
        return "⬅️ Obstacle on Right - Turn Left"
    elif avg_x < -0.2:
        return "➡️ Obstacle on Left - Turn Right"
    else:
        return "🛑 Obstacle Ahead - Stop or Turn"

print("[INFO] Radar Obstacle Detection Active...")
while True:
    try:
        byte_data = data_serial.read(4096)
        idx = find_magic_word(byte_data)

        if idx != -1 and len(byte_data) > idx + 48:
            pkt_len = struct.unpack('<I', byte_data[idx+12:idx+16])[0]
            if len(byte_data) >= idx + pkt_len:
                num_obj = struct.unpack('<H', byte_data[idx+28:idx+30])[0]
                obj_start = idx + 48
                points = []

                for i in range(num_obj):
                    try:
                        x, y, z, v = struct.unpack('<ffff', byte_data[obj_start+i*16:obj_start+(i+1)*16])
                        points.append([x, y])
                    except:
                        continue

                msg = detect_obstacle(points)
                print(f"[MSG] {msg}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        break
    except Exception as e:
        print(f"[ERROR] {e}")

data_serial.close()
