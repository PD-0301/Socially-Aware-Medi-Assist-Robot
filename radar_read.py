import serial
import struct
import numpy as np
import matplotlib.pyplot as plt

# === Radar Serial Port Setup ===
data_serial = serial.Serial('COM9', baudrate=921600, timeout=0.5)
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# === Visualization Setup ===
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(0, 4)  # forward only
ax.set_title("AWR1843BOOST Radar - Point Cloud View")
ax.set_xlabel("X (Left-Right)")
ax.set_ylabel("Y (Forward)")

def find_magic_word(data):
    return data.find(MAGIC_WORD)

print("[INFO] Starting radar stream...")
while True:
    try:
        byte_data = data_serial.read(4096)
        idx = find_magic_word(byte_data)

        if idx != -1 and len(byte_data) > idx + 48:
            pkt_len = struct.unpack('<I', byte_data[idx+12:idx+16])[0]
            if len(byte_data) >= idx + pkt_len:
                num_obj = struct.unpack('<H', byte_data[idx+28:idx+30])[0]
                print(f"[DEBUG] Objects Detected: {num_obj}")
                obj_start = idx + 48

                points = []
                for i in range(num_obj):
                    try:
                        x, y, z, v = struct.unpack('<ffff', byte_data[obj_start+i*16:obj_start+(i+1)*16])
                        points.append([x, y, z])
                    except:
                        pass

                np_points = np.array(points)

                # === PLOT ===
                ax.clear()
                ax.set_xlim(-2, 2)
                ax.set_ylim(0, 4)
                ax.set_title("AWR1843BOOST Radar - Point Cloud View")
                ax.set_xlabel("X (Left-Right)")
                ax.set_ylabel("Y (Forward)")

                if len(points) > 0:
                    ax.scatter(np_points[:, 0], np_points[:, 1], c='blue', s=20)
                else:
                    ax.text(0, 2, "No points", ha='center')

                plt.pause(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        break
    except Exception as e:
        print(f"[ERROR] {e}")

data_serial.close()
plt.close()
